"""API connection registry — register + inspect external HTTP APIs.

Two entry points:
  - register_api(...)  : insert a row into `api_connections`
  - inspect_api(id)    : probe the endpoint, ask a small model to write a
                         usage_prompt, save it back to the row.

Authentication credentials live in the `secrets` table and are referenced by
name (auth_secret_ref). Inspection never logs the resolved secret.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any
from urllib.parse import urljoin

import requests

from infra.nocodb_client import NocodbClient
from tools._org import resolve_org_id

_log = logging.getLogger("integrations.api")

TABLE = "api_connections"
SECRETS_TABLE = "secrets"

INSPECT_PATHS = (
    "/openapi.json",
    "/swagger.json",
    "/swagger/v1/swagger.json",
    "/api-docs",
    "/v1/openapi.json",
    "/.well-known/openapi.json",
)
INSPECT_TIMEOUT_S = 10


# ---------- secrets ----------

def _resolve_secret(client: NocodbClient, org_id: int, name: str) -> str | None:
    if not name or SECRETS_TABLE not in client.tables:
        return None
    rows = client._get_paginated(SECRETS_TABLE, params={
        "where": f"(org_id,eq,{org_id})~and(name,eq,{name})",
        "limit": 1,
    })
    if not rows:
        return None
    return rows[0].get("value") or rows[0].get("value_encrypted")


def _auth_headers(auth_type: str, secret_value: str | None, extra: dict) -> dict[str, str]:
    headers: dict[str, str] = {}
    if not auth_type or auth_type == "none" or not secret_value:
        return headers
    if auth_type == "bearer":
        headers["Authorization"] = f"Bearer {secret_value}"
    elif auth_type == "basic":
        import base64
        username = extra.get("username", "")
        token = base64.b64encode(f"{username}:{secret_value}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"
    elif auth_type == "api_key_header":
        header_name = extra.get("header_name", "X-API-Key")
        headers[header_name] = secret_value
    return headers


def _auth_query(auth_type: str, secret_value: str | None, extra: dict) -> dict[str, str]:
    if auth_type == "api_key_query" and secret_value:
        return {extra.get("query_name", "api_key"): secret_value}
    return {}


# ---------- register ----------

def register_api(
    name: str,
    base_url: str,
    org_id: int = 1,
    auth_type: str = "none",
    auth_secret_ref: str = "",
    auth_extra_json: dict | None = None,
    default_headers_json: dict | None = None,
    openapi_url: str = "",
    description: str = "",
    allowed_methods: str = "GET",
    rate_limit_per_min: int = 60,
    timeout_seconds: int = 30,
) -> dict:
    """Insert a new api_connections row. Returns the inserted row."""
    client = NocodbClient()
    if TABLE not in client.tables:
        raise RuntimeError(f"{TABLE} table missing — see docs/new-tables.md")
    payload = {
        "org_id": resolve_org_id(org_id),
        "name": name.strip(),
        "base_url": base_url.rstrip("/"),
        "auth_type": auth_type,
        "auth_secret_ref": auth_secret_ref,
        "auth_extra_json": json.dumps(auth_extra_json or {}),
        "default_headers_json": json.dumps(default_headers_json or {}),
        "openapi_url": openapi_url,
        "description": description,
        "allowed_methods": allowed_methods,
        "rate_limit_per_min": rate_limit_per_min,
        "timeout_seconds": timeout_seconds,
        "verification_status": "unverified",
    }
    row = client._post(TABLE, payload)
    _log.info("api registered  name=%s id=%s", name, row.get("Id"))
    return row


# ---------- inspect ----------

def _probe(url: str, headers: dict, params: dict) -> dict:
    try:
        r = requests.get(url, headers=headers, params=params, timeout=INSPECT_TIMEOUT_S)
        body = r.text[:4000]
        ctype = r.headers.get("Content-Type", "")
        return {"url": url, "status": r.status_code, "content_type": ctype, "body": body}
    except Exception as e:
        return {"url": url, "error": type(e).__name__ + ": " + str(e)[:200]}


def _find_openapi(base_url: str, headers: dict, params: dict, hint: str) -> dict | None:
    candidates = [hint] if hint else []
    candidates += [urljoin(base_url + "/", p.lstrip("/")) for p in INSPECT_PATHS]
    for url in candidates:
        if not url:
            continue
        r = _probe(url, headers, params)
        if r.get("status") == 200 and "json" in (r.get("content_type") or ""):
            try:
                spec = json.loads(r["body"])
                if isinstance(spec, dict) and ("openapi" in spec or "swagger" in spec):
                    return {"url": url, "spec": spec}
            except Exception:
                pass
    return None


def _summarise_openapi(spec: dict, max_endpoints: int = 25) -> dict:
    info = spec.get("info") or {}
    paths = spec.get("paths") or {}
    endpoints: list[str] = []
    for path, methods in list(paths.items())[:max_endpoints]:
        if not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if method.lower() not in ("get", "post", "put", "patch", "delete"):
                continue
            summary = (op or {}).get("summary") or (op or {}).get("operationId") or ""
            endpoints.append(f"{method.upper()} {path} — {summary}".strip(" —"))
    return {
        "title": info.get("title", ""),
        "version": info.get("version", ""),
        "description": (info.get("description") or "")[:500],
        "endpoint_count": sum(
            1 for m in paths.values() if isinstance(m, dict) for k in m
            if k.lower() in ("get", "post", "put", "patch", "delete")
        ),
        "endpoints_sample": endpoints,
    }


_USAGE_SYSTEM = """You write concise usage notes for an HTTP API so an LLM agent can call it correctly.
Output PLAIN TEXT (no markdown headers). Keep under 400 words.
Cover: what the API does, auth pattern, base URL, 5–10 most useful endpoints with method+path+purpose, common params, and one full example call. Be terse."""


def _write_usage_prompt(connection: dict, inspection: dict) -> str:
    """Ask a small model to produce a usage_prompt from inspection data."""
    try:
        from infra.config import get_function_config, no_think_params
        from shared.model_pool import acquire_role
    except Exception:
        return _fallback_usage_prompt(connection, inspection)

    try:
        cfg = get_function_config("tool_planner")
        role = cfg.get("role", "t3_tool")
    except Exception:
        role = "t3_tool"

    user_payload = {
        "name": connection.get("name"),
        "base_url": connection.get("base_url"),
        "auth_type": connection.get("auth_type"),
        "description": connection.get("description"),
        "inspection": inspection,
    }

    try:
        with acquire_role(role, priority=True) as (url, model_id):
            if not url:
                return _fallback_usage_prompt(connection, inspection)
            r = requests.post(
                f"{url}/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": _USAGE_SYSTEM},
                        {"role": "user", "content": json.dumps(user_payload)[:12000]},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 700,
                    **no_think_params(),
                },
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        _log.warning("usage prompt model call failed — falling back", exc_info=True)
        return _fallback_usage_prompt(connection, inspection)


def _fallback_usage_prompt(connection: dict, inspection: dict) -> str:
    lines = [
        f"API: {connection.get('name')}",
        f"Base URL: {connection.get('base_url')}",
        f"Auth: {connection.get('auth_type') or 'none'}",
    ]
    if desc := connection.get("description"):
        lines.append(f"Purpose: {desc}")
    spec = (inspection.get("openapi") or {}).get("summary") or {}
    if spec:
        lines.append(f"Title: {spec.get('title')}  Version: {spec.get('version')}")
        if spec.get("description"):
            lines.append(spec["description"])
        if eps := spec.get("endpoints_sample"):
            lines.append("Endpoints:")
            lines.extend(f"  - {e}" for e in eps[:15])
    return "\n".join(lines)


def inspect_api(connection_id: int) -> dict:
    """Probe the API, build inspection summary, generate usage_prompt, persist."""
    client = NocodbClient()
    if TABLE not in client.tables:
        raise RuntimeError(f"{TABLE} table missing")
    rows = client._get_paginated(TABLE, params={"where": f"(Id,eq,{connection_id})", "limit": 1})
    if not rows:
        raise ValueError(f"connection {connection_id} not found")
    conn = rows[0]
    org_id = resolve_org_id(conn.get("org_id"))

    auth_extra = _safe_json(conn.get("auth_extra_json"), {})
    headers = _safe_json(conn.get("default_headers_json"), {})
    secret_value = _resolve_secret(client, org_id, conn.get("auth_secret_ref") or "")
    headers.update(_auth_headers(conn.get("auth_type") or "none", secret_value, auth_extra))
    params = _auth_query(conn.get("auth_type") or "none", secret_value, auth_extra)

    base = (conn.get("base_url") or "").rstrip("/")
    inspection: dict[str, Any] = {"probed_at": int(time.time())}

    inspection["root"] = _probe(base, headers, params)

    openapi = _find_openapi(base, headers, params, conn.get("openapi_url") or "")
    if openapi:
        inspection["openapi"] = {
            "url": openapi["url"],
            "summary": _summarise_openapi(openapi["spec"]),
        }

    usage_prompt = _write_usage_prompt(conn, inspection)

    root_status = inspection["root"].get("status")
    reachable = bool(root_status and root_status < 500)
    status = "verified" if reachable else "failed"

    update = {
        "usage_prompt": usage_prompt,
        "inspection_summary_json": json.dumps(inspection)[:60000],
        "verified_at": _iso_now(),
        "verification_status": status,
    }
    client._patch(TABLE, conn["Id"], update)

    _log.info("api inspected  id=%s status=%s openapi=%s", conn["Id"], status, bool(openapi))
    return {"connection_id": conn["Id"], "status": status, "usage_prompt": usage_prompt, "inspection": inspection}


# ---------- helpers ----------

def _safe_json(raw, default):
    if isinstance(raw, dict):
        return raw
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def _iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
