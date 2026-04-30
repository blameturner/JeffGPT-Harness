"""Connectors router — APIs, SMTP, and Secrets management.

Thin REST wrapper around the NocoDB tables and the registry helpers in
``tools/integrations``. Frontend Connectors tab calls these.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from infra.nocodb_client import NocodbClient
from tools._org import resolve_org_id
from tools.integrations import api_registry, smtp_registry

_log = logging.getLogger("main.connectors")
router = APIRouter(prefix="/connectors", tags=["connectors"])

API_TABLE = "api_connections"
SMTP_TABLE = "smtp_accounts"
SECRETS_TABLE = "secrets"


def _client() -> NocodbClient:
    return NocodbClient()


def _require_table(c: NocodbClient, table: str):
    if table not in c.tables:
        raise HTTPException(status_code=503, detail=f"{table} table missing — see docs/new-tables.md")


def _redact_secret(row: dict) -> dict:
    out = dict(row)
    val = out.pop("value", None) or out.pop("value_encrypted", None) or ""
    out["value_length"] = len(val) if isinstance(val, str) else 0
    out["value_masked"] = "•" * min(8, out["value_length"]) if out["value_length"] else ""
    return out


# ============================================================
# Unified connectors view (apis + smtp)
# ============================================================
#
# The Connectors UI tab shows one combined list across kinds. IDs are
# namespaced (`api-{N}`, `smtp-{N}`) so the per-row endpoints can
# dispatch back to the right table without ambiguity.

def _split_connector_id(connector_id: str) -> tuple[str, int]:
    if "-" not in connector_id:
        raise HTTPException(status_code=400, detail="connector id must be 'api-N' or 'smtp-N'")
    kind, _, raw = connector_id.partition("-")
    if kind not in ("api", "smtp"):
        raise HTTPException(status_code=400, detail=f"unknown connector kind '{kind}'")
    try:
        numeric = int(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="connector id suffix must be int")
    return kind, numeric


@router.get("")
def list_connectors(org_id: int = 1, limit: int = 200):
    """Unified list across `api_connections` + `smtp_accounts`.

    Response shape matches the Connectors UI: each row exposes a
    namespaced `id` (`api-N`/`smtp-N`), `kind`, and a normalised
    `status`. `last_call_at` and `error_count_24h` are placeholder
    nulls/zeros until a call-log table is added.
    """
    c = _client()
    org = resolve_org_id(org_id)
    out: list[dict] = []

    if API_TABLE in c.tables:
        rows = c._get_paginated(
            API_TABLE,
            params={"where": f"(org_id,eq,{org})", "limit": limit},
        )
        for r in rows:
            out.append({
                "id": f"api-{r['Id']}",
                "name": r.get("name"),
                "kind": "api",
                "status": r.get("verification_status") or "unverified",
                "last_call_at": r.get("verified_at"),
                "error_count_24h": 0,
            })

    if SMTP_TABLE in c.tables:
        rows = c._get_paginated(
            SMTP_TABLE,
            params={"where": f"(org_id,eq,{org})", "limit": limit},
        )
        for r in rows:
            out.append({
                "id": f"smtp-{r['Id']}",
                "name": r.get("name"),
                "kind": "smtp",
                "status": r.get("verification_status") or "unverified",
                "last_call_at": r.get("verified_at"),
                "error_count_24h": 0,
            })

    return {"connectors": out, "count": len(out)}


@router.get("/{connector_id}/calls")
def list_connector_calls(connector_id: str, limit: int = 50):
    """Recent call log for a connector.

    Currently returns an empty list — there is no call-log table yet.
    The shape (`{calls: [{id, ts, endpoint, duration_ms, status_code,
    ok, error}]}`) is fixed so the UI table renders.
    """
    _split_connector_id(connector_id)  # validate format
    return {"calls": [], "count": 0, "limit": limit}


@router.post("/{connector_id}/test")
def test_connector(connector_id: str):
    """Generic connector test dispatcher. Routes to inspect (api) or
    test_smtp (smtp) based on the namespaced id."""
    kind, numeric = _split_connector_id(connector_id)
    if kind == "api":
        try:
            return api_registry.inspect_api(numeric)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            _log.error("inspect failed", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    # smtp
    try:
        return smtp_registry.test_smtp(numeric)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        _log.error("smtp test failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# APIs
# ============================================================

class ApiCreate(BaseModel):
    name: str
    base_url: str
    org_id: int = 1
    description: str = ""
    auth_type: str = "none"
    auth_secret_ref: str = ""
    auth_extra_json: dict = Field(default_factory=dict)
    default_headers_json: dict = Field(default_factory=dict)
    default_query_json: dict = Field(default_factory=dict)
    openapi_url: str = ""
    allowed_methods: str = "GET"
    allowed_paths_regex: str = ""
    rate_limit_per_min: int = 60
    timeout_seconds: int = 30


class ApiTestCall(BaseModel):
    method: str = "GET"
    path: str = ""
    params: dict = Field(default_factory=dict)
    headers: dict = Field(default_factory=dict)
    body: Any = None


@router.get("/apis")
def list_apis(org_id: int = 1, q: str = "", status: str = "", limit: int = 200):
    c = _client()
    _require_table(c, API_TABLE)
    where_parts = [f"(org_id,eq,{resolve_org_id(org_id)})"]
    if status:
        where_parts.append(f"(verification_status,eq,{status})")
    if q:
        where_parts.append(f"(name,like,%{q}%)")
    rows = c._get_paginated(API_TABLE, params={"where": "~and".join(where_parts), "limit": limit})
    return {"apis": rows, "count": len(rows)}


@router.post("/apis")
def create_api(payload: ApiCreate):
    try:
        row = api_registry.register_api(
            name=payload.name,
            base_url=payload.base_url,
            org_id=payload.org_id,
            auth_type=payload.auth_type,
            auth_secret_ref=payload.auth_secret_ref,
            auth_extra_json=payload.auth_extra_json,
            default_headers_json=payload.default_headers_json,
            openapi_url=payload.openapi_url,
            description=payload.description,
            allowed_methods=payload.allowed_methods,
            rate_limit_per_min=payload.rate_limit_per_min,
            timeout_seconds=payload.timeout_seconds,
        )
        return row
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        _log.error("api create failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/apis/{api_id}")
def get_api(api_id: int):
    c = _client()
    _require_table(c, API_TABLE)
    rows = c._get_paginated(API_TABLE, params={"where": f"(Id,eq,{api_id})", "limit": 1})
    if not rows:
        raise HTTPException(status_code=404, detail="not found")
    row = rows[0]
    row["used_by_agent_count"] = _count_agents_using(c, "connected_apis", api_id)
    return row


@router.patch("/apis/{api_id}")
def patch_api(api_id: int, payload: dict):
    c = _client()
    _require_table(c, API_TABLE)
    blocked = {"verified_at", "verification_status", "inspection_summary_json", "Id", "id", "CreatedAt", "UpdatedAt"}
    update = {k: v for k, v in payload.items() if k not in blocked}
    for k in ("auth_extra_json", "default_headers_json", "default_query_json"):
        if k in update and isinstance(update[k], (dict, list)):
            update[k] = json.dumps(update[k])
    return c._patch(API_TABLE, api_id, update)


@router.delete("/apis/{api_id}")
def delete_api(api_id: int):
    c = _client()
    _require_table(c, API_TABLE)
    using = _count_agents_using(c, "connected_apis", api_id)
    if using:
        raise HTTPException(status_code=409, detail=f"in use by {using} agent(s)")
    requests.delete(
        f"{c.url}/{c.tables[API_TABLE]}/{api_id}",
        headers=c.headers, timeout=10,
    ).raise_for_status()
    return {"ok": True, "deleted": api_id}


@router.post("/apis/{api_id}/inspect")
def inspect_api_route(api_id: int):
    try:
        return api_registry.inspect_api(api_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        _log.error("inspect failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/apis/{api_id}/test-call")
def test_call_api(api_id: int, call: ApiTestCall):
    c = _client()
    _require_table(c, API_TABLE)
    rows = c._get_paginated(API_TABLE, params={"where": f"(Id,eq,{api_id})", "limit": 1})
    if not rows:
        raise HTTPException(status_code=404, detail="connection not found")
    conn = rows[0]
    org_id = resolve_org_id(conn.get("org_id"))

    method = (call.method or "GET").upper()
    allowed = [m.strip().upper() for m in (conn.get("allowed_methods") or "GET").split(",")]
    if method not in allowed:
        raise HTTPException(status_code=403, detail=f"method {method} not allowed")
    paths_re = conn.get("allowed_paths_regex")
    if paths_re and not re.search(paths_re, call.path or ""):
        raise HTTPException(status_code=403, detail=f"path not allowed by regex")

    headers: dict = {}
    try:
        headers.update(json.loads(conn.get("default_headers_json") or "{}"))
    except Exception:
        pass
    headers.update(call.headers or {})

    auth_extra = {}
    try:
        auth_extra = json.loads(conn.get("auth_extra_json") or "{}")
    except Exception:
        pass
    secret = api_registry._resolve_secret(c, org_id, conn.get("auth_secret_ref") or "")
    headers.update(api_registry._auth_headers(conn.get("auth_type") or "none", secret, auth_extra))
    params = dict(call.params or {})
    params.update(api_registry._auth_query(conn.get("auth_type") or "none", secret, auth_extra))

    base = (conn.get("base_url") or "").rstrip("/")
    full_url = base + (call.path or "")
    try:
        r = requests.request(
            method, full_url,
            headers=headers, params=params,
            json=call.body if isinstance(call.body, (dict, list)) else None,
            data=call.body if isinstance(call.body, (str, bytes)) else None,
            timeout=int(conn.get("timeout_seconds") or 30),
        )
        return {
            "status": r.status_code,
            "elapsed_ms": int(r.elapsed.total_seconds() * 1000),
            "headers": dict(r.headers),
            "body": r.text[:20000],
            "url": r.url,
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {type(e).__name__}: {e}")


# ============================================================
# SMTP
# ============================================================

class SmtpCreate(BaseModel):
    name: str
    host: str
    port: int
    username: str
    password_secret_ref: str
    from_email: str
    org_id: int = 1
    description: str = ""
    use_tls: bool = True
    use_starttls: bool = True
    imap_host: str = ""
    imap_port: int = 993
    imap_username: str = ""
    imap_password_secret_ref: str = ""


@router.get("/smtp")
def list_smtp(org_id: int = 1, q: str = "", status: str = "", limit: int = 200):
    c = _client()
    _require_table(c, SMTP_TABLE)
    where_parts = [f"(org_id,eq,{resolve_org_id(org_id)})"]
    if status:
        where_parts.append(f"(verification_status,eq,{status})")
    if q:
        where_parts.append(f"(name,like,%{q}%)")
    rows = c._get_paginated(SMTP_TABLE, params={"where": "~and".join(where_parts), "limit": limit})
    return {"smtp": rows, "count": len(rows)}


@router.post("/smtp")
def create_smtp(payload: SmtpCreate):
    try:
        return smtp_registry.register_smtp(
            name=payload.name,
            host=payload.host,
            port=payload.port,
            username=payload.username,
            password_secret_ref=payload.password_secret_ref,
            from_email=payload.from_email,
            org_id=payload.org_id,
            description=payload.description,
            use_tls=payload.use_tls,
            use_starttls=payload.use_starttls,
            imap_host=payload.imap_host,
            imap_port=payload.imap_port,
            imap_username=payload.imap_username,
            imap_password_secret_ref=payload.imap_password_secret_ref,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        _log.error("smtp create failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/smtp/{smtp_id}")
def get_smtp(smtp_id: int):
    c = _client()
    _require_table(c, SMTP_TABLE)
    rows = c._get_paginated(SMTP_TABLE, params={"where": f"(Id,eq,{smtp_id})", "limit": 1})
    if not rows:
        raise HTTPException(status_code=404, detail="not found")
    row = rows[0]
    row["used_by_agent_count"] = _count_agents_using(c, "connected_smtp", smtp_id)
    return row


@router.patch("/smtp/{smtp_id}")
def patch_smtp(smtp_id: int, payload: dict):
    c = _client()
    _require_table(c, SMTP_TABLE)
    blocked = {"verified_at", "verification_status", "verification_note", "last_test_message_id",
               "Id", "id", "CreatedAt", "UpdatedAt"}
    update = {k: v for k, v in payload.items() if k not in blocked}
    return c._patch(SMTP_TABLE, smtp_id, update)


@router.delete("/smtp/{smtp_id}")
def delete_smtp(smtp_id: int):
    c = _client()
    _require_table(c, SMTP_TABLE)
    using = _count_agents_using(c, "connected_smtp", smtp_id)
    if using:
        raise HTTPException(status_code=409, detail=f"in use by {using} agent(s)")
    requests.delete(f"{c.url}/{c.tables[SMTP_TABLE]}/{smtp_id}", headers=c.headers, timeout=10).raise_for_status()
    return {"ok": True, "deleted": smtp_id}


@router.post("/smtp/{smtp_id}/test")
def test_smtp_route(smtp_id: int):
    try:
        return smtp_registry.test_smtp(smtp_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        _log.error("smtp test failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Secrets
# ============================================================

class SecretCreate(BaseModel):
    name: str
    value: str
    org_id: int = 1
    kind: str = "api_key"
    description: str = ""
    expires_at: str = ""


class SecretRotate(BaseModel):
    value: str


@router.get("/secrets")
def list_secrets(org_id: int = 1, q: str = "", limit: int = 200):
    c = _client()
    _require_table(c, SECRETS_TABLE)
    where_parts = [f"(org_id,eq,{resolve_org_id(org_id)})"]
    if q:
        where_parts.append(f"(name,like,%{q}%)")
    rows = c._get_paginated(SECRETS_TABLE, params={"where": "~and".join(where_parts), "limit": limit})
    return {"secrets": [_redact_secret(r) for r in rows], "count": len(rows)}


@router.post("/secrets")
def create_secret(payload: SecretCreate):
    c = _client()
    _require_table(c, SECRETS_TABLE)
    if not re.fullmatch(r"[a-z0-9_]+", payload.name):
        raise HTTPException(status_code=400, detail="name must be lowercase letters/digits/underscore")
    body = {
        "name": payload.name,
        "org_id": resolve_org_id(payload.org_id),
        "kind": payload.kind,
        "value": payload.value,
        "description": payload.description,
    }
    if payload.expires_at:
        body["expires_at"] = payload.expires_at
    row = c._post(SECRETS_TABLE, body)
    return _redact_secret(row)


@router.get("/secrets/{secret_id}")
def get_secret(secret_id: int):
    c = _client()
    _require_table(c, SECRETS_TABLE)
    rows = c._get_paginated(SECRETS_TABLE, params={"where": f"(Id,eq,{secret_id})", "limit": 1})
    if not rows:
        raise HTTPException(status_code=404, detail="not found")
    row = _redact_secret(rows[0])
    row["used_by"] = _secret_referrers(c, rows[0].get("name"))
    return row


@router.patch("/secrets/{secret_id}")
def patch_secret(secret_id: int, payload: dict):
    c = _client()
    _require_table(c, SECRETS_TABLE)
    blocked = {"value", "value_encrypted", "rotated_at", "Id", "id", "CreatedAt", "UpdatedAt"}
    update = {k: v for k, v in payload.items() if k not in blocked}
    row = c._patch(SECRETS_TABLE, secret_id, update)
    return _redact_secret(row)


@router.delete("/secrets/{secret_id}")
def delete_secret(secret_id: int):
    c = _client()
    _require_table(c, SECRETS_TABLE)
    rows = c._get_paginated(SECRETS_TABLE, params={"where": f"(Id,eq,{secret_id})", "limit": 1})
    if not rows:
        raise HTTPException(status_code=404, detail="not found")
    refs = _secret_referrers(c, rows[0].get("name"))
    if refs:
        raise HTTPException(status_code=409, detail={"reason": "in use", "referrers": refs})
    requests.delete(f"{c.url}/{c.tables[SECRETS_TABLE]}/{secret_id}", headers=c.headers, timeout=10).raise_for_status()
    return {"ok": True, "deleted": secret_id}


@router.post("/secrets/{secret_id}/reveal")
def reveal_secret(secret_id: int):
    """Return the cleartext value once. Caller is responsible for handling
    securely; this is logged."""
    c = _client()
    _require_table(c, SECRETS_TABLE)
    rows = c._get_paginated(SECRETS_TABLE, params={"where": f"(Id,eq,{secret_id})", "limit": 1})
    if not rows:
        raise HTTPException(status_code=404, detail="not found")
    _log.warning("secret revealed  id=%s name=%s", secret_id, rows[0].get("name"))
    return {"value": rows[0].get("value") or rows[0].get("value_encrypted") or ""}


@router.post("/secrets/{secret_id}/rotate")
def rotate_secret(secret_id: int, payload: SecretRotate):
    c = _client()
    _require_table(c, SECRETS_TABLE)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    row = c._patch(SECRETS_TABLE, secret_id, {"value": payload.value, "rotated_at": now})
    _log.info("secret rotated  id=%s", secret_id)
    return _redact_secret(row)


# ============================================================
# helpers
# ============================================================

def _count_agents_using(c: NocodbClient, column: str, ref_id: int) -> int:
    if "agents" not in c.tables:
        return 0
    rows = c._get_paginated("agents", params={"where": f"({column},like,%{ref_id}%)", "limit": 200})
    needle = str(ref_id)
    count = 0
    for r in rows:
        ids = [p.strip() for p in str(r.get(column) or "").split(",") if p.strip()]
        if needle in ids:
            count += 1
    return count


def _secret_referrers(c: NocodbClient, name: str) -> list[dict]:
    out: list[dict] = []
    if not name:
        return out
    if API_TABLE in c.tables:
        rows = c._get_paginated(API_TABLE, params={"where": f"(auth_secret_ref,eq,{name})", "limit": 100})
        out += [{"kind": "api_connection", "id": r["Id"], "name": r.get("name")} for r in rows]
    if SMTP_TABLE in c.tables:
        rows = c._get_paginated(SMTP_TABLE, params={
            "where": f"(password_secret_ref,eq,{name})", "limit": 100,
        })
        out += [{"kind": "smtp_account", "id": r["Id"], "name": r.get("name")} for r in rows]
        rows = c._get_paginated(SMTP_TABLE, params={
            "where": f"(imap_password_secret_ref,eq,{name})", "limit": 100,
        })
        out += [{"kind": "smtp_account_imap", "id": r["Id"], "name": r.get("name")} for r in rows]
    return out
