"""Bridge between the harvest fetcher and the existing api_connections /
secrets registry. Resolves a `connection_id` to (auth_headers, auth_query,
default_headers).

Secrets are resolved at fetch time and **never** retained beyond the
duration of the call. Body content is passed through `_redact_secrets`
before any LLM call (see extractor).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from infra.nocodb_client import NocodbClient

_log = logging.getLogger("harvest.connectors")

API_TABLE = "api_connections"
SECRETS_TABLE = "secrets"


def _safe_json(raw, default):
    if isinstance(raw, dict):
        return raw
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


def resolve(connection_id: int, *, org_id: Optional[int] = None,
            client: Optional[NocodbClient] = None) -> tuple[dict, dict, dict, str | None]:
    """Resolve a connection to (auth_headers, auth_query_params,
    default_headers, base_url).

    Returns empty dicts and None on any failure — the caller should
    proceed unauthenticated rather than fail the whole fetch.
    """
    if not connection_id:
        return {}, {}, {}, None
    client = client or NocodbClient()
    try:
        rows = client._get(
            API_TABLE,
            params={"where": f"(Id,eq,{int(connection_id)})", "limit": 1},
        ).get("list", [])
    except Exception:
        _log.warning("api_connections lookup failed id=%s", connection_id, exc_info=True)
        return {}, {}, {}, None
    if not rows:
        return {}, {}, {}, None
    conn = rows[0]
    auth_type = (conn.get("auth_type") or "none").strip().lower()
    secret_ref = (conn.get("auth_secret_ref") or "").strip()
    extra = _safe_json(conn.get("auth_extra_json"), {})
    default_headers = _safe_json(conn.get("default_headers_json"), {})
    base_url = (conn.get("base_url") or "").strip() or None

    secret_value: str | None = None
    if secret_ref and SECRETS_TABLE in client.tables:
        scope_org = int(org_id or conn.get("org_id") or 1)
        try:
            sec_rows = client._get_paginated(
                SECRETS_TABLE,
                params={
                    "where": f"(org_id,eq,{scope_org})~and(name,eq,{secret_ref})",
                    "limit": 1,
                },
            )
            if sec_rows:
                secret_value = (
                    sec_rows[0].get("value")
                    or sec_rows[0].get("value_encrypted")
                )
        except Exception:
            _log.warning("secret resolution failed name=%s", secret_ref, exc_info=True)

    auth_headers: dict[str, str] = {}
    auth_query: dict[str, str] = {}

    if auth_type == "bearer" and secret_value:
        auth_headers["Authorization"] = f"Bearer {secret_value}"
    elif auth_type == "basic" and secret_value:
        import base64
        username = extra.get("username", "")
        tok = base64.b64encode(f"{username}:{secret_value}".encode()).decode()
        auth_headers["Authorization"] = f"Basic {tok}"
    elif auth_type == "api_key_header" and secret_value:
        header_name = extra.get("header_name", "X-API-Key")
        auth_headers[header_name] = secret_value
    elif auth_type == "api_key_query" and secret_value:
        auth_query[extra.get("query_name", "api_key")] = secret_value

    return auth_headers, auth_query, default_headers, base_url


_REDACT_PATTERNS = [
    # Bearer / token-shaped strings
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}"),
    re.compile(r"(?i)\bsk-[A-Za-z0-9]{20,}\b"),                # OpenAI-style
    re.compile(r"(?i)\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),       # Slack
    re.compile(r"(?i)\bgh[pousr]_[A-Za-z0-9]{20,}\b"),          # GitHub
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                       # AWS
    re.compile(r"\b[A-Za-z0-9]{40,}\b"),                       # generic long token (last resort, kept narrow)
    # PII-ish
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                       # SSN
    re.compile(r"\b\d{13,19}\b"),                               # credit card-ish
]


def redact_secrets(text: str) -> str:
    """Pass-through redactor for body content sent to LLMs.

    Conservative: misses creative custom tokens, but masks the obvious
    AWS/GitHub/OpenAI/Slack patterns plus generic long alnum runs.
    """
    if not text:
        return text
    out = text
    for pat in _REDACT_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    return out
