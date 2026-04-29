"""Context assembly. The single function that determines output quality."""
from __future__ import annotations

import json
import logging
from datetime import date

from infra.nocodb_client import NocodbClient

_log = logging.getLogger("agents.context")


def _interpolate(text: str, variables: dict) -> str:
    if not text:
        return ""
    out = text
    base = {"date": date.today().isoformat(), **(variables or {})}
    for k, v in base.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _csv_ids(value) -> list[int]:
    if not value:
        return []
    return [int(p) for p in str(value).split(",") if p.strip().isdigit()]


def _csv_strs(value) -> list[str]:
    if not value:
        return []
    return [p.strip() for p in str(value).split(",") if p.strip()]


def _safe_json(raw, default):
    if isinstance(raw, dict):
        return raw
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def resolve_apis(client: NocodbClient, csv_ids_value) -> list[dict]:
    """Pull api_connections rows by id; return minimal dicts with usage_prompt."""
    if "api_connections" not in client.tables:
        return []
    ids = _csv_ids(csv_ids_value)
    if not ids:
        return []
    out: list[dict] = []
    for cid in ids:
        rows = client._get_paginated("api_connections", params={
            "where": f"(Id,eq,{cid})",
            "limit": 1,
        })
        if rows:
            out.append(rows[0])
    return out


def resolve_smtp(client: NocodbClient, csv_ids_value) -> list[dict]:
    if "smtp_accounts" not in client.tables:
        return []
    ids = _csv_ids(csv_ids_value)
    if not ids:
        return []
    out: list[dict] = []
    for cid in ids:
        rows = client._get_paginated("smtp_accounts", params={
            "where": f"(Id,eq,{cid})",
            "limit": 1,
        })
        if rows:
            out.append(rows[0])
    return out


def build_system_prompt(agent: dict) -> str:
    """Persona + system_prompt_template + pinned_context, with {var} interpolation."""
    variables = _safe_json(agent.get("prompt_variables_json"), {})
    parts: list[str] = []
    if persona := agent.get("persona"):
        parts.append(_interpolate(persona, variables))
    if tmpl := agent.get("system_prompt_template"):
        parts.append(_interpolate(tmpl, variables))
    if pinned := agent.get("pinned_context"):
        parts.append(_interpolate(pinned, variables))
    return "\n\n".join(p for p in parts if p)


def build_user_context(
    client: NocodbClient,
    agent: dict,
    assignment: dict | None,
    type_specific_input: str = "",
) -> str:
    """Brief + API usage prompts + RAG (later) + type-specific input + task."""
    variables = _safe_json(agent.get("prompt_variables_json"), {})
    parts: list[str] = []
    if brief := agent.get("brief"):
        parts.append(f"BRIEF: {_interpolate(brief, variables)}")

    apis = resolve_apis(client, agent.get("connected_apis"))
    for api in apis:
        if usage := api.get("usage_prompt"):
            parts.append(f"\n# API: {api['name']}\n{usage}")

    smtps = resolve_smtp(client, agent.get("connected_smtp"))
    for s in smtps:
        parts.append(f"\n# SMTP: {s['name']} (from={s.get('from_email')})")

    if type_specific_input:
        parts.append(type_specific_input)

    if assignment and (task := assignment.get("task")):
        parts.append(f"\nTASK:\n{task}")

    return "\n\n".join(p for p in parts if p)


def parse_csv(value) -> list[str]:
    return _csv_strs(value)


def parse_json(raw, default=None):
    return _safe_json(raw, default if default is not None else {})
