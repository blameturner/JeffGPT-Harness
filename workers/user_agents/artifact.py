"""Artifact read/write helpers with versioning.

Every write to a target row goes through here so we get a row in
`artifact_versions` for rollback.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from infra.nocodb_client import NocodbClient

_log = logging.getLogger("agents.artifact")
VERSIONS_TABLE = "artifact_versions"


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def read(client: NocodbClient, table: str, row_id: int, column: str) -> str:
    if table not in client.tables:
        raise ValueError(f"unknown table: {table}")
    rows = client._get_paginated(table, params={"where": f"(Id,eq,{row_id})", "limit": 1})
    if not rows:
        return ""
    return rows[0].get(column) or ""


def write(
    client: NocodbClient,
    *,
    agent_id: int,
    assignment_id: int,
    table: str,
    row_id: int,
    column: str,
    new_text: str,
    edit_mode: str = "replace",
    forbidden_tables: set[str] | None = None,
    dry_run: bool = False,
) -> dict:
    """Read current → compute next → write → version. Returns {before, after, mode}."""
    if forbidden_tables and table in forbidden_tables:
        raise PermissionError(f"table forbidden: {table}")
    if table not in client.tables:
        raise ValueError(f"unknown table: {table}")

    before = read(client, table, row_id, column)

    if edit_mode == "append":
        after = (before + ("\n\n" if before else "") + new_text).strip()
    elif edit_mode == "patch_section":
        after = _patch_section(before, new_text)
    else:
        after = new_text

    if dry_run:
        _log.info("dry-run skip write  table=%s row=%d col=%s", table, row_id, column)
        return {"before": before, "after": after, "mode": edit_mode, "dry_run": True}

    client._patch(table, row_id, {column: after})

    if VERSIONS_TABLE in client.tables:
        try:
            client._post(VERSIONS_TABLE, {
                "agent_id": agent_id,
                "assignment_id": assignment_id,
                "table_name": table,
                "row_id": row_id,
                "column_name": column,
                "before_text": before,
                "after_text": after,
                "created_at": _iso_now(),
            })
        except Exception:
            _log.warning("artifact version write failed", exc_info=True)

    return {"before": before, "after": after, "mode": edit_mode}


def insert(
    client: NocodbClient,
    *,
    table: str,
    payload: dict,
    forbidden_tables: set[str] | None = None,
    dry_run: bool = False,
) -> dict:
    if forbidden_tables and table in forbidden_tables:
        raise PermissionError(f"table forbidden: {table}")
    if dry_run:
        return {"dry_run": True, "would_insert": payload}
    return client._post(table, payload)


def _patch_section(body: str, patch: str) -> str:
    """Replace a markdown section delimited by `## <name>` headers.

    `patch` must start with `## <name>`. If the section exists, replace it.
    Otherwise append.
    """
    lines = patch.strip().split("\n")
    if not lines or not lines[0].startswith("## "):
        return (body + "\n\n" + patch).strip()
    header = lines[0]
    if header in body:
        body_lines = body.split("\n")
        start = body_lines.index(header)
        end = len(body_lines)
        for i in range(start + 1, len(body_lines)):
            if body_lines[i].startswith("## "):
                end = i
                break
        return "\n".join(body_lines[:start] + lines + body_lines[end:]).strip()
    return (body + "\n\n" + patch).strip()
