"""Consolidated read-side helpers for the `daily_digests` NocoDB table.

Centralises the lookup + markdown-read logic that used to live in both
``app/routers/home.py`` and ``workers/chat/home.py``. Adds a
``markdown_available`` flag so callers can distinguish "no digest row yet"
from "digest row exists but markdown file isn't reachable from this
process" (common when the API container has a different filesystem view
than the worker that wrote the file).
"""
from __future__ import annotations

import logging
from pathlib import Path

from infra.config import NOCODB_TABLE_DAILY_DIGESTS
from infra.nocodb_client import NocodbClient

_log = logging.getLogger("home.digest_reader")


def latest_digest(client: NocodbClient, org_id: int) -> dict | None:
    if NOCODB_TABLE_DAILY_DIGESTS not in client.tables:
        return None
    try:
        rows = client._get_paginated(NOCODB_TABLE_DAILY_DIGESTS, params={
            "where": f"(org_id,eq,{org_id})",
            "sort": "-digest_date,-CreatedAt",
            "limit": 1,
        })
    except Exception:
        _log.warning("daily_digests fetch failed  org=%d", org_id, exc_info=True)
        return None
    return rows[0] if rows else None


def digest_for_date(client: NocodbClient, org_id: int, date: str) -> dict | None:
    if NOCODB_TABLE_DAILY_DIGESTS not in client.tables:
        return None
    try:
        data = client._get(NOCODB_TABLE_DAILY_DIGESTS, params={
            "where": f"(org_id,eq,{org_id})~and(digest_date,eq,{date})",
            "limit": 1,
        })
    except Exception:
        _log.warning("daily_digest date fetch failed  org=%d date=%s", org_id, date, exc_info=True)
        return None
    rows = data.get("list", [])
    return rows[0] if rows else None


def list_digests(client: NocodbClient, org_id: int, limit: int = 7) -> list[dict]:
    if NOCODB_TABLE_DAILY_DIGESTS not in client.tables:
        return []
    try:
        return client._get_paginated(NOCODB_TABLE_DAILY_DIGESTS, params={
            "where": f"(org_id,eq,{org_id})",
            "sort": "-digest_date,-CreatedAt",
            "limit": limit,
        })
    except Exception:
        _log.warning("daily_digest list failed  org=%d", org_id, exc_info=True)
        return []


def read_markdown(row: dict | None) -> tuple[str, bool]:
    """Return ``(markdown, available)``. ``available=False`` means the path is
    either missing, empty, or unreadable from this process."""
    if not row:
        return "", False
    path = (row.get("markdown_path") or "").strip()
    if not path:
        return "", False
    try:
        p = Path(path).expanduser()
        if p.is_file():
            return p.read_text(encoding="utf-8"), True
        _log.warning("digest markdown file missing  path=%s", path)
    except Exception:
        _log.warning("digest markdown read failed  path=%s", path, exc_info=True)
    return "", False


def as_payload(row: dict | None, include_markdown: bool = True) -> dict | None:
    """Shape a digest row for API responses."""
    if not row:
        return None
    markdown, available = (read_markdown(row) if include_markdown else ("", bool(row.get("markdown_path"))))
    return {
        "id": row.get("Id"),
        "date": row.get("digest_date"),
        "markdown": markdown if include_markdown else None,
        "markdown_available": available,
        "path": row.get("markdown_path"),
        "cluster_count": row.get("cluster_count"),
        "source_count": row.get("source_count"),
        "created_at": row.get("CreatedAt"),
    }
