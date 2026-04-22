"""Consolidated read-side helpers for the `daily_digests` NocoDB table.

Markdown is stored inline in the ``markdown`` column so API and worker
processes don't need a shared filesystem.
"""
from __future__ import annotations

import logging

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
    """Return ``(markdown, available)``. ``available=False`` means the row has
    no markdown content."""
    if not row:
        return "", False
    md = (row.get("markdown") or "").strip()
    if md:
        return md, True
    return "", False


def as_payload(row: dict | None, include_markdown: bool = True) -> dict | None:
    """Shape a digest row for API responses."""
    if not row:
        return None
    markdown, available = read_markdown(row)
    return {
        "id": row.get("Id"),
        "date": row.get("digest_date"),
        "markdown": markdown if include_markdown else None,
        "markdown_available": available,
        "cluster_count": row.get("cluster_count"),
        "source_count": row.get("source_count"),
        "created_at": row.get("CreatedAt"),
    }
