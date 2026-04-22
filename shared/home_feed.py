"""Chronological feed for the home dashboard.

Merges four streams into one timeline:

- ``digest`` — entries from the ``daily_digests`` table
- ``insight`` — published rows from the ``insights`` table
- ``question`` — pending / recently-answered ``assistant_questions`` rows
- ``run`` — notable entries from ``agent_runs`` (completed, non-empty summary)

All reads are bounded and best-effort; any single stream failing just yields
an empty contribution so the feed still renders.
"""
from __future__ import annotations

import logging
from typing import Any

from infra.config import NOCODB_TABLE_ASSISTANT_QUESTIONS, NOCODB_TABLE_DAILY_DIGESTS
from infra.nocodb_client import NocodbClient
from shared import digest_reader, home_questions, insights as insights_mod

_log = logging.getLogger("home.feed")


def _safe_sort_key(item: dict) -> str:
    return str(item.get("created_at") or "")


def _digest_items(client: NocodbClient, org_id: int, limit: int) -> list[dict]:
    rows = digest_reader.list_digests(client, org_id, limit=limit)
    return [{
        "kind": "digest",
        "id": r.get("Id"),
        "title": f"Daily digest — {r.get('digest_date')}",
        "snippet": f"{r.get('cluster_count') or 0} clusters · {r.get('source_count') or 0} sources",
        "created_at": r.get("CreatedAt") or r.get("digest_date"),
        "ref": {"date": r.get("digest_date")},
    } for r in rows]


def _insight_items(org_id: int, limit: int) -> list[dict]:
    rows = insights_mod.list_recent(org_id, limit=limit)
    return [{
        "kind": "insight",
        "id": r.get("id"),
        "title": r.get("title") or r.get("topic") or "Insight",
        "snippet": (r.get("summary") or r.get("body_markdown") or "")[:400],
        "created_at": r.get("created_at"),
        "ref": {"topic": r.get("topic"), "trigger": r.get("trigger")},
    } for r in rows]


def _question_items(client: NocodbClient, org_id: int, limit: int) -> list[dict]:
    if NOCODB_TABLE_ASSISTANT_QUESTIONS not in client.tables:
        return []
    try:
        rows = client._get_paginated(NOCODB_TABLE_ASSISTANT_QUESTIONS, params={
            "where": f"(org_id,eq,{org_id})",
            "sort": "-CreatedAt",
            "limit": limit,
        })
    except Exception:
        _log.warning("feed: questions stream failed", exc_info=True)
        return []
    out = []
    for r in rows:
        status = r.get("status") or "pending"
        out.append({
            "kind": "question",
            "id": r.get("Id"),
            "title": r.get("question_text") or "Question",
            "snippet": f"status={status}  " + (home_questions.render_provenance(r.get("context_ref") or "")),
            "created_at": r.get("CreatedAt"),
            "ref": {"status": status, "context_ref": r.get("context_ref")},
        })
    return out


def _run_items(client: NocodbClient, org_id: int, limit: int) -> list[dict]:
    try:
        rows = client._get_paginated("agent_runs", params={
            "where": f"(org_id,eq,{org_id})~and(status,eq,complete)",
            "sort": "-CreatedAt",
            "limit": limit,
        })
    except Exception:
        _log.warning("feed: runs stream failed", exc_info=True)
        return []
    out = []
    for r in rows:
        summary = (r.get("summary") or "").strip()
        if not summary:
            continue
        out.append({
            "kind": "run",
            "id": r.get("Id"),
            "title": f"{r.get('agent_name')} completed",
            "snippet": summary[:400],
            "created_at": r.get("CreatedAt"),
            "ref": {"agent_name": r.get("agent_name")},
        })
    return out


def build_feed(org_id: int, limit: int = 25) -> list[dict[str, Any]]:
    client = NocodbClient()
    # Over-fetch each stream modestly so the merged set still has ~limit rows.
    per_stream = max(5, limit)
    buckets: list[list[dict]] = []
    for fetch in (
        lambda: _insight_items(org_id, per_stream),
        lambda: _digest_items(client, org_id, per_stream),
        lambda: _question_items(client, org_id, per_stream),
        lambda: _run_items(client, org_id, per_stream),
    ):
        try:
            buckets.append(fetch())
        except Exception:
            _log.warning("feed stream failed", exc_info=True)
            buckets.append([])
    merged: list[dict] = []
    for b in buckets:
        merged.extend(b)
    merged.sort(key=_safe_sort_key, reverse=True)
    return merged[:limit]
