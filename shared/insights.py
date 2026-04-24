"""Persistent store for long-form research insights surfaced on the home dashboard.

An *insight* is the dashboard's headline artifact: a multi-paragraph briefing
that the system produced while the user was idle (e.g. "Since we've been
discussing Duck Creek, here's a synopsis of competing insurance platforms
and where DCT fits"). Insights are heavier than questions — they carry a
title, body markdown, the topic/entity they covered, what triggered them,
and the underlying research run id for deep-dive access.

Access shape mirrors ``shared.home_questions``: inert if the NocoDB table
isn't provisioned yet.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from infra.nocodb_client import NocodbClient

_log = logging.getLogger("home.insights")

INSIGHTS_TABLE = "insights"

STATUS_DRAFT = "draft"
STATUS_PUBLISHED = "published"
STATUS_ARCHIVED = "archived"

TRIGGER_CHAT_IDLE = "chat_idle"
TRIGGER_FALLBACK = "fallback_twice_daily"
TRIGGER_MANUAL = "manual"
TRIGGER_FOLLOWUP = "question_followup"


def _table_present(client: NocodbClient) -> bool:
    if INSIGHTS_TABLE in client.tables:
        return True
    _log.info("%s table absent — insights API inert", INSIGHTS_TABLE)
    return False


def _hydrate(row: dict) -> dict:
    return {
        "id": row.get("Id"),
        "org_id": row.get("org_id"),
        "title": row.get("title") or "",
        "topic": row.get("topic") or "",
        "body_markdown": row.get("body_markdown") or "",
        "summary": row.get("summary") or "",
        "trigger": row.get("trigger") or "",
        "status": row.get("status") or STATUS_DRAFT,
        "research_plan_id": row.get("research_plan_id"),
        "related_entities": _decode_json(row.get("related_entities")),
        "sources": _decode_json(row.get("sources")),
        "created_at": row.get("CreatedAt"),
        "surfaced_at": row.get("surfaced_at"),
    }


def _decode_json(raw: Any) -> list:
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        out = json.loads(raw)
        return out if isinstance(out, list) else []
    except (TypeError, ValueError):
        return []


def create(
    org_id: int,
    title: str,
    body_markdown: str,
    topic: str = "",
    summary: str = "",
    trigger: str = TRIGGER_MANUAL,
    research_plan_id: int | None = None,
    related_entities: list[str] | None = None,
    sources: list[dict] | None = None,
    status: str = STATUS_PUBLISHED,
) -> int | None:
    client = NocodbClient()
    if not _table_present(client):
        return None
    payload = {
        "org_id": int(org_id),
        "title": title[:255],
        "topic": (topic or "")[:255],
        "body_markdown": body_markdown,
        "summary": summary[:1000] if summary else "",
        "trigger": trigger,
        "status": status,
        "research_plan_id": research_plan_id,
        "related_entities": json.dumps(related_entities or []),
        "sources": json.dumps(sources or []),
        "surfaced_at": datetime.now(timezone.utc).isoformat() if status == STATUS_PUBLISHED else None,
    }
    try:
        row = client._post(INSIGHTS_TABLE, payload)
        iid = row.get("Id")
        _log.info("insight created  org=%d id=%s topic=%s", org_id, iid, topic[:60])
        return iid
    except Exception:
        _log.warning("insight create failed  org=%d", org_id, exc_info=True)
        return None


def list_recent(org_id: int, limit: int = 10) -> list[dict]:
    client = NocodbClient()
    if not _table_present(client):
        return []
    try:
        rows = client._get_paginated(INSIGHTS_TABLE, params={
            "where": f"(org_id,eq,{org_id})~and(status,eq,{STATUS_PUBLISHED})",
            "sort": "-CreatedAt",
            "limit": limit,
        })
    except Exception:
        _log.warning("insights list failed  org=%d", org_id, exc_info=True)
        return []
    return [_hydrate(r) for r in rows]


def get(insight_id: int) -> dict | None:
    client = NocodbClient()
    if not _table_present(client):
        return None
    try:
        data = client._get(INSIGHTS_TABLE, params={
            "where": f"(Id,eq,{insight_id})",
            "limit": 1,
        })
    except Exception:
        return None
    rows = data.get("list", [])
    return _hydrate(rows[0]) if rows else None


def append_research(plan_id: int, paper_content: str, focus: str = "") -> int:
    """Append a completed research plan's paper to every published insight that
    links to it (either as primary ``research_plan_id`` or as a ``parent_insight_id``
    follow-up plan). Returns the number of insights updated."""
    if not paper_content:
        return 0
    client = NocodbClient()
    if not _table_present(client):
        return 0

    # Primary link: insights.research_plan_id = plan_id
    targets: list[dict] = []
    try:
        rows = client._get_paginated(INSIGHTS_TABLE, params={
            "where": f"(research_plan_id,eq,{plan_id})~and(status,eq,{STATUS_PUBLISHED})",
            "limit": 20,
        })
        targets.extend(rows)
    except Exception:
        _log.warning("append_research: primary lookup failed  plan_id=%d", plan_id, exc_info=True)

    # Follow-up link: research_plans.parent_insight_id → insight
    try:
        plan_rows = client._get("research_plans", params={
            "where": f"(Id,eq,{plan_id})",
            "limit": 1,
        }).get("list", [])
        parent_id = plan_rows[0].get("parent_insight_id") if plan_rows else None
        if parent_id:
            pd = client._get(INSIGHTS_TABLE, params={
                "where": f"(Id,eq,{int(parent_id)})",
                "limit": 1,
            }).get("list", [])
            if pd and pd[0].get("status") == STATUS_PUBLISHED:
                targets.append(pd[0])
    except Exception:
        _log.warning("append_research: parent lookup failed  plan_id=%d", plan_id, exc_info=True)

    # De-dupe by insight id
    seen: set[int] = set()
    updated = 0
    heading = f"## Follow-up: {focus}" if focus else "## Deeper research (from follow-up plan)"
    block = f"\n\n---\n\n{heading}\n\n_From research plan #{plan_id}_\n\n{paper_content.strip()}\n"
    for row in targets:
        iid = row.get("Id")
        if not iid or iid in seen:
            continue
        seen.add(iid)
        existing = row.get("body_markdown") or ""
        if f"research plan #{plan_id}" in existing:
            continue
        try:
            client._patch(INSIGHTS_TABLE, iid, {
                "Id": iid,
                "body_markdown": existing + block,
            })
            updated += 1
            _log.info("insight augmented  id=%s plan_id=%d focus=%s", iid, plan_id, focus[:40])
        except Exception:
            _log.warning("insight append failed  id=%s plan_id=%d", iid, plan_id, exc_info=True)
    return updated


def latest_created_at(org_id: int) -> str | None:
    """Used by the dispatcher to decide whether to produce a new insight."""
    client = NocodbClient()
    if not _table_present(client):
        return None
    try:
        rows = client._get_paginated(INSIGHTS_TABLE, params={
            "where": f"(org_id,eq,{org_id})",
            "sort": "-CreatedAt",
            "limit": 1,
        })
    except Exception:
        return None
    if not rows:
        return None
    return rows[0].get("CreatedAt")
