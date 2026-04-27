"""Overnight research seeder.

Runs on a nightly cron. Picks 1–2 task-shaped warm topics or pending
decisions surfaced during the day and kicks off the existing research
planner for each. By morning, the brief's recall layer sees the
completed plans in `completed_research` and the brief leads with
"I read up on X overnight."

Deterministic selection — no LLM. Keeps the produce-research-overnight
pipeline cheap and predictable.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from infra.config import (
    NOCODB_TABLE_PA_OPEN_LOOPS,
    NOCODB_TABLE_PA_WARM_TOPICS,
    get_feature,
    is_feature_enabled,
)
from infra.nocodb_client import NocodbClient
from shared.pa.memory import (
    LOOP_INTENT_DECISION,
    LOOP_STATUS_OPEN,
    TOPIC_KIND_TASK,
    engagement_bias,
)

_log = logging.getLogger("research_seeder")

_RECENCY_HOURS = 24


def _cfg(key: str, default):
    return get_feature("research_seeder", key, default)


def _parse_iso(raw):
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _existing_plan_topics(client: NocodbClient, org_id: int) -> set[str]:
    if "research_plans" not in client.tables:
        return set()
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    cutoff_iso = cutoff.isoformat()
    try:
        rows = client._get_paginated("research_plans", params={
            "where": f"(org_id,eq,{org_id})~and(CreatedAt,gt,{cutoff_iso})",
            "limit": 100,
        })
    except Exception:
        return set()
    return {
        (r.get("topic") or "").strip().lower()
        for r in rows if (r.get("topic") or "").strip()
    }


def _candidate_topics(client: NocodbClient, org_id: int, now: datetime) -> list[str]:
    """Recently-touched task warm-topics, ranked by warmth × engagement bias."""
    if NOCODB_TABLE_PA_WARM_TOPICS not in client.tables:
        return []
    try:
        rows = client._get_paginated(NOCODB_TABLE_PA_WARM_TOPICS, params={
            "where": f"(org_id,eq,{org_id})",
            "sort": "-warmth,-last_touched_at",
            "limit": 50,
        })
    except Exception:
        return []
    cutoff = now - timedelta(hours=_RECENCY_HOURS)
    bias = 1.0
    try:
        bias = float(engagement_bias(org_id, "news_watch"))
    except Exception:
        bias = 1.0
    candidates: list[tuple[float, str]] = []
    for r in rows:
        if (r.get("kind") or "") != TOPIC_KIND_TASK:
            continue
        touched = _parse_iso(r.get("last_touched_at"))
        if touched is None or touched < cutoff:
            continue
        phrase = (r.get("entity_or_phrase") or "").strip()
        if not phrase:
            continue
        warmth = float(r.get("warmth") or 0)
        candidates.append((warmth * bias, phrase))
    candidates.sort(key=lambda p: p[0], reverse=True)
    return [c[1] for c in candidates]


def _candidate_decisions(client: NocodbClient, org_id: int, now: datetime) -> list[str]:
    """Decision-pending loops created in the last 24h."""
    if NOCODB_TABLE_PA_OPEN_LOOPS not in client.tables:
        return []
    cutoff = now - timedelta(hours=_RECENCY_HOURS)
    cutoff_iso = cutoff.isoformat()
    try:
        rows = client._get_paginated(NOCODB_TABLE_PA_OPEN_LOOPS, params={
            "where": (
                f"(org_id,eq,{org_id})~and(intent,eq,{LOOP_INTENT_DECISION})"
                f"~and(status,eq,{LOOP_STATUS_OPEN})~and(CreatedAt,gt,{cutoff_iso})"
            ),
            "sort": "-CreatedAt",
            "limit": 20,
        })
    except Exception:
        return []
    return [(r.get("text") or "").strip() for r in rows if (r.get("text") or "").strip()]


def run_research_seeder(org_id: int, now: datetime | None = None) -> dict:
    out: dict = {
        "status": "ok",
        "org_id": int(org_id),
        "queued": 0,
        "candidates_topics": 0,
        "candidates_decisions": 0,
    }
    if not is_feature_enabled("pa"):
        out["status"] = "skipped"
        out["reason"] = "pa_disabled"
        return out
    if not _cfg("enabled", True):
        out["status"] = "skipped"
        out["reason"] = "research_seeder_disabled"
        return out
    if int(org_id) <= 0:
        out["status"] = "error"
        out["reason"] = "invalid_org_id"
        return out

    if now is None:
        now = datetime.now(timezone.utc)

    client = NocodbClient()
    existing = _existing_plan_topics(client, int(org_id))

    topics = _candidate_topics(client, int(org_id), now)
    decisions = _candidate_decisions(client, int(org_id), now)
    out["candidates_topics"] = len(topics)
    out["candidates_decisions"] = len(decisions)

    # Interleave topics and decisions; topics generally outrank a single overnight
    # decision but if both exist we want to cover both surfaces.
    ordered: list[str] = []
    for pair in zip(topics, decisions):
        ordered.extend(pair)
    leftover = topics[len(decisions):] + decisions[len(topics):]
    ordered.extend(leftover)

    max_topics = int(_cfg("max_topics_per_night", 2) or 2)
    queued_topics: list[str] = []
    for topic in ordered:
        if len(queued_topics) >= max_topics:
            break
        if not topic or topic.lower() in existing:
            continue
        try:
            from tools.research.research_planner import create_research_plan
        except Exception:
            _log.error("research_seeder: planner import failed", exc_info=True)
            out["status"] = "error"
            out["reason"] = "planner_import_failed"
            break
        try:
            result = create_research_plan(topic, org_id=int(org_id))
        except Exception:
            _log.warning("research_seeder: create_research_plan raised  topic=%r", topic, exc_info=True)
            continue
        if (result or {}).get("status") == "queued":
            queued_topics.append(topic)
            existing.add(topic.lower())
            _log.info(
                "research_seeder queued  org=%d topic=%r plan_id=%s",
                org_id, topic, (result or {}).get("plan_id"),
            )

    out["queued"] = len(queued_topics)
    out["queued_topics"] = queued_topics
    return out
