"""Pick a topic the user actually cares about for a long-form briefing.

Replaces the old graph-degree-driven picker (2026-04-27). The old picker
ranked candidates by how well-connected they were in the entity graph;
that's structural, not episodic, and produced exactly the "abstract
topic about something I once searched" failure mode the user complained
about.

The new picker reads the recall layer and ranks candidates by *evidence
the user is currently engaged with the topic*:

  1. ``thread_of_day`` — the conversation they've been most active in
     over the last 24–72h. Topic = its title or the most-mentioned
     entity in its tail. Highest priority.
  2. Decision-pending loops naming a topic. The user is *waiting on a
     decision* about it; a deep brief is genuinely useful.
  3. Hot warm topics (``last_touched_at`` ≤ 3d) of kind ``task`` or
     ``interest``, not in mute_keys, not in engagement_blocks, not
     covered in a recent insight or daily brief.
  4. Active project facts (``pa_user_facts.kind="project"``) — slower-
     moving but always relevant.

If none of these produce a candidate, returns ``None``. We don't
fabricate a topic from low-degree graph nodes any more — silence is
better than a bad pick.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from shared.pa.recall import RecallPayload, build_recall

_log = logging.getLogger("insight.topic")

_HOT_TOPIC_HOURS = 72
_RECENT_INSIGHT_DAYS = 14
_RECENT_BRIEF_DAYS = 7

_GENERIC_PHRASES = {
    "general", "general overview", "productivity", "tech", "technology",
    "software", "stuff", "things", "the user", "my work", "this", "that",
}

_ANGLE_BY_INTENT = {
    "decision_pending": "decision support",
    "todo": "execution context",
    "event": "preparation",
    "waiting_on_other": "alternatives if blocked",
    "worry": "risk landscape",
}


def _slug(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _is_useful_topic(phrase: str) -> bool:
    if not phrase:
        return False
    s = _slug(phrase)
    if s in _GENERIC_PHRASES:
        return False
    if len(s) < 3:
        return False
    if len(s) > 80:
        return False
    return True


def _recent_covered(payload: RecallPayload, client) -> set[str]:
    """Topics already covered by an insight or daily brief in the last
    week or two — don't re-pick them."""
    covered: set[str] = set()
    cutoff = datetime.now(timezone.utc) - timedelta(days=_RECENT_INSIGHT_DAYS)
    cutoff_iso = cutoff.isoformat()
    try:
        rows = client._get_paginated("insights", params={
            "where": f"(org_id,eq,{payload.open_loops_user[0].id if False else 0})",
            "sort": "-CreatedAt",
            "limit": 30,
        })
    except Exception:
        rows = []
    # We don't have org_id on payload directly here; rely on recent_briefs from payload
    for b in payload.recent_briefs:
        topic = (b.get("topic") or "").strip()
        if topic:
            covered.add(_slug(topic))
    # Pull recent insights via shared module to keep query consistent
    try:
        from shared import insights as insights_mod
        # No direct org_id on payload; caller must use _recent_covered_by_org instead.
    except Exception:
        pass
    return covered


def _recent_covered_by_org(org_id: int) -> set[str]:
    covered: set[str] = set()
    try:
        from shared import insights as insights_mod
        rows = insights_mod.list_recent(org_id, limit=30)
    except Exception:
        rows = []
    for r in rows:
        topic = (r.get("topic") or "").strip()
        if topic:
            covered.add(_slug(topic))
    return covered


def _candidates_from_recall(payload: RecallPayload) -> list[dict]:
    """Build (score, topic, rationale, related, angle) candidates from recall.

    Higher score = stronger evidence the user cares right now.
    """
    out: list[dict] = []
    seen: set[str] = set()

    block_set = {b.lower() for b in payload.engagement_blocks}

    def _add(topic: str, score: float, rationale: str, related: list[str], angle: str) -> None:
        if not _is_useful_topic(topic):
            return
        slug = _slug(topic)
        if slug in seen:
            return
        if any(slug in m.lower() or m.lower() in slug for m in payload.mute_keys):
            return
        if slug in block_set or topic in block_set:
            return
        seen.add(slug)
        out.append({
            "topic": topic.strip()[:120],
            "score": score,
            "rationale": rationale[:300],
            "related": [r for r in related if r and r != topic][:5],
            "angle": angle,
        })

    # 1. Thread of day → highest priority
    if payload.thread_of_day is not None:
        t = payload.thread_of_day
        title = (t.title or "").strip()
        # Title often is a topic phrase already
        if title:
            related = [w.get("phrase", "") for w in payload.warm_topics[:5] if w.get("phrase")]
            _add(
                title,
                score=10.0 + min(t.msgs_24h, 20) * 0.2,
                rationale=f"You've been active in '{title}' — {t.msgs_24h} messages in the last 24h.",
                related=related,
                angle="continuation of active work",
            )

    # 2. Decision-pending loops
    for loop in payload.open_loops_user:
        if loop.intent != "decision_pending":
            continue
        text = loop.text.strip()
        if not _is_useful_topic(text):
            continue
        score = 8.0
        if loop.is_overdue:
            score += 2.0
        score += min(loop.age_hours / 24.0, 5.0) * 0.5
        _add(
            text,
            score=score,
            rationale=f"You flagged a decision on '{text}' — open for {int(loop.age_hours)}h.",
            related=[],
            angle=_ANGLE_BY_INTENT.get(loop.intent, "decision support"),
        )

    # 3. Hot warm topics
    cutoff = datetime.now(timezone.utc) - timedelta(hours=_HOT_TOPIC_HOURS)
    for topic in payload.warm_topics:
        kind = topic.get("kind") or ""
        if kind not in ("task", "interest", "user_stated"):
            continue
        phrase = (topic.get("phrase") or "").strip()
        if not phrase:
            continue
        warmth = float(topic.get("warmth") or 0)
        score = 5.0 + warmth * 3.0
        # boost if recently touched
        last_raw = topic.get("last_touched_at") or ""
        try:
            last = datetime.fromisoformat(str(last_raw).replace("Z", "+00:00"))
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            if last >= cutoff:
                hours_ago = (datetime.now(timezone.utc) - last).total_seconds() / 3600.0
                score += max(0, (_HOT_TOPIC_HOURS - hours_ago) / _HOT_TOPIC_HOURS) * 2.0
        except (TypeError, ValueError):
            pass
        related = [t.get("phrase", "") for t in payload.warm_topics if t.get("phrase") and t.get("phrase") != phrase][:4]
        _add(
            phrase,
            score=score,
            rationale=f"You've kept coming back to {phrase} (warmth {warmth:.2f}).",
            related=related,
            angle="competitive landscape" if kind == "task" else "deeper understanding",
        )

    # 4. Project facts — slower-moving but reliably relevant
    for fact in payload.projects_and_routines:
        if fact.get("kind") != "project":
            continue
        value = (fact.get("value") or "").strip()
        if not _is_useful_topic(value):
            continue
        _add(
            value,
            score=4.0,
            rationale=f"Tied to your active project: {value}.",
            related=[],
            angle="project context",
        )

    # 5. Todo loops with named entity
    for loop in payload.open_loops_user:
        if loop.intent != "todo":
            continue
        text = loop.text.strip()
        if not _is_useful_topic(text):
            continue
        score = 3.5
        if loop.is_overdue:
            score += 1.5
        _add(
            text,
            score=score,
            rationale=f"On your todo list: {text}.",
            related=[],
            angle=_ANGLE_BY_INTENT.get(loop.intent, "execution context"),
        )

    return out


def pick_topic(org_id: int) -> dict[str, Any] | None:
    """Return ``{topic, related_entities, rationale, angle}`` or None.

    Returns None when there's no genuinely-engaged signal — the insight
    producer treats this as "stay silent, try later".
    """
    if int(org_id) <= 0:
        return None
    try:
        payload = build_recall(int(org_id))
    except Exception:
        _log.warning("topic_picker: build_recall failed  org=%d", org_id, exc_info=True)
        return None

    candidates = _candidates_from_recall(payload)
    if not candidates:
        _log.info("topic_picker: no episodic signal  org=%d — staying silent", org_id)
        return None

    covered = _recent_covered_by_org(org_id)
    candidates = [c for c in candidates if _slug(c["topic"]) not in covered]
    if not candidates:
        _log.info("topic_picker: all candidates already covered  org=%d", org_id)
        return None

    candidates.sort(key=lambda c: c["score"], reverse=True)
    pick = candidates[0]

    _log.info(
        "topic_picker pick  org=%d topic=%r score=%.2f angle=%s",
        org_id, pick["topic"], pick["score"], pick["angle"],
    )

    return {
        "topic": pick["topic"],
        "related_entities": pick["related"],
        "rationale": pick["rationale"],
        "angle": pick["angle"],
    }
