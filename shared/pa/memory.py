"""Data access for the PA memory layer.

All four tables are treated as optional at runtime — if the NocoDB schema
hasn't been provisioned yet, reads return empty and writes log + no-op so
the rest of the PA stack can ship without blocking on DB work.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from infra.config import (
    NOCODB_TABLE_PA_OPEN_LOOPS,
    NOCODB_TABLE_PA_WARM_TOPICS,
    NOCODB_TABLE_PA_USER_FACTS,
    NOCODB_TABLE_PA_ASSISTANT_MOVES,
)
from infra.nocodb_client import NocodbClient

_log = logging.getLogger("pa.memory")


# ── status / kind enums ───────────────────────────────────────────────────────

LOOP_STATUS_OPEN = "open"
LOOP_STATUS_NUDGED = "nudged"
LOOP_STATUS_RESOLVED = "resolved"
LOOP_STATUS_DROPPED = "dropped"

LOOP_INTENT_TODO = "todo"
LOOP_INTENT_EVENT = "event"
LOOP_INTENT_DECISION = "decision_pending"
LOOP_INTENT_WAITING = "waiting_on_other"
LOOP_INTENT_WORRY = "worry"

TOPIC_KIND_TASK = "task"
TOPIC_KIND_INTEREST = "interest"
TOPIC_KIND_STATED = "user_stated"

FACT_CONFIDENCE_OBSERVED = "observed"
FACT_CONFIDENCE_STATED = "stated"
FACT_CONFIDENCE_CONFIRMED = "confirmed"

MOVE_MODE_REACTIVE = "reactive_inline"
MOVE_MODE_PROACTIVE = "proactive"
MOVE_MODE_BACKGROUND = "background_drop"

MOVE_KIND_CLOSURE = "closure"
MOVE_KIND_CONNECT = "connect"
MOVE_KIND_NEWS = "news_watch"
MOVE_KIND_SERENDIPITY = "serendipity"
MOVE_KIND_RESURFACE = "resurface"


# ── helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _table_ready(client: NocodbClient, table: str) -> bool:
    try:
        if table in client.tables:
            return True
    except Exception:
        _log.warning("pa table probe failed  table=%s", table, exc_info=True)
        return False
    _log.info("pa table not provisioned  table=%s — skipping", table)
    return False


_WHERE_UNSAFE = set(",)(~")


def _safe_where_value(value: str) -> str:
    """NocoDB v1 where-parser is sensitive to unescaped punctuation. For
    text values coming from the LLM, strip the characters that break
    parsing."""
    if not value:
        return ""
    return "".join(c for c in str(value) if c not in _WHERE_UNSAFE)


def _dumps(value: Any) -> str:
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return "[]"


def _loads(raw: Any) -> Any:
    if raw is None or raw == "":
        return None
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


# ── open_loops ────────────────────────────────────────────────────────────────

def list_open_loops(
    org_id: int,
    status: str | None = LOOP_STATUS_OPEN,
    limit: int = 50,
) -> list[dict]:
    client = NocodbClient()
    if not _table_ready(client, NOCODB_TABLE_PA_OPEN_LOOPS):
        return []
    parts = [f"(org_id,eq,{int(org_id)})"]
    if status:
        parts.append(f"(status,eq,{status})")
    try:
        return client._get_paginated(NOCODB_TABLE_PA_OPEN_LOOPS, params={
            "where": "~and".join(parts),
            "sort": "-CreatedAt",
            "limit": min(max(1, limit), 200),
        })
    except Exception:
        _log.warning("list_open_loops failed  org=%d", org_id, exc_info=True)
        return []


def find_loop_by_text(org_id: int, text: str) -> dict | None:
    """Fuzzy match on text — used by extractor to dedupe / resolve."""
    if not text:
        return None
    client = NocodbClient()
    if not _table_ready(client, NOCODB_TABLE_PA_OPEN_LOOPS):
        return None
    needle = text.strip().lower()[:80]
    # fetch open loops, match in Python (NocoDB's LIKE parser is fragile)
    try:
        rows = list_open_loops(org_id, status=LOOP_STATUS_OPEN, limit=100)
    except Exception:
        return None
    for r in rows:
        candidate = (r.get("text") or "").strip().lower()
        if not candidate:
            continue
        if needle in candidate or candidate in needle:
            return r
    return None


def create_open_loop(
    org_id: int,
    text: str,
    intent: str = LOOP_INTENT_TODO,
    when_hint: str = "",
    due_at: str | None = None,
    source_message_id: int | None = None,
) -> dict | None:
    text = (text or "").strip()
    if not text or org_id <= 0:
        return None
    client = NocodbClient()
    if not _table_ready(client, NOCODB_TABLE_PA_OPEN_LOOPS):
        return None
    existing = find_loop_by_text(org_id, text)
    if existing:
        return existing
    payload = {
        "org_id": int(org_id),
        "text": text[:500],
        "intent": intent,
        "when_hint": (when_hint or "")[:80],
        "status": LOOP_STATUS_OPEN,
        "nudge_count": 0,
    }
    if due_at:
        payload["due_at"] = due_at
    if source_message_id:
        payload["created_from_message_id"] = int(source_message_id)
    try:
        row = client._post(NOCODB_TABLE_PA_OPEN_LOOPS, payload)
        _log.info("open_loop created  org=%d id=%s text=%r", org_id, row.get("Id"), text[:80])
        return row
    except Exception:
        _log.warning("create_open_loop failed  org=%d", org_id, exc_info=True)
        return None


def mark_loop_nudged(loop_id: int) -> None:
    client = NocodbClient()
    if not _table_ready(client, NOCODB_TABLE_PA_OPEN_LOOPS):
        return
    try:
        row = client._get(NOCODB_TABLE_PA_OPEN_LOOPS, params={
            "where": f"(Id,eq,{int(loop_id)})", "limit": 1,
        }).get("list", [])
        current_count = int((row[0] if row else {}).get("nudge_count") or 0)
        client._patch(NOCODB_TABLE_PA_OPEN_LOOPS, int(loop_id), {
            "Id": int(loop_id),
            "status": LOOP_STATUS_NUDGED,
            "last_nudged_at": _now_iso(),
            "nudge_count": current_count + 1,
        })
    except Exception:
        _log.warning("mark_loop_nudged failed  id=%d", loop_id, exc_info=True)


def resolve_loop(loop_id: int, note: str = "") -> None:
    client = NocodbClient()
    if not _table_ready(client, NOCODB_TABLE_PA_OPEN_LOOPS):
        return
    try:
        client._patch(NOCODB_TABLE_PA_OPEN_LOOPS, int(loop_id), {
            "Id": int(loop_id),
            "status": LOOP_STATUS_RESOLVED,
            "resolution_note": (note or "")[:500],
        })
    except Exception:
        _log.warning("resolve_loop failed  id=%d", loop_id, exc_info=True)


def drop_loop(loop_id: int, reason: str = "") -> None:
    client = NocodbClient()
    if not _table_ready(client, NOCODB_TABLE_PA_OPEN_LOOPS):
        return
    try:
        client._patch(NOCODB_TABLE_PA_OPEN_LOOPS, int(loop_id), {
            "Id": int(loop_id),
            "status": LOOP_STATUS_DROPPED,
            "resolution_note": (reason or "")[:500],
        })
    except Exception:
        _log.warning("drop_loop failed  id=%d", loop_id, exc_info=True)


# ── warm_topics ───────────────────────────────────────────────────────────────

def list_warm_topics(org_id: int, limit: int = 10, min_warmth: float = 0.1) -> list[dict]:
    client = NocodbClient()
    if not _table_ready(client, NOCODB_TABLE_PA_WARM_TOPICS):
        return []
    try:
        rows = client._get_paginated(NOCODB_TABLE_PA_WARM_TOPICS, params={
            "where": f"(org_id,eq,{int(org_id)})",
            "sort": "-warmth,-last_touched_at",
            "limit": min(max(1, limit), 100),
        })
    except Exception:
        _log.warning("list_warm_topics failed  org=%d", org_id, exc_info=True)
        return []
    return [r for r in rows if float(r.get("warmth") or 0) >= min_warmth]


def find_topic_by_phrase(org_id: int, phrase: str) -> dict | None:
    if not phrase:
        return None
    client = NocodbClient()
    if not _table_ready(client, NOCODB_TABLE_PA_WARM_TOPICS):
        return None
    needle = phrase.strip().lower()
    try:
        rows = client._get_paginated(NOCODB_TABLE_PA_WARM_TOPICS, params={
            "where": f"(org_id,eq,{int(org_id)})",
            "limit": 200,
        })
    except Exception:
        return None
    for r in rows:
        cand = (r.get("entity_or_phrase") or "").strip().lower()
        if cand and (cand == needle or cand in needle or needle in cand):
            return r
    return None


def upsert_warm_topic(
    org_id: int,
    entity_or_phrase: str,
    kind: str = TOPIC_KIND_INTEREST,
    warmth_boost: float = 0.3,
    max_warmth: float = 1.0,
) -> dict | None:
    phrase = (entity_or_phrase or "").strip()
    if not phrase or org_id <= 0:
        return None
    client = NocodbClient()
    if not _table_ready(client, NOCODB_TABLE_PA_WARM_TOPICS):
        return None
    existing = find_topic_by_phrase(org_id, phrase)
    now = _now_iso()
    if existing:
        current = float(existing.get("warmth") or 0)
        new_warmth = min(max_warmth, max(current, 0.0) + warmth_boost)
        try:
            client._patch(NOCODB_TABLE_PA_WARM_TOPICS, int(existing["Id"]), {
                "Id": int(existing["Id"]),
                "warmth": round(new_warmth, 3),
                "last_touched_at": now,
            })
            existing["warmth"] = new_warmth
            existing["last_touched_at"] = now
        except Exception:
            _log.warning("upsert_warm_topic patch failed  id=%s", existing.get("Id"), exc_info=True)
        return existing
    try:
        row = client._post(NOCODB_TABLE_PA_WARM_TOPICS, {
            "org_id": int(org_id),
            "entity_or_phrase": phrase[:255],
            "kind": kind,
            "warmth": round(min(max_warmth, max(0.0, warmth_boost)), 3),
            "last_touched_at": now,
            "sources": "[]",
        })
        _log.info("warm_topic created  org=%d phrase=%r kind=%s", org_id, phrase[:80], kind)
        return row
    except Exception:
        _log.warning("upsert_warm_topic insert failed  org=%d", org_id, exc_info=True)
        return None


def set_topic_brief(topic_id: int, brief: str, sources: list[dict] | None = None) -> None:
    client = NocodbClient()
    if not _table_ready(client, NOCODB_TABLE_PA_WARM_TOPICS):
        return
    payload: dict[str, Any] = {
        "Id": int(topic_id),
        "background_brief": (brief or "")[:4000],
        "sources": _dumps(sources or []),
    }
    try:
        client._patch(NOCODB_TABLE_PA_WARM_TOPICS, int(topic_id), payload)
    except Exception:
        _log.warning("set_topic_brief failed  id=%d", topic_id, exc_info=True)


def decay_warm_topics(org_id: int, half_life_hours: float = 48.0) -> None:
    """Exponentially decay warmth based on time since last_touched_at.

    Called periodically; safe to run often — no-op when last_touched_at is
    within a small delta.
    """
    client = NocodbClient()
    if not _table_ready(client, NOCODB_TABLE_PA_WARM_TOPICS):
        return
    try:
        rows = client._get_paginated(NOCODB_TABLE_PA_WARM_TOPICS, params={
            "where": f"(org_id,eq,{int(org_id)})",
            "limit": 200,
        })
    except Exception:
        return
    now = datetime.now(timezone.utc)
    for r in rows:
        try:
            last = r.get("last_touched_at") or r.get("CreatedAt")
            if not last:
                continue
            ts = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            hours = (now - ts).total_seconds() / 3600.0
            if hours < 1:
                continue
            current = float(r.get("warmth") or 0)
            new_warmth = current * (0.5 ** (hours / half_life_hours))
            if abs(new_warmth - current) < 0.02:
                continue
            client._patch(NOCODB_TABLE_PA_WARM_TOPICS, int(r["Id"]), {
                "Id": int(r["Id"]),
                "warmth": round(max(0.0, new_warmth), 3),
            })
        except Exception:
            _log.debug("decay skipped row  id=%s", r.get("Id"), exc_info=True)


def topic_sources(topic: dict) -> list[dict]:
    raw = _loads(topic.get("sources"))
    return raw if isinstance(raw, list) else []


# ── user_facts ────────────────────────────────────────────────────────────────

def list_user_facts(org_id: int, kind: str | None = None, limit: int = 100) -> list[dict]:
    client = NocodbClient()
    if not _table_ready(client, NOCODB_TABLE_PA_USER_FACTS):
        return []
    parts = [f"(org_id,eq,{int(org_id)})"]
    if kind:
        parts.append(f"(kind,eq,{kind})")
    try:
        return client._get_paginated(NOCODB_TABLE_PA_USER_FACTS, params={
            "where": "~and".join(parts),
            "sort": "-last_seen_at",
            "limit": min(max(1, limit), 500),
        })
    except Exception:
        _log.warning("list_user_facts failed  org=%d", org_id, exc_info=True)
        return []


def upsert_user_fact(
    org_id: int,
    kind: str,
    key: str,
    value: str,
    confidence: str = FACT_CONFIDENCE_OBSERVED,
    source_ref: str = "",
) -> dict | None:
    if org_id <= 0 or not kind or not key or not value:
        return None
    client = NocodbClient()
    if not _table_ready(client, NOCODB_TABLE_PA_USER_FACTS):
        return None
    safe_kind = _safe_where_value(kind)
    safe_key = _safe_where_value(key[:100])
    try:
        rows = client._get(NOCODB_TABLE_PA_USER_FACTS, params={
            "where": f"(org_id,eq,{int(org_id)})~and(kind,eq,{safe_kind})~and(key,eq,{safe_key})",
            "limit": 1,
        }).get("list", [])
    except Exception:
        rows = []
    now = _now_iso()
    if rows:
        row = rows[0]
        try:
            client._patch(NOCODB_TABLE_PA_USER_FACTS, int(row["Id"]), {
                "Id": int(row["Id"]),
                "value": value[:2000],
                "confidence": confidence,
                "source_ref": (source_ref or "")[:255],
                "last_seen_at": now,
            })
        except Exception:
            _log.warning("upsert_user_fact patch failed  id=%s", row.get("Id"), exc_info=True)
        return row
    try:
        return client._post(NOCODB_TABLE_PA_USER_FACTS, {
            "org_id": int(org_id),
            "kind": kind,
            "key": key[:100],
            "value": value[:2000],
            "confidence": confidence,
            "source_ref": (source_ref or "")[:255],
            "last_seen_at": now,
        })
    except Exception:
        _log.warning("upsert_user_fact insert failed  org=%d", org_id, exc_info=True)
        return None


def delete_user_fact(fact_id: int) -> None:
    client = NocodbClient()
    if not _table_ready(client, NOCODB_TABLE_PA_USER_FACTS):
        return
    try:
        # NocoDB v1 DELETE: requires full URL; use a minimal patch-then-ignore
        # approach by marking confidence=deleted; leave the row for audit.
        client._patch(NOCODB_TABLE_PA_USER_FACTS, int(fact_id), {
            "Id": int(fact_id),
            "confidence": "deleted",
            "last_seen_at": _now_iso(),
        })
    except Exception:
        _log.warning("delete_user_fact failed  id=%d", fact_id, exc_info=True)


# ── assistant_moves ───────────────────────────────────────────────────────────

def log_move(
    org_id: int,
    move_kind: str,
    mode: str,
    input_refs: dict | None = None,
    question_id: int | None = None,
) -> dict | None:
    client = NocodbClient()
    if not _table_ready(client, NOCODB_TABLE_PA_ASSISTANT_MOVES):
        return None
    try:
        return client._post(NOCODB_TABLE_PA_ASSISTANT_MOVES, {
            "org_id": int(org_id),
            "move_kind": move_kind,
            "mode": mode,
            "input_refs": _dumps(input_refs or {}),
            "question_id": int(question_id) if question_id else None,
            "engaged": None,
        })
    except Exception:
        _log.warning("log_move failed  org=%d kind=%s", org_id, move_kind, exc_info=True)
        return None


def recent_moves(org_id: int, since_hours: float = 24.0, limit: int = 40) -> list[dict]:
    client = NocodbClient()
    if not _table_ready(client, NOCODB_TABLE_PA_ASSISTANT_MOVES):
        return []
    try:
        rows = client._get_paginated(NOCODB_TABLE_PA_ASSISTANT_MOVES, params={
            "where": f"(org_id,eq,{int(org_id)})",
            "sort": "-CreatedAt",
            "limit": min(max(1, limit), 200),
        })
    except Exception:
        return []
    now = datetime.now(timezone.utc)
    out = []
    for r in rows:
        try:
            created = r.get("CreatedAt") or r.get("created_at")
            if not created:
                continue
            ts = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if (now - ts).total_seconds() / 3600.0 <= since_hours:
                out.append(r)
        except Exception:
            continue
    return out


def last_move_at(org_id: int, mode: str | None = None) -> datetime | None:
    rows = recent_moves(org_id, since_hours=720.0, limit=50)
    for r in rows:
        if mode and r.get("mode") != mode:
            continue
        created = r.get("CreatedAt") or r.get("created_at")
        if not created:
            continue
        try:
            ts = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts
        except Exception:
            continue
    return None


def move_engaged(move_id: int, engaged: bool) -> None:
    client = NocodbClient()
    if not _table_ready(client, NOCODB_TABLE_PA_ASSISTANT_MOVES):
        return
    try:
        client._patch(NOCODB_TABLE_PA_ASSISTANT_MOVES, int(move_id), {
            "Id": int(move_id),
            "engaged": 1 if engaged else 0,
        })
    except Exception:
        _log.warning("move_engaged failed  id=%d", move_id, exc_info=True)


def move_kind_cooldown_ok(org_id: int, move_kind: str, cooldown_hours: float) -> bool:
    """True when the last move of this kind is older than ``cooldown_hours``
    (or there is none)."""
    rows = recent_moves(org_id, since_hours=cooldown_hours + 1, limit=40)
    now = datetime.now(timezone.utc)
    for r in rows:
        if r.get("move_kind") != move_kind:
            continue
        created = r.get("CreatedAt") or r.get("created_at")
        if not created:
            continue
        try:
            ts = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if (now - ts).total_seconds() / 3600.0 < cooldown_hours:
                return False
        except Exception:
            continue
    return True


def engagement_bias(org_id: int, move_kind: str, window_days: int = 14) -> float:
    """Returns a 0.5–1.5 multiplier reflecting how often the user engages
    with this move kind. Neutral when no history."""
    rows = recent_moves(org_id, since_hours=window_days * 24, limit=100)
    relevant = [r for r in rows if r.get("move_kind") == move_kind and r.get("engaged") is not None]
    if len(relevant) < 3:
        return 1.0
    engaged = sum(1 for r in relevant if int(r.get("engaged") or 0) == 1)
    ratio = engaged / len(relevant)
    return 0.5 + ratio  # 0.5 (never engages) → 1.5 (always engages)
