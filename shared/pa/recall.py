"""Episodic recall layer for the home assistant.

Single read-only function that gathers everything the daily-brief and
anchored-asks producers need from existing tables. No new schema. The
result is pre-filtered by user-stated mute preferences and engagement
history so producers consume a clean payload and never re-apply filters.

Design principle: every field has to point at *evidence the user cares*
(an open loop, a recent conversation, a stated preference, completed
research). Graph-shape signals (sparse concepts, shortest-path bridges)
are deliberately absent — they were the source of the "abstract question
about a topic I never cared about" failure mode.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from infra.config import NOCODB_TABLE_ASSISTANT_QUESTIONS, NOCODB_TABLE_CONVERSATIONS
from infra.nocodb_client import NocodbClient
from shared.pa.memory import (
    LOOP_STATUS_OPEN,
    LOOP_STATUS_NUDGED,
)

_log = logging.getLogger("pa.recall")

# ── tunables (kept here, not config.json — these are recall semantics) ────────

_YESTERDAY_TAIL_HOURS = 36
_YESTERDAY_TAIL_MAX_CONVS = 5
_YESTERDAY_TAIL_MSGS_PER_CONV = 5
_THREAD_OF_DAY_MSG_LIMIT = 8
_WARM_TOPIC_RECENCY_HOURS = 24 * 7
_WARM_TOPIC_MIN_WARMTH = 0.3
_PENDING_ASKS_RECENT_HOURS = 6
_RECENT_BRIEFS_DAYS = 7
_COMPLETED_RESEARCH_HOURS = 24
_ENGAGEMENT_LOOKBACK_DAYS = 14
_ENGAGEMENT_BLOCK_THRESHOLD = 2


# ── modes ─────────────────────────────────────────────────────────────────────

MODE_MONDAY_MORNING = "monday_morning"
MODE_MIDWEEK_MORNING = "midweek_morning"
MODE_WEEKDAY_MIDDAY = "weekday_midday"
MODE_FRIDAY_PM = "friday_pm"
MODE_WEEKEND = "weekend"


# ── payload types ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TimeContext:
    now: datetime
    weekday: int                  # 0=Mon .. 6=Sun
    is_weekend: bool
    part_of_day: str              # morning | midday | afternoon | evening
    mode: str
    days_since_last_home_message: float | None


@dataclass(frozen=True)
class ConversationTail:
    conversation_id: int
    title: str
    kind: str
    last_activity_at: str
    msgs_24h: int
    msgs_72h: int
    messages: list[dict]          # [{role, content, created_at}]


@dataclass(frozen=True)
class LoopRecall:
    id: int
    text: str
    intent: str
    when_hint: str
    status: str
    nudge_count: int
    age_hours: float
    is_overdue: bool
    due_at: str | None
    source_ref: str


@dataclass(frozen=True)
class RecentStyle:
    avg_user_chars: int           # rolling avg over last N user msgs
    register: str                 # terse | conversational | long
    sample_count: int


@dataclass(frozen=True)
class RecallPayload:
    time_context: TimeContext
    yesterday_tail: list[ConversationTail]
    thread_of_day: ConversationTail | None
    open_loops_user: list[LoopRecall]
    open_loops_assistant: list[LoopRecall]
    projects_and_routines: list[dict]
    warm_topics: list[dict]
    pending_anchored_asks: list[dict]
    recent_briefs: list[dict]
    completed_research: list[dict]
    mute_keys: list[str]          # phrases to never mention
    engagement_blocks: list[str]  # entities/loop-texts the user has ignored 2+ times
    has_signal: bool              # short-circuit for the silence gate
    their_last_word: dict | None  # the user's last message in the hottest non-home conv
    recent_corrections: list[str] # "avoid:*" constraints from corrections
    recent_style: RecentStyle     # rolling user-message-length style
    recent_assistant_openings: list[str]  # first 60 chars of last N assistant home turns

    def as_prompt_dict(self) -> dict:
        """Serialisable form for prompt construction — drops dataclass overhead."""
        return {
            "time_context": {
                "now": self.time_context.now.isoformat(),
                "weekday": self.time_context.weekday,
                "is_weekend": self.time_context.is_weekend,
                "part_of_day": self.time_context.part_of_day,
                "mode": self.time_context.mode,
                "days_since_last_home_message": self.time_context.days_since_last_home_message,
            },
            "yesterday_tail": [_tail_to_dict(t) for t in self.yesterday_tail],
            "thread_of_day": _tail_to_dict(self.thread_of_day) if self.thread_of_day else None,
            "open_loops_user": [_loop_to_dict(l) for l in self.open_loops_user],
            "open_loops_assistant": [_loop_to_dict(l) for l in self.open_loops_assistant],
            "projects_and_routines": list(self.projects_and_routines),
            "warm_topics": list(self.warm_topics),
            "pending_anchored_asks": list(self.pending_anchored_asks),
            "recent_briefs": list(self.recent_briefs),
            "completed_research": list(self.completed_research),
            "mute_keys": list(self.mute_keys),
            "engagement_blocks": list(self.engagement_blocks),
            "has_signal": self.has_signal,
            "their_last_word": self.their_last_word,
            "recent_corrections": list(self.recent_corrections),
            "recent_style": {
                "avg_user_chars": self.recent_style.avg_user_chars,
                "register": self.recent_style.register,
                "sample_count": self.recent_style.sample_count,
            },
            "recent_assistant_openings": list(self.recent_assistant_openings),
        }


def _tail_to_dict(t: ConversationTail) -> dict:
    return {
        "conversation_id": t.conversation_id,
        "title": t.title,
        "kind": t.kind,
        "last_activity_at": t.last_activity_at,
        "msgs_24h": t.msgs_24h,
        "msgs_72h": t.msgs_72h,
        "messages": list(t.messages),
    }


def _loop_to_dict(l: LoopRecall) -> dict:
    return {
        "id": l.id,
        "text": l.text,
        "intent": l.intent,
        "when_hint": l.when_hint,
        "status": l.status,
        "nudge_count": l.nudge_count,
        "age_hours": round(l.age_hours, 1),
        "is_overdue": l.is_overdue,
        "due_at": l.due_at,
        "source_ref": l.source_ref,
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_iso(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _hours_between(a: datetime, b: datetime) -> float:
    return (a - b).total_seconds() / 3600.0


def _part_of_day(now: datetime) -> str:
    h = now.hour
    if h < 11:
        return "morning"
    if h < 14:
        return "midday"
    if h < 18:
        return "afternoon"
    return "evening"


def _mode_for(now: datetime, part: str) -> str:
    wd = now.weekday()
    if wd >= 5:
        return MODE_WEEKEND
    if wd == 4 and part in ("afternoon", "evening"):
        return MODE_FRIDAY_PM
    if part == "midday":
        return MODE_WEEKDAY_MIDDAY
    if wd == 0 and part == "morning":
        return MODE_MONDAY_MORNING
    if part == "morning":
        return MODE_MIDWEEK_MORNING
    # afternoon/evening on Mon–Thu fall through to midweek_morning shape
    return MODE_MIDWEEK_MORNING


def _safe_decode_json_list(raw: Any) -> list:
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        out = json.loads(raw)
        return out if isinstance(out, list) else []
    except (TypeError, ValueError):
        return []


def _matches_mute(text: str, mute_keys: list[str]) -> bool:
    if not text or not mute_keys:
        return False
    low = text.lower()
    return any(k.lower() in low for k in mute_keys if k)


def _recent_rows_with_python_cutoff(
    client: NocodbClient,
    table: str,
    *,
    org_id: int,
    cutoff: datetime,
    extra_where: list[str] | None = None,
    sort: str = "-CreatedAt",
    limit: int = 500,
) -> list[dict]:
    """Fetch recent rows while tolerating NocoDB's flaky datetime `where` parsing.

    Some deployments reject `CreatedAt,gt,<timestamp>` with HTTP 422 depending on
    timestamp shape / server version. We still try the narrow query first for
    efficiency, then fall back to an org-scoped query and apply the cutoff in
    Python so recall never goes blank.
    """
    where_parts = [f"(org_id,eq,{org_id})", *(extra_where or [])]
    cutoff_candidates = [
        cutoff.strftime("%Y-%m-%d %H:%M:%S"),
        cutoff.strftime("%Y-%m-%dT%H:%M:%S"),
        cutoff.isoformat(),
    ]
    last_422: requests.HTTPError | None = None
    for raw_cutoff in cutoff_candidates:
        try:
            return client._get_paginated(table, params={
                "where": "~and".join([*where_parts, f"(CreatedAt,gt,{raw_cutoff})"]),
                "sort": sort,
                "limit": limit,
            })
        except requests.HTTPError as exc:
            if getattr(exc.response, "status_code", None) != 422:
                raise
            last_422 = exc
        except Exception:
            raise

    if last_422 is not None:
        _log.info(
            "recent_rows fallback to python cutoff  table=%s org=%d status=%s",
            table,
            org_id,
            getattr(last_422.response, "status_code", "?"),
        )

    rows = client._get_paginated(table, params={
        "where": "~and".join(where_parts),
        "sort": sort,
        "limit": limit,
    })
    return [r for r in rows if (_parse_iso(r.get("CreatedAt")) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff]


# ── time context ──────────────────────────────────────────────────────────────

def _build_time_context(client: NocodbClient, org_id: int, now: datetime) -> TimeContext:
    part = _part_of_day(now)
    mode = _mode_for(now, part)
    days_since: float | None = None
    try:
        rows = client._get_paginated("conversations", params={
            "where": f"(org_id,eq,{org_id})~and(kind,eq,home)",
            "limit": 1,
        })
        if rows:
            conv_id = int(rows[0].get("Id") or 0)
            if conv_id:
                msg_rows = client._get_paginated("messages", params={
                    "where": f"(conversation_id,eq,{conv_id})~and(org_id,eq,{org_id})",
                    "sort": "-CreatedAt",
                    "limit": 1,
                })
                if msg_rows:
                    last = _parse_iso(msg_rows[0].get("CreatedAt"))
                    if last is not None:
                        days_since = (now - last).total_seconds() / 86400.0
    except Exception:
        _log.debug("time_context: last home message lookup failed", exc_info=True)
    return TimeContext(
        now=now,
        weekday=now.weekday(),
        is_weekend=now.weekday() >= 5,
        part_of_day=part,
        mode=mode,
        days_since_last_home_message=days_since,
    )


# ── yesterday tail + thread of day ────────────────────────────────────────────

def _build_tails(
    client: NocodbClient,
    org_id: int,
    now: datetime,
) -> tuple[list[ConversationTail], ConversationTail | None]:
    cutoff = now - timedelta(hours=_YESTERDAY_TAIL_HOURS)
    try:
        msg_rows = _recent_rows_with_python_cutoff(
            client,
            "messages",
            org_id=org_id,
            cutoff=cutoff,
            sort="-CreatedAt",
            limit=500,
        )
    except Exception:
        _log.warning("yesterday_tail: messages query failed  org=%d", org_id, exc_info=True)
        return [], None

    by_conv: dict[int, list[dict]] = {}
    for r in msg_rows:
        try:
            cid = int(r.get("conversation_id") or 0)
        except (TypeError, ValueError):
            continue
        if cid <= 0:
            continue
        by_conv.setdefault(cid, []).append(r)

    if not by_conv:
        return [], None

    try:
        conv_rows = client._get_paginated("conversations", params={
            "where": f"(org_id,eq,{org_id})",
            "limit": 200,
        })
    except Exception:
        conv_rows = []
    conv_index = {int(c.get("Id") or 0): c for c in conv_rows if c.get("Id")}

    tails: list[ConversationTail] = []
    cutoff_24h = now - timedelta(hours=24)
    cutoff_72h = now - timedelta(hours=72)
    for cid, rows in by_conv.items():
        meta = conv_index.get(cid, {})
        kind = (meta.get("kind") or "").strip()
        if kind == "home":
            # the home convo is the surface; don't echo it back to itself
            continue
        rows.sort(key=lambda r: str(r.get("CreatedAt") or ""), reverse=True)
        msgs_24h = sum(1 for r in rows if (_parse_iso(r.get("CreatedAt")) or now) >= cutoff_24h)
        msgs_72h = sum(1 for r in rows if (_parse_iso(r.get("CreatedAt")) or now) >= cutoff_72h)
        last_at = str(rows[0].get("CreatedAt") or "") if rows else ""
        msgs_for_payload = []
        for r in rows[:_YESTERDAY_TAIL_MSGS_PER_CONV]:
            content = (r.get("content") or "").strip()
            if not content:
                continue
            msgs_for_payload.append({
                "role": r.get("role") or "",
                "content": content[:1200],
                "created_at": r.get("CreatedAt") or "",
            })
        msgs_for_payload.reverse()  # chronological for prompt readability
        if not msgs_for_payload:
            continue
        tails.append(ConversationTail(
            conversation_id=cid,
            title=(meta.get("title") or "").strip(),
            kind=kind,
            last_activity_at=last_at,
            msgs_24h=msgs_24h,
            msgs_72h=msgs_72h,
            messages=msgs_for_payload,
        ))

    if not tails:
        return [], None

    tails.sort(key=lambda t: str(t.last_activity_at), reverse=True)
    tails = tails[:_YESTERDAY_TAIL_MAX_CONVS]

    # thread of day: highest score on (msgs_24h*2 + msgs_72h)
    scored = sorted(tails, key=lambda t: (t.msgs_24h * 2 + t.msgs_72h), reverse=True)
    top = scored[0] if scored else None
    if top is not None and top.msgs_24h == 0 and top.msgs_72h <= 1:
        top = None
    return tails, top


# ── loops ─────────────────────────────────────────────────────────────────────

def _row_to_loop(row: dict, now: datetime) -> LoopRecall | None:
    try:
        loop_id = int(row.get("Id") or 0)
    except (TypeError, ValueError):
        return None
    if loop_id <= 0:
        return None
    text = (row.get("text") or "").strip()
    if not text:
        return None
    created = _parse_iso(row.get("CreatedAt") or row.get("created_at"))
    age_h = _hours_between(now, created) if created else 0.0
    due_at_raw = row.get("due_at")
    due = _parse_iso(due_at_raw)
    is_overdue = bool(due and due < now)
    return LoopRecall(
        id=loop_id,
        text=text,
        intent=str(row.get("intent") or ""),
        when_hint=str(row.get("when_hint") or ""),
        status=str(row.get("status") or ""),
        nudge_count=int(row.get("nudge_count") or 0),
        age_hours=age_h,
        is_overdue=is_overdue,
        due_at=str(due_at_raw) if due_at_raw else None,
        source_ref=str(row.get("source_ref") or ""),
    )


def _build_loops(
    client: NocodbClient,
    org_id: int,
    now: datetime,
) -> tuple[list[LoopRecall], list[LoopRecall]]:
    from infra.config import NOCODB_TABLE_PA_OPEN_LOOPS
    if NOCODB_TABLE_PA_OPEN_LOOPS not in client.tables:
        return [], []
    try:
        rows = client._get_paginated(NOCODB_TABLE_PA_OPEN_LOOPS, params={
            "where": (
                f"(org_id,eq,{org_id})~and(status,eq,{LOOP_STATUS_OPEN})"
                f"~or((org_id,eq,{org_id})~and(status,eq,{LOOP_STATUS_NUDGED}))"
            ),
            "sort": "-CreatedAt",
            "limit": 100,
        })
    except Exception:
        _log.warning("recall: list open_loops failed  org=%d", org_id, exc_info=True)
        return [], []
    user_loops: list[LoopRecall] = []
    assistant_loops: list[LoopRecall] = []
    for r in rows:
        loop = _row_to_loop(r, now)
        if loop is None:
            continue
        if loop.source_ref.startswith("assistant_commitment:"):
            assistant_loops.append(loop)
        else:
            user_loops.append(loop)
    return user_loops, assistant_loops


# ── facts (projects + routines + mutes) ───────────────────────────────────────

def _build_facts(client: NocodbClient, org_id: int) -> tuple[list[dict], list[str]]:
    from infra.config import NOCODB_TABLE_PA_USER_FACTS
    if NOCODB_TABLE_PA_USER_FACTS not in client.tables:
        return [], []
    try:
        rows = client._get_paginated(NOCODB_TABLE_PA_USER_FACTS, params={
            "where": f"(org_id,eq,{org_id})",
            "limit": 300,
        })
    except Exception:
        return [], []

    facts: list[dict] = []
    mute_keys: list[str] = []
    for r in rows:
        kind = (r.get("kind") or "").strip()
        key = (r.get("key") or "").strip()
        value = (r.get("value") or "").strip()
        confidence = (r.get("confidence") or "").strip()
        if confidence == "deleted":
            continue
        if kind == "preference" and key.startswith("mute:"):
            slug = key[5:].strip()
            if slug:
                mute_keys.append(slug)
            continue
        if kind in ("routine", "project", "relationship", "constraint"):
            facts.append({
                "kind": kind,
                "key": key,
                "value": value,
                "confidence": confidence,
            })
    return facts, mute_keys


# ── warm topics ───────────────────────────────────────────────────────────────

def _build_warm_topics(client: NocodbClient, org_id: int, now: datetime) -> list[dict]:
    from infra.config import NOCODB_TABLE_PA_WARM_TOPICS
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
    cutoff = now - timedelta(hours=_WARM_TOPIC_RECENCY_HOURS)
    out: list[dict] = []
    for r in rows:
        warmth = float(r.get("warmth") or 0)
        if warmth < _WARM_TOPIC_MIN_WARMTH:
            continue
        touched = _parse_iso(r.get("last_touched_at"))
        if touched is None or touched < cutoff:
            continue
        phrase = (r.get("entity_or_phrase") or "").strip()
        if not phrase:
            continue
        out.append({
            "id": r.get("Id"),
            "phrase": phrase,
            "kind": r.get("kind") or "",
            "warmth": round(warmth, 3),
            "last_touched_at": r.get("last_touched_at"),
        })
    return out


# ── pending asks, recent briefs, completed research ───────────────────────────

def _build_pending_asks(client: NocodbClient, org_id: int, now: datetime) -> list[dict]:
    if NOCODB_TABLE_ASSISTANT_QUESTIONS not in client.tables:
        return []
    cutoff = now - timedelta(hours=_PENDING_ASKS_RECENT_HOURS)
    try:
        rows = client._get_paginated(NOCODB_TABLE_ASSISTANT_QUESTIONS, params={
            "where": f"(org_id,eq,{org_id})~and(status,eq,pending)",
            "sort": "-CreatedAt",
            "limit": 30,
        })
    except Exception:
        return []
    out: list[dict] = []
    for r in rows:
        created = _parse_iso(r.get("CreatedAt"))
        if created is None or created < cutoff:
            continue
        out.append({
            "id": r.get("Id"),
            "question_text": r.get("question_text") or "",
            "context_ref": r.get("context_ref") or "",
            "suggested_options": _safe_decode_json_list(r.get("suggested_options")),
        })
    return out


def _build_recent_briefs(client: NocodbClient, org_id: int, now: datetime) -> list[dict]:
    if "insights" not in client.tables:
        return []
    cutoff = now - timedelta(days=_RECENT_BRIEFS_DAYS)
    try:
        rows = client._get_paginated("insights", params={
            "where": f"(org_id,eq,{org_id})~and(trigger,eq,daily_brief)",
            "sort": "-CreatedAt",
            "limit": 10,
        })
    except Exception:
        return []
    out: list[dict] = []
    for r in rows:
        created = _parse_iso(r.get("CreatedAt"))
        if created is None or created < cutoff:
            continue
        out.append({
            "id": r.get("Id"),
            "title": (r.get("title") or "").strip(),
            "summary": (r.get("summary") or "").strip(),
            "topic": (r.get("topic") or "").strip(),
            "created_at": r.get("CreatedAt"),
        })
    return out


def _build_completed_research(client: NocodbClient, org_id: int, now: datetime) -> list[dict]:
    if "research_plans" not in client.tables:
        return []
    cutoff = now - timedelta(hours=_COMPLETED_RESEARCH_HOURS)
    try:
        rows = client._get_paginated("research_plans", params={
            "where": f"(org_id,eq,{org_id})~and(status,eq,completed)",
            "sort": "-CreatedAt",
            "limit": 10,
        })
    except Exception:
        return []
    out: list[dict] = []
    for r in rows:
        completed = _parse_iso(
            r.get("completed_at") or r.get("UpdatedAt") or r.get("CreatedAt")
        )
        if completed is None or completed < cutoff:
            continue
        out.append({
            "plan_id": r.get("Id"),
            "topic": (r.get("topic") or "").strip(),
            "summary": (r.get("summary") or "").strip()[:400],
            "completed_at": completed.isoformat(),
        })
    return out


# ── engagement blocks ─────────────────────────────────────────────────────────

def _build_engagement_blocks(client: NocodbClient, org_id: int, now: datetime) -> list[str]:
    from infra.config import NOCODB_TABLE_PA_ASSISTANT_MOVES
    if NOCODB_TABLE_PA_ASSISTANT_MOVES not in client.tables:
        return []
    cutoff = now - timedelta(days=_ENGAGEMENT_LOOKBACK_DAYS)
    try:
        rows = client._get_paginated(NOCODB_TABLE_PA_ASSISTANT_MOVES, params={
            "where": f"(org_id,eq,{org_id})",
            "sort": "-CreatedAt",
            "limit": 200,
        })
    except Exception:
        return []
    counts: dict[str, int] = {}
    for r in rows:
        engaged = r.get("engaged")
        if engaged is None:
            continue
        try:
            if int(engaged) != 0:
                continue
        except (TypeError, ValueError):
            continue
        created = _parse_iso(r.get("CreatedAt"))
        if created is None or created < cutoff:
            continue
        refs = r.get("input_refs")
        if isinstance(refs, str):
            try:
                refs = json.loads(refs)
            except (TypeError, ValueError):
                refs = {}
        if not isinstance(refs, dict):
            continue
        # collect any reference token that identifies a topic/loop
        for k in ("entity", "topic", "phrase", "a", "b"):
            v = refs.get(k)
            if isinstance(v, str) and v.strip():
                counts[v.strip()] = counts.get(v.strip(), 0) + 1
        if refs.get("loop_id") is not None:
            tok = f"loop:{refs.get('loop_id')}"
            counts[tok] = counts.get(tok, 0) + 1
    return [k for k, c in counts.items() if c >= _ENGAGEMENT_BLOCK_THRESHOLD]


# ── style, last word, corrections, assistant openings ────────────────────────

def _build_recent_style_and_openings(
    client: NocodbClient,
    org_id: int,
    home_conv_id: int | None,
) -> tuple[RecentStyle, list[str]]:
    """Compute rolling user-message length register and recent assistant
    opening phrases for the home conversation.

    Both are cheap reads on the existing ``messages`` table for the home
    convo. Returned together because they share the query.
    """
    style = RecentStyle(avg_user_chars=0, register="conversational", sample_count=0)
    openings: list[str] = []
    if not home_conv_id:
        return style, openings
    try:
        rows = client._get_paginated("messages", params={
            "where": f"(conversation_id,eq,{int(home_conv_id)})~and(org_id,eq,{org_id})",
            "sort": "-CreatedAt",
            "limit": 30,
        })
    except Exception:
        return style, openings
    user_lengths: list[int] = []
    for r in rows:
        role = r.get("role") or ""
        content = (r.get("content") or "").strip()
        if not content:
            continue
        if role == "user" and len(user_lengths) < 8:
            user_lengths.append(len(content))
        elif role == "assistant" and len(openings) < 5:
            head = content.split("\n", 1)[0].strip()
            openings.append(head[:80])
        if len(user_lengths) >= 8 and len(openings) >= 5:
            break
    if user_lengths:
        avg = sum(user_lengths) // len(user_lengths)
        if avg < 60:
            register = "terse"
        elif avg > 280:
            register = "long"
        else:
            register = "conversational"
        style = RecentStyle(
            avg_user_chars=avg,
            register=register,
            sample_count=len(user_lengths),
        )
    return style, openings


def _their_last_word(tails: list[ConversationTail]) -> dict | None:
    """The user's last message in the most recently-active non-home conversation."""
    for tail in tails:
        for msg in reversed(tail.messages):
            if (msg.get("role") or "") == "user":
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                return {
                    "conversation_id": tail.conversation_id,
                    "title": tail.title,
                    "content": content[:600],
                    "created_at": msg.get("created_at"),
                }
    return None


def _build_recent_corrections(client: NocodbClient, org_id: int) -> list[str]:
    """Collect ``avoid:*`` constraints the user (or extractor) has flagged.

    These are persisted as ``pa_user_facts`` rows of ``kind="constraint"``,
    ``key="avoid:<short tag>"`` — surfacing them every turn keeps the
    assistant from making the same mistake twice.
    """
    from infra.config import NOCODB_TABLE_PA_USER_FACTS
    if NOCODB_TABLE_PA_USER_FACTS not in client.tables:
        return []
    try:
        rows = client._get_paginated(NOCODB_TABLE_PA_USER_FACTS, params={
            "where": f"(org_id,eq,{org_id})~and(kind,eq,constraint)",
            "sort": "-last_seen_at",
            "limit": 30,
        })
    except Exception:
        return []
    out: list[str] = []
    for r in rows:
        if (r.get("confidence") or "") == "deleted":
            continue
        key = (r.get("key") or "").strip()
        if not key.startswith("avoid:"):
            continue
        value = (r.get("value") or "").strip()
        out.append(f"{key[6:]}: {value}" if value else key[6:])
    return out[:10]


def _home_conv_id(client: NocodbClient, org_id: int) -> int | None:
    try:
        rows = client._get_paginated("conversations", params={
            "where": f"(org_id,eq,{org_id})~and(kind,eq,home)",
            "limit": 1,
        })
        if rows:
            return int(rows[0].get("Id") or 0) or None
    except Exception:
        pass
    return None


# ── main entry ────────────────────────────────────────────────────────────────

def build_recall(org_id: int, now: datetime | None = None) -> RecallPayload:
    """Read everything the producers need. Pure read, single function call.

    Always returns a RecallPayload — never raises. Failures collapse to
    empty fields so producers can decide silence vs. partial render.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    client = NocodbClient()

    time_ctx = _build_time_context(client, org_id, now)
    tails, thread_of_day = _build_tails(client, org_id, now)
    user_loops, assistant_loops = _build_loops(client, org_id, now)
    facts, mute_keys = _build_facts(client, org_id)
    warm_topics = _build_warm_topics(client, org_id, now)
    pending_asks = _build_pending_asks(client, org_id, now)
    recent_briefs = _build_recent_briefs(client, org_id, now)
    completed_research = _build_completed_research(client, org_id, now)
    engagement_blocks = _build_engagement_blocks(client, org_id, now)
    home_id = _home_conv_id(client, org_id)
    recent_style, recent_openings = _build_recent_style_and_openings(client, org_id, home_id)
    their_last_word = _their_last_word(tails)
    recent_corrections = _build_recent_corrections(client, org_id)

    # ── apply mute + engagement filters as the final pass ─────────────────────
    block_set = {b.lower() for b in engagement_blocks}

    def _allow_topic(phrase: str, topic_id: Any = None) -> bool:
        if _matches_mute(phrase, mute_keys):
            return False
        if phrase.lower() in block_set:
            return False
        return True

    def _allow_loop(loop: LoopRecall) -> bool:
        if _matches_mute(loop.text, mute_keys):
            return False
        if f"loop:{loop.id}" in block_set:
            return False
        if loop.text.lower() in block_set:
            return False
        return True

    warm_topics = [t for t in warm_topics if _allow_topic(t.get("phrase", ""))]
    user_loops = [l for l in user_loops if _allow_loop(l)]
    assistant_loops = [l for l in assistant_loops if _allow_loop(l)]
    pending_asks = [
        a for a in pending_asks
        if not _matches_mute(a.get("question_text", ""), mute_keys)
    ]

    # ── compute has_signal (silence-gate input) ───────────────────────────────
    has_overdue = any(l.is_overdue for l in user_loops + assistant_loops)
    has_hot_conv = any(t.msgs_24h >= 1 or t.msgs_72h >= 3 for t in tails)
    has_research = bool(completed_research)
    has_event_passed = any(
        l.intent == "event" and l.is_overdue for l in user_loops
    )
    has_assistant_owed = bool(assistant_loops)
    has_signal = (
        has_overdue
        or has_hot_conv
        or has_research
        or has_event_passed
        or has_assistant_owed
    )

    return RecallPayload(
        time_context=time_ctx,
        yesterday_tail=tails,
        thread_of_day=thread_of_day,
        open_loops_user=user_loops,
        open_loops_assistant=assistant_loops,
        projects_and_routines=facts,
        warm_topics=warm_topics,
        pending_anchored_asks=pending_asks,
        recent_briefs=recent_briefs,
        completed_research=completed_research,
        mute_keys=mute_keys,
        engagement_blocks=engagement_blocks,
        has_signal=has_signal,
        their_last_word=their_last_word,
        recent_corrections=recent_corrections,
        recent_style=recent_style,
        recent_assistant_openings=recent_openings,
    )
