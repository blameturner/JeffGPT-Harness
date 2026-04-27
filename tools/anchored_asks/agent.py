"""Deterministic producer for episodically-anchored questions.

Replaces the eager loop-question queueing the extractor used to do at chat
time, the picker-driven `closure` move, and the digest's stale-loop sweep.

Triggers off concrete state in `pa_open_loops` only — never off graph
shape or low-degree concepts. Each ask points at a real loop the user
made themselves (or the assistant committed to, separately surfaced).

The caps stack: at most 3 asks per run, per-loop hard cap of 2 nudges,
plus mute-key + engagement-block filters from the recall layer.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from infra.config import get_feature, is_feature_enabled
from shared.home_questions import queue_question_deduped
from shared.pa.memory import (
    LOOP_INTENT_DECISION,
    LOOP_INTENT_EVENT,
    LOOP_INTENT_TODO,
    LOOP_INTENT_WAITING,
    MOVE_KIND_CLOSURE,
    MOVE_MODE_PROACTIVE,
    log_move,
    mark_loop_nudged,
)
from shared.pa.recall import LoopRecall, build_recall

_log = logging.getLogger("anchored_asks")

_DECISION_MIN_AGE_HOURS = 24.0
_WAITING_MIN_AGE_HOURS = 72.0
_MAX_NUDGE_COUNT = 2

_TRIGGER_EVENT = "event_passed"
_TRIGGER_DECISION = "decision_stale"
_TRIGGER_WAITING = "waiting_stale"
_TRIGGER_TODO = "todo_overdue"


def _cfg(key: str, default):
    return get_feature("anchored_asks", key, default)


def _ask_for(loop: LoopRecall) -> tuple[str, str, list[dict], str] | None:
    """Return (question_text, trigger, suggested_options, followup_action) or None.

    None means the loop doesn't trigger an ask under any rule.
    """
    text = loop.text.strip()
    if not text:
        return None

    if loop.intent == LOOP_INTENT_EVENT and loop.is_overdue:
        return (
            f'How did "{text}" go?',
            _TRIGGER_EVENT,
            [
                {"label": "Went well", "value": "went_well"},
                {"label": "Mixed", "value": "mixed"},
                {"label": "Didn't happen", "value": "skipped"},
            ],
            "",
        )

    if (
        loop.intent == LOOP_INTENT_DECISION
        and loop.age_hours >= _DECISION_MIN_AGE_HOURS
        and loop.nudge_count < _MAX_NUDGE_COUNT
    ):
        return (
            f'Decision still open: "{text}". Want me to dig in, or is this on you?',
            _TRIGGER_DECISION,
            [
                {"label": "Research this", "value": f"research:{text}"},
                {"label": "Still thinking", "value": "defer"},
                {"label": "Already decided", "value": "done"},
            ],
            f"enqueue:research:{text}",
        )

    if (
        loop.intent == LOOP_INTENT_WAITING
        and loop.age_hours >= _WAITING_MIN_AGE_HOURS
        and loop.nudge_count < _MAX_NUDGE_COUNT
    ):
        return (
            f'Still waiting on "{text}"? Any movement?',
            _TRIGGER_WAITING,
            [
                {"label": "No movement", "value": "waiting"},
                {"label": "Got the answer", "value": "resolved"},
                {"label": "Drop it", "value": "drop"},
            ],
            "",
        )

    if (
        loop.intent == LOOP_INTENT_TODO
        and loop.is_overdue
        and loop.nudge_count < _MAX_NUDGE_COUNT
    ):
        return (
            f'Did "{text}" get done?',
            _TRIGGER_TODO,
            [
                {"label": "Done", "value": "resolved"},
                {"label": "Still working", "value": "nudge"},
                {"label": "Drop it", "value": "drop"},
            ],
            "",
        )

    return None


def run_anchored_asks(org_id: int, now: datetime | None = None) -> dict:
    """Single tick. Returns a status dict for logging."""
    out: dict = {
        "status": "ok",
        "org_id": int(org_id),
        "asks_queued": 0,
        "candidates_seen": 0,
        "skipped_capped": 0,
        "skipped_no_table": 0,
    }
    if not is_feature_enabled("pa"):
        out["status"] = "skipped"
        out["reason"] = "pa_disabled"
        return out
    if not _cfg("enabled", True):
        out["status"] = "skipped"
        out["reason"] = "anchored_asks_disabled"
        return out
    if int(org_id) <= 0:
        out["status"] = "error"
        out["reason"] = "invalid_org_id"
        return out

    if now is None:
        now = datetime.now(timezone.utc)

    payload = build_recall(int(org_id), now=now)
    # Only user-side loops get asks. Assistant commitments are surfaced by
    # the brief itself ("things I owe you") so we don't ask the user about them.
    candidates = payload.open_loops_user
    out["candidates_seen"] = len(candidates)

    max_per_run = int(_cfg("max_per_run", 3) or 3)
    queued = 0
    for loop in candidates:
        if queued >= max_per_run:
            out["skipped_capped"] += 1
            continue
        triple = _ask_for(loop)
        if triple is None:
            continue
        question_text, trigger, opts, followup = triple
        context_ref = f"loop:{loop.id}:{trigger}"
        try:
            qid = queue_question_deduped(
                org_id=int(org_id),
                question_text=question_text,
                context_ref=context_ref,
                suggested_options=opts,
                followup_action=followup,
            )
        except Exception:
            _log.warning("anchored_asks: queue failed  loop=%d", loop.id, exc_info=True)
            continue
        if qid is None:
            # already queued for this loop+trigger pair — that's the dedup happy path
            continue
        try:
            log_move(
                int(org_id),
                MOVE_KIND_CLOSURE,
                MOVE_MODE_PROACTIVE,
                input_refs={"loop_id": loop.id, "trigger": trigger},
                question_id=qid,
            )
        except Exception:
            _log.debug("anchored_asks: log_move failed  loop=%d", loop.id, exc_info=True)
        try:
            mark_loop_nudged(loop.id)
        except Exception:
            _log.debug("anchored_asks: mark_loop_nudged failed  loop=%d", loop.id, exc_info=True)
        queued += 1

    out["asks_queued"] = queued
    return out
