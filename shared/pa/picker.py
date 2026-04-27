"""Scores candidate PA moves, enforces cadence rules, picks one to surface."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from infra.config import is_feature_enabled
from shared.pa.memory import (
    MOVE_KIND_CLOSURE,
    MOVE_KIND_CONNECT,
    MOVE_KIND_NEWS,
    MOVE_KIND_SERENDIPITY,
    MOVE_MODE_PROACTIVE,
    engagement_bias,
    last_move_at,
    log_move,
    mark_loop_nudged,
    move_kind_cooldown_ok,
    recent_moves,
)
from shared.pa.moves import (
    MoveCandidate,
    closure,
    connect,
    news_watch,
    serendipity,
)

_log = logging.getLogger("pa.picker")

PROACTIVE_MIN_GAP_HOURS = 4.0

COOLDOWNS: dict[str, float] = {
    MOVE_KIND_CLOSURE: 6.0,
    MOVE_KIND_CONNECT: 24.0,
    MOVE_KIND_NEWS: 8.0,
    MOVE_KIND_SERENDIPITY: 48.0,
}

_MOVES: list[tuple[str, callable]] = [
    # NOTE: emptied 2026-04-27 as part of the personal-assistant redesign.
    # The graph-shape-driven moves (serendipity, connect) and the warmth-only
    # news_watch produced "abstract questions about topics the user never
    # cared about". Closure was replaced by the deterministic
    # tools/anchored_asks producer which fires on real loop state, not on a
    # picker tick. This list is intentionally empty; the picker framework
    # remains for future moves that pass the "useful, not novelty" bar.
]


def _rejection_penalty(org_id: int) -> float:
    try:
        rows = recent_moves(org_id, since_hours=24, limit=40)
    except Exception:
        return 1.0
    # Only proactive surfaces count — inline/background drops aren't
    # user-visible asks so shouldn't feed the rejection signal.
    proactive = [r for r in rows if r.get("mode") == MOVE_MODE_PROACTIVE]
    last_three = proactive[:3]
    if len(last_three) < 3:
        return 1.0
    for r in last_three:
        engaged = r.get("engaged")
        if engaged is None:
            return 1.0
        try:
            if int(engaged) != 0:
                return 1.0
        except (TypeError, ValueError):
            return 1.0
    return 2.0


def pick_proactive_move(org_id: int, ignore_global_gap: bool = False) -> MoveCandidate | None:
    """Returns the move to surface now, or None for silence.

    Enforces global cadence + per-kind cooldowns before calling move functions.
    Set ``ignore_global_gap=True`` to bypass the 4h proactive-surface gap
    (e.g. manual user trigger). Per-kind cooldowns still apply.
    Never raises.
    """
    try:
        if not is_feature_enabled("pa"):
            return None
    except Exception:
        _log.warning("pick_proactive_move: feature gate failed", exc_info=True)
        return None

    if not ignore_global_gap:
        try:
            last = last_move_at(org_id, mode=MOVE_MODE_PROACTIVE)
        except Exception:
            last = None
        if last is not None:
            try:
                gap_h = (datetime.now(timezone.utc) - last).total_seconds() / 3600.0
                if gap_h < PROACTIVE_MIN_GAP_HOURS:
                    return None
            except Exception:
                pass

    penalty = _rejection_penalty(org_id)

    scored: list[tuple[float, MoveCandidate]] = []
    for kind, fn in _MOVES:
        cooldown = COOLDOWNS.get(kind, 12.0) * penalty
        try:
            if not move_kind_cooldown_ok(org_id, kind, cooldown):
                continue
        except Exception:
            _log.warning("picker: cooldown check failed  kind=%s", kind, exc_info=True)
            continue
        try:
            cand = fn(org_id)
        except Exception:
            _log.warning("picker: move fn raised  kind=%s", kind, exc_info=True)
            continue
        if cand is None:
            continue
        try:
            bias = engagement_bias(org_id, kind)
        except Exception:
            bias = 1.0
        score = float(cand.novelty) * float(bias)
        scored.append((score, cand))

    if not scored:
        return None
    scored.sort(key=lambda p: p[0], reverse=True)
    return scored[0][1]


def record_surface(
    org_id: int,
    candidate: MoveCandidate,
    mode: str = MOVE_MODE_PROACTIVE,
    question_id: int | None = None,
) -> int | None:
    """Logs the move via log_move(). For closure moves, also marks the loop
    nudged. Returns the move row id (or None on failure)."""
    if candidate is None:
        return None
    try:
        row = log_move(
            org_id,
            candidate.kind,
            mode,
            input_refs=dict(candidate.input_refs or {}),
            question_id=question_id,
        )
    except Exception:
        _log.warning("record_surface: log_move failed  org=%d kind=%s",
                     org_id, candidate.kind, exc_info=True)
        row = None

    if candidate.kind == MOVE_KIND_CLOSURE:
        loop_id = (candidate.input_refs or {}).get("loop_id")
        if loop_id:
            try:
                mark_loop_nudged(int(loop_id))
            except Exception:
                _log.warning("record_surface: mark_loop_nudged failed  id=%s",
                             loop_id, exc_info=True)

    if not row:
        return None
    try:
        return int(row.get("Id")) if row.get("Id") is not None else None
    except (TypeError, ValueError):
        return None
