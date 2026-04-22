"""Activity-aware trigger for the insight producer.

Fires periodically (every few minutes) and decides whether to enqueue a
``insight_produce`` job onto the tool queue:

- **Chat-idle trigger**: if the user's last chat/code activity was >= N hours
  ago AND no insight has been produced since then, produce now.
- **Fallback trigger**: if there's been no activity at all in the last M
  hours AND no insight in the last M hours, produce anyway. Keeps the
  dashboard fresh when the user is away for days.

All decisions are org-scoped against the ``insights`` NocoDB table.
The dispatcher *never* runs the producer itself — it only enqueues.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

from infra.config import get_feature, is_feature_enabled
from shared import insights as insights_mod
from tools._org import default_org_id, resolve_org_id

_log = logging.getLogger("insight.dispatcher")


_DEFAULT_IDLE_HOURS = 2
_DEFAULT_FALLBACK_HOURS = 12

# Backoff after a producer failure ("no_topic", "thin_synthesis", etc). Keyed
# by org_id. Stops the dispatcher hammering topic_picker every 10 min on a
# fresh install with nothing to brief on.
_BACKOFF_S_ON_FAILURE = 3600  # 1h
_org_backoff: dict[int, float] = {}
_backoff_lock = threading.Lock()


def _in_backoff(org_id: int) -> bool:
    with _backoff_lock:
        until = _org_backoff.get(org_id)
    return bool(until and until > time.time())


def note_producer_result(org_id: int, status: str) -> None:
    """Called by the insight producer after each run. Any non-'ok' outcome
    parks the org for one hour so we don't burn LLM calls retrying a
    dry well every dispatcher tick."""
    if status == "ok":
        with _backoff_lock:
            _org_backoff.pop(org_id, None)
        return
    with _backoff_lock:
        _org_backoff[org_id] = time.time() + _BACKOFF_S_ON_FAILURE
    _log.info("insight dispatcher: backoff %ds for org=%d after status=%s",
              _BACKOFF_S_ON_FAILURE, org_id, status)


def _parse_iso(value) -> datetime | None:
    if not value:
        return None
    try:
        s = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _hours_since(ts: datetime | None) -> float:
    if ts is None:
        return float("inf")
    return (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0


def _decide(org_id: int) -> tuple[bool, str]:
    idle_hours = float(get_feature("insights", "chat_idle_hours", _DEFAULT_IDLE_HOURS))
    fallback_hours = float(get_feature("insights", "fallback_hours", _DEFAULT_FALLBACK_HOURS))

    try:
        from workers.tool_queue import seconds_since_chat
        chat_idle_hours = seconds_since_chat() / 3600.0
    except Exception:
        chat_idle_hours = float("inf")

    last_insight_ts = _parse_iso(insights_mod.latest_created_at(org_id))
    hours_since_insight = _hours_since(last_insight_ts)

    # Primary: chat-idle after activity (must have had recent activity AND gone quiet)
    if chat_idle_hours != float("inf") and chat_idle_hours >= idle_hours:
        # Only produce if we haven't already produced since the chat idle started
        if hours_since_insight > chat_idle_hours:
            return True, insights_mod.TRIGGER_CHAT_IDLE
    # Fallback: no activity in a long time AND no recent insight
    if chat_idle_hours == float("inf") or chat_idle_hours >= fallback_hours:
        if hours_since_insight >= fallback_hours:
            return True, insights_mod.TRIGGER_FALLBACK

    return False, f"skip  chat_idle_h={chat_idle_hours:.2f} last_insight_h={hours_since_insight:.2f}"


def jumpstart_insights(org_id: int | None = None) -> dict:
    """APScheduler tick. Enqueues at most one job onto the tool queue."""
    if not is_feature_enabled("insights"):
        return {"status": "disabled"}

    try:
        from infra.nocodb_client import NocodbClient
        client = NocodbClient()
        org = resolve_org_id(org_id or default_org_id(client))
    except Exception:
        _log.warning("insight dispatcher: org resolution failed", exc_info=True)
        return {"status": "no_org"}

    if _in_backoff(org):
        _log.debug("insight dispatcher: org=%d in backoff", org)
        return {"status": "backoff", "org_id": org}

    should, reason = _decide(org)
    if not should:
        _log.debug("insight dispatcher %s  org=%d", reason, org)
        return {"status": "skipped", "reason": reason, "org_id": org}

    try:
        from workers.tool_queue import get_tool_queue
        tq = get_tool_queue()
    except Exception:
        _log.warning("insight dispatcher: tool queue unavailable", exc_info=True)
        return {"status": "no_queue"}

    if not tq:
        return {"status": "no_queue"}

    try:
        job_id = tq.submit(
            "insight_produce",
            {"org_id": org, "trigger": reason},
            source="insight_dispatcher",
            org_id=org,
        )
        _log.info("insight enqueued  org=%d trigger=%s job=%s", org, reason, job_id)
        return {"status": "queued", "job_id": job_id, "org_id": org, "trigger": reason}
    except Exception as e:
        _log.warning("insight submit failed  org=%d err=%s", org, e, exc_info=True)
        return {"status": "submit_failed", "error": str(e)}
