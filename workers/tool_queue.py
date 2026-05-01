from __future__ import annotations

import collections
import json
import logging
import threading
import time
import uuid
from math import ceil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from infra.config import JOB_QUEUE_POLL_INTERVAL, JOB_QUEUE_STALE_TIMEOUT
from tools.contract import ToolName
from tools._org import resolve_org_id

_log = logging.getLogger("tool_queue")

NOCODB_TABLE = "tool_jobs"

# Single chat-idle threshold for all background jobs. Interactive/bypass jobs
# skip the gate entirely. Bumped to 120s default: a back-and-forth chat
# regularly pauses 30+ seconds while the user reads/types, and the prior
# value caused background work to resume mid-conversation, contending with
# the chat model and dragging stream time from seconds to minutes.
# Default idle window after an interactive turn before background queue workers
# can claim non-bypass jobs. 30 minutes by default for strict isolation.
_DEFAULT_BACKGROUND_CHAT_IDLE_S = 1800.0
_DEFAULT_MAX_ATTEMPTS = 1
_DEFAULT_RETRY_BACKOFF_S = 5.0

_last_chat_activity: float = 0.0
_chat_active_count: int = 0  # number of currently-streaming chat turns
_chat_turn_oldest_started_at: float = 0.0  # wall-clock of the OLDEST active turn
_activity_lock = threading.Lock()

# ContextVar set by the worker around each handler call. Lets handlers (and
# anything they call) emit progress + check cancellation without needing
# the job_id threaded through every function. Handlers retrieve the queue
# from get_tool_queue() and call report_progress / is_job_cancelled.
import contextvars
_current_job_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_tool_job_id", default=None,
)


def report_progress(
    message: str,
    *,
    kind: str = "",
    step: int = 0,
    total: int = 0,
) -> None:
    """Stamp a human-readable progress line on the currently-running tool
    job. ``kind`` / ``step`` / ``total`` give the UI structured data to
    render a real progress bar (e.g. ``kind='search', step=3, total=8``).
    No-op outside a handler. Always logs regardless of DB success so the
    live log stream still shows it."""
    job_id = _current_job_id.get()
    pretty = message
    if step or total:
        pretty = f"[{step or '?'}/{total or '?'}] {message}" if step or total else message
    if not job_id:
        _log.info("progress (no job): %s", pretty[:200])
        return
    _log.info("progress  job=%s  %s%s", job_id[:12],
              f"{kind}: " if kind else "", pretty[:300])
    try:
        q = get_tool_queue()
        if q is not None:
            q.update_progress(job_id, message, kind=kind, step=step, total=total)
    except Exception:
        _log.debug("report_progress: queue update failed", exc_info=True)


def is_job_cancelled() -> bool:
    """Cooperative cancellation check. Long-running handlers should call
    this between phases (e.g. between search queries, between section
    writes) and abort cleanly when it returns True. No-op (returns False)
    outside a tool-queue handler.
    """
    job_id = _current_job_id.get()
    if not job_id:
        return False
    try:
        q = get_tool_queue()
        return bool(q is not None and q.is_cancelled(job_id))
    except Exception:
        return False


def current_job_id() -> str | None:
    """Returns the currently-running tool job's id, or None if not inside
    a handler. Used by handlers that fan work out to thread pools — capture
    this before submitting and re-set it inside each worker via
    :func:`bind_job_id`, since contextvars don't propagate across threads."""
    return _current_job_id.get()


def bind_job_id(job_id: str | None):
    """Context manager that sets the current job_id contextvar. Use inside
    threads spawned from a handler so ``report_progress`` and
    ``is_job_cancelled`` continue to work in the worker thread."""
    import contextlib

    @contextlib.contextmanager
    def _cm():
        if job_id is None:
            yield
            return
        tok = _current_job_id.set(job_id)
        try:
            yield
        finally:
            _current_job_id.reset(tok)

    return _cm()


class JobCancelled(Exception):
    """Raised by handlers (or the framework) when a running job is
    cancelled cooperatively. Caught at the top of the dispatch loop and
    turned into a ``status='cancelled'`` outcome."""
    pass

# A "chat turn" should never legitimately be active for longer than this.
# If the counter is still elevated past this window, we assume the counter
# leaked (chat thread crashed without decrementing) and treat the chat as
# inactive — otherwise every background Huey worker is blocked forever at
# _block_while_chat_active.
_CHAT_TURN_STALE_S = 900.0   # 15 minutes


def touch_chat_activity():
    """Record a chat tick. Use this for momentary signals (search start, end
    of stream, etc). Resets the idle clock; queue workers won't claim new
    jobs for ``background_chat_idle_seconds`` after this."""
    global _last_chat_activity
    with _activity_lock:
        _last_chat_activity = time.time()


def begin_chat_turn() -> None:
    """Mark a chat turn as actively streaming. While the count is > 0, queue
    workers MUST treat chat as live regardless of how long ago the last
    ``touch_chat_activity`` call was — long LLM streams legitimately go
    minutes between activity ticks and we don't want background jobs
    creeping in mid-stream and contending for the model backend.

    Pair every ``begin_chat_turn`` with exactly one ``end_chat_turn``."""
    global _chat_active_count, _last_chat_activity, _chat_turn_oldest_started_at
    with _activity_lock:
        if _chat_active_count == 0:
            _chat_turn_oldest_started_at = time.time()
        _chat_active_count += 1
        _last_chat_activity = time.time()


def end_chat_turn() -> None:
    """Companion to :func:`begin_chat_turn`. The idle clock starts FROM HERE,
    not from when the turn began, so the configured idle window guarantees
    a real quiet period after the stream closes."""
    global _chat_active_count, _last_chat_activity, _chat_turn_oldest_started_at
    with _activity_lock:
        _chat_active_count = max(0, _chat_active_count - 1)
        _last_chat_activity = time.time()
        if _chat_active_count == 0:
            _chat_turn_oldest_started_at = 0.0


def reset_chat_active(reason: str = "manual") -> int:
    """Force the chat-active counter to zero. Used by admin UI when the
    counter leaked (e.g. a chat handler crashed without decrementing) and
    every background worker is now blocked at the gate. Returns the count
    that was discarded so the operator can confirm what was reset.
    """
    global _chat_active_count, _chat_turn_oldest_started_at, _last_chat_activity
    with _activity_lock:
        prev = _chat_active_count
        _chat_active_count = 0
        _chat_turn_oldest_started_at = 0.0
        _last_chat_activity = time.time()
    if prev:
        _log.warning("reset_chat_active  prev=%d reason=%s", prev, reason)
    return prev


def chat_active_state() -> dict:
    """Snapshot of the chat-active gate for the admin UI. Surfaces both the
    counter and how long the oldest active turn has been live so the user
    can spot a leaked counter (e.g. ``count=1`` for an hour with no chat
    happening)."""
    with _activity_lock:
        count = _chat_active_count
        started_at = _chat_turn_oldest_started_at
        last_activity = _last_chat_activity
    age_s = (time.time() - started_at) if started_at > 0 else 0
    return {
        "count": count,
        "oldest_turn_age_seconds": int(age_s),
        "stale_threshold_seconds": int(_CHAT_TURN_STALE_S),
        "is_stale": bool(count > 0 and age_s > _CHAT_TURN_STALE_S),
        "seconds_since_last_activity": (
            None if last_activity == 0 else int(time.time() - last_activity)
        ),
    }


def is_chat_active() -> bool:
    """True when at least one chat turn is currently streaming.

    Returns False if the counter has been > 0 longer than
    ``_CHAT_TURN_STALE_S`` — a chat turn legitimately can't last 15 minutes,
    so a sustained elevated counter means the chat thread crashed without
    decrementing. Without this guard a single leak would block every
    background worker forever (the symptom we just hit). The counter is
    NOT auto-cleared here — operators see the stale state via
    :func:`chat_active_state` and can call :func:`reset_chat_active`.
    """
    with _activity_lock:
        if _chat_active_count <= 0:
            return False
        if _chat_turn_oldest_started_at <= 0:
            return True
        age = time.time() - _chat_turn_oldest_started_at
        return age <= _CHAT_TURN_STALE_S


def yield_to_chat(max_wait_s: float = 60.0, poll_s: float = 1.0) -> bool:
    """Cooperative checkpoint for long-running background tasks.

    Call this between phases (e.g. before each LLM call inside a research
    plan, or between scrape batches). If a chat turn is active, blocks for
    up to ``max_wait_s`` waiting for it to finish; returns True if it had
    to wait, False if chat was already idle.

    Tasks that respect this gate will pause mid-flight when the user
    starts typing, freeing the model backend for the chat reply.
    """
    if not is_chat_active():
        return False
    waited = 0.0
    while is_chat_active() and waited < max_wait_s:
        time.sleep(poll_s)
        waited += poll_s
    return True


def seconds_since_chat() -> float:
    with _activity_lock:
        if _chat_active_count > 0:
            # Honour the same staleness check as is_chat_active(): if the
            # counter is elevated past the legitimate-turn window we treat
            # the chat as inactive — otherwise the queue worker loop would
            # also block forever on a leaked counter (it gates on
            # `seconds_since_chat() < idle_gate`, not on is_chat_active()).
            if (
                _chat_turn_oldest_started_at <= 0
                or (time.time() - _chat_turn_oldest_started_at) <= _CHAT_TURN_STALE_S
            ):
                return 0.0
            # Stale: fall through and return real elapsed since last tick.
        if _last_chat_activity == 0:
            return float("inf")
        return time.time() - _last_chat_activity


def force_background_ready(reason: str = "manual restart") -> dict:
    """Immediately open the background gate and wake all worker loops.

    Used by an explicit operator action when the 30-minute idle window should
    be bypassed and queued background work resumed right now.
    """
    global _chat_active_count, _chat_turn_oldest_started_at, _last_chat_activity
    now = time.time()
    with _activity_lock:
        prev_count = _chat_active_count
        prev_last = _last_chat_activity
        _chat_active_count = 0
        _chat_turn_oldest_started_at = 0.0
        # 0 means seconds_since_chat() -> inf, which clears idle gating.
        _last_chat_activity = 0.0

    woke = 0
    q = get_tool_queue()
    if q is not None:
        for ev in q._wake_events.values():
            ev.set()
            woke += 1

    seconds_before = float("inf") if prev_last == 0 else max(0.0, now - prev_last)
    _log.warning(
        "force_background_ready  reason=%s prev_count=%d prev_seconds_since=%.1f woke=%d",
        reason,
        prev_count,
        seconds_before,
        woke,
    )
    return {
        "reason": reason,
        "previous_count": int(prev_count),
        "previous_seconds_since_last_activity": (
            None if prev_last == 0 else round(seconds_before, 1)
        ),
        "woke_worker_types": int(woke),
    }


def _background_idle_gate() -> float:
    try:
        from infra.config import get_feature
        raw = get_feature("tool_queue", "background_chat_idle_seconds", _DEFAULT_BACKGROUND_CHAT_IDLE_S)
        val = float(raw)
        return val if val >= 0 else _DEFAULT_BACKGROUND_CHAT_IDLE_S
    except Exception:
        return _DEFAULT_BACKGROUND_CHAT_IDLE_S


def _max_attempts_for_job_type(job_type: str) -> int:
    try:
        from infra.config import get_feature
        per_type = get_feature("tool_queue", "job_type_max_attempts", None)
        if isinstance(per_type, dict):
            val = per_type.get(job_type)
            if val is not None:
                return max(1, int(val))
        raw = get_feature("tool_queue", "default_max_attempts", _DEFAULT_MAX_ATTEMPTS)
        return max(1, int(raw))
    except Exception:
        return _DEFAULT_MAX_ATTEMPTS


def _retry_backoff_s(job_type: str, attempt: int) -> float:
    try:
        from infra.config import get_feature
        per_type = get_feature("tool_queue", "job_type_retry_backoff_seconds", None)
        if isinstance(per_type, dict):
            val = per_type.get(job_type)
            if val is not None:
                base = max(0.0, float(val))
                return min(base * max(1, attempt), 300.0)
        raw = get_feature("tool_queue", "default_retry_backoff_seconds", _DEFAULT_RETRY_BACKOFF_S)
        base = max(0.0, float(raw))
        return min(base * max(1, attempt), 300.0)
    except Exception:
        return _DEFAULT_RETRY_BACKOFF_S


def _is_interactive_source(source: str) -> bool:
    s = (source or "").strip().lower()
    return s == "chat" or s == "code" or s.startswith("chat_") or s.startswith("code_")


def _job_type_jumps_queue(job_type: str) -> bool:
    """Per-type toggle from config.json: tool_queue.queue_jumpers[<type>] = true
    means jobs of this type bypass the chat-idle gate (jump the queue)."""
    try:
        from infra.config import get_feature
        jumpers = get_feature("tool_queue", "queue_jumpers", None)
        if isinstance(jumpers, dict):
            return bool(jumpers.get(job_type))
    except Exception:
        pass
    return False


def _bypass_idle(job: "ToolJob") -> bool:
    """Jobs that skip the background chat-idle gate: interactive chat/code jobs,
    types configured as queue jumpers in config.json, and anything explicitly
    requesting bypass via payload."""
    if _is_interactive_source(job.source):
        return True
    if _job_type_jumps_queue(job.type):
        return True
    payload = job.payload if isinstance(job.payload, dict) else {}
    return bool(payload.get("bypass_idle"))


@dataclass
class HandlerConfig:
    handler: Callable[[dict], dict]
    max_workers: int = 1
    priority_default: int = 3
    dedup_key: str | None = None
    source: str = ""


@dataclass
class ToolJob:
    job_id: str
    type: str
    status: str = "queued"
    priority: int = 3
    # Free-form, human-readable progress message. Updated by handlers via
    # ``report_progress(message, kind=…, step=…, total=…)``. Surfaces in
    # queue UI so the user can see what each running job is actually doing.
    progress: str = ""
    progress_at: str = ""
    # Structured fields paired with ``progress`` so the UI can draw a real
    # progress bar instead of parsing the message. ``progress_kind`` is a
    # short label for the current phase (``plan`` / ``search`` / ``synth``
    # / ``review`` / ``publish``); ``progress_step`` / ``progress_total``
    # are positive integers when known, 0 otherwise.
    progress_kind: str = ""
    progress_step: int = 0
    progress_total: int = 0
    # Free-form tags. Handlers can attach (`["paper", "client_acme"]`)
    # to enable UI filtering. Survives across retries.
    tags: list[str] = field(default_factory=list)
    # Parent linkage for fan-out flows (e.g. research_planner → research_agent
    # → graph_extract). Lets the UI render a tree and cancel-cascade.
    parent_job_id: str = ""
    # Set when the job has been moved to dead-letter after exceeding the
    # configured failure budget. Reset on manual replay.
    dead_lettered_at: str = ""
    source: str = ""
    org_id: int = 1
    payload: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    error: str = ""
    claimed_by: str = ""
    started_at: str = ""
    completed_at: str = ""
    depends_on: str = ""
    nocodb_id: int | None = None

    def to_row(self) -> dict:
        # Conditionally include the new optional columns. Sending empty
        # strings for timestamp columns can confuse NocoDB; sending an
        # empty list/parent_id for unused fields is wasted bandwidth on
        # every persist. Only emit when set so PATCH semantics leave the
        # column alone for jobs that don't use these features.
        row: dict[str, Any] = {
            "job_id": self.job_id,
            "type": self.type,
            "status": self.status,
            "priority": self.priority,
            "source": self.source,
            "org_id": self.org_id,
            "payload_json": json.dumps(self.payload),
            "result_json": json.dumps(self.result) if self.result else "{}",
            "error": self.error,
            "claimed_by": self.claimed_by,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "depends_on": self.depends_on,
        }
        if self.dead_lettered_at:
            row["dead_lettered_at"] = self.dead_lettered_at
        if self.parent_job_id:
            row["parent_job_id"] = self.parent_job_id
        if self.tags:
            row["tags"] = json.dumps(self.tags)
        return row

    def _task_summary(self) -> str:
        meta = self.payload.get("metadata") or {}
        for candidate in (
            meta.get("title"),
            meta.get("name"),
            self.payload.get("task"),
            self.payload.get("topic"),
            self.payload.get("url"),
            self.payload.get("seed_url"),
            self.payload.get("query"),
            self.payload.get("message_id"),
            self.payload.get("plan_id"),
            self.payload.get("target_id"),
            self.payload.get("discovery_id"),
        ):
            if candidate not in (None, ""):
                return str(candidate)
        return ""

    def to_api(self, verbose: bool = False) -> dict:
        meta = self.payload.get("metadata") or {}
        d = {
            "job_id": self.job_id,
            "type": self.type,
            "status": self.status,
            "priority": self.priority,
            "source": self.source,
            "org_id": self.org_id,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "depends_on": self.depends_on,
            "task": self._task_summary() or None,
            "progress": self.progress or None,
            "progress_at": self.progress_at or None,
            "progress_kind": self.progress_kind or None,
            "progress_step": self.progress_step or None,
            "progress_total": self.progress_total or None,
            "tags": self.tags or None,
            "parent_job_id": self.parent_job_id or None,
            "dead_lettered_at": self.dead_lettered_at or None,
        }
        conversation_id = meta.get("conversation_id") or self.payload.get("conversation_id")
        url = meta.get("url") or self.payload.get("url") or self.payload.get("seed_url")
        title = meta.get("title") or meta.get("name")
        if conversation_id:
            d["conversation_id"] = conversation_id
        if url:
            d["url"] = url
        if title:
            d["title"] = title
        if self.result:
            d["result_status"] = self.result.get("status")
        if verbose:
            d["claimed_by"] = self.claimed_by or None
            d["nocodb_id"] = self.nocodb_id
            d["payload"] = self.payload
            d["result"] = self.result
        return d

    @staticmethod
    def from_row(row: dict) -> ToolJob:
        payload = row.get("payload_json") or "{}"
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {}
        result = row.get("result_json") or "{}"
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except Exception:
                result = {}
        return ToolJob(
            job_id=row.get("job_id") or "",
            type=row.get("type") or "",
            status=row.get("status") or "queued",
            priority=int(row.get("priority") or 3),
            source=row.get("source") or "",
            org_id=resolve_org_id(row.get("org_id")),
            payload=payload,
            result=result,
            error=row.get("error") or "",
            claimed_by=row.get("claimed_by") or "",
            started_at=row.get("started_at") or "",
            completed_at=row.get("completed_at") or "",
            depends_on=row.get("depends_on") or "",
            nocodb_id=row.get("Id"),
            progress=row.get("progress") or "",
            progress_at=row.get("progress_at") or "",
            progress_kind=row.get("progress_kind") or "",
            progress_step=int(row.get("progress_step") or 0),
            progress_total=int(row.get("progress_total") or 0),
            tags=ToolJob._decode_tags(row.get("tags")),
            parent_job_id=row.get("parent_job_id") or "",
            dead_lettered_at=row.get("dead_lettered_at") or "",
        )

    @staticmethod
    def _decode_tags(raw) -> list[str]:
        if not raw:
            return []
        if isinstance(raw, list):
            return [str(t) for t in raw if t]
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(t) for t in parsed if t]
            except Exception:
                # Fallback: comma-separated.
                return [t.strip() for t in raw.split(",") if t.strip()]
        return []


class ToolJobQueue:

    def __init__(self):
        self._handlers: dict[str, HandlerConfig] = {}
        self._workers: dict[str, list[threading.Thread]] = {}
        self._wake_events: dict[str, threading.Event] = {}
        self._stop = threading.Event()
        self._worker_id = f"{threading.current_thread().name}_{id(self)}"
        self._stale_thread: threading.Thread | None = None
        self._subscribers: list[list[dict]] = []
        self._sub_lock = threading.Lock()
        self._started_at: float = 0.0
        # Per-type pause control. Operator can flip via /tool-queue/pause-type
        # to halt a misbehaving handler kind without stopping everything.
        self._paused_types: set[str] = set()
        self._paused_types_lock = threading.Lock()
        # Per-type rolling median duration for ETA estimates. Cap each list
        # at 50 most recent durations; cheap to compute median.
        self._duration_samples: dict[str, list[float]] = {}
        self._duration_lock = threading.Lock()

    def register(self, job_type: str, config: HandlerConfig):
        self._handlers[job_type] = config
        self._wake_events[job_type] = threading.Event()

    def start(self):
        self._stop.clear()
        self._started_at = time.time()
        self._load_pending()
        for job_type, config in self._handlers.items():
            threads: list[threading.Thread] = []
            for i in range(config.max_workers):
                t = threading.Thread(
                    target=self._worker_loop,
                    args=(job_type,),
                    daemon=True,
                    name=f"tq-{job_type}-{i}",
                )
                t.start()
                threads.append(t)
            self._workers[job_type] = threads
            _log.info("started %d workers for type=%s", config.max_workers, job_type)
        self._stale_thread = threading.Thread(
            target=self._stale_checker_loop, daemon=True, name="tq-stale",
        )
        self._stale_thread.start()
        _log.info("tool job queue started  types=%s", list(self._handlers))

    def stop(self):
        self._stop.set()
        for ev in self._wake_events.values():
            ev.set()
        for threads in self._workers.values():
            for t in threads:
                t.join(timeout=5)
        self._workers.clear()
        _log.info("tool job queue stopped")

    def submit(
        self,
        job_type: str,
        payload: dict,
        source: str = "",
        org_id: int | None = None,
        priority: int | None = None,
        depends_on: str = "",
    ) -> str:
        config = self._handlers.get(job_type)
        if not config:
            raise ValueError(f"unknown job type: {job_type}")

        payload_org = payload.get("org_id") if isinstance(payload, dict) else None
        org_id = resolve_org_id(org_id if org_id else payload_org, fallback=0)
        if org_id <= 0:
            raise ValueError(f"missing org_id for job type: {job_type}")

        if config.dedup_key and not depends_on:
            dedup_val = payload.get(config.dedup_key, "")
            if dedup_val:
                existing = self._find_dedup(job_type, config.dedup_key, dedup_val)
                if existing:
                    _log.debug("dedup hit type=%s key=%s val=%s existing=%s",
                               job_type, config.dedup_key, str(dedup_val)[:60], existing)
                    return existing

        if priority is None:
            priority = config.priority_default

        job = ToolJob(
            job_id=uuid.uuid4().hex,
            type=job_type,
            status="queued",
            priority=max(1, min(5, priority)),
            source=source,
            org_id=org_id,
            payload=payload,
            depends_on=depends_on,
        )
        if not self._persist_new(job):
            raise RuntimeError(f"failed to persist job {job.job_id}")
        self._emit_event({
            "type": "job_queued",
            "job_id": job.job_id,
            "job_type": job_type,
            "priority": job.priority,
        })
        _log.info("submit  id=%s type=%s priority=%d source=%s depends_on=%s",
                   job.job_id, job_type, job.priority, source, depends_on or "-")
        self._wake_events.get(job_type, threading.Event()).set()
        return job.job_id

    def submit_batch(self, items: list[dict]) -> list[str]:
        ids = []
        for item in items:
            jid = self.submit(
                job_type=item.get("type", "scrape"),
                payload=item.get("payload", {}),
                source=item.get("source", ""),
                org_id=item.get("org_id"),
                priority=item.get("priority"),
                depends_on=item.get("depends_on", ""),
            )
            ids.append(jid)
        return ids

    def status(self) -> dict:
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                return {"error": "table not found"}
            rows = db._get_paginated(NOCODB_TABLE, params={"limit": 500})
        except Exception:
            return {"error": "db query failed"}

        counts: dict[str, dict[str, int]] = {}
        for row in rows:
            jt = row.get("type") or "unknown"
            st = row.get("status") or "unknown"
            counts.setdefault(jt, {})
            counts[jt][st] = counts[jt].get(st, 0) + 1

        workers: dict[str, int] = {}
        for jt, threads in self._workers.items():
            workers[jt] = sum(1 for t in threads if t.is_alive())

        idle_raw = seconds_since_chat()
        idle = idle_raw
        if idle == float("inf"):
            idle = -1  # no chat activity yet
        idle_display = idle

        gate = _background_idle_gate()
        chat_live = is_chat_active()
        if chat_live:
            backoff_state = "chat_active"
        elif idle < 0 or idle >= gate:
            backoff_state = "clear"
        else:
            backoff_state = "waiting_for_idle"
        if idle_display >= 0 and idle_display > gate:
            # UI countdowns often compute threshold-idle; clamp display to avoid
            # negative values while preserving raw idle for diagnostics.
            idle_display = gate
        remaining_s = 0
        if backoff_state == "waiting_for_idle":
            remaining_s = max(0, int(round(gate - max(0.0, idle_raw))))

        return {
            "counts": counts,
            "workers": workers,
            "backoff": {
                "state": backoff_state,
                "chat_active": chat_live,
                "idle_seconds": round(idle_display, 0),
                "idle_seconds_raw": round(idle, 0),
                "threshold": gate,
                "remaining_seconds": remaining_s,
            },
        }

    def get_job(self, job_id: str) -> ToolJob | None:
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                return None
            rows = db._get(NOCODB_TABLE, params={
                "where": f"(job_id,eq,{job_id})",
                "limit": 1,
            }).get("list", [])
            if rows:
                return ToolJob.from_row(rows[0])
        except Exception:
            pass
        return None

    def list_jobs(
        self,
        job_type: str = "",
        status: str = "",
        source: str = "",
        limit: int = 50,
        org_id: int | None = None,
        verbose: bool = False,
    ) -> list[dict]:
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                return []
            where_parts: list[str] = []
            if job_type:
                where_parts.append(f"(type,eq,{job_type})")
            if status:
                where_parts.append(f"(status,eq,{status})")
            if source:
                where_parts.append(f"(source,eq,{source})")
            if org_id is not None:
                where_parts.append(f"(org_id,eq,{int(org_id)})")
            params: dict[str, Any] = {
                "sort": "-CreatedAt",
                "limit": limit,
            }
            if where_parts:
                params["where"] = "~and".join(where_parts)
            rows = db._get(NOCODB_TABLE, params=params).get("list", [])
            out: list[dict] = []
            for r in rows:
                job = ToolJob.from_row(r)
                api = job.to_api(verbose=verbose)
                # Annotate running jobs with an ETA estimate based on the
                # rolling median duration for that type minus elapsed.
                # Not present on jobs we have no samples for yet.
                if job.status == "running" and job.started_at:
                    median = self.median_duration_for(job.type)
                    if median is not None:
                        try:
                            started = datetime.fromisoformat(job.started_at).timestamp()
                            elapsed = max(0.0, time.time() - started)
                            remaining = max(0.0, median - elapsed)
                            api["median_duration_s"] = round(median, 1)
                            api["elapsed_s"] = round(elapsed, 1)
                            api["eta_seconds"] = int(remaining)
                            api["over_median"] = elapsed > median * 1.5
                        except Exception:
                            pass
                out.append(api)
            return out
        except Exception:
            return []

    def cancel(self, job_id: str) -> bool:
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                return False
            rows = db._get(NOCODB_TABLE, params={
                "where": f"(job_id,eq,{job_id})~and(status,eq,queued)",
                "limit": 1,
            }).get("list", [])
            if not rows:
                return False
            noco_id = rows[0].get("Id")
            db._patch(NOCODB_TABLE, noco_id, {
                "Id": noco_id,
                "status": "cancelled",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
            self._emit_event({"type": "job_cancelled", "job_id": job_id})
            return True
        except Exception:
            return False

    def set_priority(self, job_id: str, priority: int) -> bool:
        """Re-prioritise a queued job. Bounded 1–5."""
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                return False
            rows = db._get(NOCODB_TABLE, params={
                "where": f"(job_id,eq,{job_id})~and(status,eq,queued)",
                "limit": 1,
            }).get("list", [])
            if not rows:
                return False
            noco_id = rows[0].get("Id")
            db._patch(NOCODB_TABLE, noco_id, {
                "Id": noco_id,
                "priority": max(1, min(5, int(priority))),
            })
            ev = self._wake_events.get(rows[0].get("type") or "")
            if ev:
                ev.set()
            return True
        except Exception:
            return False

    def update_tags(self, job_id: str, *, add: list[str], remove: list[str]) -> bool:
        """Add/remove tags on a job. Tags are JSON-encoded in NocoDB."""
        try:
            job = self.get_job(job_id)
            if not job:
                return False
            current = set(job.tags or [])
            for t in add or []:
                if t:
                    current.add(str(t))
            for t in remove or []:
                current.discard(str(t))
            if not job.nocodb_id:
                return False
            self._db()._patch(NOCODB_TABLE, job.nocodb_id, {
                "Id": job.nocodb_id,
                "tags": json.dumps(sorted(current)),
            })
            return True
        except Exception:
            return False

    def list_children(self, job_id: str, limit: int = 50) -> list["ToolJob"]:
        """Jobs whose ``parent_job_id`` equals ``job_id``."""
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                return []
            rows = db._get(NOCODB_TABLE, params={
                "where": f"(parent_job_id,eq,{job_id})",
                "limit": limit,
            }).get("list", [])
            return [ToolJob.from_row(r) for r in rows]
        except Exception:
            return []

    def set_type_paused(self, job_type: str, paused: bool) -> None:
        """Pause/resume a single job type. Worker poll loops honour this
        and skip claim while paused; in-flight jobs are unaffected."""
        with self._paused_types_lock:
            if paused:
                self._paused_types.add(job_type)
            else:
                self._paused_types.discard(job_type)
        if not paused:
            ev = self._wake_events.get(job_type)
            if ev:
                ev.set()

    def list_paused_types(self) -> set[str]:
        with self._paused_types_lock:
            return set(self._paused_types)

    def is_type_paused(self, job_type: str) -> bool:
        with self._paused_types_lock:
            return job_type in self._paused_types

    def median_duration_for(self, job_type: str) -> float | None:
        """Return the rolling-median completion duration in seconds for a
        job type, or None if we have no samples yet. Used for ETA badges."""
        with self._duration_lock:
            samples = list(self._duration_samples.get(job_type) or [])
        if not samples:
            return None
        samples.sort()
        n = len(samples)
        return samples[n // 2] if n % 2 == 1 else 0.5 * (samples[n // 2 - 1] + samples[n // 2])

    def cancel_running(self, job_id: str, reason: str = "user terminated") -> bool:
        """Cooperative cancel for an already-running job. Marks the row
        ``status='cancelled'``; the running handler is expected to call
        :func:`is_cancelled` between phases and abort cleanly when set.

        Long-running LLM calls in flight when this fires keep going to
        completion (we never kill threads); the abort takes effect at the
        next phase boundary in the handler.
        """
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                return False
            rows = db._get(NOCODB_TABLE, params={
                "where": f"(job_id,eq,{job_id})~and(status,in,queued,running)",
                "limit": 1,
            }).get("list", [])
            if not rows:
                return False
            noco_id = rows[0].get("Id")
            db._patch(NOCODB_TABLE, noco_id, {
                "Id": noco_id,
                "status": "cancelled",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": reason[:500],
            })
            # Force the cancellation cache to flip True for this job so any
            # in-flight handler in the same process sees it on its next
            # is_cancelled() check (within the 2s TTL window the cached
            # False would otherwise dominate).
            try:
                self._cancel_cache()[job_id] = (time.time(), True)
            except Exception:
                pass
            self._emit_event({"type": "job_cancelled", "job_id": job_id, "reason": reason})
            _log.info("cancel_running  job=%s reason=%s", job_id, reason)
            return True
        except Exception:
            _log.warning("cancel_running failed  job=%s", job_id, exc_info=True)
            return False

    # Per-process cache for the cancellation flag. The research agent (and
    # other long-running handlers) call ``is_cancelled`` from inside tight
    # loops — every search query, every section. Without a cache that's a
    # NocoDB GET per iteration, multiplied by N concurrent jobs. A 2-second
    # TTL is plenty: cancellation is user-initiated and a 2s delay between
    # click and abort is fine, and cancel_running invalidates the cache
    # immediately so a self-cancel still trips on the next check.
    _CANCEL_CACHE_TTL_S = 2.0

    def _cancel_cache(self) -> dict[str, tuple[float, bool]]:
        # Lazily attached so old in-memory queues survive redeploys.
        cache = getattr(self, "_cancel_cache_dict", None)
        if cache is None:
            cache = {}
            self._cancel_cache_dict = cache  # type: ignore[attr-defined]
        return cache

    def is_cancelled(self, job_id: str) -> bool:
        """Cheap check used by long-running handlers between phases. The
        handler raises ``JobCancelled`` (or returns early) when this flips
        True. Cached for ~2 s to avoid a NocoDB GET per loop iteration."""
        cache = self._cancel_cache()
        now = time.time()
        hit = cache.get(job_id)
        if hit and (now - hit[0]) < self._CANCEL_CACHE_TTL_S:
            return hit[1]
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                cache[job_id] = (now, False)
                return False
            rows = db._get(NOCODB_TABLE, params={
                "where": f"(job_id,eq,{job_id})",
                "limit": 1,
                "fields": "status",
            }).get("list", [])
            cancelled = bool(rows and (rows[0].get("status") == "cancelled"))
            cache[job_id] = (now, cancelled)
            return cancelled
        except Exception:
            # Never let a transient DB hiccup signal cancellation.
            return bool(hit and hit[1])

    # Cache job_id → nocodb_id so update_progress doesn't do a GET-then-PATCH
    # for every progress line; the row's PK doesn't change.
    def _noco_id_for(self, job_id: str) -> int | None:
        cache = getattr(self, "_noco_id_cache", None)
        if cache is None:
            cache = {}
            self._noco_id_cache = cache  # type: ignore[attr-defined]
        if job_id in cache:
            return cache[job_id]
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                return None
            rows = db._get(NOCODB_TABLE, params={
                "where": f"(job_id,eq,{job_id})",
                "limit": 1,
                "fields": "Id",
            }).get("list", [])
            if rows:
                noco_id = int(rows[0].get("Id") or 0) or None
                if noco_id:
                    cache[job_id] = noco_id
                return noco_id
        except Exception:
            pass
        return None

    # Per-job throttle on PATCH writes. The SSE event always fires (cheap),
    # but two PATCHes within a 1.5s window collapse: handlers calling
    # report_progress in tight loops won't hammer NocoDB. The final
    # progress message in a burst is the one persisted (skipped writes
    # happen during the burst; the next call after the window flushes).
    _PROGRESS_THROTTLE_S = 1.5

    def _last_progress_at(self) -> dict[str, float]:
        cache = getattr(self, "_progress_throttle_cache", None)
        if cache is None:
            cache = {}
            self._progress_throttle_cache = cache  # type: ignore[attr-defined]
        return cache

    def update_progress(
        self,
        job_id: str,
        message: str,
        *,
        kind: str = "",
        step: int = 0,
        total: int = 0,
    ) -> None:
        """Stamp a human-readable progress message on the running job row.

        Optional structured fields (``kind`` / ``step`` / ``total``) let the
        UI render a real progress bar instead of parsing the text. Best-
        effort write; SSE event always emits regardless of DB success.
        The PATCH is single-call (cached PK lookup) and throttled per-job
        so tight loops don't multiply database traffic.
        """
        ts = datetime.now(timezone.utc).isoformat()
        # Per-job PATCH throttle — emit SSE always, write to DB ≤ 1× per window.
        throttle = self._last_progress_at()
        now = time.time()
        last = throttle.get(job_id, 0.0)
        skip_db = (now - last) < self._PROGRESS_THROTTLE_S
        if not skip_db:
            throttle[job_id] = now
        noco_id = self._noco_id_for(job_id) if not skip_db else None
        patch: dict[str, Any] = {
            "progress": message[:500],
            "progress_at": ts,
        }
        if kind:
            patch["progress_kind"] = kind[:60]
        if step:
            patch["progress_step"] = int(step)
        if total:
            patch["progress_total"] = int(total)
        if noco_id is not None:
            try:
                self._db()._patch(NOCODB_TABLE, noco_id, {"Id": noco_id, **patch})
            except Exception:
                _log.debug("update_progress: nocodb patch failed  job=%s", job_id, exc_info=True)
        # SSE event regardless of DB success — keeps the UI live.
        self._emit_event({
            "type": "job_progress",
            "job_id": job_id,
            "message": message[:500],
            "kind": kind or None,
            "step": int(step) if step else None,
            "total": int(total) if total else None,
            "ts": ts,
        })

    def subscribe(self) -> collections.deque:
        buf: collections.deque = collections.deque()
        with self._sub_lock:
            self._subscribers.append(buf)
        return buf

    def unsubscribe(self, buf: collections.deque):
        with self._sub_lock:
            try:
                self._subscribers.remove(buf)
            except ValueError:
                pass

    def _emit_event(self, event: dict):
        with self._sub_lock:
            for sub in self._subscribers:
                sub.append(event)

    def _worker_loop(self, job_type: str):
        wake = self._wake_events[job_type]
        worker_id = threading.current_thread().name
        _log.info("worker %s started", worker_id)

        while not self._stop.is_set():
            wake.wait(timeout=JOB_QUEUE_POLL_INTERVAL)
            wake.clear()

            if self._stop.is_set():
                break

            # Per-type pause: operator wants this handler kind dormant.
            # Skip claim entirely; in-flight jobs of this type continue.
            if self.is_type_paused(job_type):
                continue

            queued_head = self._peek_next_queued(job_type)
            head_bypass = bool(queued_head and _bypass_idle(queued_head))

            # Chat-idle gate: non-bypass jobs wait until chat is quiet.
            if not head_bypass:
                idle = seconds_since_chat()
                gate = _background_idle_gate()
                if idle < gate:
                    continue

            job = self._claim_next(job_type, worker_id)
            if not job:
                continue

            # Re-check gate post-claim in case chat activity touched between peek and claim.
            if not _bypass_idle(job):
                idle = seconds_since_chat()
                gate = _background_idle_gate()
                if idle < gate:
                    wait_secs = min(gate - idle, 60)
                    _log.info(
                        "queue %s: job %s needs %.0fs idle, currently %.0fs — sleeping %.0fs",
                        worker_id, job.job_id[:12], gate, idle, wait_secs,
                    )
                    self._unclaim(job)
                    self._stop.wait(timeout=wait_secs)
                    continue

            if job.depends_on:
                dep = self.get_job(job.depends_on)
                dep_status = dep.status if dep else "not_found"
                # Hard-fail if the dependency itself failed or vanished — without
                # this guard the job re-claims, re-checks, and unclaims forever
                # (every poll interval), pegging NocoDB and never making progress.
                if dep_status in {"failed", "cancelled", "not_found"}:
                    _log.warning(
                        "queue %s: job %s dependency %s is %s — failing dependent job",
                        worker_id, job.job_id[:12], job.depends_on[:12], dep_status,
                    )
                    job.status = "failed"
                    job.error = f"dependency {job.depends_on} ended with status={dep_status}"[:500]
                    job.completed_at = datetime.now(timezone.utc).isoformat()
                    self._persist_update(job)
                    continue
                if dep_status != "completed":
                    # Still in flight — release the claim and let another poll cycle re-check.
                    _log.debug("queue %s: job %s waiting on dependency %s (status=%s)",
                               worker_id, job.job_id[:12], job.depends_on[:12], dep_status)
                    self._unclaim(job)
                    continue
                if dep.result:
                    job.payload.update(dep.result)

            # mirror job-level org_id into payload — handlers read from payload
            if job.org_id:
                job.payload["org_id"] = job.org_id
            elif job.payload.get("org_id"):
                job.org_id = int(job.payload["org_id"])

            _log.info("queue %s: RUNNING  job=%s  type=%s  priority=%d  source=%s  org=%d",
                       worker_id, job.job_id[:12], job_type, job.priority, job.source or "-", job.org_id)
            if self._dispatch_to_huey(job, worker_id):
                continue

            # Huey-only execution path: if dispatch fails, put the job back and retry.
            # This prevents split execution implementations (inline + Huey).
            _log.error(
                "queue %s: huey dispatch unavailable, re-queueing job=%s type=%s",
                worker_id, job.job_id[:12], job.type,
            )
            self._unclaim(job)
            # Wake this worker type immediately once we re-queue so recovery does
            # not wait up to JOB_QUEUE_POLL_INTERVAL (default 300s).
            wake.set()
            self._stop.wait(timeout=2)

    def _dispatch_to_huey(self, job: ToolJob, worker_id: str) -> bool:
        try:
            from infra.huey_runtime import enqueue_tool_job, is_huey_consumer_running, get_huey
            if not is_huey_consumer_running():
                _log.error(
                    "queue %s: huey consumer NOT RUNNING — cannot dispatch  job=%s  type=%s",
                    worker_id, job.job_id[:12], job.type,
                )
                return False
            ok = enqueue_tool_job(job.job_id)
            if ok:
                # Surface Huey backlog so a job sitting "running" in NocoDB
                # without handler logs is recognisably a Huey-backlog issue
                # rather than a silently-stuck handler.
                pending = "?"
                try:
                    h = get_huey()
                    if h is not None:
                        pending = str(h.pending_count())
                except Exception:
                    pass
                _log.info(
                    "queue %s: DISPATCHED  job=%s  type=%s  org=%d  huey_pending_after=%s",
                    worker_id, job.job_id[:12], job.type, job.org_id, pending,
                )
                self._emit_event({
                    "type": "job_dispatched",
                    "job_id": job.job_id,
                    "job_type": job.type,
                })
                return True
        except Exception:
            _log.error("queue %s: huey dispatch failed  job=%s", worker_id, job.job_id[:12], exc_info=True)
        return False

    def execute_claimed_job(self, job_id: str) -> dict:
        """Execute a previously-claimed running job. Called by Huey worker tasks."""
        worker_id = threading.current_thread().name
        _log.info("huey-pickup %s: PICKED UP  job=%s", worker_id, job_id[:12])
        job = self.get_job(job_id)
        if not job:
            _log.warning("huey-pickup %s: job %s NOT FOUND in queue", worker_id, job_id[:12])
            return {"status": "not_found", "job_id": job_id}
        if job.status != "running":
            _log.warning(
                "huey-pickup %s: job %s SKIPPED  status=%s (expected running — was the row reset by stale reaper?)",
                worker_id, job_id[:12], job.status,
            )
            return {"status": "skipped", "reason": f"status_{job.status}", "job_id": job_id}
        if not _bypass_idle(job):
            idle = seconds_since_chat()
            gate = _background_idle_gate()
            if idle < gate:
                self._unclaim(job)
                remaining_s = max(0, int(round(gate - max(0.0, idle))))
                _log.info(
                    "huey-pickup %s: DEFERRED  job=%s type=%s idle=%.0fs gate=%.0fs remaining=%ds",
                    worker_id,
                    job_id[:12],
                    job.type,
                    idle,
                    gate,
                    remaining_s,
                )
                self._emit_event({
                    "type": "job_deferred_chat_active",
                    "job_id": job.job_id,
                    "job_type": job.type,
                    "idle_seconds": round(idle, 1),
                    "threshold_seconds": round(gate, 1),
                    "remaining_seconds": remaining_s,
                })
                return {
                    "status": "deferred",
                    "reason": "chat_active",
                    "job_id": job_id,
                    "idle_seconds": round(idle, 1),
                    "threshold": round(gate, 1),
                    "remaining_seconds": remaining_s,
                }
        config = self._handlers.get(job.type)
        if not config:
            _log.error(
                "huey-pickup %s: job %s NO HANDLER for type=%s — registered=%s",
                worker_id, job_id[:12], job.type, list(self._handlers.keys()),
            )
            job.status = "failed"
            job.error = f"no handler for job type {job.type}"[:500]
            job.completed_at = datetime.now(timezone.utc).isoformat()
            self._persist_update(job)
            return {"status": "failed", "job_id": job_id, "error": job.error}
        return self._execute_job(job, config, worker_id)

    def _execute_job(self, job: ToolJob, config: HandlerConfig, worker_id: str) -> dict:
        t0 = time.time()
        max_attempts = _max_attempts_for_job_type(job.type)
        attempts = 0
        last_result: dict = {}
        while attempts < max_attempts:
            attempts += 1
            try:
                from shared.models import model_usage_scope

                # Log the safe-to-show identity fields from the payload so a
                # `result_status=not_found` line can be diagnosed without
                # digging through NocoDB. Other payload values are not logged
                # (they may carry user content or oversized strings).
                _identity_fields = ("plan_id", "kind", "conversation_id", "org_id", "insight_id")
                _identity = {k: (job.payload or {}).get(k)
                             for k in _identity_fields if k in (job.payload or {})}
                _log.info(
                    "queue %s: HANDLER START  job=%s  type=%s  attempt=%d/%d  payload=%s  payload_keys=%s",
                    worker_id, job.job_id[:12], job.type, attempts, max_attempts,
                    _identity, sorted((job.payload or {}).keys()),
                )
                handler_t0 = time.time()
                # Set the contextvar so handlers can emit progress and
                # check cancellation via report_progress() / is_job_cancelled()
                # without having to thread the job_id through every call.
                _job_token = _current_job_id.set(job.job_id)
                try:
                    with model_usage_scope(org_id=job.org_id, source=f"tool_queue:{job.type}"):
                        result = config.handler(job.payload)
                except JobCancelled as cancel_exc:
                    result = {
                        "status": "cancelled",
                        "reason": str(cancel_exc) or "cancelled cooperatively",
                    }
                finally:
                    _current_job_id.reset(_job_token)
                handler_elapsed = round(time.time() - handler_t0, 1)
                _log.info(
                    "queue %s: HANDLER RETURN  job=%s  type=%s  %.1fs  result_status=%s",
                    worker_id, job.job_id[:12], job.type, handler_elapsed,
                    (result or {}).get("status") if isinstance(result, dict) else "<non-dict>",
                )

                # Normalise non-dict returns to a failure rather than silently
                # "completing" — a handler that returned a string error or None
                # was previously treated as success and the row was marked
                # completed with empty result, which masked real bugs.
                if not isinstance(result, dict):
                    coerced = {"status": "failed", "error": f"handler returned non-dict: {type(result).__name__}={str(result)[:200]}"}
                    _log.warning(
                        "queue %s: handler returned non-dict for job=%s type=%s — coercing to failure",
                        worker_id, job.job_id[:12], job.type,
                    )
                    result = coerced

                status_val = str(result.get("status") or "").lower()
                if status_val == "cancelled":
                    # Cooperative cancel: don't retry, don't mark failed —
                    # the operator chose to stop this work. Persist as
                    # cancelled with the supplied reason for visibility.
                    job.status = "cancelled"
                    job.result = result
                    job.error = str(result.get("reason") or "cancelled")[:500]
                    break
                if status_val in {"failed", "error"}:
                    last_result = result
                    job.error = str(last_result.get("reason") or last_result.get("error") or status_val)[:500]
                    if attempts < max_attempts:
                        delay = _retry_backoff_s(job.type, attempts)
                        _log.warning(
                            "queue %s: RETRYING  job=%s  type=%s  attempt=%d/%d  error=%s  delay=%.1fs",
                            worker_id, job.job_id[:12], job.type, attempts, max_attempts, job.error, delay,
                        )
                        if self._stop.wait(timeout=delay):
                            break
                        continue
                    job.status = "failed"
                    job.result = last_result
                    job.dead_lettered_at = datetime.now(timezone.utc).isoformat()
                else:
                    job.status = "completed"
                    job.result = result
                    break
            except Exception as e:
                job.error = str(e)[:500]
                if attempts < max_attempts:
                    delay = _retry_backoff_s(job.type, attempts)
                    _log.warning(
                        "queue %s: RETRYING  job=%s  type=%s  attempt=%d/%d  error=%s  delay=%.1fs",
                        worker_id, job.job_id[:12], job.type, attempts, max_attempts, e, delay,
                    )
                    if self._stop.wait(timeout=delay):
                        break
                    continue
                job.status = "failed"
                job.dead_lettered_at = datetime.now(timezone.utc).isoformat()
            except BaseException as be:
                # SystemExit / KeyboardInterrupt / GeneratorExit must not leave
                # the row stuck at status=running. Mark failed, persist, then
                # re-raise so the interpreter can shut down. The stale reaper
                # would clean this up eventually but we lose hours of latency.
                _log.critical(
                    "queue %s: handler raised BaseException for job=%s type=%s — marking failed and re-raising",
                    worker_id, job.job_id[:12], job.type, exc_info=True,
                )
                job.error = f"BaseException: {type(be).__name__}: {str(be)[:300]}"
                job.status = "failed"
                job.result = {"status": "failed", "error": job.error}
                job.completed_at = datetime.now(timezone.utc).isoformat()
                try:
                    self._persist_update(job)
                except Exception:
                    pass
                raise

        if job.status not in {"completed", "failed", "cancelled"}:
            job.status = "failed"
        if not isinstance(job.result, dict):
            job.result = {}
        job.result.setdefault("attempt_count", attempts)
        job.completed_at = datetime.now(timezone.utc).isoformat()
        elapsed = round(time.time() - t0, 1)

        if job.status == "completed":
            _log.info("queue %s: COMPLETED  job=%s  type=%s  %.1fs attempts=%d",
                      worker_id, job.job_id[:12], job.type, elapsed, attempts)
            # Sample duration for ETA estimates. Cap at 50 most recent.
            with self._duration_lock:
                samples = self._duration_samples.setdefault(job.type, [])
                samples.append(float(elapsed))
                if len(samples) > 50:
                    del samples[: len(samples) - 50]
            self._emit_event({
                "type": "job_completed",
                "job_id": job.job_id,
                "job_type": job.type,
                "duration_s": elapsed,
            })
        elif job.status == "cancelled":
            _log.warning("queue %s: CANCELLED  job=%s  type=%s  reason=%s  %.1fs",
                         worker_id, job.job_id[:12], job.type, job.error, elapsed)
            self._emit_event({
                "type": "job_cancelled",
                "job_id": job.job_id,
                "job_type": job.type,
                "reason": job.error[:200],
            })
        else:
            _log.error("queue %s: FAILED  job=%s  type=%s  error=%s  %.1fs attempts=%d",
                       worker_id, job.job_id[:12], job.type, job.error, elapsed, attempts)
            self._emit_event({
                "type": "job_failed",
                "job_id": job.job_id,
                "job_type": job.type,
                "error": job.error[:200],
            })

        self._persist_update(job)
        self._wake_dependents(job.job_id)
        return {
            "status": job.status,
            "job_id": job.job_id,
            "job_type": job.type,
            "error": job.error,
            "result": job.result,
        }

    def _stale_checker_loop(self):
        while not self._stop.is_set():
            self._stop.wait(timeout=60)
            if self._stop.is_set():
                break
            try:
                self._reset_stale_jobs()
            except Exception:
                _log.error("stale checker error", exc_info=True)

    def _reset_stale_jobs(self):
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                return
            # Dynamic stale sizing for long fan-out jobs.
            research_agent_dynamic_mult = 8
            research_planner_dynamic_mult = 4
            try:
                from infra.config import get_feature
                # research_agent can run multiple web-search calls (one per query)
                # then synthesis + critic in the same handler invocation.
                rq_timeout = int(get_feature("research", "web_search_per_query_timeout_s", 180) or 180)
                rq_max = int(get_feature("research", "max_queries", 20) or 20)
                rs_timeout = int(get_feature("research", "synthesis_timeout_s", 1200) or 1200)
                rc_timeout = int(get_feature("research", "critic_timeout_s", 480) or 480)
                research_required_s = max(600, rq_timeout * max(1, rq_max) + rs_timeout + rc_timeout + 300)
                # Cap at 6h so stale reset still recovers truly wedged jobs.
                research_agent_dynamic_mult = min(
                    72,
                    max(8, int(ceil(research_required_s / max(1, JOB_QUEUE_STALE_TIMEOUT)))),
                )

                # research_planner performs one bounded planner LLM call.
                rp_timeout = int(get_feature("research", "planner_timeout_s", 1800) or 1800)
                planner_required_s = max(600, rp_timeout + 300)
                research_planner_dynamic_mult = min(
                    24,
                    max(4, int(ceil(planner_required_s / max(1, JOB_QUEUE_STALE_TIMEOUT)))),
                )
            except Exception:
                research_agent_dynamic_mult = 8
                research_planner_dynamic_mult = 4

            # Per-type stale-timeout overrides from config — operators can
            # extend a single type's window without code changes.
            try:
                from infra.config import get_feature
                per_type_timeouts = get_feature("tool_queue", "stale_timeouts", {}) or {}
            except Exception:
                per_type_timeouts = {}

            rows = db._get_paginated(NOCODB_TABLE, params={
                "where": "(status,eq,running)",
                "limit": 100,
            })
            now = time.time()
            reset_types: set[str] = set()
            for row in rows:
                started = row.get("started_at") or ""
                if not started:
                    continue
                try:
                    started_ts = datetime.fromisoformat(started).timestamp()
                except Exception:
                    continue
                # Heartbeat-aware staleness: a handler that stamped a
                # progress message in the last few minutes is clearly making
                # progress; the previous logic would reset it just because
                # `started_at` was old. Prefer the latest of (started_at,
                # progress_at) so long-running handlers (research_agent,
                # harvest_run, etc.) only get reset when they truly stall.
                progress_ts = 0.0
                progress_at_raw = row.get("progress_at") or ""
                if progress_at_raw:
                    try:
                        progress_ts = datetime.fromisoformat(progress_at_raw).timestamp()
                    except Exception:
                        progress_ts = 0.0
                last_activity_ts = max(started_ts, progress_ts)

                job_type = row.get("type") or ""
                _STALE_MULTIPLIERS = {
                    "graph_extract": 4,               # 20m — LLM inference 7-16min
                    "research_planner": research_planner_dynamic_mult,
                    "research_agent": research_agent_dynamic_mult,
                    # research_review and research_op share research_agent's
                    # workload profile — both can fire multiple bounded LLM
                    # calls + web search per job. Without an entry here the
                    # reaper resets them mid-flight every ~5 minutes and the
                    # row never reaches "completed".
                    "research_review": research_agent_dynamic_mult,
                    "research_op": research_agent_dynamic_mult,
                    # Harvest jobs run a per-URL fetch+extract loop bounded
                    # by policy.timeout_total_s (≤2h). Use the same long
                    # multiplier so the reaper does not reset them mid-walk.
                    "harvest_run": research_agent_dynamic_mult,
                    "harvest_finalise": 4,
                    "scrape_page": 3,                 # 15m — fetch + chunk + embed
                    "pathfinder_extract": 3,          # 15m — fetch + link extract
                    "summarise_page": 3,              # 15m — summariser LLM
                    "extract_relationships": 4,       # 20m — relationship extraction LLM
                    "discover_agent_run": 3,          # 15m — Chroma sample + LLM + SearXNG queries
                    "simulation_run": 6,              # 30m — N persona turns + debrief LLM
                }
                # Per-type config override wins over the multiplier table.
                cfg_timeout = per_type_timeouts.get(job_type) if isinstance(per_type_timeouts, dict) else None
                if isinstance(cfg_timeout, (int, float)) and cfg_timeout > 0:
                    timeout = float(cfg_timeout)
                else:
                    timeout = JOB_QUEUE_STALE_TIMEOUT * _STALE_MULTIPLIERS.get(job_type, 1)
                # Heartbeat-aware: stale window measured from latest activity
                # (progress_at if newer than started_at), not just claim time.
                stuck_for = now - last_activity_ts
                if stuck_for > timeout:
                    noco_id = row.get("Id")
                    db._patch(NOCODB_TABLE, noco_id, {
                        "Id": noco_id,
                        "status": "queued",
                        "claimed_by": "",
                        "started_at": "",
                    })
                    if job_type:
                        reset_types.add(job_type)
                    _log.warning(
                        "reset stale job %s (type=%s, stuck %.0fs since last activity, timeout=%.0fs)",
                        row.get("job_id"), row.get("type"), stuck_for, timeout,
                    )
            for jt in reset_types:
                ev = self._wake_events.get(jt)
                if ev:
                    ev.set()
            if reset_types:
                _log.info("stale reset wake  types=%s", sorted(reset_types))
        except Exception:
            _log.error("stale job reset failed", exc_info=True)

    @staticmethod
    def _db():
        from infra.nocodb_client import NocodbClient
        return NocodbClient()

    def _persist_new(self, job: ToolJob) -> bool:
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                _log.warning("table %s not found — job %s not persisted", NOCODB_TABLE, job.job_id)
                return False
            row = db._post(NOCODB_TABLE, job.to_row())
            job.nocodb_id = row.get("Id")
            return True
        except Exception:
            _log.error("persist_new failed job=%s", job.job_id, exc_info=True)
            return False

    def _persist_update(self, job: ToolJob):
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables or not job.nocodb_id:
                return
            data = job.to_row()
            data["Id"] = job.nocodb_id
            db._patch(NOCODB_TABLE, job.nocodb_id, data)
        except Exception:
            _log.error("persist_update failed job=%s", job.job_id, exc_info=True)

    def _claim_next(self, job_type: str, worker_id: str) -> ToolJob | None:
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                return None

            rows = db._get(NOCODB_TABLE, params={
                "where": f"(type,eq,{job_type})~and(status,eq,queued)",
                "sort": "CreatedAt",
                "limit": 1,
            }).get("list", [])

            if not rows:
                return None

            row = rows[0]
            noco_id = row.get("Id")
            now = datetime.now(timezone.utc).isoformat()

            db._patch(NOCODB_TABLE, noco_id, {
                "Id": noco_id,
                "status": "running",
                "claimed_by": worker_id,
                "started_at": now,
            })

            # re-fetch to verify claim won the race (nocodb has no CAS)
            verify = db._get(NOCODB_TABLE, params={
                "where": f"(Id,eq,{noco_id})",
                "limit": 1,
            }).get("list", [])

            if not verify:
                return None
            v = verify[0]
            if v.get("claimed_by") != worker_id or v.get("status") != "running":
                _log.debug("claim race lost for noco_id=%s worker=%s", noco_id, worker_id)
                return None

            job = ToolJob.from_row(v)
            job.nocodb_id = noco_id
            return job

        except Exception:
            _log.error("claim_next failed type=%s", job_type, exc_info=True)
            return None

    def _peek_next_queued(self, job_type: str) -> ToolJob | None:
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                return None
            rows = db._get(NOCODB_TABLE, params={
                "where": f"(type,eq,{job_type})~and(status,eq,queued)",
                "sort": "CreatedAt",
                "limit": 1,
            }).get("list", [])
            if not rows:
                return None
            return ToolJob.from_row(rows[0])
        except Exception:
            return None

    def _unclaim(self, job: ToolJob):
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables or not job.nocodb_id:
                return
            db._patch(NOCODB_TABLE, job.nocodb_id, {
                "Id": job.nocodb_id,
                "status": "queued",
                "claimed_by": "",
                "started_at": "",
            })
        except Exception:
            _log.error("unclaim failed job=%s", job.job_id, exc_info=True)

    def _find_dedup(self, job_type: str, key: str, value: str) -> str | None:
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                return None

            # url dedup has to filter in python — nocodb where-clauses choke on urls with special chars
            if key == "url":
                rows = db._get_paginated(NOCODB_TABLE, params={
                    "where": f"(type,eq,{job_type})~and(status,in,queued,running)",
                    "limit": 200,
                })
                for row in rows:
                    pj = row.get("payload_json") or "{}"
                    if isinstance(pj, str):
                        try:
                            p = json.loads(pj)
                        except Exception:
                            continue
                    else:
                        p = pj
                    if p.get(key) == value:
                        return row.get("job_id")
                return None

            return None
        except Exception:
            return None

    def _wake_dependents(self, completed_job_id: str):
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                return
            rows = db._get(NOCODB_TABLE, params={
                "where": f"(depends_on,eq,{completed_job_id})~and(status,eq,queued)",
                "limit": 50,
            }).get("list", [])
            if rows:
                types_to_wake = {r.get("type") for r in rows}
                for jt in types_to_wake:
                    ev = self._wake_events.get(jt)
                    if ev:
                        ev.set()
                _log.debug("woke dependents for job=%s types=%s", completed_job_id, types_to_wake)
        except Exception:
            _log.error("wake_dependents failed for job=%s", completed_job_id, exc_info=True)

    def _load_pending(self):
        # crash recovery: any rows left as running from a previous process are orphaned
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                _log.info("table %s not found — starting with empty queue", NOCODB_TABLE)
                return
            rows = db._get_paginated(NOCODB_TABLE, params={
                "where": "(status,eq,running)",
                "limit": 200,
            })
            for row in rows:
                noco_id = row.get("Id")
                db._patch(NOCODB_TABLE, noco_id, {
                    "Id": noco_id,
                    "status": "queued",
                    "claimed_by": "",
                    "started_at": "",
                })
            if rows:
                _log.info("reset %d stale running jobs to queued on startup", len(rows))
        except Exception:
            _log.error("load_pending failed", exc_info=True)


_instance: ToolJobQueue | None = None


def get_tool_queue() -> ToolJobQueue | None:
    return _instance


def _set_instance(q: ToolJobQueue):
    global _instance
    _instance = q
