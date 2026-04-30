"""Huey runtime — SqliteHuey with a threaded consumer and a health watchdog.

Public surface (what the rest of the app uses):

    init_huey()                — create the Huey instance and register tasks
    start_huey_consumer()      — start the consumer thread + health monitor
    enqueue_tool_job(job_id)   — producer-side enqueue
    is_huey_consumer_running() — Thread.is_alive() check
    get_huey_health()           — snapshot for /tool-queue/status
    get_huey()                  — accessor for queue depth, etc.
    shutdown_huey()             — stop the consumer cleanly

The architecture is a strict producer/consumer pair on a single SQLite
queue. The tool-queue worker threads claim NocoDB rows and call
``enqueue_tool_job``; the Huey consumer thread pulls from SQLite and
runs the ``run_tool_job`` task, which calls back into the tool queue's
``execute_claimed_job`` to dispatch the actual handler.

A background health monitor enqueues a heartbeat task once per minute.
If it does not execute within HEARTBEAT_TIMEOUT_S the consumer is
considered wedged and restarted in place. ``Thread.is_alive()`` is not
sufficient: threads can be alive yet stuck on internal locks.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable

from huey import SqliteHuey
from huey.consumer import Consumer

from infra.config import (
    HUEY_CONSUMER_WORKERS,
    HUEY_ENABLED,
    HUEY_SQLITE_PATH,
    HUEY_TASK_RETRIES,
    HUEY_TASK_RETRY_DELAY_S,
)

_log = logging.getLogger("huey.runtime")

# Suppress Huey's per-second worker/scheduler/monitor heartbeat DEBUG spam.
for _noisy in (
    "huey.consumer",
    "huey.consumer.Scheduler",
    "huey.consumer.Worker",
    "huey.consumer.Monitor",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ── tuning ──────────────────────────────────────────────────────────────────

# Health monitor: fire a heartbeat every 60s, expect it to run within 90s.
# Worst-case detection latency for a wedged consumer is therefore ~90s.
_HEARTBEAT_INTERVAL_S = 60
_HEARTBEAT_TIMEOUT_S = 90
_MAX_RESTART_ATTEMPTS = 3


class _ThreadedConsumer(Consumer):
    """Huey consumer that skips signal wiring (we run from a background thread).

    Huey's default `_set_signal_handlers` only works on the main interpreter
    thread; calling it from a worker thread raises ``ValueError: signal only
    works in main thread``. We disable it — graceful shutdown is driven by
    ``shutdown_huey()`` from the FastAPI lifespan.
    """

    def _set_signal_handlers(self):  # type: ignore[override]
        return


# ── module state ────────────────────────────────────────────────────────────
#
# All consumer/monitor lifecycle state lives at module scope. Compound
# updates (e.g. swapping the consumer) are protected by `_lifecycle_lock`.
# Heartbeat timestamps are GIL-atomic floats — see comment below.

_huey: SqliteHuey | None = None
_run_tool_job: Callable[[str], object] | None = None
_heartbeat_task: Callable[[], object] | None = None
_consumer: Consumer | None = None
_consumer_thread: threading.Thread | None = None
_monitor_thread: threading.Thread | None = None

# `_lifecycle_lock` guards the consumer/monitor lifecycle (start, stop,
# restart). It is NOT held during heartbeat writes — a Huey worker thread
# executing the heartbeat task must never block on this lock, otherwise a
# concurrent `_restart_consumer` (which holds the lock and calls
# ``consumer.stop(graceful=True)``, waiting for that very worker to finish)
# would deadlock.
_lifecycle_lock = threading.Lock()
_monitor_stop = threading.Event()
# Single float reads/writes are atomic under CPython's GIL, so the heartbeat
# timestamps are safe to read/write without a lock. A concurrent restart
# might briefly observe a stale value, which is acceptable for monitoring.
_heartbeat_fired_at: float = 0.0   # last time the monitor enqueued a heartbeat
_heartbeat_ran_at: float = 0.0     # last time a heartbeat task body executed


# ── lifecycle ───────────────────────────────────────────────────────────────

def init_huey() -> SqliteHuey | None:
    """Create the Huey instance and register tasks. Idempotent."""
    global _huey
    if not HUEY_ENABLED:
        _log.info("huey disabled (HUEY_ENABLED=False)")
        return None
    if _huey is not None:
        return _huey
    _ensure_sqlite_path(HUEY_SQLITE_PATH)
    _huey = SqliteHuey(name="harness", filename=HUEY_SQLITE_PATH, fsync=True)
    _register_tasks(_huey)
    _log.info("huey ready  db=%s", HUEY_SQLITE_PATH)
    return _huey


def start_huey_consumer() -> bool:
    """Start the consumer thread and health monitor. Idempotent."""
    if not HUEY_ENABLED:
        return False
    if _huey is None and init_huey() is None:
        return False
    with _lifecycle_lock:
        if _consumer_thread is not None and _consumer_thread.is_alive():
            return True
        _spawn_consumer_locked()
        _start_health_monitor_locked()
    return True


def shutdown_huey() -> None:
    """Stop the consumer and monitor cleanly. Idempotent."""
    global _huey, _run_tool_job, _heartbeat_task, _monitor_thread
    _monitor_stop.set()
    with _lifecycle_lock:
        _stop_consumer_locked()
        mt = _monitor_thread
    if mt is not None and mt.is_alive():
        mt.join(timeout=5)
    _monitor_thread = None
    _huey = None
    _run_tool_job = None
    _heartbeat_task = None


# ── producer ────────────────────────────────────────────────────────────────

def enqueue_tool_job(job_id: str) -> bool:
    """Enqueue a tool-queue job for the consumer.

    Returns False if Huey is disabled or not initialised. Otherwise calls the
    Huey task (which writes to SQLite) and returns True. Network/SQLite
    errors propagate so the caller can react.
    """
    if not HUEY_ENABLED or _run_tool_job is None:
        return False
    _run_tool_job(job_id)
    return True


# ── status ──────────────────────────────────────────────────────────────────

def get_huey() -> SqliteHuey | None:
    return _huey


def is_huey_consumer_running() -> bool:
    """Returns True if the consumer thread object exists and is_alive().

    NOTE: this does not detect a wedged-but-alive thread. Use
    ``get_huey_health()['consumer_healthy']`` for that.
    """
    return bool(_consumer_thread and _consumer_thread.is_alive())


def get_huey_health() -> dict:
    """Snapshot of consumer + heartbeat state for the status endpoint."""
    fired = _heartbeat_fired_at
    ran = _heartbeat_ran_at
    now = time.time()
    alive = is_huey_consumer_running()
    healthy = (
        alive
        and ran > 0
        and (now - ran) < (_HEARTBEAT_INTERVAL_S + _HEARTBEAT_TIMEOUT_S)
    )
    return {
        "consumer_running": alive,
        "consumer_healthy": healthy,
        "heartbeat_last_fired_s_ago": round(now - fired, 1) if fired else None,
        "heartbeat_last_ran_s_ago": round(now - ran, 1) if ran else None,
    }


# ── internals ───────────────────────────────────────────────────────────────

def _ensure_sqlite_path(path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.commit()
    finally:
        conn.close()


def _register_tasks(huey: SqliteHuey) -> None:
    """Bind the tool-queue dispatcher task to Huey. Idempotent."""
    global _run_tool_job
    if _run_tool_job is not None:
        return

    @huey.task(
        retries=max(0, int(HUEY_TASK_RETRIES or 0)),
        retry_delay=max(0, int(HUEY_TASK_RETRY_DELAY_S or 0)),
    )
    def run_tool_job(job_id: str) -> dict:
        from workers.tool_queue import get_tool_queue
        q = get_tool_queue()
        if q is None:
            raise RuntimeError("tool queue not initialised")
        return q.execute_claimed_job(job_id)

    @huey.task()
    def heartbeat() -> str:
        # No lock — see _lifecycle_lock comment. Single float assignment is
        # GIL-atomic, and the worker must not block on the lifecycle lock
        # because a concurrent restart would deadlock against worker drain.
        global _heartbeat_ran_at
        _heartbeat_ran_at = time.time()
        return "ok"

    global _heartbeat_task
    _run_tool_job = run_tool_job
    _heartbeat_task = heartbeat


def _spawn_consumer_locked() -> None:
    """Create and start a fresh consumer thread. Caller holds _lifecycle_lock."""
    global _consumer, _consumer_thread
    if _huey is None:
        raise RuntimeError("init_huey must be called first")
    workers = max(1, int(HUEY_CONSUMER_WORKERS or 1))
    _consumer = _ThreadedConsumer(_huey, workers=workers, scheduler_interval=5)
    _consumer_thread = threading.Thread(
        target=_consumer.run, daemon=True, name="huey-consumer",
    )
    _consumer_thread.start()
    _log.info("huey consumer started  workers=%d", workers)


def _stop_consumer_locked() -> None:
    """Stop the current consumer. Caller holds _lifecycle_lock. Never raises."""
    global _consumer, _consumer_thread
    if _consumer is None:
        return
    stop = getattr(_consumer, "stop", None)
    if callable(stop):
        try:
            stop(graceful=True)
        except TypeError:
            # Older Huey versions without the `graceful` kwarg.
            try:
                stop()
            except Exception:
                pass
        except RuntimeError as e:
            # Raised on very fast start→stop cycles when worker threads
            # have not finished initialising. Safe to ignore.
            if "before it is started" not in str(e):
                _log.warning("huey consumer stop error", exc_info=True)
        except Exception:
            _log.warning("huey consumer stop error", exc_info=True)
    _consumer = None
    _consumer_thread = None


def _restart_consumer() -> None:
    """Replace a wedged consumer in place. Acquires _lifecycle_lock."""
    _log.warning("huey consumer restart triggered")
    with _lifecycle_lock:
        _stop_consumer_locked()
        _spawn_consumer_locked()


def _start_health_monitor_locked() -> None:
    """Start the health watchdog thread. Caller holds _lifecycle_lock."""
    global _monitor_thread
    if _monitor_thread is not None and _monitor_thread.is_alive():
        return
    if _heartbeat_task is None:
        _log.warning("huey health monitor: heartbeat task not registered, skipping")
        return
    _monitor_stop.clear()
    _monitor_thread = threading.Thread(
        target=_monitor_loop,
        args=(_heartbeat_task,),
        daemon=True,
        name="huey-health-monitor",
    )
    _monitor_thread.start()
    _log.info(
        "huey health monitor started  interval=%ds  timeout=%ds",
        _HEARTBEAT_INTERVAL_S, _HEARTBEAT_TIMEOUT_S,
    )


def _monitor_loop(heartbeat_task: Callable[[], object]) -> None:
    """Body of the health watchdog.

    Cycle:
      1. Enqueue a heartbeat task; record fired_at.
      2. Wait HEARTBEAT_TIMEOUT_S.
      3. If the heartbeat ran (ran_at >= fired_at), counter resets.
         Else log CRITICAL and restart the consumer (up to MAX attempts).
      4. Wait the rest of HEARTBEAT_INTERVAL_S.

    All sleeps go through `_monitor_stop.wait()` so shutdown is prompt.
    """
    global _heartbeat_fired_at
    failures = 0
    exhausted_logged = False  # rate-limit the "attempts exhausted" CRITICAL

    while not _monitor_stop.is_set():
        # Fire heartbeat. No lock: single assignment is GIL-atomic and we
        # must not hold the lifecycle lock during enqueue — that would
        # serialise against any in-flight restart attempt.
        try:
            _heartbeat_fired_at = time.time()
            heartbeat_task()
        except Exception:
            _log.error("huey heartbeat enqueue failed", exc_info=True)
            if _monitor_stop.wait(_HEARTBEAT_INTERVAL_S):
                return
            continue

        # Wait for the heartbeat to run (or timeout)
        if _monitor_stop.wait(_HEARTBEAT_TIMEOUT_S):
            return
        fired = _heartbeat_fired_at
        ran = _heartbeat_ran_at

        if ran >= fired:
            if failures > 0 or exhausted_logged:
                _log.warning("HUEY consumer recovered (heartbeat ran successfully)")
            failures = 0
            exhausted_logged = False
        else:
            failures += 1
            age = time.time() - fired
            _log.critical(
                "HUEY CONSUMER NOT CONSUMING — heartbeat fired %.0fs ago, never ran "
                "(thread_alive=%s, failures=%d/%d).",
                age, is_huey_consumer_running(), failures, _MAX_RESTART_ATTEMPTS,
            )
            if failures <= _MAX_RESTART_ATTEMPTS:
                try:
                    _restart_consumer()
                except Exception:
                    _log.error("huey consumer restart failed", exc_info=True)
            elif not exhausted_logged:
                # Log the give-up message exactly once per failure streak,
                # not every minute thereafter.
                _log.critical(
                    "HUEY: %d restart attempts exhausted. Manual intervention required. "
                    "Will continue monitoring; further alerts suppressed until recovery.",
                    _MAX_RESTART_ATTEMPTS,
                )
                exhausted_logged = True

        # Wait the rest of the interval before next probe
        remaining = max(0, _HEARTBEAT_INTERVAL_S - _HEARTBEAT_TIMEOUT_S)
        if remaining and _monitor_stop.wait(remaining):
            return
