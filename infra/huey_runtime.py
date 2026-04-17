from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Callable, Any

from huey import SqliteHuey

from infra.config import (
    HUEY_CONSUMER_WORKERS,
    HUEY_ENABLED,
    HUEY_SQLITE_PATH,
    HUEY_TASK_RETRIES,
    HUEY_TASK_RETRY_DELAY_S,
)

_log = logging.getLogger("huey.runtime")

_huey: SqliteHuey | None = None
_run_tool_job_task: Callable[[str], Any] | None = None
_consumer = None
_consumer_thread: threading.Thread | None = None


def _ensure_sqlite_path(path: str) -> None:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Create the SQLite file eagerly so container startup guarantees persistence target exists.
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.commit()
    finally:
        conn.close()


def init_huey() -> SqliteHuey | None:
    global _huey
    if not HUEY_ENABLED:
        _log.info("huey disabled by HUEY_ENABLED")
        return None
    if _huey is not None:
        return _huey

    _ensure_sqlite_path(HUEY_SQLITE_PATH)
    _huey = SqliteHuey(
        name="harness",
        filename=HUEY_SQLITE_PATH,
        fsync=True,
    )
    _register_tasks()
    _log.info("huey sqlite runtime ready  db=%s", HUEY_SQLITE_PATH)
    return _huey


def _register_tasks() -> None:
    global _run_tool_job_task
    if _huey is None or _run_tool_job_task is not None:
        return

    @_huey.task(
        retries=max(0, int(HUEY_TASK_RETRIES or 0)),
        retry_delay=max(0, int(HUEY_TASK_RETRY_DELAY_S or 0)),
    )
    def _run_tool_job(job_id: str) -> dict:
        from workers.tool_queue import get_tool_queue

        q = get_tool_queue()
        if q is None:
            raise RuntimeError("tool queue instance unavailable")
        return q.execute_claimed_job(job_id)

    _run_tool_job_task = _run_tool_job


def enqueue_tool_job(job_id: str) -> bool:
    if not HUEY_ENABLED:
        return False
    if _huey is None:
        init_huey()
    if _run_tool_job_task is None:
        return False
    _run_tool_job_task(job_id)
    return True


def is_huey_consumer_running() -> bool:
    return bool(_consumer_thread and _consumer_thread.is_alive())


def start_huey_consumer() -> bool:
    global _consumer, _consumer_thread
    if not HUEY_ENABLED:
        return False
    if _consumer_thread and _consumer_thread.is_alive():
        return True
    if _huey is None:
        init_huey()
    if _huey is None:
        return False

    from huey.consumer import Consumer

    workers = max(1, int(HUEY_CONSUMER_WORKERS or 1))
    _consumer = Consumer(_huey, workers=workers)
    _consumer_thread = threading.Thread(target=_consumer.run, daemon=True, name="huey-consumer")
    _consumer_thread.start()
    _log.info("huey consumer started  workers=%d", workers)
    return True


def get_huey() -> SqliteHuey | None:
    return _huey


def shutdown_huey() -> None:
    global _huey, _consumer, _consumer_thread, _run_tool_job_task
    if _consumer is not None:
        stop = getattr(_consumer, "stop", None)
        if callable(stop):
            try:
                stop(graceful=True)
            except TypeError:
                stop()
            except Exception:
                _log.warning("huey consumer stop failed", exc_info=True)
    if _consumer_thread is not None and _consumer_thread.is_alive():
        _consumer_thread.join(timeout=5)
    _consumer = None
    _consumer_thread = None
    _run_tool_job_task = None
    _huey = None

