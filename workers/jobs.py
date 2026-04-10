import json
import logging
import threading
import time
import uuid
from typing import Callable, Iterator, Optional

_log = logging.getLogger("jobs")

# Single-process job store. To scale to multiple workers, back this with Redis
# (streams or pub/sub + list) — the public API here is designed to swap.

_GC_AFTER_SECONDS = 300
_MAX_EVENTS_PER_JOB = 10_000
_WAIT_TIMEOUT = 15.0


class Job:
    __slots__ = ("id", "events", "done", "error", "finished_at", "cond")

    def __init__(self, job_id: str):
        self.id = job_id
        self.events: list[dict] = []
        self.done = False
        self.error: Optional[str] = None
        self.finished_at: Optional[float] = None
        self.cond = threading.Condition()


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def _gc_locked(self):
        now = time.time()
        stale = [
            jid for jid, j in self._jobs.items()
            if j.done and j.finished_at and now - j.finished_at > _GC_AFTER_SECONDS
        ]
        for jid in stale:
            self._jobs.pop(jid, None)
        if stale:
            _log.debug("gc'd %d finished jobs", len(stale))

    def create(self) -> Job:
        with self._lock:
            self._gc_locked()
            job = Job(uuid.uuid4().hex)
            self._jobs[job.id] = job
            return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def append(self, job: Job, event: dict):
        with job.cond:
            if len(job.events) < _MAX_EVENTS_PER_JOB:
                job.events.append(event)
            job.cond.notify_all()

    def finish(self, job: Job, error: Optional[str] = None):
        with job.cond:
            job.error = error
            job.done = True
            job.finished_at = time.time()
            job.cond.notify_all()
        if error:
            _log.error("job %s finished with error: %s", job.id, error)
        else:
            _log.debug("job %s finished  events=%d", job.id, len(job.events))


STORE = JobStore()


def run_in_background(job: Job, generator_factory: Callable[[], Iterator[dict]]):
    _log.debug("job %s started", job.id)

    def _worker():
        try:
            for event in generator_factory():
                STORE.append(job, event)
        except Exception as e:
            STORE.append(job, {"type": "error", "message": str(e)})
            STORE.finish(job, error=str(e))
            return
        STORE.finish(job)

    threading.Thread(target=_worker, daemon=True).start()


def stream_events(job_id: str, cursor: int = 0) -> Iterator[str]:
    job = STORE.get(job_id)
    if job is None:
        yield f"data: {json.dumps({'type': 'error', 'message': 'job not found or expired'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    idx = max(0, cursor)
    while True:
        with job.cond:
            if idx >= len(job.events) and not job.done:
                job.cond.wait(timeout=_WAIT_TIMEOUT)
            pending = job.events[idx:]
            done_now = job.done and idx + len(pending) >= len(job.events)

        for ev in pending:
            yield f"id: {idx}\ndata: {json.dumps(ev)}\n\n"
            idx += 1

        if done_now:
            yield "data: [DONE]\n\n"
            return

        if not pending:
            yield ": keepalive\n\n"