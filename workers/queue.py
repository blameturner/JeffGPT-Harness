"""
Job queue for the T1 Secondary model (Gemma 4 31B Dense, port 8081).

Single-threaded worker processes one job at a time through the slow high-quality
model. Jobs are priority-sorted (1=highest) and persisted to NocoDB so they
survive harness restarts.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

import requests

from config import get_model_url, no_think_params

_log = logging.getLogger("queue")

DEFAULT_TIMEOUTS = {
    "summarise": 300,
    "enrich": 600,
    "analyse": 900,
    "agent": 1800,
    "research": 3600,
}

AVG_TIMES = {
    "summarise": 120,
    "enrich": 180,
    "analyse": 300,
    "agent": 600,
    "research": 900,
}


@dataclass
class QueueJob:
    id: str
    type: str
    payload: dict
    priority: int = 3
    status: str = "queued"
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    result: str = ""
    error: str = ""
    callback_url: str = ""
    timeout_s: int = 0
    nocodb_id: int | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("nocodb_id", None)
        d.pop("created_at", None)
        if isinstance(d.get("payload"), dict):
            d["payload"] = json.dumps(d["payload"])
        return d

    def to_api(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "priority": self.priority,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "timeout_s": self.timeout_s,
        }


class JobQueue:
    def __init__(self):
        self._queue: list[QueueJob] = []
        self._current: QueueJob | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._cancel_flag = threading.Event()
        self._worker: threading.Thread | None = None
        self._subscribers: list[list[dict]] = []
        self._sub_lock = threading.Lock()
        self._model_url: str | None = None
        self._completion_history: list[tuple[str, float]] = []

    def start(self):
        self._model_url = get_model_url("t1_secondary")
        if not self._model_url:
            _log.warning("t1_secondary model not available — queue disabled")
            return
        _log.info("queue starting  model_url=%s", self._model_url)
        self._load_from_db()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="queue-worker")
        self._worker.start()

    def _load_from_db(self):
        try:
            from nocodb_client import NocodbClient
            db = NocodbClient()
            if "queue_jobs" not in db.tables:
                _log.info("queue_jobs table not found — starting with empty queue")
                return
            rows = db._get("queue_jobs", params={
                "where": "(status,in,queued,running)",
                "sort": "priority,CreatedAt",
                "limit": 200,
            }).get("list", [])
            for row in rows:
                payload = row.get("payload") or "{}"
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except Exception:
                        payload = {}
                job = QueueJob(
                    id=row.get("job_id") or str(uuid.uuid4().hex),
                    type=row.get("type") or "unknown",
                    payload=payload,
                    priority=int(row.get("priority") or 3),
                    status="queued",
                    created_at=row.get("created_at") or "",
                    timeout_s=int(row.get("timeout_s") or 0),
                    callback_url=row.get("callback_url") or "",
                    nocodb_id=row.get("Id"),
                )
                if not job.timeout_s:
                    job.timeout_s = DEFAULT_TIMEOUTS.get(job.type, 600)
                self._queue.append(job)
            if rows:
                _log.info("queue loaded %d persisted jobs from NocoDB", len(rows))
                self._sort_queue()
        except Exception:
            _log.error("queue failed to load from NocoDB", exc_info=True)

    def _sort_queue(self):
        self._queue.sort(key=lambda j: (j.priority, j.created_at))

    def _persist_job(self, job: QueueJob):
        try:
            from nocodb_client import NocodbClient
            db = NocodbClient()
            if "queue_jobs" not in db.tables:
                return
            data = job.to_dict()
            if job.nocodb_id:
                db._patch("queue_jobs", job.nocodb_id, {"Id": job.nocodb_id, **data})
            else:
                row = db._post("queue_jobs", data)
                job.nocodb_id = row.get("Id")
        except Exception:
            _log.error("queue persist failed  job=%s", job.id, exc_info=True)

    def _emit_event(self, event: dict):
        _log.info("queue event  %s", event.get("type", ""))
        with self._sub_lock:
            for sub in self._subscribers:
                sub.append(event)

    def subscribe(self) -> list[dict]:
        buf: list[dict] = []
        with self._sub_lock:
            self._subscribers.append(buf)
        return buf

    def unsubscribe(self, buf: list[dict]):
        with self._sub_lock:
            try:
                self._subscribers.remove(buf)
            except ValueError:
                pass

    def submit(
        self,
        job_type: str,
        payload: dict,
        priority: int = 3,
        callback_url: str = "",
        timeout_s: int = 0,
    ) -> QueueJob:
        priority = max(1, min(5, priority))
        if not timeout_s:
            timeout_s = DEFAULT_TIMEOUTS.get(job_type, 600)
        job = QueueJob(
            id=uuid.uuid4().hex,
            type=job_type,
            payload=payload,
            priority=priority,
            status="queued",
            created_at=datetime.now(timezone.utc).isoformat(),
            timeout_s=timeout_s,
            callback_url=callback_url,
        )
        with self._lock:
            self._queue.append(job)
            self._sort_queue()
            position = self._queue.index(job) + 1
        self._persist_job(job)
        self._emit_event({
            "type": "job_queued",
            "job_id": job.id,
            "job_type": job.type,
            "position": position,
            "queue_length": len(self._queue),
        })
        _log.info("queue submit  id=%s type=%s priority=%d position=%d", job.id, job.type, job.priority, position)
        self._wake.set()
        return job

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            for i, job in enumerate(self._queue):
                if job.id == job_id:
                    job.status = "cancelled"
                    job.completed_at = datetime.now(timezone.utc).isoformat()
                    self._queue.pop(i)
                    self._persist_job(job)
                    self._emit_event({"type": "job_cancelled", "job_id": job_id})
                    _log.info("queue cancel  id=%s (was queued)", job_id)
                    return True
            if self._current and self._current.id == job_id:
                self._cancel_flag.set()
                _log.info("queue cancel  id=%s (running — flagged for cancel)", job_id)
                return True
        return False

    def update_priority(self, job_id: str, new_priority: int) -> bool:
        new_priority = max(1, min(5, new_priority))
        with self._lock:
            for job in self._queue:
                if job.id == job_id:
                    job.priority = new_priority
                    self._sort_queue()
                    self._persist_job(job)
                    _log.info("queue priority  id=%s new=%d", job_id, new_priority)
                    return True
        return False

    def get_job(self, job_id: str) -> QueueJob | None:
        with self._lock:
            if self._current and self._current.id == job_id:
                return self._current
            for job in self._queue:
                if job.id == job_id:
                    return job
        try:
            from nocodb_client import NocodbClient
            db = NocodbClient()
            if "queue_jobs" not in db.tables:
                return None
            rows = db._get("queue_jobs", params={
                "where": f"(job_id,eq,{job_id})",
                "limit": 1,
            }).get("list", [])
            if rows:
                row = rows[0]
                return QueueJob(
                    id=row.get("job_id", ""),
                    type=row.get("type", ""),
                    payload={},
                    priority=int(row.get("priority") or 3),
                    status=row.get("status", ""),
                    created_at=row.get("created_at", ""),
                    started_at=row.get("started_at", ""),
                    completed_at=row.get("completed_at", ""),
                    result=row.get("result", ""),
                    error=row.get("error", ""),
                    timeout_s=int(row.get("timeout_s") or 0),
                    nocodb_id=row.get("Id"),
                )
        except Exception:
            pass
        return None

    def list_jobs(self, limit: int = 50) -> list[dict]:
        with self._lock:
            jobs = []
            if self._current:
                jobs.append(self._current.to_api())
            for job in self._queue[:limit]:
                jobs.append(job.to_api())
        return jobs

    def status(self) -> dict:
        with self._lock:
            current = None
            if self._current:
                elapsed = time.time() - (
                    datetime.fromisoformat(self._current.started_at).timestamp()
                    if self._current.started_at else time.time()
                )
                current = {
                    "job_id": self._current.id,
                    "type": self._current.type,
                    "elapsed_s": int(elapsed),
                }
            return {
                "queue_length": len(self._queue),
                "current_job": current,
                "estimated_wait_s": self._estimate_wait(),
                "model_url": self._model_url,
            }

    def _estimate_wait(self) -> int:
        wait = 0
        if self._current and self._current.started_at:
            try:
                elapsed = time.time() - datetime.fromisoformat(self._current.started_at).timestamp()
            except Exception:
                elapsed = 0
            expected = self._avg_time(self._current.type)
            wait += max(0, int(expected - elapsed))
        for job in self._queue:
            wait += self._avg_time(job.type)
        return wait

    def _avg_time(self, job_type: str) -> int:
        recent = [dur for t, dur in self._completion_history if t == job_type]
        if recent:
            return int(sum(recent) / len(recent))
        return AVG_TIMES.get(job_type, 300)

    def _worker_loop(self):
        _log.info("queue worker started")
        while True:
            self._wake.wait(timeout=30)
            self._wake.clear()

            job: QueueJob | None = None
            with self._lock:
                if self._queue:
                    job = self._queue.pop(0)
                    self._current = job

            if not job:
                continue

            self._cancel_flag.clear()
            job.status = "running"
            job.started_at = datetime.now(timezone.utc).isoformat()
            self._persist_job(job)
            self._emit_event({
                "type": "job_started",
                "job_id": job.id,
                "job_type": job.type,
                "estimated_s": self._avg_time(job.type),
            })

            _log.info("queue run  id=%s type=%s priority=%d timeout=%ds", job.id, job.type, job.priority, job.timeout_s)
            t0 = time.time()

            try:
                result = self._execute_job(job)
                duration = round(time.time() - t0, 1)

                if self._cancel_flag.is_set():
                    job.status = "cancelled"
                    job.completed_at = datetime.now(timezone.utc).isoformat()
                    job.result = result or ""
                    _log.info("queue cancelled  id=%s after %.1fs", job.id, duration)
                    self._emit_event({"type": "job_cancelled", "job_id": job.id})
                else:
                    job.status = "completed"
                    job.completed_at = datetime.now(timezone.utc).isoformat()
                    job.result = result or ""
                    _log.info("queue completed  id=%s type=%s chars=%d %.1fs", job.id, job.type, len(job.result), duration)
                    self._completion_history.append((job.type, duration))
                    if len(self._completion_history) > 100:
                        self._completion_history = self._completion_history[-50:]
                    self._emit_event({
                        "type": "job_completed",
                        "job_id": job.id,
                        "duration_s": duration,
                    })
            except Exception as e:
                duration = round(time.time() - t0, 1)
                job.status = "failed"
                job.completed_at = datetime.now(timezone.utc).isoformat()
                job.error = str(e)
                _log.error("queue failed  id=%s type=%s error=%s %.1fs", job.id, job.type, e, duration)
                self._emit_event({
                    "type": "job_failed",
                    "job_id": job.id,
                    "error": str(e)[:200],
                })

            self._persist_job(job)
            self._fire_callback(job)

            with self._lock:
                self._current = None

    def _execute_job(self, job: QueueJob) -> str:
        payload = job.payload
        messages = payload.get("messages") or []
        if not messages:
            prompt = payload.get("prompt", "")
            system = payload.get("system", "")
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": "t1_secondary",
            "messages": messages,
            "temperature": float(payload.get("temperature", 0.3)),
            "max_tokens": int(payload.get("max_tokens", 2048)),
            **no_think_params(),
        }

        resp = requests.post(
            f"{self._model_url}/v1/chat/completions",
            json=body,
            timeout=(30, job.timeout_s),
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage") or {}
        _log.info("queue model response  id=%s tokens=%s chars=%d",
                   job.id, usage.get("total_tokens", "?"), len(content))
        return content

    def _fire_callback(self, job: QueueJob):
        if not job.callback_url:
            return
        def _bg():
            try:
                requests.post(
                    job.callback_url,
                    json={
                        "job_id": job.id,
                        "status": job.status,
                        "result": job.result[:5000] if job.result else "",
                        "error": job.error,
                    },
                    timeout=10,
                )
            except Exception:
                _log.warning("queue callback failed  id=%s url=%s", job.id, job.callback_url)
        threading.Thread(target=_bg, daemon=True).start()
