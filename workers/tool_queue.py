"""
General-purpose tool job queue backed by NocoDB.

Manages typed jobs with per-type handler functions and configurable worker
pools.  Scraping runs in parallel (I/O bound), summarisation runs one at a
time (model bound), and future tool types plug in via `register()`.

Jobs are persisted to the ``tool_jobs`` NocoDB table so they survive harness
restarts.  Priority ordering (1 = highest) lets user-facing work jump ahead
of background enrichment.

Job chaining:
  submit_pipeline() creates a scrape job and a dependent summarise job.
  When the scrape job completes the summarise worker merges its result
  into the dependent job's payload and processes it.
"""

from __future__ import annotations

import collections
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from config import JOB_QUEUE_POLL_INTERVAL, JOB_QUEUE_STALE_TIMEOUT

_log = logging.getLogger("tool_queue")

NOCODB_TABLE = "tool_jobs"

# ---------------------------------------------------------------------------
# Chat activity tracking — queue workers yield to active sessions
# ---------------------------------------------------------------------------
# Priority thresholds: a job only runs if enough seconds have elapsed since
# the last chat/code activity for its priority tier.
#   priority 1 (research):     30s  (user-requested, wait for stream to finish)
#   priority 2 (deep search):  120s (user-requested, 2 min backoff for resources)
#   priority 3  (normal):      120s (2 min)
#   priority 4+ (enrichment):  600s (10 min, true background work)
_PRIORITY_BACKOFF: dict[int, float] = {1: 30, 2: 120, 3: 120}
_DEFAULT_BACKOFF = 600.0  # priority 4-5

_last_chat_activity: float = 0.0
_activity_lock = threading.Lock()


def touch_chat_activity():
    """Call from chat/code endpoints to signal an active session."""
    global _last_chat_activity
    with _activity_lock:
        _last_chat_activity = time.time()


def seconds_since_chat() -> float:
    with _activity_lock:
        if _last_chat_activity == 0:
            return float("inf")
        return time.time() - _last_chat_activity


def _backoff_for_priority(priority: int) -> float:
    return _PRIORITY_BACKOFF.get(priority, _DEFAULT_BACKOFF)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class HandlerConfig:
    handler: Callable[[dict], dict]
    max_workers: int = 1
    priority_default: int = 3
    dedup_key: str | None = None


@dataclass
class ToolJob:
    job_id: str
    type: str
    status: str = "queued"
    priority: int = 3
    source: str = ""
    org_id: int = 0
    payload: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    error: str = ""
    claimed_by: str = ""
    started_at: str = ""
    completed_at: str = ""
    depends_on: str = ""
    nocodb_id: int | None = None

    def to_row(self) -> dict:
        return {
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

    def to_api(self) -> dict:
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
        }
        if meta.get("conversation_id"):
            d["conversation_id"] = meta["conversation_id"]
        if meta.get("url"):
            d["url"] = meta["url"]
        if meta.get("title") or meta.get("name"):
            d["title"] = meta.get("title") or meta.get("name")
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
            org_id=int(row.get("org_id") or 0),
            payload=payload,
            result=result,
            error=row.get("error") or "",
            claimed_by=row.get("claimed_by") or "",
            started_at=row.get("started_at") or "",
            completed_at=row.get("completed_at") or "",
            depends_on=row.get("depends_on") or "",
            nocodb_id=row.get("Id"),
        )


# ---------------------------------------------------------------------------
# ToolJobQueue
# ---------------------------------------------------------------------------

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

    # ---- Registration ----

    def register(self, job_type: str, config: HandlerConfig):
        self._handlers[job_type] = config
        self._wake_events[job_type] = threading.Event()

    # ---- Lifecycle ----

    def start(self):
        self._stop.clear()
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

    # ---- Producer API ----

    def submit(
        self,
        job_type: str,
        payload: dict,
        source: str = "",
        org_id: int = 0,
        priority: int | None = None,
        depends_on: str = "",
    ) -> str:
        config = self._handlers.get(job_type)
        if not config:
            raise ValueError(f"unknown job type: {job_type}")

        # Dedup check
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
        self._persist_new(job)
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
                org_id=item.get("org_id", 0),
                priority=item.get("priority"),
                depends_on=item.get("depends_on", ""),
            )
            ids.append(jid)
        return ids

    def submit_pipeline(
        self,
        url: str,
        org_id: int,
        collection: str,
        source: str = "",
        priority: int = 3,
        metadata: dict | None = None,
        summarise_function: str = "deep_search_summarise",
    ) -> list[str]:
        """Submit a scrape→summarise pipeline.  Returns [scrape_id, summarise_id]."""
        meta = metadata or {}
        scrape_id = self.submit(
            job_type="scrape",
            payload={
                "url": url,
                "snippet": meta.get("snippet", ""),
                "title": meta.get("title", ""),
            },
            source=source,
            org_id=org_id,
            priority=priority,
        )
        # Build summarise metadata — pass through ALL upstream metadata
        # so handlers can access scrape_target_id, category, cycle_id, etc.
        summarise_meta = {
            "url": url,
            "name": meta.get("title") or url,
            "source": source or "web_search",
            "queries": ", ".join(meta.get("queries", []))[:500],
            "conversation_id": meta.get("conversation_id"),
            "scrape_target_id": meta.get("scrape_target_id"),
            "category": meta.get("category"),
            "cycle_id": meta.get("cycle_id"),
        }
        summarise_id = self.submit(
            job_type="summarise",
            payload={
                "url": url,
                "query": " ".join(meta.get("queries", [])) if meta.get("queries") else url,
                "org_id": org_id,
                "collection": collection,
                "function_name": summarise_function,
                "metadata": summarise_meta,
            },
            source=source,
            org_id=org_id,
            priority=priority,
            depends_on=scrape_id,
        )
        return [scrape_id, summarise_id]

    # ---- Status API ----

    def status(self) -> dict:
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                return {"error": "table not found"}
            rows = db._get(NOCODB_TABLE, params={"limit": 500}).get("list", [])
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

        idle = seconds_since_chat()
        if idle == float("inf"):
            idle = -1  # no chat activity recorded yet
        backoff_state = "active"
        if idle < 0 or idle >= _DEFAULT_BACKOFF:
            backoff_state = "clear"
        elif idle >= _PRIORITY_BACKOFF.get(2, 300):
            backoff_state = "priority_1_2_only"
        elif idle >= _PRIORITY_BACKOFF.get(1, 120):
            backoff_state = "priority_1_only"

        return {
            "counts": counts,
            "workers": workers,
            "backoff": {
                "state": backoff_state,
                "idle_seconds": round(idle, 0),
                "thresholds": {
                    "research": _PRIORITY_BACKOFF.get(1, 120),
                    "deep_search": _PRIORITY_BACKOFF.get(2, 300),
                    "background": _DEFAULT_BACKOFF,
                },
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

    def list_jobs(self, job_type: str = "", status: str = "", source: str = "", limit: int = 50) -> list[dict]:
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
            params: dict[str, Any] = {
                "sort": "-CreatedAt",
                "limit": limit,
            }
            if where_parts:
                params["where"] = "~and".join(where_parts)
            rows = db._get(NOCODB_TABLE, params=params).get("list", [])
            return [ToolJob.from_row(r).to_api() for r in rows]
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

    # ---- Event stream ----

    def subscribe(self) -> collections.deque:
        """Return a thread-safe deque buffer that receives job events."""
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

    # ---- Worker loop ----

    def _worker_loop(self, job_type: str):
        config = self._handlers[job_type]
        wake = self._wake_events[job_type]
        worker_id = threading.current_thread().name
        _log.info("worker %s started", worker_id)

        while not self._stop.is_set():
            wake.wait(timeout=JOB_QUEUE_POLL_INTERVAL)
            wake.clear()

            if self._stop.is_set():
                break

            # Scrape jobs are I/O-only (no model) — skip backoff for them.
            # Model-bound jobs (summarise etc.) must respect chat activity.
            if job_type != "scrape":
                idle = seconds_since_chat()
                # Use the LOWEST possible backoff (priority 1 = 120s) as
                # the pre-claim gate so high-priority jobs aren't blocked.
                # The per-job check after claiming enforces the real
                # threshold and sleeps instead of churning if too early.
                min_gate = _backoff_for_priority(1)
                if idle < min_gate:
                    # Only log occasionally to avoid spam
                    if int(idle) % 60 < int(JOB_QUEUE_POLL_INTERVAL) + 1:
                        _log.debug("queue %s: chat active — backing off (idle=%.0fs, gate=%.0fs)",
                                   worker_id, idle, min_gate)
                    continue

            job = self._claim_next(job_type, worker_id)
            if not job:
                continue

            # Per-job backoff: enforce the actual threshold for this job's
            # priority.  Priority 1 (research) = 120s, 2 (deep search) =
            # 300s, 3+ (enrichment) = 600s.
            if job_type != "scrape":
                idle = seconds_since_chat()
                required = _backoff_for_priority(job.priority)
                if idle < required:
                    wait_secs = min(required - idle, 60)
                    _log.info("queue %s: job %s (priority=%d) needs %.0fs idle, currently %.0fs — sleeping %.0fs",
                              worker_id, job.job_id[:12], job.priority, required, idle, wait_secs)
                    self._unclaim(job)
                    self._stop.wait(timeout=wait_secs)
                    continue

            # Dependency check
            if job.depends_on:
                dep = self.get_job(job.depends_on)
                if not dep or dep.status != "completed":
                    dep_status = dep.status if dep else "not_found"
                    _log.debug("queue %s: job %s waiting on dependency %s (status=%s)",
                               worker_id, job.job_id[:12], job.depends_on[:12], dep_status)
                    self._unclaim(job)
                    continue
                if dep.result:
                    job.payload.update(dep.result)

            # Ensure org_id is always available in the payload from the
            # authoritative job-level field so handlers never see org=0
            # when the caller set it on submit().
            if job.org_id:
                job.payload["org_id"] = job.org_id
            elif job.payload.get("org_id"):
                job.org_id = int(job.payload["org_id"])

            _log.info("queue %s: RUNNING  job=%s  type=%s  priority=%d  source=%s  org=%d",
                       worker_id, job.job_id[:12], job_type, job.priority, job.source or "-", job.org_id)
            t0 = time.time()

            try:
                result = config.handler(job.payload)
                job.status = "completed"
                job.result = result or {}
                job.completed_at = datetime.now(timezone.utc).isoformat()
                elapsed = round(time.time() - t0, 1)
                _log.info("queue %s: COMPLETED  job=%s  type=%s  %.1fs",
                           worker_id, job.job_id[:12], job_type, elapsed)
                self._emit_event({
                    "type": "job_completed",
                    "job_id": job.job_id,
                    "job_type": job_type,
                    "duration_s": elapsed,
                })
            except Exception as e:
                job.status = "failed"
                job.error = str(e)[:500]
                job.completed_at = datetime.now(timezone.utc).isoformat()
                elapsed = round(time.time() - t0, 1)
                _log.error("queue %s: FAILED  job=%s  type=%s  error=%s  %.1fs",
                           worker_id, job.job_id[:12], job_type, e, elapsed)
                self._emit_event({
                    "type": "job_failed",
                    "job_id": job.job_id,
                    "job_type": job_type,
                    "error": str(e)[:200],
                })

            self._persist_update(job)
            # Wake workers for dependent jobs
            self._wake_dependents(job.job_id)

    # ---- Stale checker ----

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
            rows = db._get(NOCODB_TABLE, params={
                "where": "(status,eq,running)",
                "limit": 100,
            }).get("list", [])
            now = time.time()
            for row in rows:
                started = row.get("started_at") or ""
                if not started:
                    continue
                try:
                    started_ts = datetime.fromisoformat(started).timestamp()
                except Exception:
                    continue
                # Long-running jobs need extended stale timeouts.
                # graph_extract: 7-16 min (LLM inference) → 4x
                # research: up to 60 min (iterative multi-source) → 12x
                job_type = row.get("type") or ""
                _STALE_MULTIPLIERS = {"graph_extract": 4, "deep_search": 8, "research": 20}
                timeout = JOB_QUEUE_STALE_TIMEOUT * _STALE_MULTIPLIERS.get(job_type, 1)
                if now - started_ts > timeout:
                    noco_id = row.get("Id")
                    db._patch(NOCODB_TABLE, noco_id, {
                        "Id": noco_id,
                        "status": "queued",
                        "claimed_by": "",
                        "started_at": "",
                    })
                    _log.warning("reset stale job %s (type=%s, stuck %.0fs, timeout=%.0fs)",
                                 row.get("job_id"), row.get("type"),
                                 now - started_ts, timeout)
        except Exception:
            _log.error("stale job reset failed", exc_info=True)

    # ---- NocoDB operations ----

    @staticmethod
    def _db():
        from nocodb_client import NocodbClient
        return NocodbClient()

    def _persist_new(self, job: ToolJob):
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                _log.warning("table %s not found — job %s not persisted", NOCODB_TABLE, job.job_id)
                return
            row = db._post(NOCODB_TABLE, job.to_row())
            job.nocodb_id = row.get("Id")
        except Exception:
            _log.error("persist_new failed job=%s", job.job_id, exc_info=True)

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
                "sort": "priority,CreatedAt",
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

            # Re-fetch to verify we got the claim
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

    def _unclaim(self, job: ToolJob):
        """Put a job back to queued (dependency not ready)."""
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
        """Check if a job with this dedup key value is already pending/running."""
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                return None

            # For URL dedup, query by type+status and filter payload in Python.
            # NocoDB where-clauses can't reliably match URLs with special chars.
            if key == "url":
                rows = db._get(NOCODB_TABLE, params={
                    "where": f"(type,eq,{job_type})~and(status,in,queued,running)",
                    "limit": 200,
                }).get("list", [])
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

            # For non-URL keys, search payload_json directly is impractical.
            # Future: add a dedup_hash column.
            return None
        except Exception:
            return None

    def _wake_dependents(self, completed_job_id: str):
        """Wake workers for jobs that depend on the completed job."""
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                return
            rows = db._get(NOCODB_TABLE, params={
                "where": f"(depends_on,eq,{completed_job_id})~and(status,eq,queued)",
                "limit": 50,
            }).get("list", [])
            if rows:
                # Wake all relevant worker types
                types_to_wake = {r.get("type") for r in rows}
                for jt in types_to_wake:
                    ev = self._wake_events.get(jt)
                    if ev:
                        ev.set()
                _log.debug("woke dependents for job=%s types=%s", completed_job_id, types_to_wake)
        except Exception:
            _log.error("wake_dependents failed for job=%s", completed_job_id, exc_info=True)

    def _load_pending(self):
        """On startup, reset any running jobs back to queued (crash recovery)."""
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                _log.info("table %s not found — starting with empty queue", NOCODB_TABLE)
                return
            rows = db._get(NOCODB_TABLE, params={
                "where": "(status,eq,running)",
                "limit": 200,
            }).get("list", [])
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


# ---------------------------------------------------------------------------
# Built-in handlers
# ---------------------------------------------------------------------------

def _handle_scrape(payload: dict) -> dict:
    """Scrape a URL using the shared scraping pipeline."""
    from workers.search.scraping import scrape_page

    url = payload.get("url", "")
    snippet = payload.get("snippet", "")

    if not url:
        _log.warning("queue scrape: empty URL — skipped")
        return {"url": url, "text": snippet, "path": "empty_url"}

    _log.info("queue scrape: starting  url=%s", url[:80])
    meta: dict = {}
    try:
        text = scrape_page(url, snippet, None, meta)
    except Exception as e:
        _log.warning("queue scrape: FAILED  url=%s  error=%s", url[:80], e)
        text = snippet
        meta["path"] = "error"

    path = meta.get("path", "unknown")
    text_len = len(text or "")
    _log.info("queue scrape: complete  url=%s  path=%s  chars=%d", url[:80], path, text_len)

    return {
        "url": url,
        "text": (text or "")[:20000],
        "path": path,
    }


def _update_enrichment_target_on_error(metadata: dict, error_msg: str):
    """Mark an enrichment scrape target as errored with exponential backoff."""
    target_id = metadata.get("scrape_target_id")
    source_type = metadata.get("source", "")
    if not target_id or source_type != "enrichment":
        return
    try:
        from datetime import datetime, timedelta, timezone
        from workers.enrichment.db import EnrichmentDB
        db = EnrichmentDB()
        rows = db._get("scrape_targets", params={
            "where": f"(Id,eq,{target_id})", "limit": 1,
        }).get("list", [])
        source_row = rows[0] if rows else {}
        consecutive = int(source_row.get("consecutive_failures") or 0) + 1
        freq_hours = float(source_row.get("frequency_hours") or 24)
        # Exponential backoff: double the wait each failure, cap at 16x
        backoff_mult = min(2 ** consecutive, 16)
        now_utc = datetime.now(timezone.utc)
        retry_at = now_utc + timedelta(hours=freq_hours * backoff_mult)
        db.update_scrape_target(
            int(target_id),
            status="error",
            next_crawl_at=retry_at.isoformat(),
            consecutive_failures=consecutive,
            last_scrape_error=error_msg[:500],
        )
        _log.info("enrichment target error  id=%s consecutive=%d retry=%s", target_id, consecutive, retry_at.isoformat())
    except Exception:
        _log.warning("failed to update scrape target %s on error", target_id, exc_info=True)


def _handle_summarise(payload: dict) -> dict:
    """Summarise text and store in ChromaDB.  Uses the function_name from
    payload to select the right model config (deep_search_summarise,
    enrichment_summarise, research_summarise, etc.)."""
    from config import get_function_config
    from workers.enrichment.models import model_call

    text = payload.get("text", "")
    query = payload.get("query", "")
    url = payload.get("url", "")
    org_id = int(payload.get("org_id") or 0)
    collection = payload.get("collection", "web_search")
    metadata = payload.get("metadata") or {}
    function_name = payload.get("function_name", "deep_search_summarise")

    source_type = metadata.get("source", "")
    scrape_target_id = metadata.get("scrape_target_id", "")
    _log.info("queue summarise: starting  url=%s  source=%s  target=%s  func=%s  text_chars=%d",
              url[:80], source_type, scrape_target_id or "-", function_name, len(text))

    if not text or len(text) < 50:
        _log.info("queue summarise: skipped — text too short (%d chars)  url=%s", len(text), url[:80])
        _update_enrichment_target_on_error(metadata, "scrape returned no usable text")
        return {"summary": text, "chunks": 0}

    cfg = get_function_config(function_name)
    max_input = cfg.get("max_input_chars", 12000)
    max_tokens = cfg.get("max_tokens", 300)

    # Deep/research summarise gets a more thorough prompt
    if "deep" in function_name or "research" in function_name:
        prompt = (
            f"Provide a comprehensive analysis of the following web page. "
            f"Focus on information relevant to: {query}\n\n"
            f"Include:\n"
            f"- Key facts, data points, statistics, dates\n"
            f"- Main arguments or conclusions\n"
            f"- Notable quotes or claims with attribution\n"
            f"- Methodology or evidence quality where applicable\n"
            f"- Any caveats, limitations, or biases\n\n"
            f"URL: {url}\n\n"
            f"Content:\n{text[:max_input]}"
        )
    else:
        prompt = (
            "Summarise the following web page content. Focus ONLY on information "
            f"relevant to: {query}\n\n"
            "Rules:\n"
            "- Keep under 300 words.\n"
            "- Include specific facts, numbers, dates, names.\n"
            "- Skip navigation, boilerplate, cookie notices, unrelated content.\n\n"
            f"URL: {url}\n\n"
            f"Content:\n{text[:max_input]}"
        )

    summary, _tokens = model_call(function_name, prompt)
    if not summary:
        _log.warning("queue summarise: model returned empty — using raw text fallback  url=%s", url[:80])
        summary = text[:2000]

    _log.info("queue summarise: model complete  url=%s  tokens=%d  summary_chars=%d", url[:80], _tokens, len(summary))

    chunks = 0
    if org_id and summary:
        try:
            from memory import remember
            ids = remember(
                text=summary,
                metadata=metadata,
                org_id=org_id,
                collection_name=collection,
            )
            chunks = len(ids or [])
        except Exception as e:
            _log.warning("ChromaDB store failed for %s: %s", url[:80], e)

    _log.info("queue summarise: stored to ChromaDB  url=%s  collection=%s  chunks=%d", url[:80], collection, chunks)

    # Deliver result back to the originating conversation as a message.
    conversation_id = metadata.get("conversation_id")
    if conversation_id and source_type in ("deep_search", "research"):
        _deliver_to_conversation(conversation_id, org_id, url, summary)
        _log.info("queue summarise: delivered to conversation  conv=%s  url=%s", conversation_id, url[:60])

    # Queue graph extraction so relationships land in FalkorDB.
    # Runs at lowest priority (5) when models are idle.
    _graph_sources = ("enrichment", "deep_search", "research")
    if source_type in _graph_sources and summary and org_id:
        try:
            tq = get_tool_queue()
            if tq:
                graph_job_id = tq.submit(
                    job_type="graph_extract",
                    payload={
                        "user_text": f"Source: {url}",
                        "assistant_text": summary,
                        "conversation_id": 0,
                        "org_id": org_id,
                    },
                    source=source_type,
                    org_id=org_id,
                    priority=5,
                )
                _log.info("queue summarise: graph extraction queued  source=%s  job=%s  url=%s", source_type, graph_job_id, url[:60])
        except Exception:
            _log.error("queue summarise: graph extraction queue FAILED  source=%s  url=%s", source_type, url[:60], exc_info=True)

    # Update the scrape target so it's not re-scraped until next_crawl_at.
    if scrape_target_id and source_type == "enrichment":
        try:
            from datetime import datetime, timedelta, timezone
            from workers.enrichment.db import EnrichmentDB
            db = EnrichmentDB()
            source_row = None
            try:
                rows = db._get("scrape_targets", params={
                    "where": f"(Id,eq,{scrape_target_id})", "limit": 1,
                }).get("list", [])
                source_row = rows[0] if rows else None
            except Exception:
                pass
            freq_hours = float((source_row or {}).get("frequency_hours") or 24)
            now_utc = datetime.now(timezone.utc)
            next_at = now_utc + timedelta(hours=freq_hours)
            db.update_scrape_target(
                int(scrape_target_id),
                last_scraped_at=now_utc.isoformat(),
                status="ok",
                next_crawl_at=next_at.isoformat(),
                consecutive_failures=0,
            )
            _log.info("queue summarise: target complete  id=%s  status=ok  next_crawl=%s", scrape_target_id, next_at.isoformat())
        except Exception:
            _log.error("queue summarise: target update FAILED  id=%s", scrape_target_id, exc_info=True)

    return {"summary": summary[:3000], "chunks": chunks}


def _deliver_to_conversation(conversation_id: int, org_id: int, url: str, summary: str):
    """Post a completed deep search result back to the conversation.

    Uses role="assistant" with model="deep_search" so the frontend renders
    it as a visible message (system messages are typically hidden).
    """
    try:
        from nocodb_client import NocodbClient
        db = NocodbClient()
        content = (
            f"[Deep search result]\n"
            f"Source: {url}\n\n"
            f"{summary[:2000]}"
        )
        db.add_message(
            conversation_id=conversation_id,
            org_id=org_id,
            role="assistant",
            content=content,
            model="deep_search",
            tokens_input=0,
            tokens_output=0,
            search_used=True,
            search_status="completed",
            search_confidence="high",
            search_source_count=1,
        )
        _log.info("delivered result to conversation=%s url=%s", conversation_id, url[:60])
    except Exception:
        _log.warning("failed to deliver result to conversation=%s", conversation_id, exc_info=True)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: ToolJobQueue | None = None


def get_tool_queue() -> ToolJobQueue | None:
    return _instance


def _handle_graph_extract(payload: dict) -> dict:
    """Extract knowledge graph relationships from a chat/enrichment turn."""
    from workers.chat.graph import extract_and_write_graph

    user_text = payload.get("user_text") or ""
    assistant_text = payload.get("assistant_text") or ""
    conversation_id = payload.get("conversation_id") or 0
    org_id = int(payload.get("org_id") or 0)

    if not user_text and not assistant_text:
        _log.info("queue graph_extract: skipped — empty input  org=%d", org_id)
        return {"written": 0, "error": "empty input"}

    _log.info("queue graph_extract: starting  org=%d  text_chars=%d",
              org_id, len(user_text) + len(assistant_text))
    extract_and_write_graph(user_text, assistant_text, conversation_id, org_id)
    _log.info("queue graph_extract: complete  org=%d", org_id)
    return {"status": "ok"}


def _handle_deep_search(payload: dict) -> dict:
    """Deep search: gather ~10 usable sources → T1 summarise each → synthesise → deliver.

    Starts with the plan URLs, scrapes them, and if not enough have usable
    content, runs additional SearXNG searches to fill the gap.  Keeps going
    until it has TARGET_SOURCES usable pages or exhausts all queries.
    """
    import asyncio as _aio
    from memory import remember
    from workers.enrichment.models import model_call
    from workers.chat.graph import extract_and_write_graph
    from tools.framework.executors.web_search import _scrape_one, _summarise_one

    plan = payload.get("plan") or {}
    org_id = int(payload.get("org_id") or 0)
    conversation_id = payload.get("conversation_id")

    queries = plan.get("queries", [])
    plan_urls = plan.get("urls", [])
    question = " ".join(queries[:3]) if queries else ""

    if not plan_urls or not org_id:
        _log.info("queue deep_search: skipped — no urls or org  org=%d", org_id)
        return {"status": "skipped", "error": "no urls or org"}

    _log.info("queue deep_search: starting  org=%d  urls=%d  queries=%d",
              org_id, len(plan_urls), len(queries))

    loop = _aio.new_event_loop()
    _aio.set_event_loop(loop)
    all_summaries: list[dict] = []
    with_text: list[dict] = []

    try:
        # 1. Scrape all plan URLs in parallel (return_exceptions so one bad URL doesn't kill the batch)
        try:
            results = loop.run_until_complete(
                _aio.gather(*[_scrape_one(r) for r in plan_urls], return_exceptions=True)
            )
            scraped = []
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    _log.warning("queue deep_search: scrape exception  url=%s  error=%s",
                                 plan_urls[i].get("url", "?")[:60], r)
                else:
                    scraped.append(r)
        except Exception:
            _log.error("queue deep_search: scrape failed", exc_info=True)
            scraped = []

        for s in scraped:
            text_len = len(s.get("text") or "")
            if text_len >= 200:
                with_text.append(s)
            else:
                _log.info("queue deep_search: scrape too short  url=%s  chars=%d  path=%s",
                          s.get("url", "?")[:60], text_len, s.get("path", "?"))

        _log.info("queue deep_search: scraped  total=%d usable=%d", len(scraped), len(with_text))

        if not with_text:
            _log.warning("queue deep_search: no usable sources")
            if conversation_id:
                try:
                    from nocodb_client import NocodbClient
                    db = NocodbClient()
                    db.add_message(
                        conversation_id=int(conversation_id),
                        org_id=org_id,
                        role="assistant",
                        content=(
                            "[Deep search failed]\n\n"
                            "Could not extract usable content from any source. "
                            "Try different search queries or topics."
                        ),
                        model="deep_search",
                        search_status="failed",
                        search_confidence="none",
                    )
                except Exception:
                    _log.error("queue deep_search: failure delivery failed", exc_info=True)
            return {"status": "no_sources"}

        # 2. Summarise each usable source with T1 secondary
        for source in with_text:
            try:
                result = loop.run_until_complete(
                    _summarise_one(source["url"], source["text"], question, "deep_search_summarise", priority=False)
                )
                summary_text = result.get("summary", "")
                if summary_text and len(summary_text) >= 50:
                    all_summaries.append({
                        "url": source["url"],
                        "title": source.get("title", ""),
                        "summary": summary_text,
                    })

                    # Store each source to ChromaDB
                    try:
                        remember(
                            text=summary_text,
                            metadata={
                                "url": source["url"],
                                "type": "deep_search",
                                "conversation_id": conversation_id,
                            },
                            org_id=org_id,
                            collection_name="web_search",
                        )
                    except Exception:
                        _log.warning("queue deep_search: chroma store failed  url=%s", source["url"][:60], exc_info=True)

                    # Graph extraction per source
                    try:
                        extract_and_write_graph(
                            f"Deep search source: {source['url']}",
                            summary_text,
                            conversation_id or 0,
                            org_id,
                        )
                    except Exception:
                        _log.debug("queue deep_search: graph extract failed  url=%s", source["url"][:60], exc_info=True)

                    _log.info("queue deep_search: summarised  url=%s  chars=%d  total=%d",
                              source["url"][:60], len(summary_text), len(all_summaries))
                else:
                    _log.info("queue deep_search: summary too short, dropped  url=%s  chars=%d",
                              source["url"][:60], len(summary_text))
            except Exception:
                _log.warning("queue deep_search: summarise failed  url=%s", source["url"][:60], exc_info=True)

    finally:
        loop.close()

    # 3. Synthesise a coherent response from all summaries
    if not all_summaries:
        _log.warning("queue deep_search: no summaries gathered")
        if conversation_id:
            try:
                from nocodb_client import NocodbClient
                db = NocodbClient()
                db.add_message(
                    conversation_id=int(conversation_id),
                    org_id=org_id,
                    role="assistant",
                    content=(
                        "[Deep search failed]\n\n"
                        "Sources were found but none could be summarised successfully. "
                        "Try different search queries or topics."
                    ),
                    model="deep_search",
                    search_status="failed",
                    search_confidence="none",
                )
            except Exception:
                _log.error("queue deep_search: no-summaries delivery failed", exc_info=True)
        return {"status": "no_summaries"}

    evidence_block = ""
    for i, s in enumerate(all_summaries, 1):
        evidence_block += f"\n[{i}] {s['url']}\nTitle: {s['title']}\n{s['summary']}\n"

    synth_prompt = (
        f"You are synthesising web research results into a clear, comprehensive answer.\n\n"
        f"SEARCH QUERIES: {', '.join(queries[:5])}\n\n"
        f"Below are {len(all_summaries)} sources analysed in depth. "
        f"Synthesise them into a well-structured response that:\n"
        f"- Directly addresses the search queries\n"
        f"- Cites sources by number [1], [2] etc.\n"
        f"- Highlights key facts, data points, and conclusions\n"
        f"- Notes any contradictions or gaps between sources\n"
        f"- Distinguishes established facts from opinions\n\n"
        f"SOURCES:\n{evidence_block[:18000]}"
    )

    _log.info("queue deep_search: synthesising  sources=%d", len(all_summaries))
    report, synth_tokens = model_call("deep_search_synthesise", synth_prompt)
    _log.info("queue deep_search: synthesis complete  tokens=%d  chars=%d", synth_tokens, len(report or ""))

    if not report:
        report = "Deep search synthesis failed. Individual source summaries were stored but could not be compiled."

    # Store synthesis to ChromaDB
    try:
        remember(
            text=f"DEEP SEARCH: {', '.join(queries[:3])}\n\n{report}",
            metadata={
                "type": "deep_search_report",
                "queries": ", ".join(queries[:5]),
                "sources": len(all_summaries),
                "conversation_id": conversation_id,
            },
            org_id=org_id,
            collection_name="web_search",
        )
    except Exception:
        _log.error("queue deep_search: synthesis chroma store failed", exc_info=True)

    # Graph extraction on synthesis
    try:
        extract_and_write_graph(
            f"Deep search: {', '.join(queries[:3])}",
            report[:8000],
            conversation_id or 0,
            org_id,
        )
    except Exception:
        _log.debug("queue deep_search: synthesis graph extract failed", exc_info=True)

    # Deliver to conversation — put sources before report so truncation
    # cuts the report body, not the source URL list.
    if conversation_id:
        try:
            from nocodb_client import NocodbClient
            db = NocodbClient()
            source_urls = "\n".join(f"[{i+1}] {s['url']}" for i, s in enumerate(all_summaries))
            content = (
                f"[Deep search complete]\n"
                f"Sources analysed: {len(all_summaries)}\n\n"
                f"Sources:\n{source_urls}\n\n"
                f"{report}"
            )
            db.add_message(
                conversation_id=int(conversation_id),
                org_id=org_id,
                role="assistant",
                content=content[:16000],
                model="deep_search",
                tokens_input=0,
                tokens_output=synth_tokens,
                search_used=True,
                search_status="completed",
                search_confidence="high",
                search_source_count=len(all_summaries),
            )
            _log.info("queue deep_search: delivered to conversation=%s  sources=%d", conversation_id, len(all_summaries))
        except Exception:
            _log.error("queue deep_search: delivery failed  conv=%s", conversation_id, exc_info=True)

    _log.info("queue deep_search: complete  sources=%d  synth_tokens=%d", len(all_summaries), synth_tokens)
    return {"status": "ok", "sources": len(all_summaries), "report_chars": len(report)}


def _handle_research(payload: dict) -> dict:
    """Run iterative research: search → scrape → summarise → assess → refine → repeat.

    Runs inside a tool queue worker thread.  All async work is batched into
    a single asyncio event loop per phase to avoid the fragility of repeated
    asyncio.run() calls.
    """
    import asyncio as _aio
    from memory import remember
    from workers.enrichment.models import model_call
    from workers.chat.graph import extract_and_write_graph
    from tools.framework.executors.web_search import _scrape_one, _summarise_one
    from tools.framework.executors.research import (
        _assess_progress, MAX_ITERATIONS, MAX_SOURCES_PER_ITERATION, MAX_TOTAL_SOURCES,
        _SYNTHESISE_PROMPT,
    )

    plan = payload.get("plan") or {}
    org_id = int(payload.get("org_id") or 0)
    conversation_id = payload.get("conversation_id")

    question = plan.get("question", "")
    objective = plan.get("objective", "")
    queries = list(plan.get("queries", []))
    lookout = plan.get("lookout", [])
    criteria = plan.get("completion_criteria", [])

    if not queries or not org_id:
        _log.info("queue research: skipped — no queries or org  org=%d", org_id)
        return {"status": "skipped", "error": "no queries or org"}

    _log.info("queue research: starting  org=%d  question=%s  queries=%d",
              org_id, question[:60], len(queries))

    # Single event loop for all async work in this handler.
    loop = _aio.new_event_loop()
    _aio.set_event_loop(loop)

    all_summaries: list[dict] = []  # {url, title, summary}
    iteration = 0

    try:
        while iteration < MAX_ITERATIONS and queries:
            iteration += 1
            _log.info("queue research: iteration %d/%d  queries=%d  total_sources=%d",
                      iteration, MAX_ITERATIONS, len(queries), len(all_summaries))

            # 1. Search — gather more URLs than _search_all's cap of 5
            try:
                from workers.search.urls import _is_blocklisted
                from tools.framework.executors.web_search import _search_one
                all_raw = loop.run_until_complete(
                    _aio.gather(*[_search_one(q) for q in queries])
                )
                seen_search: set[str] = {s["url"] for s in all_summaries}
                results = []
                for result_set in all_raw:
                    if isinstance(result_set, Exception):
                        continue
                    for r in result_set:
                        url = (r.get("url") or "").strip()
                        if not url or url in seen_search or _is_blocklisted(url):
                            continue
                        seen_search.add(url)
                        results.append(r)
            except Exception:
                _log.error("queue research: search failed iteration=%d", iteration, exc_info=True)
                break

            new_results = results[:MAX_SOURCES_PER_ITERATION]

            if not new_results:
                _log.info("queue research: no new URLs found in iteration %d — stopping", iteration)
                break

            _log.info("queue research: iteration %d  new_urls=%d", iteration, len(new_results))

            # 2. Scrape (parallel, return_exceptions so one bad URL doesn't kill the batch)
            try:
                raw_scraped = loop.run_until_complete(
                    _aio.gather(*[_scrape_one(r) for r in new_results], return_exceptions=True)
                )
                scraped = []
                for i, r in enumerate(raw_scraped):
                    if isinstance(r, Exception):
                        _log.warning("queue research: scrape exception  url=%s  error=%s",
                                     new_results[i].get("url", "?")[:60], r)
                    else:
                        scraped.append(r)
            except Exception:
                _log.error("queue research: scrape failed iteration=%d", iteration, exc_info=True)
                scraped = []

            with_text = []
            for s in scraped:
                text_len = len(s.get("text") or "")
                if text_len >= 200:
                    with_text.append(s)
                else:
                    _log.info("queue research: scrape too short  url=%s  chars=%d  path=%s",
                              s.get("url", "?")[:60], text_len, s.get("path", "?"))

            if not with_text:
                _log.info("queue research: all sources empty in iteration %d", iteration)
                if iteration < MAX_ITERATIONS:
                    queries = queries[len(queries)//2:]
                    continue
                break

            # 3. Summarise each source with T1 secondary
            for source in with_text:
                if len(all_summaries) >= MAX_TOTAL_SOURCES:
                    break
                try:
                    result = loop.run_until_complete(
                        _summarise_one(source["url"], source["text"], question, "research_summarise", priority=False)
                    )
                    summary_text = result.get("summary", "")
                    if summary_text and len(summary_text) >= 50:
                        entry = {
                            "url": source["url"],
                            "title": source.get("title", ""),
                            "summary": summary_text,
                        }
                        all_summaries.append(entry)

                        try:
                            remember(
                                text=summary_text,
                                metadata={
                                    "url": source["url"],
                                    "type": "research",
                                    "question": question[:500],
                                    "conversation_id": conversation_id,
                                },
                                org_id=org_id,
                                collection_name="research",
                            )
                        except Exception:
                            _log.warning("queue research: chroma store failed  url=%s", source["url"][:60], exc_info=True)

                        try:
                            extract_and_write_graph(
                                f"Research source: {source['url']}",
                                summary_text,
                                conversation_id or 0,
                                org_id,
                            )
                        except Exception:
                            _log.debug("queue research: graph extract failed  url=%s", source["url"][:60], exc_info=True)

                        _log.info("queue research: summarised  url=%s  chars=%d  total=%d",
                                  source["url"][:60], len(summary_text), len(all_summaries))
                    else:
                        _log.info("queue research: summary too short, dropped  url=%s  chars=%d",
                                  source["url"][:60], len(summary_text))
                except Exception:
                    _log.warning("queue research: summarise failed  url=%s", source["url"][:60], exc_info=True)

            # 4. Assess progress
            if iteration < MAX_ITERATIONS and all_summaries:
                try:
                    assessment = _assess_progress(objective, lookout, criteria, all_summaries)
                    _log.info("queue research: assessment  complete=%s  gaps=%s  new_queries=%d",
                              assessment.get("complete"), assessment.get("gaps", []),
                              len(assessment.get("new_queries", [])))

                    # Strict boolean check — string "true" from T3 won't stop iteration
                    if assessment.get("complete") is True:
                        _log.info("queue research: completion criteria met — stopping iteration")
                        break

                    new_queries = assessment.get("new_queries", [])
                    if new_queries:
                        # Filter to actual strings only — T3 can return nulls, ints, bools
                        queries = [q.strip() for q in new_queries if isinstance(q, str) and q.strip()][:6]
                        if not queries:
                            _log.info("queue research: new_queries were all invalid — stopping iteration")
                            break
                    else:
                        _log.info("queue research: no new queries suggested — stopping iteration")
                        break
                except Exception:
                    _log.warning("queue research: assessment failed — continuing with existing queries", exc_info=True)

    finally:
        loop.close()

    # 5. Synthesise final report with T1 secondary
    if not all_summaries:
        _log.warning("queue research: no summaries gathered — cannot synthesise")
        if conversation_id:
            try:
                from nocodb_client import NocodbClient
                db = NocodbClient()
                db.add_message(
                    conversation_id=int(conversation_id),
                    org_id=org_id,
                    role="assistant",
                    content=(
                        f"[Research failed]\n"
                        f"Question: {question}\n\n"
                        f"Could not find sufficient sources after {iteration} round(s). "
                        f"Try rephrasing the question or broadening the scope."
                    ),
                    model="research",
                    tokens_input=0, tokens_output=0,
                    search_used=True, search_status="failed",
                    search_confidence="none", search_source_count=0,
                )
            except Exception:
                _log.error("queue research: failure delivery failed  conv=%s", conversation_id, exc_info=True)
        return {"status": "no_sources", "iterations": iteration}

    evidence_block = ""
    for i, s in enumerate(all_summaries, 1):
        evidence_block += f"\n[{i}] {s['url']}\nTitle: {s['title']}\n{s['summary']}\n"

    synth_prompt = _SYNTHESISE_PROMPT.format(
        question=question,
        objective=objective,
        source_count=len(all_summaries),
        iterations=iteration,
        evidence_block=evidence_block[:21000],
    )

    _log.info("queue research: synthesising  sources=%d  iterations=%d", len(all_summaries), iteration)

    report, synth_tokens = model_call("research_synthesise", synth_prompt)
    _log.info("queue research: synthesis complete  tokens=%d  chars=%d", synth_tokens, len(report or ""))

    if not report:
        report = "Research synthesis failed. Raw evidence was gathered but could not be compiled into a report."

    # Store final report to ChromaDB
    try:
        remember(
            text=f"RESEARCH REPORT: {question}\n\n{report}",
            metadata={
                "type": "research_report",
                "question": question[:500],
                "sources": len(all_summaries),
                "iterations": iteration,
                "conversation_id": conversation_id,
            },
            org_id=org_id,
            collection_name="research",
        )
    except Exception:
        _log.error("queue research: final report chroma store failed", exc_info=True)

    # Graph extraction on the final report
    try:
        extract_and_write_graph(
            f"Research question: {question}",
            report[:8000],
            conversation_id or 0,
            org_id,
        )
    except Exception:
        _log.debug("queue research: final report graph extract failed", exc_info=True)

    # Deliver to conversation
    if conversation_id:
        try:
            from nocodb_client import NocodbClient
            db = NocodbClient()
            source_urls = "\n".join(f"[{i+1}] {s['url']}" for i, s in enumerate(all_summaries))
            content = (
                f"[Research complete]\n"
                f"Question: {question}\n"
                f"Sources analysed: {len(all_summaries)} across {iteration} round(s)\n\n"
                f"Sources:\n{source_urls}\n\n"
                f"{report}"
            )
            db.add_message(
                conversation_id=int(conversation_id),
                org_id=org_id,
                role="assistant",
                content=content[:16000],
                model="research",
                tokens_input=0,
                tokens_output=synth_tokens,
                search_used=True,
                search_status="completed",
                search_confidence="high",
                search_source_count=len(all_summaries),
            )
            _log.info("queue research: delivered to conversation=%s  sources=%d", conversation_id, len(all_summaries))
        except Exception:
            _log.error("queue research: delivery failed  conv=%s", conversation_id, exc_info=True)

    _log.info("queue research: complete  question=%s  sources=%d  iterations=%d",
              question[:60], len(all_summaries), iteration)

    return {
        "status": "ok",
        "sources": len(all_summaries),
        "iterations": iteration,
        "report_chars": len(report),
    }


def _set_instance(q: ToolJobQueue):
    global _instance
    _instance = q
