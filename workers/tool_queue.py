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

from infra.config import JOB_QUEUE_POLL_INTERVAL, JOB_QUEUE_STALE_TIMEOUT

_log = logging.getLogger("tool_queue")

NOCODB_TABLE = "tool_jobs"

# priority tiers gate jobs by seconds-since-last-chat-activity.
# Defaults below are used only if `features.tool_queue` isn't in config.json;
# the config-driven overrides are the source of truth at runtime.
_DEFAULT_PRIORITY_BACKOFF: dict[int, float] = {1: 30, 2: 60, 3: 60, 4: 120, 5: 120}
_DEFAULT_BACKOFF = 300.0

_last_chat_activity: float = 0.0
_activity_lock = threading.Lock()


def touch_chat_activity():
    global _last_chat_activity
    with _activity_lock:
        _last_chat_activity = time.time()


def seconds_since_chat() -> float:
    with _activity_lock:
        if _last_chat_activity == 0:
            return float("inf")
        return time.time() - _last_chat_activity


def _backoff_for_priority(priority: int) -> float:
    # Read config each time so edits to config.json take effect without restart-of-workers
    try:
        from infra.config import get_feature
        per_pri = get_feature("tool_queue", "priority_backoff_seconds", None)
        if isinstance(per_pri, dict):
            val = per_pri.get(str(priority), per_pri.get(priority))
            if val is not None:
                return float(val)
        default = get_feature("tool_queue", "default_backoff_seconds", None)
        if default is not None:
            return float(default)
    except Exception:
        pass
    return _DEFAULT_PRIORITY_BACKOFF.get(priority, _DEFAULT_BACKOFF)


# legacy aliases used elsewhere in this module (kept for compatibility)
_PRIORITY_BACKOFF = _DEFAULT_PRIORITY_BACKOFF


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

    def register(self, job_type: str, config: HandlerConfig):
        self._handlers[job_type] = config
        self._wake_events[job_type] = threading.Event()

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
            idle = -1  # no chat activity yet

        p1 = _backoff_for_priority(1)
        p2 = _backoff_for_priority(2)
        background = _backoff_for_priority(5)
        backoff_state = "active"
        if idle < 0 or idle >= background:
            backoff_state = "clear"
        elif idle >= p2:
            backoff_state = "priority_1_2_only"
        elif idle >= p1:
            backoff_state = "priority_1_only"

        return {
            "counts": counts,
            "workers": workers,
            "backoff": {
                "state": backoff_state,
                "idle_seconds": round(idle, 0),
                "thresholds": {
                    "priority_1": p1,
                    "priority_2": p2,
                    "background": background,
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
        config = self._handlers[job_type]
        wake = self._wake_events[job_type]
        worker_id = threading.current_thread().name
        _log.info("worker %s started", worker_id)

        while not self._stop.is_set():
            wake.wait(timeout=JOB_QUEUE_POLL_INTERVAL)
            wake.clear()

            if self._stop.is_set():
                break

            # scrape is pure I/O so it ignores chat backoff; model-bound jobs must not
            if job_type != "scrape":
                idle = seconds_since_chat()
                # pre-claim gate uses the smallest backoff so p1 isn't blocked here;
                # real priority-specific threshold is enforced post-claim below
                min_gate = _backoff_for_priority(1)
                if idle < min_gate:
                    if int(idle) % 60 < int(JOB_QUEUE_POLL_INTERVAL) + 1:
                        _log.debug("queue %s: chat active — backing off (idle=%.0fs, gate=%.0fs)",
                                   worker_id, idle, min_gate)
                    continue

            job = self._claim_next(job_type, worker_id)
            if not job:
                continue

            # per-job backoff: unclaim and sleep if priority threshold not met
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

            # mirror job-level org_id into payload — handlers read from payload
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
            self._wake_dependents(job.job_id)

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
                # graph_extract runs LLM inference 7-16min so it needs an extended stale timeout
                job_type = row.get("type") or ""
                _STALE_MULTIPLIERS = {"graph_extract": 4}
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

    @staticmethod
    def _db():
        from infra.nocodb_client import NocodbClient
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


_instance: ToolJobQueue | None = None


def get_tool_queue() -> ToolJobQueue | None:
    return _instance


def _set_instance(q: ToolJobQueue):
    global _instance
    _instance = q
