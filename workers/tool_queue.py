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
# skip the gate entirely. Local CPU infra: keep this low (seconds, not minutes)
# so background work runs seamlessly between chat turns.
_DEFAULT_BACKGROUND_CHAT_IDLE_S = 30.0
_DEFAULT_MAX_ATTEMPTS = 1
_DEFAULT_RETRY_BACKOFF_S = 5.0
_DEFAULT_TYPE_STARVATION_OVERRIDE_S = 180.0
_DEFAULT_FAIRNESS_CADENCE_CLAIMS = 2
_DEFAULT_FAIRNESS_SCAN_LIMIT = 200

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


def _priority_yield_max_seconds(job_type: str, my_priority: int) -> float:
    """Max time a worker will keep yielding to higher-priority queued jobs.

    Prevents permanent starvation when a steady p3 stream would otherwise block
    all p4/p5 work forever. A value <= 0 disables the cap.
    """
    try:
        from infra.config import get_feature
        per_type = get_feature("tool_queue", "job_type_priority_yield_max_seconds", None)
        if isinstance(per_type, dict):
            val = per_type.get(job_type)
            if val is not None:
                return float(val)
        # Conservative defaults: keep strict yield for p1-p3, allow p4/p5 to
        # make occasional progress when backlog is continuous.
        raw = get_feature(
            "tool_queue",
            "priority_yield_max_seconds",
            120 if my_priority >= 4 else 0,
        )
        return float(raw)
    except Exception:
        return 120.0 if my_priority >= 4 else 0.0


def _is_interactive_source(source: str) -> bool:
    s = (source or "").strip().lower()
    return s == "chat" or s == "code" or s.startswith("chat_") or s.startswith("code_")


def _bypass_idle(job: "ToolJob") -> bool:
    """Jobs that skip the background chat-idle gate: interactive chat/code jobs,
    user-initiated priority-1-3 jobs tagged with bypass_idle, and everything
    explicitly requesting bypass via payload."""
    if _is_interactive_source(job.source):
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
        self._started_at: float = 0.0
        self._last_served_at_by_type: dict[str, float] = {}
        self._fairness_lock = threading.Lock()
        self._fairness_claim_counter: int = 0
        self._forced_type_once: str = ""

    @staticmethod
    def _type_starvation_override_s() -> float:
        try:
            from infra.config import get_feature

            raw = get_feature(
                "tool_queue",
                "type_starvation_override_seconds",
                _DEFAULT_TYPE_STARVATION_OVERRIDE_S,
            )
            val = float(raw)
            return val if val > 0 else _DEFAULT_TYPE_STARVATION_OVERRIDE_S
        except Exception:
            return _DEFAULT_TYPE_STARVATION_OVERRIDE_S

    @staticmethod
    def _fairness_cadence_claims() -> int:
        try:
            from infra.config import get_feature

            raw = get_feature(
                "tool_queue",
                "fairness_cadence_claims",
                _DEFAULT_FAIRNESS_CADENCE_CLAIMS,
            )
            val = int(raw)
            return val if val > 0 else _DEFAULT_FAIRNESS_CADENCE_CLAIMS
        except Exception:
            return _DEFAULT_FAIRNESS_CADENCE_CLAIMS

    @staticmethod
    def _fairness_scan_limit() -> int:
        try:
            from infra.config import get_feature

            raw = get_feature(
                "tool_queue",
                "fairness_scan_limit",
                _DEFAULT_FAIRNESS_SCAN_LIMIT,
            )
            val = int(raw)
            return max(20, min(1000, val))
        except Exception:
            return _DEFAULT_FAIRNESS_SCAN_LIMIT

    def _mark_type_served(self, job_type: str) -> None:
        with self._fairness_lock:
            self._last_served_at_by_type[job_type] = time.time()

    def _type_starved(self, job_type: str) -> bool:
        threshold = self._type_starvation_override_s()
        if threshold <= 0:
            return False
        with self._fairness_lock:
            last = self._last_served_at_by_type.get(job_type, self._started_at or 0.0)
        if last <= 0:
            return True
        return (time.time() - last) >= threshold

    def _forced_type_for(self, job_type: str) -> bool:
        with self._fairness_lock:
            return self._forced_type_once == job_type

    def _clear_forced_type(self, job_type: str) -> None:
        with self._fairness_lock:
            if self._forced_type_once == job_type:
                self._forced_type_once = ""

    def _pick_stale_queued_type(self) -> str | None:
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                return None
            rows = db._get(NOCODB_TABLE, params={
                "where": "(status,eq,queued)",
                "sort": "CreatedAt",
                "limit": self._fairness_scan_limit(),
            }).get("list", [])
            if not rows:
                return None

            oldest_by_type: dict[str, dict] = {}
            for row in rows:
                jt = str(row.get("type") or "").strip()
                # Only consider currently-registered handlers; unknown legacy
                # types cannot be served and would wedge forced fairness.
                if jt and jt in self._handlers and jt not in oldest_by_type:
                    oldest_by_type[jt] = row
            if not oldest_by_type:
                return None

            now = time.time()
            with self._fairness_lock:
                served = dict(self._last_served_at_by_type)
                started = self._started_at or now

            best_type: str | None = None
            best_score = -1.0
            for jt in oldest_by_type:
                last = served.get(jt, started)
                score = now - last if last > 0 else now
                if score > best_score:
                    best_type = jt
                    best_score = score
            return best_type
        except Exception:
            return None

    def _record_run_and_maybe_arm_fairness(self, run_type: str) -> None:
        self._mark_type_served(run_type)

        cadence = self._fairness_cadence_claims()
        if cadence <= 0:
            return

        with self._fairness_lock:
            self._fairness_claim_counter += 1
            if self._forced_type_once == run_type:
                self._forced_type_once = ""

            if self._forced_type_once:
                return
            due = (self._fairness_claim_counter % cadence) == 0

        if not due:
            return

        forced = self._pick_stale_queued_type()
        if not forced or forced == run_type:
            return

        with self._fairness_lock:
            if self._forced_type_once:
                return
            self._forced_type_once = forced

        ev = self._wake_events.get(forced)
        if not ev:
            self._clear_forced_type(forced)
            return
        ev.set()
        _log.info(
            "queue fairness: armed forced_type=%s after %d claims",
            forced,
            self._fairness_claim_counter,
        )

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

        idle = seconds_since_chat()
        if idle == float("inf"):
            idle = -1  # no chat activity yet

        gate = _background_idle_gate()
        backoff_state = "clear" if (idle < 0 or idle >= gate) else "waiting_for_idle"

        return {
            "counts": counts,
            "workers": workers,
            "backoff": {
                "state": backoff_state,
                "idle_seconds": round(idle, 0),
                "threshold": gate,
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
            return [ToolJob.from_row(r).to_api(verbose=verbose) for r in rows]
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

    def _higher_priority_queued(self, my_priority: int) -> bool:
        """True if any queued job across ALL types has priority strictly higher
        (lower number) than `my_priority`. Used so low-priority workers pause
        while a high-priority job is waiting — the per-type worker model
        otherwise lets a p5 pathfinder_crawl run in parallel with a p3
        research_agent, which is what the user saw as "chaos".

        We fetch the single lowest-priority-number queued row (sort asc, limit 1)
        and compare in Python rather than relying on the `lt` NocoDB operator,
        which isn't used anywhere else in this codebase and is unverified."""
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                return False
            rows = db._get(NOCODB_TABLE, params={
                "where": "(status,eq,queued)",
                "sort": "priority",
                "limit": 1,
            }).get("list", [])
            if not rows:
                return False
            top = int(rows[0].get("priority") or 5)
            return top < my_priority
        except Exception:
            return False

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

            queued_head = self._peek_next_queued(job_type)
            forced_for_me = self._forced_type_for(job_type)
            head_bypass = bool(queued_head and _bypass_idle(queued_head))

            if forced_for_me and queued_head is None:
                self._clear_forced_type(job_type)
                forced_for_me = False

            # Single chat-idle gate: non-bypass jobs wait until chat is quiet.
            if not head_bypass:
                idle = seconds_since_chat()
                gate = _background_idle_gate()
                if idle < gate:
                    continue

            # Cross-type priority yield: if a higher-priority job is queued
            # anywhere, back off so its worker gets first claim.
            my_priority = config.priority_default
            if my_priority > 1 and not head_bypass and not forced_for_me:
                yielded = False
                yield_started = time.time()
                yield_cap_s = _priority_yield_max_seconds(job_type, my_priority)
                while self._higher_priority_queued(my_priority):
                    if queued_head and self._type_starved(job_type):
                        _log.info(
                            "queue %s: fairness override — %s idle too long; claiming queued work",
                            worker_id,
                            job_type,
                        )
                        break
                    if yield_cap_s > 0 and (time.time() - yield_started) >= yield_cap_s:
                        _log.info(
                            "queue %s: fairness override — yielded %.0fs; claiming %s work",
                            worker_id, yield_cap_s, job_type,
                        )
                        break
                    if not yielded:
                        _log.info(
                            "queue %s: yielding — higher-priority job queued (my p=%d)",
                            worker_id, my_priority,
                        )
                        yielded = True
                    if self._stop.wait(timeout=10):
                        return
                if yielded:
                    _log.info("queue %s: resuming — higher-priority queue drained", worker_id)
            elif forced_for_me and queued_head:
                _log.info(
                    "queue %s: cadence fairness — forcing type=%s claim before higher priority",
                    worker_id,
                    job_type,
                )

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
            if self._dispatch_to_huey(job, worker_id):
                self._record_run_and_maybe_arm_fairness(job.type)
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
            from infra.huey_runtime import enqueue_tool_job, is_huey_consumer_running
            if not is_huey_consumer_running():
                return False
            ok = enqueue_tool_job(job.job_id)
            if ok:
                _log.info(
                    "queue %s: DISPATCHED  job=%s  type=%s  org=%d",
                    worker_id, job.job_id[:12], job.type, job.org_id,
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
        job = self.get_job(job_id)
        if not job:
            return {"status": "not_found", "job_id": job_id}
        if job.status != "running":
            return {"status": "skipped", "reason": f"status_{job.status}", "job_id": job_id}
        config = self._handlers.get(job.type)
        if not config:
            job.status = "failed"
            job.error = f"no handler for job type {job.type}"[:500]
            job.completed_at = datetime.now(timezone.utc).isoformat()
            self._persist_update(job)
            return {"status": "failed", "job_id": job_id, "error": job.error}
        return self._execute_job(job, config, threading.current_thread().name)

    def _execute_job(self, job: ToolJob, config: HandlerConfig, worker_id: str) -> dict:
        t0 = time.time()
        max_attempts = _max_attempts_for_job_type(job.type)
        attempts = 0
        last_result: dict = {}
        while attempts < max_attempts:
            attempts += 1
            try:
                from shared.models import model_usage_scope

                with model_usage_scope(org_id=job.org_id, source=f"tool_queue:{job.type}"):
                    result = config.handler(job.payload)
                result = result or {}
                status_val = str((result.get("status") if isinstance(result, dict) else "") or "").lower()
                if status_val in {"failed", "error"}:
                    last_result = result if isinstance(result, dict) else {}
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
                else:
                    job.status = "completed"
                    job.result = result if isinstance(result, dict) else {}
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

        if job.status not in {"completed", "failed"}:
            job.status = "failed"
        if not isinstance(job.result, dict):
            job.result = {}
        job.result.setdefault("attempt_count", attempts)
        job.completed_at = datetime.now(timezone.utc).isoformat()
        elapsed = round(time.time() - t0, 1)

        if job.status == "completed":
            _log.info("queue %s: COMPLETED  job=%s  type=%s  %.1fs attempts=%d",
                      worker_id, job.job_id[:12], job.type, elapsed, attempts)
            self._emit_event({
                "type": "job_completed",
                "job_id": job.job_id,
                "job_type": job.type,
                "duration_s": elapsed,
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
                # Jobs with multiple LLM calls + web scraping need longer stale windows
                # than the 300s default, or they'll be reset mid-flight while the handler
                # is still working.
                job_type = row.get("type") or ""
                _STALE_MULTIPLIERS = {
                    "graph_extract": 4,               # 20m — LLM inference 7-16min
                    "research_planner": research_planner_dynamic_mult,
                    "research_agent": research_agent_dynamic_mult,
                    "scrape_page": 3,                 # 15m — fetch + chunk + embed
                    "pathfinder_extract": 3,          # 15m — fetch + link extract
                    "summarise_page": 3,              # 15m — summariser LLM
                    "extract_relationships": 4,       # 20m — relationship extraction LLM
                    "discover_agent_run": 3,          # 15m — Chroma sample + LLM + SearXNG queries
                }
                timeout = JOB_QUEUE_STALE_TIMEOUT * _STALE_MULTIPLIERS.get(job_type, 1)
                if now - started_ts > timeout:
                    noco_id = row.get("Id")
                    db._patch(NOCODB_TABLE, noco_id, {
                        "Id": noco_id,
                        "status": "queued",
                        "claimed_by": "",
                        "started_at": "",
                    })
                    if job_type:
                        reset_types.add(job_type)
                    _log.warning("reset stale job %s (type=%s, stuck %.0fs, timeout=%.0fs)",
                                 row.get("job_id"), row.get("type"),
                                 now - started_ts, timeout)
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

    def _peek_next_queued(self, job_type: str) -> ToolJob | None:
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
