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
            "result_json": json.dumps(self.result) if self.result else "",
            "error": self.error,
            "claimed_by": self.claimed_by,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "depends_on": self.depends_on,
        }

    def to_api(self) -> dict:
        return {
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
        summarise_id = self.submit(
            job_type="summarise",
            payload={
                "url": url,
                "query": " ".join(meta.get("queries", [])) if meta.get("queries") else url,
                "org_id": org_id,
                "collection": collection,
                "metadata": {
                    "url": url,
                    "name": meta.get("title") or url,
                    "source": "web_search",
                    "queries": ", ".join(meta.get("queries", []))[:500],
                },
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

        return {"counts": counts, "workers": workers}

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

    def list_jobs(self, job_type: str = "", status: str = "", limit: int = 50) -> list[dict]:
        try:
            db = self._db()
            if NOCODB_TABLE not in db.tables:
                return []
            where_parts: list[str] = []
            if job_type:
                where_parts.append(f"(type,eq,{job_type})")
            if status:
                where_parts.append(f"(status,eq,{status})")
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

            job = self._claim_next(job_type, worker_id)
            if not job:
                continue

            # Dependency check
            if job.depends_on:
                dep = self.get_job(job.depends_on)
                if not dep or dep.status != "completed":
                    # Dependency not ready — put back
                    self._unclaim(job)
                    continue
                # Merge dependency result into payload
                if dep.result:
                    job.payload.update(dep.result)

            _log.info("worker %s running job=%s type=%s priority=%d",
                       worker_id, job.job_id, job_type, job.priority)
            t0 = time.time()

            try:
                result = config.handler(job.payload)
                job.status = "completed"
                job.result = result or {}
                job.completed_at = datetime.now(timezone.utc).isoformat()
                elapsed = round(time.time() - t0, 1)
                _log.info("worker %s completed job=%s type=%s %.1fs",
                           worker_id, job.job_id, job_type, elapsed)
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
                _log.error("worker %s failed job=%s type=%s error=%s %.1fs",
                           worker_id, job.job_id, job_type, e, elapsed)
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
                if now - started_ts > JOB_QUEUE_STALE_TIMEOUT:
                    noco_id = row.get("Id")
                    db._patch(NOCODB_TABLE, noco_id, {
                        "Id": noco_id,
                        "status": "queued",
                        "claimed_by": "",
                        "started_at": "",
                    })
                    _log.warning("reset stale job %s (type=%s, stuck %.0fs)",
                                 row.get("job_id"), row.get("type"),
                                 now - started_ts)
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
        return {"url": url, "text": snippet, "path": "empty_url"}

    meta: dict = {}
    try:
        text = scrape_page(url, snippet, None, meta)
    except Exception as e:
        _log.warning("scrape handler failed for %s: %s", url[:80], e)
        text = snippet
        meta["path"] = "error"

    return {
        "url": url,
        "text": (text or "")[:20000],
        "path": meta.get("path", "unknown"),
    }


def _handle_summarise(payload: dict) -> dict:
    """Summarise text via RWKV and store in ChromaDB."""
    from config import get_function_config
    from workers.enrichment.models import model_call

    text = payload.get("text", "")
    query = payload.get("query", "")
    url = payload.get("url", "")
    org_id = int(payload.get("org_id") or 0)
    collection = payload.get("collection", "web_search")
    metadata = payload.get("metadata") or {}

    if not text or len(text) < 50:
        _log.debug("summarise handler: text too short for %s", url[:80])
        return {"summary": text, "chunks": 0}

    cfg = get_function_config("search_summarise")
    max_input = cfg.get("max_input_chars", 12000)

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

    summary, _tokens = model_call("search_summarise", prompt)
    if not summary:
        summary = text[:1500]

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

    return {"summary": summary[:1500], "chunks": chunks}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: ToolJobQueue | None = None


def get_tool_queue() -> ToolJobQueue | None:
    return _instance


def _set_instance(q: ToolJobQueue):
    global _instance
    _instance = q
