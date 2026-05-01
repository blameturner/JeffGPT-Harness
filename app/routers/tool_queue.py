import json
import logging
import threading
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from infra.config import HUEY_CONSUMER_WORKERS, HUEY_ENABLED, HUEY_SQLITE_PATH
from infra.huey_runtime import get_huey, get_huey_health, is_huey_consumer_running
from workers.tool_queue import NOCODB_TABLE

_log = logging.getLogger("main.tool_queue")

router = APIRouter(prefix="/tool-queue")


def _get_queue(request: Request):
    q = getattr(request.app.state, "tool_queue", None)
    if q is None:
        raise HTTPException(status_code=503, detail="Tool queue not initialised")
    return q


def _huey_status() -> dict:
    h = get_huey()
    pending: int | str = "?"
    scheduled: int | str = "?"
    if h is not None:
        try:
            pending = h.pending_count()
        except Exception:
            pending = "error"
        try:
            scheduled = h.scheduled_count()
        except Exception:
            scheduled = "error"
    out = {
        "enabled": bool(HUEY_ENABLED),
        "consumer_running": is_huey_consumer_running(),
        "workers": int(HUEY_CONSUMER_WORKERS or 1),
        "sqlite_path": HUEY_SQLITE_PATH,
        "queue_ready": bool(h is not None),
        "pending_count": pending,
        "scheduled_count": scheduled,
    }
    try:
        out["health"] = get_huey_health()
    except Exception:
        out["health"] = {"error": "health probe unavailable"}
    return out


def _scheduler_status(request: Request) -> dict:
    sched = getattr(request.app.state, "scheduler", None)
    running = bool(sched and sched.running)
    agent_jobs: list[dict] = []
    enrichment_jobs: list[dict] = []
    if sched:
        for job in sched.get_jobs():
            payload = {
                "id": job.id,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            }
            if job.id.startswith("agent_schedule_"):
                agent_jobs.append(payload)
            elif job.id in {
                "enrichment_scrape_dispatcher",
                "pathfinder_dispatcher",
                "discover_agent_dispatcher",
            }:
                enrichment_jobs.append(payload)
    next_run = None
    for ej in agent_jobs:
        if ej["next_run"] and (next_run is None or ej["next_run"] < next_run):
            next_run = ej["next_run"]
    next_enrichment_run = None
    for ej in enrichment_jobs:
        if ej["next_run"] and (next_enrichment_run is None or ej["next_run"] < next_enrichment_run):
            next_enrichment_run = ej["next_run"]
    return {
        "running": running,
        "next_run": next_run,
        "next_enrichment_run": next_enrichment_run,
        "agent_schedules": agent_jobs,
        "enrichment_schedules": enrichment_jobs,
    }


@router.get("/status")
def status(request: Request):
    q = _get_queue(request)
    out = q.status()
    out["huey"] = _huey_status()
    return out


@router.get("/runtime")
def runtime_status(request: Request):
    _get_queue(request)
    return {
        "tool_queue_ready": True,
        "huey": _huey_status(),
    }


# Per-(org_id, limit) TTL cache for the heavy dashboard endpoint. The
# Console polls this from multiple panels every 4 seconds; without the
# cache simultaneous polls all hit NocoDB. 1.5 s feels live and collapses
# bursts to one read.
_DASHBOARD_CACHE_TTL_S = 1.5
_dashboard_cache: dict[tuple, tuple[float, dict]] = {}
_dashboard_cache_lock = threading.Lock()


@router.get("/dashboard")
def dashboard(
    request: Request,
    org_id: int | None = None,
    limit: int = 20,
):
    q = _get_queue(request)
    limit = min(max(1, limit), 100)
    cache_key = (org_id if org_id is not None else 0, limit)
    now = time.time()
    with _dashboard_cache_lock:
        hit = _dashboard_cache.get(cache_key)
        if hit and (now - hit[0]) < _DASHBOARD_CACHE_TTL_S:
            return hit[1]

    recent_jobs = q.list_jobs(limit=limit, org_id=org_id, verbose=True)
    active_jobs = [j for j in recent_jobs if j["status"] in ("queued", "running")]
    payload = {
        "queue": q.status(),
        "runtime": {
            "tool_queue_ready": True,
            "huey": _huey_status(),
        },
        "controls": {
            "restart_background": {
                "endpoint": "/tool-queue/restart-background",
                "confirm_prompt": (
                    "Start background processing now? This bypasses the "
                    "30-minute interactive backoff and queued jobs may begin immediately."
                ),
            }
        },
        "scheduler": _scheduler_status(request),
        "recent_jobs": recent_jobs,
        "active_summary": {
            "active": len(active_jobs),
            "queued": sum(1 for j in active_jobs if j["status"] == "queued"),
            "running": sum(1 for j in active_jobs if j["status"] == "running"),
            "org_id": org_id,
        },
    }
    with _dashboard_cache_lock:
        _dashboard_cache[cache_key] = (now, payload)
        if len(_dashboard_cache) > 16:
            oldest = sorted(_dashboard_cache.items(), key=lambda kv: kv[1][0])[:4]
            for k, _ in oldest:
                _dashboard_cache.pop(k, None)
    return payload


@router.get("/jobs")
def list_jobs(
    request: Request,
    type: str = "",
    status: str = "",
    source: str = "",
    limit: int = 50,
    org_id: int | None = None,
    verbose: bool = False,
):
    q = _get_queue(request)
    return {
        "jobs": q.list_jobs(
            job_type=type,
            status=status,
            source=source,
            limit=limit,
            org_id=org_id,
            verbose=verbose,
        )
    }


@router.get("/active")
def active_jobs(
    request: Request,
    conversation_id: int | None = None,
    source: str = "",
    org_id: int | None = None,
):
    q = _get_queue(request)
    jobs = q.list_jobs(source=source, limit=200, org_id=org_id, verbose=False)
    active = [
        j for j in jobs
        if j["status"] in ("queued", "running")
        and (
            conversation_id is None
            or j.get("conversation_id") == conversation_id
        )
    ]
    return {
        "active": len(active),
        "queued": sum(1 for j in active if j["status"] == "queued"),
        "running": sum(1 for j in active if j["status"] == "running"),
        "conversation_id": conversation_id,
        "source": source or None,
        "org_id": org_id,
    }


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    q = _get_queue(request)
    job = q.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_api(verbose=True)


class PriorityUpdate(BaseModel):
    priority: int


@router.patch("/jobs/{job_id}/priority")
def update_priority(job_id: str, body: PriorityUpdate, request: Request):
    q = _get_queue(request)
    job = q.get_job(job_id)
    if not job or job.status != "queued":
        raise HTTPException(status_code=404, detail="Job not found or not queued")
    try:
        db = q._db()
        if job.nocodb_id:
            db._patch(NOCODB_TABLE, job.nocodb_id, {
                "Id": job.nocodb_id,
                "priority": max(1, min(5, body.priority)),
            })
            return {"updated": True, "priority": body.priority}
    except Exception:
        pass
    raise HTTPException(status_code=500, detail="Failed to update priority")


@router.post("/jobs/{job_id}/run-now")
def run_job_now(job_id: str, request: Request):
    """Force a queued job to run immediately: bumps priority to 1, flips
    ``bypass_idle`` on the payload so chat-idle backoff doesn't hold it,
    and wakes the worker thread for its type.

    Idempotent-ish: re-pressing on a job that's still queued just refreshes
    the flags. Fails 409 if the job is running/completed/failed/cancelled —
    use ``/retry`` for those.
    """
    q = _get_queue(request)
    job = q.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "queued":
        raise HTTPException(
            status_code=409,
            detail=f"Job is {job.status} — use /retry to re-queue it",
        )
    try:
        db = q._db()
        payload = dict(job.payload or {})
        payload["bypass_idle"] = True
        patch: dict = {
            "Id": job.nocodb_id,
            "priority": 1,
            "payload_json": json.dumps(payload),
        }
        if job.nocodb_id:
            db._patch(NOCODB_TABLE, job.nocodb_id, patch)
        # Wake the worker thread for this type so it claims the row without
        # waiting for the next poll.
        ev = q._wake_events.get(job.type)
        if ev is not None:
            ev.set()
        return {
            "status": "dispatched",
            "job_id": job_id,
            "type": job.type,
            "priority": 1,
            "bypass_idle": True,
        }
    except Exception as e:
        _log.warning("run_job_now failed  job=%s", job_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"run-now failed: {e}")


@router.post("/jobs/{job_id}/retry")
def retry_job(job_id: str, request: Request):
    q = _get_queue(request)
    job = q.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="Job is already active")
    try:
        new_job_id = q.submit(
            job.type,
            dict(job.payload or {}),
            source=f"{job.source or job.type}_retry",
            org_id=job.org_id,
            priority=job.priority,
            depends_on=job.depends_on,
        )
        return {
            "status": "queued",
            "previous_job_id": job_id,
            "job_id": new_job_id,
            "type": job.type,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retry failed: {e}")


@router.delete("/jobs/{job_id}")
def cancel_job(job_id: str, request: Request):
    q = _get_queue(request)
    ok = q.cancel(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found or not cancellable")
    return {"cancelled": True}


class BulkAction(BaseModel):
    job_ids: list[str]
    action: str  # "cancel" | "retry" | "set_priority" | "tag" | "untag"
    priority: int | None = None
    tags: list[str] | None = None
    reason: str | None = None


@router.post("/bulk")
def bulk_action(body: BulkAction, request: Request):
    """Apply ``action`` to every ``job_id`` in one call. Returns per-id
    outcome so the UI can show partial-success cleanly."""
    q = _get_queue(request)
    out: list[dict] = []
    action = (body.action or "").strip().lower()
    if action not in {"cancel", "retry", "set_priority", "tag", "untag"}:
        raise HTTPException(status_code=400, detail=f"unknown action: {body.action}")
    for jid in body.job_ids:
        try:
            if action == "cancel":
                # cancel_running covers both queued and running
                ok = q.cancel_running(jid, reason=body.reason or "bulk cancel")
                out.append({"job_id": jid, "ok": ok})
            elif action == "retry":
                job = q.get_job(jid)
                if not job:
                    out.append({"job_id": jid, "ok": False, "error": "not found"})
                    continue
                if job.status in {"queued", "running"}:
                    out.append({"job_id": jid, "ok": False, "error": f"job is {job.status}"})
                    continue
                new_id = q.submit(
                    job.type, dict(job.payload or {}),
                    source=f"{job.source or job.type}_bulkretry",
                    org_id=job.org_id, priority=job.priority,
                    depends_on=job.depends_on,
                )
                out.append({"job_id": jid, "ok": True, "new_job_id": new_id})
            elif action == "set_priority":
                if body.priority is None:
                    out.append({"job_id": jid, "ok": False, "error": "priority required"})
                    continue
                ok = q.set_priority(jid, int(body.priority))
                out.append({"job_id": jid, "ok": ok})
            elif action == "tag" or action == "untag":
                ok = q.update_tags(jid, add=body.tags or [] if action == "tag" else [],
                                   remove=body.tags or [] if action == "untag" else [])
                out.append({"job_id": jid, "ok": ok})
        except Exception as e:
            out.append({"job_id": jid, "ok": False, "error": str(e)[:200]})
    return {"results": out}


class ClearQueueRequest(BaseModel):
    # Empty body = "clear queued only". Set to True to also cooperatively
    # cancel everything currently running.
    include_running: bool = False
    job_type: str | None = None  # narrow to one type
    org_id: int | None = None
    reason: str = "queue cleared from console"


@router.post("/clear")
def clear_queue(body: ClearQueueRequest, request: Request):
    """Clear queued jobs (and optionally running). Used as the queue's
    "stop / clear all" button. Running jobs are cancelled cooperatively —
    in-flight LLM calls finish, but no new phase work begins.
    Returns counts so the UI can confirm what was cleared."""
    q = _get_queue(request)
    cancelled_queued = 0
    cancelled_running = 0
    statuses = ["queued"]
    if body.include_running:
        statuses.append("running")
    for status in statuses:
        try:
            jobs = q.list_jobs(
                job_type=body.job_type or "",
                status=status,
                limit=500,
                org_id=body.org_id,
            )
        except Exception:
            jobs = []
        for j in jobs:
            try:
                if status == "queued":
                    if q.cancel(j["job_id"]):
                        cancelled_queued += 1
                else:
                    if q.cancel_running(j["job_id"], reason=body.reason):
                        cancelled_running += 1
            except Exception:
                continue
    return {
        "cancelled_queued": cancelled_queued,
        "cancelled_running": cancelled_running,
        "scope": {"job_type": body.job_type, "org_id": body.org_id},
    }


class StopAllRequest(BaseModel):
    pause: bool = True


class RestartBackgroundRequest(BaseModel):
    # Explicit confirmation guard so a UI click can present a safety prompt
    # before bypassing the 30-minute idle gate.
    confirm: bool = False
    reason: str = "manual background restart from queue UI"


@router.post("/stop-all")
def stop_all(body: StopAllRequest, request: Request):
    """Pause every registered job type at once. Workers stop claiming new
    jobs; in-flight work continues. Pair with ``/clear`` to also cancel
    running jobs. Set ``pause=False`` to resume everything."""
    q = _get_queue(request)
    types = list(q._handlers.keys())
    for t in types:
        q.set_type_paused(t, bool(body.pause))
    return {
        "paused" if body.pause else "resumed": types,
        "count": len(types),
    }


@router.post("/restart-background")
def restart_background(body: RestartBackgroundRequest, request: Request):
    """Manually bypass the interactive idle backoff and wake workers now.

    Requires an explicit confirm flag so UIs can gate this behind a prompt.
    """
    _get_queue(request)
    confirm_prompt = (
        "Start background processing now? This bypasses the 30-minute "
        "interactive backoff and queued jobs may begin immediately."
    )
    if not body.confirm:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "confirmation_required",
                "confirm_prompt": confirm_prompt,
            },
        )

    from workers.tool_queue import force_background_ready

    result = force_background_ready(reason=(body.reason or "manual restart").strip() or "manual restart")
    return {
        "status": "restarted",
        "confirm_prompt": confirm_prompt,
        **result,
    }


class TypePauseRequest(BaseModel):
    job_type: str
    paused: bool


@router.post("/pause-type")
def pause_type(body: TypePauseRequest, request: Request):
    """Pause or resume a single job type. Workers for that type sleep in
    their poll loop until resumed; in-flight jobs continue to completion.
    Useful when one job kind is misbehaving without stopping everything."""
    q = _get_queue(request)
    q.set_type_paused(body.job_type, body.paused)
    return {"job_type": body.job_type, "paused": body.paused}


@router.get("/paused-types")
def paused_types(request: Request):
    q = _get_queue(request)
    return {"paused": sorted(q.list_paused_types())}


@router.get("/dag/{job_id}")
def dag(job_id: str, request: Request, depth: int = 3):
    """Return the dependency / parent-child neighbourhood of a job up to
    ``depth`` hops in either direction. Powers the DAG drawer in the UI."""
    q = _get_queue(request)
    visited: set[str] = set()
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def _add(jid: str, hops: int):
        if not jid or jid in visited or hops < 0:
            return
        visited.add(jid)
        job = q.get_job(jid)
        if not job:
            return
        nodes[jid] = job.to_api(verbose=False)
        if job.depends_on:
            edges.append({"from": job.depends_on, "to": jid, "kind": "depends_on"})
            _add(job.depends_on, hops - 1)
        if job.parent_job_id:
            edges.append({"from": job.parent_job_id, "to": jid, "kind": "parent"})
            _add(job.parent_job_id, hops - 1)
        # children (jobs whose parent_job_id == jid)
        try:
            for child in q.list_children(jid):
                edges.append({"from": jid, "to": child.job_id, "kind": "child"})
                _add(child.job_id, hops - 1)
        except Exception:
            pass

    _add(job_id, max(1, min(int(depth), 5)))
    return {"root": job_id, "nodes": list(nodes.values()), "edges": edges}


@router.post("/jobs/{job_id}/replay")
def replay_with_edits(job_id: str, request: Request, body: dict):
    """Clone a completed/failed job into a new submission with optional
    payload overrides. Useful for re-running research with a tweaked
    topic or query list without losing the original record.
    Body shape: ``{payload_overrides: {...}, priority?: int, tags?: [..]}``.
    """
    q = _get_queue(request)
    job = q.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status in {"queued", "running"}:
        raise HTTPException(
            status_code=409,
            detail=f"job is {job.status} — cancel it first if you want to replay",
        )
    new_payload = dict(job.payload or {})
    overrides = body.get("payload_overrides") if isinstance(body, dict) else None
    if isinstance(overrides, dict):
        new_payload.update(overrides)
    try:
        new_id = q.submit(
            job.type, new_payload,
            source=f"{job.source or job.type}_replay",
            org_id=job.org_id,
            priority=int(body.get("priority") or job.priority),
            depends_on=job.depends_on,
        )
        # Tag the replay so the UI can show ancestry.
        if body.get("tags"):
            try:
                q.update_tags(new_id, add=list(body.get("tags") or []), remove=[])
            except Exception:
                pass
        return {"status": "queued", "previous_job_id": job_id, "job_id": new_id, "type": job.type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"replay failed: {e}")


def _event_stream(queue, disconnect: threading.Event):
    # deque.popleft is atomic under the GIL — producer/consumer need no shared lock here
    buf = queue.subscribe()
    try:
        while not disconnect.is_set():
            drained = False
            while buf:
                try:
                    event = buf.popleft()
                except IndexError:
                    break
                yield f"data: {json.dumps(event)}\n\n"
                drained = True
            if not drained:
                yield ": keepalive\n\n"
                time.sleep(2)
    except GeneratorExit:
        pass
    finally:
        queue.unsubscribe(buf)


@router.get("/events")
def events(request: Request):
    q = _get_queue(request)
    disconnect = threading.Event()

    async def on_disconnect():
        disconnect.set()

    from starlette.background import BackgroundTask
    response = StreamingResponse(
        _event_stream(q, disconnect),
        media_type="text/event-stream",
        background=BackgroundTask(on_disconnect),
    )
    return response
