import json
import logging
import threading
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from infra.config import HUEY_CONSUMER_WORKERS, HUEY_ENABLED, HUEY_SQLITE_PATH
from infra.huey_runtime import get_huey, is_huey_consumer_running
from workers.tool_queue import NOCODB_TABLE

_log = logging.getLogger("main.tool_queue")

router = APIRouter(prefix="/tool-queue")


def _get_queue(request: Request):
    q = getattr(request.app.state, "tool_queue", None)
    if q is None:
        raise HTTPException(status_code=503, detail="Tool queue not initialised")
    return q


def _huey_status() -> dict:
    return {
        "enabled": bool(HUEY_ENABLED),
        "consumer_running": is_huey_consumer_running(),
        "workers": int(HUEY_CONSUMER_WORKERS or 1),
        "sqlite_path": HUEY_SQLITE_PATH,
        "queue_ready": bool(get_huey() is not None),
    }


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
                "pathfinder_recrawl_dispatcher",
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


@router.get("/dashboard")
def dashboard(
    request: Request,
    org_id: int | None = None,
    limit: int = 20,
):
    q = _get_queue(request)
    limit = min(max(1, limit), 100)
    recent_jobs = q.list_jobs(limit=limit, org_id=org_id, verbose=True)
    active_jobs = [j for j in recent_jobs if j["status"] in ("queued", "running")]
    return {
        "queue": q.status(),
        "runtime": {
            "tool_queue_ready": True,
            "huey": _huey_status(),
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


@router.delete("/jobs/{job_id}")
def cancel_job(job_id: str, request: Request):
    q = _get_queue(request)
    ok = q.cancel(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found or not cancellable")
    return {"cancelled": True}


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
