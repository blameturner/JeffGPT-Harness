import json
import logging
import threading
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from workers.tool_queue import NOCODB_TABLE

_log = logging.getLogger("main.tool_queue")

router = APIRouter(prefix="/tool-queue")


def _get_queue(request: Request):
    q = getattr(request.app.state, "tool_queue", None)
    if q is None:
        raise HTTPException(status_code=503, detail="Tool queue not initialised")
    return q


@router.get("/status")
def status(request: Request):
    q = _get_queue(request)
    return q.status()


@router.get("/jobs")
def list_jobs(
    request: Request,
    type: str = "",
    status: str = "",
    source: str = "",
    limit: int = 50,
):
    q = _get_queue(request)
    return {"jobs": q.list_jobs(job_type=type, status=status, source=source, limit=limit)}


@router.get("/active")
def active_jobs(
    request: Request,
    conversation_id: int | None = None,
    source: str = "",
):
    """Lightweight endpoint for UI banners — returns counts of queued/running
    jobs, optionally filtered by conversation_id and source."""
    q = _get_queue(request)
    jobs = q.list_jobs(source=source, limit=200)
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
    }


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    q = _get_queue(request)
    job = q.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_api()


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
    """Thread-safe SSE generator with disconnect detection.

    The subscriber buffer is a ``collections.deque`` — ``popleft()`` is
    atomic and O(1) under CPython's GIL, so no explicit lock is needed
    between the producer (``_emit_event``, which holds ``_sub_lock`` when
    appending) and this single consumer.
    """
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
