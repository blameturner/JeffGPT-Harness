import json
import logging
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
    return {"jobs": q.list_jobs(job_type=type, status=status, limit=limit)}


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
    # Update priority in NocoDB
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


def _event_stream(queue):
    buf = queue.subscribe()
    try:
        while True:
            while buf:
                event = buf.pop(0)
                yield f"data: {json.dumps(event)}\n\n"
            time.sleep(2)
            yield ": keepalive\n\n"
    finally:
        queue.unsubscribe(buf)


@router.get("/events")
def events(request: Request):
    q = _get_queue(request)
    return StreamingResponse(_event_stream(q), media_type="text/event-stream")
