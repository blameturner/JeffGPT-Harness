import json
import logging
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

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
