import json
import logging
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

_log = logging.getLogger("main.queue")

router = APIRouter(prefix="/queue")


def _get_queue(request: Request):
    q = getattr(request.app.state, "queue", None)
    if q is None:
        raise HTTPException(status_code=503, detail="Queue not initialised")
    return q


class SubmitRequest(BaseModel):
    type: str
    payload: dict
    priority: int = 3
    callback_url: str = ""
    timeout_s: int = 0


class PriorityUpdate(BaseModel):
    priority: int


@router.post("/submit")
def submit(body: SubmitRequest, request: Request):
    q = _get_queue(request)
    job = q.submit(
        job_type=body.type,
        payload=body.payload,
        priority=body.priority,
        callback_url=body.callback_url,
        timeout_s=body.timeout_s,
    )
    status = q.status()
    return {
        "job_id": job.id,
        "position": status["queue_length"],
        "estimated_wait_s": status["estimated_wait_s"],
    }


@router.get("/status")
def status(request: Request):
    q = _get_queue(request)
    return q.status()


@router.get("/jobs")
def list_jobs(request: Request, limit: int = 50):
    q = _get_queue(request)
    return {"jobs": q.list_jobs(limit=limit)}


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
        raise HTTPException(status_code=404, detail="Job not found or already completed")
    return {"cancelled": True}


@router.patch("/jobs/{job_id}/priority")
def update_priority(job_id: str, body: PriorityUpdate, request: Request):
    q = _get_queue(request)
    ok = q.update_priority(job_id, body.priority)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found or already running")
    return {"updated": True, "priority": body.priority}


@router.post("/jobs/{job_id}/cancel")
def cancel_running(job_id: str, request: Request):
    q = _get_queue(request)
    ok = q.cancel(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found")
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
