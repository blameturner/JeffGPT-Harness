import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from nocodb_client import NocodbClient
from workers.generator_agent import GeneratorAgent
from workers.jobs import STORE, run_in_background, stream_events

_log = logging.getLogger("main.agents")

router = APIRouter()


class RunRequest(BaseModel):
    agent_name: str
    org_id: int
    task: str
    product: str = ""


@router.post("/run")
def run_agent(request: RunRequest):
    _log.info("POST /run  agent=%s org=%d", request.agent_name, request.org_id)
    try:
        agent = GeneratorAgent(request.agent_name, request.org_id)
        result = agent.run(request.task, request.product)
        if result is None:
            _log.warning("POST /run  agent=%s produced no output", request.agent_name)
            raise HTTPException(status_code=500, detail="Agent ran but failed to produce output")
        _log.info("POST /run ok  agent=%s", request.agent_name)
        return {
            "success": True,
            "agent": request.agent_name,
            "org_id": request.org_id,
            "product": request.product,
            "output": result.model_dump(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        _log.error("POST /run failed  agent=%s", request.agent_name, exc_info=True)
        raise HTTPException(status_code=500, detail="internal error")


@router.post("/run/stream")
def run_agent_stream(request: RunRequest):
    _log.info("POST /run/stream  agent=%s org=%d", request.agent_name, request.org_id)
    try:
        agent = GeneratorAgent(request.agent_name, request.org_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    job = STORE.create()

    def worker(j):
        for event in agent.run_streaming(request.task, request.product):
            STORE.append(j, event)

    run_in_background(job, worker)
    return {"job_id": job.id}


@router.get("/stream/{job_id}")
def stream(job_id: str, cursor: int = 0):
    # Resumable SSE: clients reconnect with ?cursor=N to replay missed events.
    return StreamingResponse(stream_events(job_id, cursor), media_type="text/event-stream")


@router.get("/agents")
def list_agents(org_id: int, limit: int = 200):
    try:
        db = NocodbClient()
        rows = db.list_agents(org_id=org_id, limit=limit)
        agents = [
            {
                "Id": r["Id"],
                "name": r.get("name"),
                "display_name": r.get("display_name"),
                "model": r.get("model"),
                "status": r.get("status"),
            }
            for r in rows
            if r.get("status") in (None, "active")
        ]
        return {"agents": agents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scheduler/reload")
def scheduler_reload():
    from scheduler import reload_agent_schedules
    return reload_agent_schedules()


@router.post("/scheduler/trigger")
def scheduler_trigger():
    import threading
    from workers.enrichment_agent import run_enrichment_cycle
    threading.Thread(target=run_enrichment_cycle, daemon=True).start()
    return {"status": "triggered"}


@router.get("/scheduler/status")
def scheduler_status(request: Request):
    from workers.enrichment_agent import sources_due_count, get_last_run
    sched = getattr(request.app.state, "scheduler", None)
    running = bool(sched and sched.running)
    next_run = None
    if sched:
        job = sched.get_job("enrichment_cycle")
        if job and job.next_run_time:
            next_run = job.next_run_time.isoformat()
    return {
        "running": running,
        "next_run": next_run,
        "last_run": get_last_run(),
        "sources_due": sources_due_count(),
    }
