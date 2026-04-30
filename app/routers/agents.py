import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from infra.nocodb_client import NocodbClient
from workers.user_agents.generator_agent import GeneratorAgent
from shared.jobs import STORE, run_in_background, stream_events

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
    # cursor=N replays missed events on reconnect — don't remove without coordinating with client
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


@router.get("/agents/{agent_name}")
def get_agent_detail(agent_name: str, org_id: int):
    """Full agent metadata. Used by the agent-detail panel: persona, model,
    prompt template, status, schedule hints. Strips internal-only fields."""
    try:
        db = NocodbClient()
        row = db.get_agent(agent_name, org_id=org_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if not row:
        raise HTTPException(status_code=404, detail="agent not found")
    keep = {
        "Id", "name", "display_name", "description", "persona",
        "system_prompt_template", "model", "status",
        "max_runs_per_day", "max_tokens_per_day", "max_concurrent_runs",
        "CreatedAt", "UpdatedAt",
    }
    return {"agent": {k: v for k, v in row.items() if k in keep}}


@router.get("/agents/{agent_name}/runs")
def list_agent_runs(agent_name: str, org_id: int, limit: int = 20):
    """Recent runs for an agent. Status, summary, tokens, duration — enough
    for the UI to show 'last 20 runs' without loading every event row."""
    try:
        db = NocodbClient()
        rows = db._get_paginated("agent_runs", params={
            "where": f"(org_id,eq,{org_id})~and(agent_name,eq,{agent_name})",
            "sort": "-CreatedAt",
            "limit": min(max(1, limit), 100),
        })
    except Exception as e:
        _log.warning("list_agent_runs failed  agent=%s org=%d", agent_name, org_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    runs = [
        {
            "id": r.get("Id"),
            "status": r.get("status"),
            "summary": (r.get("summary") or "")[:500],
            "duration_seconds": r.get("duration_seconds"),
            "tokens_input": r.get("tokens_input"),
            "tokens_output": r.get("tokens_output"),
            "conversation_id": r.get("conversation_id"),
            "created_at": r.get("CreatedAt"),
        }
        for r in rows
    ]
    return {"agent_name": agent_name, "runs": runs}


@router.post("/scheduler/reload")
def scheduler_reload():
    from scheduler import reload_agent_schedules
    return reload_agent_schedules()


@router.get("/scheduler/status")
def scheduler_status(request: Request):
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
