from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from workers.generator_agent import GeneratorAgent
from workers.chat_agent import ChatAgent
from workers.code_agent import CodeAgent
from workers.jobs import STORE, run_in_background, stream_events
from workers.styles import (
    CHAT_DEFAULT_STYLE,
    CODE_DEFAULT_STYLE,
    list_chat_styles,
    list_code_styles,
)
from config import MODELS, refresh_models
from nocodb_client import NocodbClient
from contextlib import asynccontextmanager
import log

log.setup()
_log = log.get("harness")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _log.info("mstag-harness starting")
    from scheduler import start_scheduler
    sched = start_scheduler()
    app.state.scheduler = sched
    _log.info("scheduler running")
    _log.info("ready")
    try:
        yield
    finally:
        sched.shutdown(wait=False)
        _log.info("shutdown complete")

app = FastAPI(title="MSTAG Harness", version="1.0.0", lifespan=lifespan)


class RunRequest(BaseModel):
    agent_name: str
    org_id: int
    task: str
    product: str = ""


@app.get("/health")
async def health():
    return {"status": "ok", "service": "MSTAG Harness"}


@app.post("/run")
def run_agent(request: RunRequest):
    try:
        agent = GeneratorAgent(request.agent_name, request.org_id)
        result = agent.run(request.task, request.product)
        if result is None:
            raise HTTPException(status_code=500, detail="Agent ran but failed to produce output")
        return {
            "success": True,
            "agent": request.agent_name,
            "org_id": request.org_id,
            "product": request.product,
            "output": result.model_dump(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run/stream")
def run_agent_stream(request: RunRequest):
    try:
        agent = GeneratorAgent(request.agent_name, request.org_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    job = STORE.create()
    run_in_background(job, lambda: agent.run_streaming(request.task, request.product))
    return {"job_id": job.id}


class ChatRequest(BaseModel):
    org_id: int
    model: str
    message: str
    conversation_id: int | None = None
    system: str | None = None
    temperature: float = 0.7
    max_tokens: int = 8192
    rag_enabled: bool | None = None
    rag_collection: str | None = None
    knowledge_enabled: bool | None = None
    search_enabled: bool = False
    search_consent_declined: bool = False
    response_style: str | None = None


class ConversationUpdate(BaseModel):
    title: str | None = None
    code_checklist: list | None = None


class CodeRequest(BaseModel):
    org_id: int
    model: str
    message: str
    mode: str = "plan"
    approved_plan: str | None = None
    files: list[dict] | None = None
    conversation_id: int | None = None
    title: str | None = None
    codebase_collection: str | None = None
    response_style: str | None = None
    temperature: float = 0.2
    max_tokens: int = 8192


@app.get("/models")
def list_models():
    catalog = MODELS or refresh_models()
    seen: set[str] = set()
    models: list[dict] = []
    for entry in catalog.values():
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        if not role or role in seen:
            continue
        seen.add(role)
        models.append({
            "name": role,
            "role": role,
            "model_id": entry.get("model_id"),
            "url": entry.get("url"),
        })
    return {"models": models}


@app.get("/styles")
def get_styles(surface: str | None = None):
    out: dict = {}
    if surface in (None, "chat"):
        out["chat"] = {"default": CHAT_DEFAULT_STYLE, "styles": list_chat_styles()}
    if surface in (None, "code"):
        out["code"] = {"default": CODE_DEFAULT_STYLE, "styles": list_code_styles()}
    if not out:
        raise HTTPException(status_code=400, detail="surface must be 'chat' or 'code'")
    return out


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        agent = ChatAgent(
            model=request.model,
            org_id=request.org_id,
            search_enabled=request.search_enabled,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    job = STORE.create()
    run_in_background(
        job,
        lambda: agent.send_streaming(
            user_message=request.message,
            conversation_id=request.conversation_id,
            system=request.system,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            rag_enabled=request.rag_enabled,
            rag_collection=request.rag_collection,
            knowledge_enabled=request.knowledge_enabled,
            search_consent_declined=request.search_consent_declined,
            response_style=request.response_style,
        ),
        on_complete=agent.persist_from_events,
    )
    return {"job_id": job.id}


@app.post("/code")
def code(request: CodeRequest):
    try:
        agent = CodeAgent(
            model=request.model,
            org_id=request.org_id,
            mode=request.mode,  # type: ignore[arg-type]
            approved_plan=request.approved_plan,
            files=request.files,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    job = STORE.create()
    run_in_background(
        job,
        lambda: agent.run_streaming(
            user_message=request.message,
            conversation_id=request.conversation_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            title=request.title,
            codebase_collection=request.codebase_collection,
            response_style=request.response_style,
        ),
        on_complete=agent.persist_from_events,
    )
    return {"job_id": job.id}


@app.get("/stream/{job_id}")
def stream(job_id: str, cursor: int = 0):
    # Resumable SSE: clients reconnect with ?cursor=N to replay missed events.
    return StreamingResponse(stream_events(job_id, cursor), media_type="text/event-stream")


@app.get("/code/conversations")
def list_code_conversations(org_id: int, limit: int = 50):
    try:
        db = NocodbClient()
        return {"conversations": db.list_code_conversations(org_id, limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/code/conversations/{conversation_id}")
def get_code_conversation(conversation_id: int):
    try:
        db = NocodbClient()
        convo = db.get_code_conversation(conversation_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Code conversation not found")
        return convo
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/code/conversations/{conversation_id}/messages")
def get_code_messages(conversation_id: int, limit: int = 500):
    try:
        db = NocodbClient()
        return {"messages": db.list_code_messages(conversation_id, limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/code/conversations/{conversation_id}/workspace")
def get_code_workspace(conversation_id: int):
    try:
        db = NocodbClient()
        msgs = db.list_code_messages(conversation_id)
        for m in reversed(msgs):
            if m.get("role") != "user":
                continue
            raw = m.get("files_json")
            if not raw:
                continue
            if isinstance(raw, list):
                return {"files": raw}
            if isinstance(raw, str):
                try:
                    import json as _json
                    data = _json.loads(raw)
                    if isinstance(data, list):
                        return {"files": data}
                except Exception:
                    continue
        return {"files": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/code/conversations/{conversation_id}")
def update_code_conversation(conversation_id: int, body: ConversationUpdate):
    try:
        db = NocodbClient()
        convo = db.get_code_conversation(conversation_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Code conversation not found")
        updates: dict = {}
        if body.title is not None:
            updates["title"] = body.title.strip() or "Untitled"
        if body.code_checklist is not None:
            updates["code_checklist"] = body.code_checklist
        if not updates:
            return convo
        return db.update_code_conversation(conversation_id, updates)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents")
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


@app.get("/conversations")
def list_conversations(org_id: int, limit: int = 50):
    try:
        db = NocodbClient()
        return {"conversations": db.list_conversations(org_id, limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _distinct_nonempty(rows: list[dict], field: str) -> list[str]:
    return sorted({
        (r.get(field) or "").strip()
        for r in rows
        if (r.get(field) or "").strip()
    })


def _counter(rows: list[dict], field: str) -> dict:
    out: dict = {}
    for r in rows:
        key = (r.get(field) or "").strip() or "unknown"
        out[key] = out.get(key, 0) + 1
    return out


@app.patch("/conversations/{conversation_id}")
def update_conversation(conversation_id: int, body: ConversationUpdate):
    try:
        db = NocodbClient()
        convo = db.get_conversation(conversation_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")
        updates: dict = {}
        if body.title is not None:
            updates["title"] = body.title.strip() or "Untitled"
        if not updates:
            return convo
        return db.update_conversation(conversation_id, updates)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{conversation_id}/summary")
def conversation_summary(conversation_id: int):
    try:
        db = NocodbClient()
        convo = db.get_conversation(conversation_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")

        messages = db.list_messages(conversation_id)
        observations = db.list_observations_for_conversation(conversation_id)
        runs = db.list_runs_for_conversation(conversation_id)
        outputs = db.list_outputs_for_conversation(conversation_id)
        tasks = db.list_tasks_for_conversation(conversation_id)

        msg_tokens_in = sum(int(m.get("tokens_input") or 0) for m in messages)
        msg_tokens_out = sum(int(m.get("tokens_output") or 0) for m in messages)
        run_tokens_in = sum(int(r.get("tokens_input") or 0) for r in runs)
        run_tokens_out = sum(int(r.get("tokens_output") or 0) for r in runs)
        run_context_tokens = sum(int(r.get("context_tokens") or 0) for r in runs)
        run_duration = sum(float(r.get("duration_seconds") or 0) for r in runs)

        tokens_input = msg_tokens_in + run_tokens_in
        tokens_output = msg_tokens_out + run_tokens_out

        role_counts = _counter(messages, "role")
        user_msgs = [m for m in messages if m.get("role") == "user"]
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        chars_user = sum(len(m.get("content") or "") for m in user_msgs)
        chars_assistant = sum(len(m.get("content") or "") for m in assistant_msgs)

        timestamps = [m.get("CreatedAt") for m in messages if m.get("CreatedAt")]
        first_message_at = min(timestamps) if timestamps else None
        last_message_at = max(timestamps) if timestamps else None

        models_used = _distinct_nonempty(messages, "model") or _distinct_nonempty(runs, "model_name")

        themes = _distinct_nonempty(observations, "domain")
        obs_types = _distinct_nonempty(observations, "type")
        obs_confidences = _counter(observations, "confidence")
        obs_statuses = _counter(observations, "status")
        theme_counts = _counter(observations, "domain")

        agents_used = _distinct_nonempty(runs, "agent_name")
        run_statuses = _counter(runs, "status")

        task_statuses = _counter(tasks, "status")

        return {
            "conversation": convo,
            "message_count": len(messages),
            "role_counts": role_counts,
            "observation_count": len(observations),
            "run_count": len(runs),
            "output_count": len(outputs),
            "task_count": len(tasks),
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "tokens_total": tokens_input + tokens_output,
            "tokens_breakdown": {
                "messages_input": msg_tokens_in,
                "messages_output": msg_tokens_out,
                "runs_input": run_tokens_in,
                "runs_output": run_tokens_out,
                "runs_context": run_context_tokens,
            },
            "first_message_at": first_message_at,
            "last_message_at": last_message_at,
            "run_duration_seconds": round(run_duration, 2),
            "chars_user": chars_user,
            "chars_assistant": chars_assistant,
            "models_used": models_used,
            "agents_used": agents_used,
            "themes": themes,
            "theme_counts": theme_counts,
            "observation_types": obs_types,
            "observation_confidences": obs_confidences,
            "observation_statuses": obs_statuses,
            "run_statuses": run_statuses,
            "task_statuses": task_statuses,
            "observations": observations,
            "runs": runs,
            "outputs": outputs,
            "tasks": tasks,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{conversation_id}/messages")
def get_messages(conversation_id: int):
    try:
        db = NocodbClient()
        convo = db.get_conversation(conversation_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {
            "conversation": convo,
            "messages": db.list_messages(conversation_id),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


WORKER_TYPES = [
    {"id": "generator", "name": "Generator", "description": "Produces structured output for a task."},
    {"id": "monitor", "name": "Monitor", "description": "Watches a data source and flags changes."},
    {"id": "evaluator", "name": "Evaluator", "description": "Scores or critiques outputs against criteria."},
    {"id": "researcher", "name": "Researcher", "description": "Performs multi-source web research."},
    {"id": "memory", "name": "Memory", "description": "Writes and retrieves knowledge from ChromaDB / FalkorDB."},
    {"id": "enrichment", "name": "Enrichment", "description": "Scrapes whitelisted sources and enriches the knowledge base."},
    {"id": "code", "name": "Code", "description": "Plans, writes, and debugs code."},
]


@app.get("/workers/types")
def worker_types():
    return {"types": WORKER_TYPES}


@app.post("/scheduler/reload")
def scheduler_reload():
    from scheduler import reload_agent_schedules
    return reload_agent_schedules()


@app.post("/scheduler/trigger")
def scheduler_trigger():
    import threading
    from workers.enrichment_agent import run_enrichment_cycle
    threading.Thread(target=run_enrichment_cycle, daemon=True).start()
    return {"status": "triggered"}


@app.get("/scheduler/status")
def scheduler_status():
    from workers.enrichment_agent import sources_due_count
    sched = getattr(app.state, "scheduler", None)
    running = bool(sched and sched.running)
    next_run = None
    if sched:
        job = sched.get_job("enrichment_cycle")
        if job and job.next_run_time:
            next_run = job.next_run_time.isoformat()
    return {
        "running": running,
        "next_run": next_run,
        "sources_due": sources_due_count(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3800, reload=True)