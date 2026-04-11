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


@app.post("/run/stream")
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
    # §7 — per-conversation opt-out for contextual enrichment. Default
    # behaviour is opt-in (True); the frontend properties screen exposes
    # a toggle. When False, messages classified as contextual_enrichment
    # are downgraded to chitchat in chat_agent and no search fires.
    contextual_grounding_enabled: bool | None = None


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
    knowledge_enabled: bool | None = None
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
    _log.info("POST /chat  model=%s org=%d conv=%s search=%s", request.model, request.org_id, request.conversation_id, request.search_enabled)
    try:
        agent = ChatAgent(
            model=request.model,
            org_id=request.org_id,
            search_enabled=request.search_enabled,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    job = STORE.create()
    run_in_background(job, lambda j: agent.run_job(
        j,
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
    ))
    return {"job_id": job.id}


@app.post("/code")
def code(request: CodeRequest):
    _log.info("POST /code  model=%s org=%d mode=%s conv=%s", request.model, request.org_id, request.mode, request.conversation_id)
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
    run_in_background(job, lambda j: agent.run_job(
        j,
        user_message=request.message,
        conversation_id=request.conversation_id,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        title=request.title,
        codebase_collection=request.codebase_collection,
        response_style=request.response_style,
        knowledge_enabled=request.knowledge_enabled,
    ))
    return {"job_id": job.id}


@app.get("/collections")
def list_collections(org_id: int | None = None):
    from memory import client
    try:
        cols = client.list_collections()
        result = []
        for c in cols:
            name = c.name
            count = c.count()
            if org_id is not None and not name.startswith(f"org_{org_id}_"):
                continue
            result.append({"name": name, "records": count})
        return {"collections": sorted(result, key=lambda x: x["name"])}
    except Exception as e:
        _log.error("list_collections failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class CodebaseCreate(BaseModel):
    org_id: int
    name: str
    description: str | None = None


class CodebaseFileUpload(BaseModel):
    files: list[dict]


@app.get("/codebases")
def list_codebases(org_id: int):
    from memory import client
    try:
        db = NocodbClient()
        prefix = f"org_{org_id}_codebase_"
        codebases = []
        if "knowledge_sources" in db.tables:
            rows = db._get("knowledge_sources", params={
                "where": f"(org_id,eq,{org_id})~and(type,eq,codebase)",
                "limit": 200,
            }).get("list", [])
            for row in rows:
                collection_name = row.get("collection_name") or ""
                record_count = 0
                try:
                    col = client.get_or_create_collection(collection_name)
                    record_count = col.count()
                except Exception:
                    pass
                codebases.append({
                    "id": row["Id"],
                    "name": row.get("name"),
                    "description": row.get("description"),
                    "collection_name": collection_name,
                    "records": record_count,
                    "source": row.get("source"),
                    "created_at": row.get("CreatedAt"),
                })
        return {"codebases": codebases}
    except Exception as e:
        _log.error("list_codebases failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/codebases")
def create_codebase(body: CodebaseCreate):
    from config import scoped_collection
    try:
        db = NocodbClient()
        collection_name = scoped_collection(body.org_id, f"codebase_{body.name.lower().replace(' ', '_')}")
        row = db._post("knowledge_sources", {
            "org_id": body.org_id,
            "name": body.name,
            "description": body.description or "",
            "type": "codebase",
            "collection_name": collection_name,
            "source": "manual",
        })
        _log.info("codebase created  name=%s collection=%s", body.name, collection_name)
        return row
    except Exception as e:
        _log.error("create_codebase failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/codebases/{codebase_id}/index")
def index_codebase_files(codebase_id: int, body: CodebaseFileUpload):
    import base64
    from memory import remember
    try:
        db = NocodbClient()
        if "knowledge_sources" not in db.tables:
            raise HTTPException(status_code=404, detail="knowledge_sources table not found")
        rows = db._get("knowledge_sources", params={
            "where": f"(Id,eq,{codebase_id})",
            "limit": 1,
        }).get("list", [])
        if not rows:
            raise HTTPException(status_code=404, detail="codebase not found")
        row = rows[0]
        collection_name = row["collection_name"]
        org_id = int(row["org_id"])

        indexed = 0
        for f in body.files:
            name = f.get("name", "unknown")
            content = f.get("content") or ""
            if f.get("content_b64"):
                content = base64.b64decode(f["content_b64"]).decode("utf-8", errors="replace")
            if not content.strip():
                continue
            text = f"FILE: {name}\n\n{content}"
            try:
                remember(
                    text=text,
                    metadata={"file": name, "codebase_id": codebase_id, "type": "codebase"},
                    org_id=org_id,
                    collection_name=collection_name,
                )
                indexed += 1
            except Exception:
                _log.error("index failed for file %s in codebase %d", name, codebase_id, exc_info=True)

        _log.info("codebase indexed  id=%d files=%d/%d", codebase_id, indexed, len(body.files))
        return {"indexed": indexed, "total": len(body.files)}
    except HTTPException:
        raise
    except Exception as e:
        _log.error("index_codebase_files failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
        if body.contextual_grounding_enabled is not None:
            updates["contextual_grounding_enabled"] = bool(body.contextual_grounding_enabled)
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
        messages = db.list_messages(conversation_id)
        sources_by_msg: dict[int, list[dict]] = {}
        if any(int(m.get("search_source_count") or 0) > 0 for m in messages):
            all_sources = db.list_message_search_sources(conversation_id=conversation_id)
            for src in all_sources:
                mid = src.get("message_id")
                if mid is not None:
                    sources_by_msg.setdefault(mid, []).append(src)
        for msg in messages:
            raw_sources = sources_by_msg.get(msg.get("Id"), [])
            for src in raw_sources:
                src["index"] = (src.get("source_index") or 0) + 1
                src["used_in_answer"] = bool(src.get("used_in_answer"))
            msg["search_sources"] = raw_sources
        return {
            "conversation": convo,
            "messages": messages,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/messages/{message_id}/search-sources")
def get_message_search_sources(message_id: int):
    try:
        db = NocodbClient()
        sources = db.list_message_search_sources(message_id=message_id)
        for src in sources:
            src["index"] = (src.get("source_index") or 0) + 1
            src["used_in_answer"] = bool(src.get("used_in_answer"))
        return {"sources": sources}
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
    from workers.enrichment_agent import sources_due_count, get_last_run
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
        "last_run": get_last_run(),
        "sources_due": sources_due_count(),
    }


class EnrichmentAgentCreate(BaseModel):
    org_id: int
    name: str
    description: str | None = None
    category: str | None = None
    token_budget: int = 50000
    cron_expression: str | None = None
    timezone: str = "Australia/Sydney"
    active: bool = True


class EnrichmentAgentUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    category: str | None = None
    token_budget: int | None = None
    cron_expression: str | None = None
    timezone: str | None = None
    active: bool | None = None


@app.get("/enrichment/agents")
def list_enrichment_agents(org_id: int | None = None):
    from workers.enrichment_agent import EnrichmentDB
    try:
        db = EnrichmentDB()
        return {"agents": db.list_enrichment_agents(org_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enrichment/agents")
def create_enrichment_agent(body: EnrichmentAgentCreate):
    from workers.enrichment_agent import EnrichmentDB
    try:
        db = EnrichmentDB()
        agent = db.create_enrichment_agent(body.model_dump())
        return agent
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/enrichment/agents/{agent_id}")
def update_enrichment_agent(agent_id: int, body: EnrichmentAgentUpdate):
    from workers.enrichment_agent import EnrichmentDB
    try:
        db = EnrichmentDB()
        updates = {k: v for k, v in body.model_dump().items() if v is not None}
        if not updates:
            return db.get_enrichment_agent(agent_id)
        return db.update_enrichment_agent(agent_id, updates)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enrichment/agents/{agent_id}/trigger")
def trigger_enrichment_agent(agent_id: int):
    import threading
    from workers.enrichment_agent import run_enrichment_cycle
    threading.Thread(target=run_enrichment_cycle, args=[agent_id], daemon=True).start()
    return {"status": "triggered", "agent_id": agent_id}


@app.get("/enrichment/agents/{agent_id}/status")
def enrichment_agent_status(agent_id: int):
    from workers.enrichment_agent import sources_due_count, get_last_run
    sched = getattr(app.state, "scheduler", None)
    next_run = None
    if sched:
        job = sched.get_job(f"enrichment_agent_{agent_id}")
        if job and job.next_run_time:
            next_run = job.next_run_time.isoformat()
    return {
        "agent_id": agent_id,
        "next_run": next_run,
        "last_run": get_last_run(agent_id),
        "sources_due": sources_due_count(agent_id),
    }


class SourceCreate(BaseModel):
    org_id: int
    url: str
    name: str
    category: str = "documentation"
    frequency_hours: float = 24
    active: bool = True
    enrichment_agent_id: int | None = None
    use_playwright: bool = False


class SourceUpdate(BaseModel):
    name: str | None = None
    url: str | None = None
    category: str | None = None
    frequency_hours: float | None = None
    active: bool | None = None
    enrichment_agent_id: int | None = None
    use_playwright: bool | None = None


@app.get("/enrichment/sources")
def list_sources(org_id: int, agent_id: int | None = None, active_only: bool = False):
    from workers.enrichment_agent import EnrichmentDB
    try:
        db = EnrichmentDB()
        return {"sources": db.list_sources(org_id, enrichment_agent_id=agent_id, active_only=active_only)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enrichment/sources")
def create_source(body: SourceCreate):
    from workers.enrichment_agent import EnrichmentDB
    try:
        db = EnrichmentDB()
        return db.create_source(body.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/enrichment/sources/{source_id}")
def get_source(source_id: int):
    from workers.enrichment_agent import EnrichmentDB
    try:
        db = EnrichmentDB()
        source = db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        return source
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/enrichment/sources/{source_id}")
def update_source(source_id: int, body: SourceUpdate):
    from workers.enrichment_agent import EnrichmentDB
    try:
        db = EnrichmentDB()
        updates = {k: v for k, v in body.model_dump().items() if v is not None}
        if not updates:
            return db.get_source(source_id)
        db.update_scrape_target(source_id, **updates)
        return db.get_source(source_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/enrichment/sources/{source_id}")
def delete_source(source_id: int):
    from workers.enrichment_agent import EnrichmentDB
    try:
        db = EnrichmentDB()
        db.delete_source(source_id)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enrichment/sources/{source_id}/flush")
def flush_source(source_id: int):
    from workers.enrichment_agent import EnrichmentDB
    try:
        db = EnrichmentDB()
        return db.flush_source(source_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/enrichment/sources/{source_id}/log")
def source_log(source_id: int, limit: int = 50):
    from workers.enrichment_agent import EnrichmentDB
    try:
        db = EnrichmentDB()
        return {"log": db.list_log(scrape_target_id=source_id, limit=limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/enrichment/log")
def enrichment_log(org_id: int | None = None, limit: int = 100):
    from workers.enrichment_agent import EnrichmentDB
    try:
        db = EnrichmentDB()
        return {"log": db.list_log(org_id=org_id, limit=limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SuggestionUpdate(BaseModel):
    status: str | None = None


@app.get("/enrichment/suggestions")
def list_suggestions(org_id: int, status: str | None = None):
    from workers.enrichment_agent import EnrichmentDB
    try:
        db = EnrichmentDB()
        return {"suggestions": db.list_suggestions(org_id, status=status)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/enrichment/suggestions/{suggestion_id}")
def get_suggestion(suggestion_id: int):
    from workers.enrichment_agent import EnrichmentDB
    try:
        db = EnrichmentDB()
        suggestion = db.get_suggestion(suggestion_id)
        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        return suggestion
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/enrichment/suggestions/{suggestion_id}")
def update_suggestion(suggestion_id: int, body: SuggestionUpdate):
    from workers.enrichment_agent import EnrichmentDB
    try:
        db = EnrichmentDB()
        updates = {k: v for k, v in body.model_dump().items() if v is not None}
        if not updates:
            return db.get_suggestion(suggestion_id)
        return db.update_suggestion(suggestion_id, updates)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SuggestionApprove(BaseModel):
    enrichment_agent_id: int | None = None


@app.post("/enrichment/suggestions/{suggestion_id}/approve")
def approve_suggestion(suggestion_id: int, body: SuggestionApprove | None = None):
    from workers.enrichment_agent import EnrichmentDB
    try:
        db = EnrichmentDB()
        suggestion = db.get_suggestion(suggestion_id)
        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        org_id = int(suggestion.get("org_id") or 1)
        agent_id = body.enrichment_agent_id if body else None
        source = db.approve_suggestion(suggestion_id, org_id, enrichment_agent_id=agent_id)
        return {"ok": True, "source": source}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enrichment/suggestions/{suggestion_id}/reject")
def reject_suggestion(suggestion_id: int):
    from workers.enrichment_agent import EnrichmentDB
    try:
        db = EnrichmentDB()
        return db.update_suggestion(suggestion_id, {"status": "rejected"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BulkSuggestionAction(BaseModel):
    parent_target: int
    enrichment_agent_id: int | None = None


@app.post("/enrichment/suggestions/approve-by-parent")
def approve_suggestions_by_parent(body: BulkSuggestionAction):
    from workers.enrichment_agent import EnrichmentDB
    from config import NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS
    try:
        db = EnrichmentDB()
        suggestions = db._get(
            NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS,
            params={
                "where": f"(parent_target,eq,{body.parent_target})~and(status,eq,pending)",
                "limit": 200,
            },
        ).get("list", [])
        if not suggestions:
            return {"ok": True, "approved": 0, "sources": []}
        sources = []
        for s in suggestions:
            org_id = int(s.get("org_id") or 1)
            source = db.approve_suggestion(s["Id"], org_id, enrichment_agent_id=body.enrichment_agent_id)
            sources.append(source)
        return {"ok": True, "approved": len(sources), "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enrichment/suggestions/reject-by-parent")
def reject_suggestions_by_parent(body: BulkSuggestionAction):
    from workers.enrichment_agent import EnrichmentDB
    from config import NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS
    try:
        db = EnrichmentDB()
        suggestions = db._get(
            NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS,
            params={
                "where": f"(parent_target,eq,{body.parent_target})~and(status,eq,pending)",
                "limit": 200,
            },
        ).get("list", [])
        for s in suggestions:
            db.update_suggestion(s["Id"], {"status": "rejected"})
        return {"ok": True, "rejected": len(suggestions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/usage")
def stats_usage(org_id: int, period: str = "30d"):
    from datetime import datetime, timedelta, timezone
    try:
        db = NocodbClient()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    now = datetime.now(timezone.utc)
    if period == "7d":
        start = now - timedelta(days=7)
    elif period == "30d":
        start = now - timedelta(days=30)
    else:
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)

    period_start = start.strftime("%Y-%m-%d")
    period_end = now.strftime("%Y-%m-%d")

    try:
        msg_where = f"(org_id,eq,{org_id})~and(CreatedAt,gte,{start.isoformat()})"
        messages = db._get("messages", params={"where": msg_where, "limit": 5000}).get("list", [])
    except Exception:
        messages = []

    try:
        run_where = f"(org_id,eq,{org_id})~and(CreatedAt,gte,{start.isoformat()})"
        runs = db._get("agent_runs", params={"where": run_where, "limit": 5000}).get("list", [])
    except Exception:
        runs = []

    # --- totals ---
    total_tokens_in = sum(int(m.get("tokens_input") or 0) for m in messages)
    total_tokens_in += sum(int(r.get("tokens_input") or 0) for r in runs)
    total_tokens_out = sum(int(m.get("tokens_output") or 0) for m in messages)
    total_tokens_out += sum(int(r.get("tokens_output") or 0) for r in runs)
    total_requests = len(messages) + len(runs)
    total_conversations = len({m.get("conversation_id") for m in messages if m.get("conversation_id")})
    failed_runs = [r for r in runs if r.get("status") == "failed"]
    total_errors = len(failed_runs)

    # --- by_model (with percentiles) ---
    by_model: dict = {}
    for m in messages:
        model = (m.get("model") or "unknown").strip() or "unknown"
        entry = by_model.setdefault(model, {
            "model_name": model, "requests": 0, "tokens_input": 0,
            "tokens_output": 0, "durations": [], "error_count": 0,
        })
        entry["requests"] += 1
        entry["tokens_input"] += int(m.get("tokens_input") or 0)
        entry["tokens_output"] += int(m.get("tokens_output") or 0)
    for r in runs:
        model = (r.get("model_name") or "unknown").strip() or "unknown"
        entry = by_model.setdefault(model, {
            "model_name": model, "requests": 0, "tokens_input": 0,
            "tokens_output": 0, "durations": [], "error_count": 0,
        })
        entry["requests"] += 1
        entry["tokens_input"] += int(r.get("tokens_input") or 0)
        entry["tokens_output"] += int(r.get("tokens_output") or 0)
        dur = float(r.get("duration_seconds") or 0)
        if dur > 0:
            entry["durations"].append(dur)
        if r.get("status") == "failed":
            entry["error_count"] += 1

    def _percentile(sorted_vals: list[float], pct: float) -> float:
        if not sorted_vals:
            return 0.0
        idx = int(len(sorted_vals) * pct)
        idx = min(idx, len(sorted_vals) - 1)
        return round(sorted_vals[idx], 2)

    by_model_list = []
    for entry in sorted(by_model.values(), key=lambda x: x["requests"], reverse=True):
        avg_tokens = (entry["tokens_input"] + entry["tokens_output"]) // max(entry["requests"], 1)
        durations = sorted(entry["durations"])
        avg_dur = round(sum(durations) / len(durations), 2) if durations else 0.0
        by_model_list.append({
            "model_name": entry["model_name"],
            "requests": entry["requests"],
            "tokens_input": entry["tokens_input"],
            "tokens_output": entry["tokens_output"],
            "avg_tokens_per_request": avg_tokens,
            "avg_duration_seconds": avg_dur,
            "p50_duration_seconds": _percentile(durations, 0.50),
            "p95_duration_seconds": _percentile(durations, 0.95),
            "p99_duration_seconds": _percentile(durations, 0.99),
            "time_to_first_token_ms": 0,  # not tracked yet
            "error_count": entry["error_count"],
            "error_rate": round(entry["error_count"] / max(entry["requests"], 1), 4),
        })

    # --- by_day ---
    by_day: dict = {}
    for m in messages:
        day = (m.get("CreatedAt") or "")[:10]
        if not day:
            continue
        d = by_day.setdefault(day, {"date": day, "requests": 0, "tokens_input": 0, "tokens_output": 0, "errors": 0})
        d["requests"] += 1
        d["tokens_input"] += int(m.get("tokens_input") or 0)
        d["tokens_output"] += int(m.get("tokens_output") or 0)
    for r in runs:
        day = (r.get("CreatedAt") or "")[:10]
        if not day:
            continue
        d = by_day.setdefault(day, {"date": day, "requests": 0, "tokens_input": 0, "tokens_output": 0, "errors": 0})
        d["requests"] += 1
        d["tokens_input"] += int(r.get("tokens_input") or 0)
        d["tokens_output"] += int(r.get("tokens_output") or 0)
        if r.get("status") == "failed":
            d["errors"] += 1
    by_day_list = sorted(by_day.values(), key=lambda x: x["date"])

    # --- by_hour (heatmap) ---
    by_hour: dict = {}
    for m in messages:
        ts = m.get("CreatedAt") or ""
        if len(ts) < 13:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            key = (dt.hour, dt.isoweekday() % 7)  # 0=Sun
            bucket = by_hour.setdefault(key, {"hour": dt.hour, "day_of_week": dt.isoweekday() % 7, "requests": 0})
            bucket["requests"] += 1
        except (ValueError, AttributeError):
            continue
    by_hour_list = sorted(by_hour.values(), key=lambda x: (x["day_of_week"], x["hour"]))

    # --- by_style ---
    by_style: dict = {}
    for m in messages:
        style = (m.get("response_style") or "").strip() or "default"
        by_style[style] = by_style.get(style, 0) + 1
    by_style_list = [{"style": k, "requests": v} for k, v in sorted(by_style.items(), key=lambda x: -x[1])]

    # --- top_conversations ---
    convos: dict = {}
    for m in messages:
        cid = m.get("conversation_id")
        if not cid:
            continue
        c = convos.setdefault(cid, {"conversation_id": cid, "title": "", "message_count": 0, "total_tokens": 0, "last_active": ""})
        c["message_count"] += 1
        c["total_tokens"] += int(m.get("tokens_input") or 0) + int(m.get("tokens_output") or 0)
        ts = m.get("CreatedAt") or ""
        if ts > c["last_active"]:
            c["last_active"] = ts
    top_conversations = sorted(convos.values(), key=lambda x: x["message_count"], reverse=True)[:10]
    if top_conversations:
        try:
            conv_rows = db._get("conversations", params={
                "where": "~or".join(f"(Id,eq,{c['conversation_id']})" for c in top_conversations),
                "limit": 10,
            }).get("list", [])
            title_map = {r["Id"]: r.get("title") or "" for r in conv_rows}
            for c in top_conversations:
                c["title"] = title_map.get(c["conversation_id"], "")
        except Exception:
            pass

    # --- agent_runs ---
    successful_runs = [r for r in runs if r.get("status") == "complete"]
    by_agent: dict = {}
    for r in runs:
        name = r.get("agent_name") or "unknown"
        a = by_agent.setdefault(name, {"agent_name": name, "runs": 0, "successful": 0, "total_steps": 0})
        a["runs"] += 1
        if r.get("status") == "complete":
            a["successful"] += 1
        a["total_steps"] += int(r.get("steps") or 0)
    agent_runs_section = {
        "total_runs": len(runs),
        "successful": len(successful_runs),
        "failed": len(failed_runs),
        "avg_steps": round(sum(int(r.get("steps") or 0) for r in runs) / max(len(runs), 1), 1),
        "by_agent": [
            {
                "agent_name": a["agent_name"],
                "runs": a["runs"],
                "success_rate": round(a["successful"] / max(a["runs"], 1), 3),
                "avg_steps": round(a["total_steps"] / max(a["runs"], 1), 1),
            }
            for a in sorted(by_agent.values(), key=lambda x: x["runs"], reverse=True)
        ],
    }

    # --- enrichment ---
    enrichment_stats = {
        "total_cycles": 0, "total_sources_scraped": 0, "total_tokens_used": 0,
        "suggestions_generated": 0, "suggestions_approved": 0,
    }
    try:
        from workers.enrichment_agent import EnrichmentDB
        edb = EnrichmentDB()
        logs = edb.list_log(org_id=org_id, limit=5000)
        enrichment_stats["total_cycles"] = sum(1 for l in logs if l.get("event_type") == "cycle_start")
        enrichment_stats["total_sources_scraped"] = sum(1 for l in logs if l.get("event_type") == "source_scraped")
        enrichment_stats["total_tokens_used"] = sum(int(l.get("tokens_used") or 0) for l in logs)
        suggestions = edb.list_suggestions(org_id)
        enrichment_stats["suggestions_generated"] = len(suggestions)
        enrichment_stats["suggestions_approved"] = sum(1 for s in suggestions if s.get("status") == "approved")
    except Exception:
        _log.debug("enrichment stats failed", exc_info=True)

    return {
        "total_requests": total_requests,
        "total_tokens_input": total_tokens_in,
        "total_tokens_output": total_tokens_out,
        "total_conversations": total_conversations,
        "total_errors": total_errors,
        "period_start": period_start,
        "period_end": period_end,
        "by_model": by_model_list,
        "by_day": by_day_list,
        "by_hour": by_hour_list,
        "by_style": by_style_list,
        "top_conversations": top_conversations,
        "agent_runs": agent_runs_section,
        "enrichment": enrichment_stats,
    }


@app.get("/graph/snapshot")
def graph_snapshot(org_id: int, limit: int = 20):
    from graph import get_graph
    try:
        g = get_graph(org_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        node_result = g.query(
            "MATCH (n) OPTIONAL MATCH (n)-[r]-() "
            "WITH n, labels(n)[0] AS lbl, count(r) AS deg "
            "RETURN n.name, lbl, deg ORDER BY deg DESC LIMIT $lim",
            {"lim": limit},
        )
    except Exception:
        node_result = type("R", (), {"result_set": []})()

    nodes = []
    for row in node_result.result_set:
        if not row or not row[0]:
            continue
        nodes.append({
            "id": row[0],
            "label": row[0],
            "type": (row[1] or "unknown").lower(),
            "degree": int(row[2] or 0),
        })

    node_ids = {n["id"] for n in nodes}
    edges = []
    if node_ids:
        try:
            edge_result = g.query(
                "MATCH (a)-[r]->(b) WHERE a.name IN $names OR b.name IN $names "
                "RETURN a.name, type(r), b.name LIMIT $lim",
                {"names": list(node_ids), "lim": limit * 5},
            )
            for row in edge_result.result_set:
                if row and len(row) == 3:
                    edges.append({"source": row[0], "target": row[2], "relation": row[1]})
        except Exception:
            _log.debug("edge query failed", exc_info=True)

    summary = {"total_nodes": 0, "total_edges": 0, "node_types": {}}
    try:
        count_result = g.query("MATCH (n) RETURN labels(n)[0] AS lbl, count(n) AS cnt")
        for row in count_result.result_set:
            lbl = (row[0] or "unknown").lower()
            cnt = int(row[1] or 0)
            summary["node_types"][lbl] = cnt
            summary["total_nodes"] += cnt
    except Exception:
        pass
    try:
        edge_count = g.query("MATCH ()-[r]->() RETURN count(r)")
        summary["total_edges"] = int(edge_count.result_set[0][0] or 0)
    except Exception:
        pass

    return {"nodes": nodes, "edges": edges, "summary": summary}


@app.get("/chroma/snapshot")
def chroma_snapshot(org_id: int):
    from memory import client
    try:
        cols = client.list_collections()
        prefix = f"org_{org_id}_"
        collections = []
        total_docs = 0
        for c in cols:
            if not c.name.startswith(prefix):
                continue
            count = c.count()
            collections.append({"name": c.name, "count": count})
            total_docs += count
        collections.sort(key=lambda x: x["name"])
        return {
            "collections": collections,
            "total_documents": total_docs,
            "total_collections": len(collections),
        }
    except Exception as e:
        _log.error("chroma snapshot failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3800, reload=True)