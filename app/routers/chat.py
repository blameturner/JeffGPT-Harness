import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.schemas import ConversationUpdate
from nocodb_client import NocodbClient
from workers.chat_agent import ChatAgent
from workers.jobs import STORE, run_in_background

_log = logging.getLogger("main.chat")

router = APIRouter()


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


@router.post("/chat")
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


@router.get("/collections")
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


@router.get("/conversations")
def list_conversations(org_id: int, limit: int = 50):
    try:
        db = NocodbClient()
        return {"conversations": db.list_conversations(org_id, limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/conversations/{conversation_id}")
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


@router.get("/conversations/{conversation_id}/summary")
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


@router.get("/conversations/{conversation_id}/messages")
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


@router.get("/messages/{message_id}/search-sources")
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
