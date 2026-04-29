import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.schemas import ConversationUpdate, ChatMemoryItemCreate, ChatMemoryItemUpdate
from infra.nocodb_client import NocodbClient
from workers.chat.agent import ChatAgent
from shared.jobs import STORE, run_in_background

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
    response_style: str | None = None
    search_mode: Literal["disabled", "basic", "standard"] = "basic"
    search_consent_confirmed: bool = False


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
    _log.info("POST /chat  model=%s org=%d conv=%s search_mode=%s", request.model, request.org_id, request.conversation_id, request.search_mode)
    try:
        agent = ChatAgent(
            model=request.model,
            org_id=request.org_id,
            search_enabled=request.search_mode != "disabled",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    agent._search_mode = request.search_mode
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
        search_consent_confirmed=request.search_consent_confirmed,
        response_style=request.response_style,
    ))
    return {"job_id": job.id}


@router.get("/collections")
def list_collections(org_id: int | None = None):
    from infra.memory import client
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
def update_conversation(conversation_id: int, body: ConversationUpdate, org_id: int):
    try:
        db = NocodbClient()
        convo = db.get_conversation(conversation_id, org_id=org_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")
        updates: dict = {}
        if body.title is not None:
            updates["title"] = body.title.strip() or "Untitled"
        if body.contextual_grounding_enabled is not None:
            updates["contextual_grounding_enabled"] = bool(body.contextual_grounding_enabled)
        if body.system_note is not None:
            updates["system_note"] = body.system_note.strip()[:2000]
        if body.default_response_style is not None:
            from workers.chat.config import CHAT_STYLES, CHAT_DEFAULT_STYLE
            key = (body.default_response_style or "").strip().lower()
            updates["default_response_style"] = key if key in CHAT_STYLES else CHAT_DEFAULT_STYLE
        if body.polish_pass_default is not None:
            updates["polish_pass_default"] = bool(body.polish_pass_default)
        if body.strict_grounding_default is not None:
            updates["strict_grounding_default"] = bool(body.strict_grounding_default)
        if body.ask_back_default is not None:
            updates["ask_back_default"] = bool(body.ask_back_default)
        if body.memory_extract_every_n_turns is not None:
            updates["memory_extract_every_n_turns"] = max(0, min(50, int(body.memory_extract_every_n_turns)))
        if body.memory_token_budget is not None:
            updates["memory_token_budget"] = max(0, min(8000, int(body.memory_token_budget)))
        if body.saved_fragments_json is not None:
            import json as _json
            updates["saved_fragments_json"] = _json.dumps(body.saved_fragments_json)[:8000]
        if not updates:
            return convo
        return db.update_conversation(conversation_id, updates)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- chat memory endpoints --------------------------------------------------

@router.get("/conversations/{conversation_id}/memory")
def list_chat_memory(
    conversation_id: int,
    org_id: int,
    status: str | None = None,
    category: str | None = None,
    pinned_only: bool = False,
):
    """List structured memory items. Default returns all items grouped by status.

    Query params:
      status      — filter to one of `proposed`, `active`, `rejected`
      category    — filter to `fact`, `decision`, or `thread`
      pinned_only — return only pinned items
    """
    from workers.chat.memory import list_items
    try:
        db = NocodbClient()
        convo = db.get_conversation(conversation_id, org_id=org_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")
        items = list_items(
            conversation_id=conversation_id,
            org_id=org_id,
            status=status,
            category=category,
            pinned_only=pinned_only,
        )
        grouped = {"fact": [], "decision": [], "thread": []}
        proposed_count = 0
        for it in items:
            cat = it.get("category") or "fact"
            if cat in grouped:
                grouped[cat].append(it)
            if it.get("status") == "proposed":
                proposed_count += 1
        return {
            "items": items,
            "grouped": grouped,
            "counts": {
                "total": len(items),
                "facts": len(grouped["fact"]),
                "decisions": len(grouped["decision"]),
                "threads": len(grouped["thread"]),
                "proposed": proposed_count,
                "pinned": sum(1 for it in items if it.get("pinned")),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        _log.error("list_chat_memory failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations/{conversation_id}/memory")
def add_chat_memory(conversation_id: int, body: ChatMemoryItemCreate, org_id: int):
    from workers.chat.memory import add_item, CATEGORIES, STATUS_ACTIVE, STATUS_PROPOSED, STATUS_REJECTED
    try:
        db = NocodbClient()
        convo = db.get_conversation(conversation_id, org_id=org_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if body.category not in CATEGORIES:
            raise HTTPException(status_code=400, detail=f"category must be one of {CATEGORIES}")
        if body.status not in (STATUS_ACTIVE, STATUS_PROPOSED, STATUS_REJECTED):
            raise HTTPException(status_code=400, detail="invalid status")
        item = add_item(
            conversation_id=conversation_id,
            org_id=org_id,
            category=body.category,
            text=body.text,
            pinned=body.pinned,
            status=body.status,
            source_message_id=body.source_message_id,
            confidence=body.confidence,
        )
        if not item:
            raise HTTPException(status_code=500, detail="memory add failed")
        return {"item": item}
    except HTTPException:
        raise
    except Exception as e:
        _log.error("add_chat_memory failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/conversations/{conversation_id}/memory/{item_id}")
def update_chat_memory(conversation_id: int, item_id: int, body: ChatMemoryItemUpdate, org_id: int):
    from workers.chat.memory import update_item
    try:
        db = NocodbClient()
        convo = db.get_conversation(conversation_id, org_id=org_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")
        changes = body.model_dump(exclude_none=True)
        if not changes:
            raise HTTPException(status_code=400, detail="no fields to update")
        item = update_item(item_id, **changes)
        if not item:
            raise HTTPException(status_code=404, detail="memory item not found")
        return {"item": item}
    except HTTPException:
        raise
    except Exception as e:
        _log.error("update_chat_memory failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{conversation_id}/memory/{item_id}")
def delete_chat_memory(conversation_id: int, item_id: int, org_id: int):
    from workers.chat.memory import delete_item
    try:
        db = NocodbClient()
        convo = db.get_conversation(conversation_id, org_id=org_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")
        ok = delete_item(item_id)
        if not ok:
            raise HTTPException(status_code=500, detail="delete failed")
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        _log.error("delete_chat_memory failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations/{conversation_id}/memory/extract")
def trigger_memory_extract(conversation_id: int, org_id: int):
    """Manually trigger structured extraction over the current conversation.

    Useful when the user has just edited messages or wants a fresh sweep.
    Items are written as `status="proposed"` for review.
    """
    from workers.chat.memory import (
        extract_structured_delta,
        list_items,
        persist_extracted_delta,
    )
    try:
        db = NocodbClient()
        convo = db.get_conversation(conversation_id, org_id=org_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")
        messages = db.list_messages(conversation_id, org_id=org_id)
        older_text = ""
        for m in messages[-30:]:
            role = m.get("role") or "user"
            if role not in ("user", "assistant"):
                continue
            content = (m.get("content") or "")[:1500]
            older_text += f"{role}: {content}\n\n"
        if not older_text.strip():
            return {"persisted": 0, "note": "no content to extract from"}
        existing = list_items(conversation_id=conversation_id, org_id=org_id, limit=200)
        delta = extract_structured_delta(older_text, existing)
        persisted = 0
        if delta:
            persisted = persist_extracted_delta(
                conversation_id=conversation_id,
                org_id=org_id,
                delta=delta,
            )
        return {"persisted": persisted, "delta": delta or {}}
    except HTTPException:
        raise
    except Exception as e:
        _log.error("trigger_memory_extract failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}/summary")
def conversation_summary(conversation_id: int, org_id: int):
    try:
        db = NocodbClient()
        convo = db.get_conversation(conversation_id, org_id=org_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")

        messages = db.list_messages(conversation_id, org_id=org_id)
        observations = db.list_observations_for_conversation(conversation_id, org_id=org_id)
        runs = db.list_runs_for_conversation(conversation_id, org_id=org_id)
        outputs = db.list_outputs_for_conversation(conversation_id, org_id=org_id)
        tasks = db.list_tasks_for_conversation(conversation_id, org_id=org_id)

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
def get_messages(conversation_id: int, org_id: int):
    try:
        db = NocodbClient()
        convo = db.get_conversation(conversation_id, org_id=org_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")
        messages = db.list_messages(conversation_id, org_id=org_id)
        sources_by_msg: dict[int, list[dict]] = {}
        if any(int(m.get("search_source_count") or 0) > 0 for m in messages):
            all_sources = db.list_message_search_sources(conversation_id=conversation_id, org_id=org_id)
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
def get_message_search_sources(message_id: int, org_id: int):
    try:
        db = NocodbClient()
        sources = db.list_message_search_sources(message_id=message_id, org_id=org_id)
        for src in sources:
            src["index"] = (src.get("source_index") or 0) + 1
            src["used_in_answer"] = bool(src.get("used_in_answer"))
        return {"sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
