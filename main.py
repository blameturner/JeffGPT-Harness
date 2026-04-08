from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agents.generator_agent import GeneratorAgent
from agents.chat_agent import ChatAgent
from config import MODELS, refresh_models
from nocodb_client import NocodbClient
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("mstag-harness starting...")
    print("ready")
    yield

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
async def run_agent(request: RunRequest):
    try:
        agent = GeneratorAgent(request.agent_name, request.org_id)
        result = agent.run(request.task, request.product)

        if result is None:
            raise HTTPException(
                status_code=500,
                detail="Agent ran but failed to produce output"
            )

        return {
            "success": True,
            "agent": request.agent_name,
            "org_id": request.org_id,
            "product": request.product,
            "output": result.model_dump()
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    org_id: int
    model: str
    message: str
    conversation_id: int | None = None
    system: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024
    # Only honoured on the first message of a conversation (when
    # conversation_id is None). Subsequent turns inherit the setting
    # from the conversation row in NocoDB.
    rag_enabled: bool | None = None
    rag_collection: str | None = None


@app.get("/models")
async def list_models():
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


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        agent = ChatAgent(model=request.model, org_id=request.org_id)
        result = agent.send(
            user_message=request.message,
            conversation_id=request.conversation_id,
            system=request.system,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            rag_enabled=request.rag_enabled,
            rag_collection=request.rag_collection,
        )
        return {
            "success": True,
            "conversation_id": result.conversation_id,
            "model": result.model,
            "output": result.output,
            "tokens_input": result.tokens_input,
            "tokens_output": result.tokens_output,
            "duration_seconds": result.duration_seconds,
            "rag_enabled": result.rag_enabled,
            "context_chars": result.context_chars,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents")
async def list_agents(org_id: int, limit: int = 200):
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
async def list_conversations(org_id: int, limit: int = 50):
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


@app.get("/conversations/{conversation_id}/summary")
async def conversation_summary(conversation_id: int):
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

        # Token aggregates — from both messages and any linked agent runs
        msg_tokens_in = sum(int(m.get("tokens_input") or 0) for m in messages)
        msg_tokens_out = sum(int(m.get("tokens_output") or 0) for m in messages)
        run_tokens_in = sum(int(r.get("tokens_input") or 0) for r in runs)
        run_tokens_out = sum(int(r.get("tokens_output") or 0) for r in runs)
        run_context_tokens = sum(int(r.get("context_tokens") or 0) for r in runs)
        run_duration = sum(float(r.get("duration_seconds") or 0) for r in runs)

        tokens_input = msg_tokens_in + run_tokens_in
        tokens_output = msg_tokens_out + run_tokens_out

        # Message-level stats
        role_counts = _counter(messages, "role")
        user_msgs = [m for m in messages if m.get("role") == "user"]
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        chars_user = sum(len(m.get("content") or "") for m in user_msgs)
        chars_assistant = sum(len(m.get("content") or "") for m in assistant_msgs)

        timestamps = [m.get("CreatedAt") for m in messages if m.get("CreatedAt")]
        first_message_at = min(timestamps) if timestamps else None
        last_message_at = max(timestamps) if timestamps else None

        # Models actually used in this conversation
        models_used = _distinct_nonempty(messages, "model") or _distinct_nonempty(runs, "model_name")

        # Observation dimensions
        themes = _distinct_nonempty(observations, "domain")
        obs_types = _distinct_nonempty(observations, "type")
        obs_confidences = _counter(observations, "confidence")
        obs_statuses = _counter(observations, "status")
        theme_counts = _counter(observations, "domain")

        # Agent runs dimensions
        agents_used = _distinct_nonempty(runs, "agent_name")
        run_statuses = _counter(runs, "status")

        # Task dimensions
        task_statuses = _counter(tasks, "status")

        return {
            "conversation": convo,

            # Volume
            "message_count": len(messages),
            "role_counts": role_counts,
            "observation_count": len(observations),
            "run_count": len(runs),
            "output_count": len(outputs),
            "task_count": len(tasks),

            # Tokens
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

            # Timing
            "first_message_at": first_message_at,
            "last_message_at": last_message_at,
            "run_duration_seconds": round(run_duration, 2),

            # Content size
            "chars_user": chars_user,
            "chars_assistant": chars_assistant,

            # Models / agents
            "models_used": models_used,
            "agents_used": agents_used,

            # Themes & observations
            "themes": themes,
            "theme_counts": theme_counts,
            "observation_types": obs_types,
            "observation_confidences": obs_confidences,
            "observation_statuses": obs_statuses,

            # Runs & tasks
            "run_statuses": run_statuses,
            "task_statuses": task_statuses,

            # Full linked rows
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
async def get_messages(conversation_id: int):
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3800, reload=True)