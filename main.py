from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agents.generator_agent import GeneratorAgent
from agents.chat_agent import ChatAgent
from config import MODELS
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


@app.get("/models")
async def list_models():
    return {
        "models": [{"name": name, "url": url} for name, url in MODELS.items()]
    }


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
        )
        return {
            "success": True,
            "conversation_id": result.conversation_id,
            "model": result.model,
            "output": result.output,
            "tokens_input": result.tokens_input,
            "tokens_output": result.tokens_output,
            "duration_seconds": result.duration_seconds,
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