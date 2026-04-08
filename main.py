from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agents.generator_agent import GeneratorAgent
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3800, reload=True)