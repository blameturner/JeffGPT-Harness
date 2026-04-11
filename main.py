from fastapi import FastAPI

from app.lifespan import lifespan
from app.routers import agents, chat, code, enrichment, health, stats

app = FastAPI(title="MSTAG Harness", version="1.0.0", lifespan=lifespan)

app.include_router(health.router)
app.include_router(agents.router)
app.include_router(chat.router)
app.include_router(code.router)
app.include_router(enrichment.router)
app.include_router(stats.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3800, reload=True)
