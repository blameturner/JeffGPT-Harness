from fastapi import FastAPI

from app.lifespan import lifespan
from app.routers import agents, chat, code, health, stats, tool_queue
from services.browser.main import app as browser_app
from services.sandbox.main import app as sandbox_app

app = FastAPI(title="MSTAG Harness", version="1.0.0", lifespan=lifespan)

app.include_router(health.router)
app.include_router(agents.router)
app.include_router(chat.router)
app.include_router(code.router)
app.include_router(stats.router)
app.include_router(tool_queue.router)

app.mount("/browser", browser_app)
app.mount("/sandbox", sandbox_app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3800, reload=True)
