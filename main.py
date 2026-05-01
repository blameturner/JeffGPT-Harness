from fastapi import FastAPI

from app.lifespan import lifespan
from app.routers import admin, agents, agents_admin, chat, code, connectors, harvest, health, home, simulation, stats, tool_queue, enrichment, projects, gitea, projects_extra, projects_analysis, projects_ai
from services.browser.main import app as browser_app
from services.sandbox.main import app as sandbox_app

app = FastAPI(title="MSTAG Harness", version="1.0.0", lifespan=lifespan)

app.include_router(health.router)
app.include_router(agents.router)
app.include_router(agents_admin.router)
app.include_router(connectors.router)
app.include_router(chat.router)
app.include_router(code.router)
app.include_router(projects_extra.public_router)  # MUST come before projects.router (static paths > /projects/{project_id})
app.include_router(projects.router)
app.include_router(projects_extra.router)
app.include_router(projects_analysis.router)
app.include_router(projects_ai.router)
app.include_router(gitea.router)
app.include_router(gitea.projects_gitea)
app.include_router(home.router)
app.include_router(stats.router)
app.include_router(tool_queue.router)
app.include_router(enrichment.router, prefix="/enrichment", tags=["enrichment"])
app.include_router(harvest.router, tags=["harvest"])
app.include_router(simulation.router)
app.include_router(admin.router)

app.mount("/browser", browser_app)
app.mount("/sandbox", sandbox_app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3800, reload=True)
