from contextlib import asynccontextmanager

from fastapi import FastAPI

import infra.log as log

log.setup()
_log = log.get("harness")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _log.info("mstag-harness starting")
    from scheduler import start_scheduler
    sched = start_scheduler()
    app.state.scheduler = sched
    _log.info("scheduler running")

    from workers.tool_queue import HandlerConfig, ToolJobQueue, _set_instance
    from tools.graph_extract import _handle_graph_extract
    from tools.planned_search.agent import run_planned_search_job
    from tools.research.agent import run_research_agent
    from tools.research.research_planner import run_research_planner_job
    tool_queue = ToolJobQueue()
    tool_queue.register("graph_extract", HandlerConfig(
        handler=_handle_graph_extract, max_workers=1, priority_default=5,
    ))
    tool_queue.register("research_planner", HandlerConfig(
        handler=lambda p: run_research_planner_job(p["plan_id"]),
        max_workers=1, priority_default=3, source="research_planner",
    ))
    tool_queue.register("research_agent", HandlerConfig(
        handler=lambda p: run_research_agent(p["plan_id"]),
        max_workers=1, priority_default=3, source="research_agent",
    ))
    tool_queue.register("planned_search_execute", HandlerConfig(
        handler=lambda p: run_planned_search_job(p["message_id"], p["org_id"]),
        max_workers=1, priority_default=2, source="planned_search",
    ))
    _set_instance(tool_queue)
    app.state.tool_queue = tool_queue
    tool_queue.start()
    _log.info("tool job queue running")

    _log.info("ready")
    try:
        yield
    finally:
        tool_queue.stop()
        sched.shutdown(wait=False)
        _log.info("shutdown complete")
