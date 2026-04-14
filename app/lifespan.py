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
    tool_queue = ToolJobQueue()
    tool_queue.register("graph_extract", HandlerConfig(
        handler=_handle_graph_extract, max_workers=1, priority_default=5,
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
