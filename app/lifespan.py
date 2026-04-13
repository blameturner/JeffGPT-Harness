from contextlib import asynccontextmanager

from fastapi import FastAPI

import log

log.setup()
_log = log.get("harness")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _log.info("mstag-harness starting")
    from scheduler import start_scheduler
    sched = start_scheduler()
    app.state.scheduler = sched
    _log.info("scheduler running")
    from workers.queue import JobQueue
    job_queue = JobQueue()
    app.state.queue = job_queue
    job_queue.start()
    _log.info("job queue running")

    from workers.tool_queue import (
        HandlerConfig, ToolJobQueue, _handle_scrape, _handle_summarise,
        _set_instance,
    )
    tool_queue = ToolJobQueue()
    tool_queue.register("scrape", HandlerConfig(
        handler=_handle_scrape, max_workers=3, priority_default=3, dedup_key="url",
    ))
    tool_queue.register("summarise", HandlerConfig(
        handler=_handle_summarise, max_workers=1, priority_default=3,
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
