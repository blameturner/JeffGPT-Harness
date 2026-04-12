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
    _log.info("ready")
    try:
        yield
    finally:
        sched.shutdown(wait=False)
        _log.info("shutdown complete")
