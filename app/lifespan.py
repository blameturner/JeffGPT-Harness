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
    from tools.enrichment.scraper import scrape_target_job
    from tools.enrichment.summariser import summarise_page_job
    from tools.enrichment.pathfinder import pathfinder_crawl_job
    from tools.enrichment.classifier import classify_relevance_job
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
    tool_queue.register("scrape_target", HandlerConfig(
        # accepts payload dict directly; picks next due target when payload is empty {}
        handler=scrape_target_job,
        max_workers=2, priority_default=5, source="scraper",
    ))
    tool_queue.register("summarise_page", HandlerConfig(
        handler=summarise_page_job,
        max_workers=1, priority_default=5, source="summariser",
    ))
    tool_queue.register("pathfinder_crawl", HandlerConfig(
        # accepts payload dict directly; picks next discovery root when payload is empty {}
        handler=pathfinder_crawl_job,
        max_workers=1, priority_default=4, source="pathfinder",
    ))
    tool_queue.register("classify_relevance", HandlerConfig(
        handler=classify_relevance_job,
        max_workers=1, priority_default=5, source="classifier",
    ))
    _set_instance(tool_queue)
    app.state.tool_queue = tool_queue
    tool_queue.start()
    _log.info("tool job queue running")

    # periodic dispatchers: jumpstart each chain if it has fully drained.
    # scrape_target_job and pathfinder_crawl_job both self-chain, so we only need a
    # heartbeat to restart them when they go idle.
    try:
        from datetime import datetime, timedelta, timezone
        from infra.config import get_feature
        from tools.enrichment.dispatcher import jumpstart_scraper, jumpstart_pathfinder
        from apscheduler.triggers.interval import IntervalTrigger
        scrape_interval = int(get_feature("scraper", "dispatch_interval_minutes", 5))
        first_run = datetime.now(timezone.utc) + timedelta(seconds=15)
        sched.add_job(
            jumpstart_scraper,
            IntervalTrigger(minutes=scrape_interval),
            id="enrichment_scrape_dispatcher",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
            next_run_time=first_run,
        )
        recrawl_interval = int(get_feature("pathfinder", "recrawl_interval_minutes", 60))
        sched.add_job(
            jumpstart_pathfinder,
            IntervalTrigger(minutes=recrawl_interval),
            id="pathfinder_recrawl_dispatcher",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
            next_run_time=first_run,
        )
        _log.info("enrichment dispatchers scheduled  scrape=%dm  recrawl=%dm", scrape_interval, recrawl_interval)
    except Exception:
        _log.error("enrichment dispatcher registration failed", exc_info=True)

    _log.info("ready")
    try:
        yield
    finally:
        tool_queue.stop()
        sched.shutdown(wait=False)
        _log.info("shutdown complete")
