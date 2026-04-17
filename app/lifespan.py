from contextlib import asynccontextmanager

from fastapi import FastAPI

import infra.log as log

log.setup()
_log = log.get("harness")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _log.info("mstag-harness starting")
    from infra.config import HUEY_ENABLED
    from infra.huey_runtime import init_huey, shutdown_huey, start_huey_consumer

    if not HUEY_ENABLED:
        raise RuntimeError("HUEY_ENABLED is false; tool queue is configured for Huey-only execution")

    huey = init_huey()
    app.state.huey = huey
    # SqliteHuey is falsy when the queue is empty; only treat None as init failure.
    if huey is None:
        raise RuntimeError("Huey initialisation failed; refusing startup in Huey-only mode")
    started = start_huey_consumer()
    if not started:
        raise RuntimeError("Huey consumer failed to start; refusing startup in Huey-only mode")
    _log.info("huey runtime ready  consumer_started=%s", started)

    from scheduler import start_scheduler
    sched = start_scheduler()
    app.state.scheduler = sched
    _log.info("scheduler running")

    from workers.tool_queue import HandlerConfig, ToolJobQueue, _set_instance
    from tools.graph_extract import _handle_graph_extract
    from tools.planned_search.agent import run_planned_search_job, run_planned_search_scrape_job
    from tools.research.agent import run_research_agent
    from tools.research.research_planner import run_research_planner_job
    from tools.enrichment.scraper import scrape_target_job
    from tools.enrichment.summariser import summarise_page_job
    from tools.enrichment.pathfinder import pathfinder_crawl_job
    from tools.enrichment.classifier import classify_relevance_job
    from tools.enrichment.discover_agent import discover_agent_job
    tool_queue = ToolJobQueue()
    # Priority tiers (lower number = picked first, shorter chat-idle needed):
    #   2 = user-facing planned_search (runs first when user approves queries)
    #   3 = research planner + agent (user-initiated, LLM-heavy)
    #   4 = summarisers, classifier, graph_extract, scrape_target (downstream work
    #       on already-known URLs — runs ahead of pathfinder so pipelines drain)
    #   5 = pathfinder (exploratory/discovery — lowest priority, runs only when
    #       everything above is quiet)
    # scrape_target is deliberately one tier ABOVE pathfinder_crawl: we'd rather
    # finish scraping known-good targets than spend compute discovering new ones.
    tool_queue.register("planned_search_execute", HandlerConfig(
        handler=lambda p: run_planned_search_job(p["message_id"], p["org_id"]),
        max_workers=1, priority_default=2, source="planned_search",
    ))
    tool_queue.register("planned_search_scrape", HandlerConfig(
        handler=run_planned_search_scrape_job,
        max_workers=2, priority_default=2, source="planned_search",
    ))
    tool_queue.register("research_planner", HandlerConfig(
        handler=lambda p: run_research_planner_job(p["plan_id"]),
        max_workers=1, priority_default=3, source="research_planner",
    ))
    tool_queue.register("research_agent", HandlerConfig(
        handler=lambda p: run_research_agent(p["plan_id"]),
        max_workers=1, priority_default=3, source="research_agent",
    ))
    tool_queue.register("summarise_page", HandlerConfig(
        handler=summarise_page_job,
        max_workers=1, priority_default=4, source="summariser",
    ))
    tool_queue.register("classify_relevance", HandlerConfig(
        handler=classify_relevance_job,
        max_workers=1, priority_default=4, source="classifier",
    ))
    tool_queue.register("graph_extract", HandlerConfig(
        handler=_handle_graph_extract, max_workers=1, priority_default=4,
    ))
    tool_queue.register("pathfinder_crawl", HandlerConfig(
        # accepts payload dict directly; picks next discovery root when payload is empty {}
        handler=pathfinder_crawl_job,
        max_workers=1, priority_default=5, source="pathfinder",
    ))
    tool_queue.register("scrape_target", HandlerConfig(
        # accepts payload dict directly; picks next due target when payload is empty {}
        handler=scrape_target_job,
        max_workers=2, priority_default=4, source="scraper",
    ))
    tool_queue.register("discover_agent_run", HandlerConfig(
        # autonomous discovery worker: samples Chroma, generates web queries,
        # and feeds new root URLs into discovery for pathfinder to expand.
        handler=discover_agent_job,
        max_workers=1, priority_default=5, source="discover_agent",
    ))
    _set_instance(tool_queue)
    app.state.tool_queue = tool_queue
    tool_queue.start()
    _log.info("tool job queue running")

    # periodic dispatchers: submit one scrape_target and one pathfinder_crawl
    # per interval if none are inflight. Self-chaining has been removed from
    # both handlers (it caused the runaway-every-1.5s bug); these APScheduler
    # IntervalTriggers are now the sole drivers.
    try:
        from datetime import datetime, timedelta, timezone
        from infra.config import get_feature
        from tools.enrichment.dispatcher import jumpstart_scraper, jumpstart_pathfinder, jumpstart_discover_agent
        from apscheduler.triggers.interval import IntervalTrigger
        scrape_interval = int(get_feature("scraper", "dispatch_interval_minutes", 5))
        # Hold dispatchers for 10 minutes after startup so background queue work
        # cannot flood immediately on boot.
        first_run = datetime.now(timezone.utc) + timedelta(minutes=10)
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
        discover_interval = int(get_feature("discover_agent", "run_interval_minutes", 20))
        sched.add_job(
            jumpstart_discover_agent,
            IntervalTrigger(minutes=discover_interval),
            id="discover_agent_dispatcher",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
            next_run_time=first_run,
        )
        _log.info(
            "enrichment dispatchers scheduled  scrape=%dm  recrawl=%dm  discover=%dm",
            scrape_interval, recrawl_interval, discover_interval,
        )
    except Exception:
        _log.error("enrichment dispatcher registration failed", exc_info=True)

    _log.info("ready")
    try:
        yield
    finally:
        tool_queue.stop()
        shutdown_huey()
        sched.shutdown(wait=False)
        _log.info("shutdown complete")
