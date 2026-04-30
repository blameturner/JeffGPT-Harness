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
    from tools.research.agent import run_research_agent, review_research_paper
    from tools.research.research_planner import run_research_planner_job
    from tools.enrichment.scraper import scrape_page_job
    from tools.enrichment.summariser import summarise_page_job
    from tools.enrichment.pathfinder import pathfinder_extract_job
    from tools.enrichment.discover_agent import discover_agent_job
    from tools.enrichment.relationships_extractor import extract_relationships_job
    from tools.digest.agent import daily_digest_job
    from tools.seed_feedback.agent import seed_feedback_job
    from tools.corpus_maintenance.agent import corpus_maintenance_job
    from tools.insight.agent import insight_produce_job
    from tools.graph_maintenance.agent import (
        graph_maintenance_job,
        graph_resolve_entities_job,
    )
    from tools.pa.background import pa_topic_research_job
    from tools.simulation.agent import run_simulation_job
    tool_queue = ToolJobQueue()
    # Priority tiers (lower = picked first):
    #   3 = graph_extract, research planner + agent
    #   4 = scrape_page, pathfinder_extract, summarise_page
    #   5 = discover_agent_run, extract_relationships (background)
    tool_queue.register("research_planner", HandlerConfig(
        handler=lambda p: run_research_planner_job(p["plan_id"]),
        max_workers=1, priority_default=3, source="research_planner",
    ))
    tool_queue.register("research_agent", HandlerConfig(
        handler=lambda p: run_research_agent(p["plan_id"]),
        max_workers=1, priority_default=3, source="research_agent",
    ))
    tool_queue.register("research_review", HandlerConfig(
        handler=lambda p: review_research_paper(p["plan_id"], p.get("instructions", "")),
        max_workers=1, priority_default=3, source="research_review",
    ))
    from tools.research.operations import run_research_op
    # Allow two ops in flight (e.g. user fires fact_check + slide_deck) without
    # one blocking the other.
    tool_queue.register("research_op", HandlerConfig(
        handler=run_research_op,
        max_workers=2, priority_default=3, source="research_op",
    ))

    # Harvest pipeline — generic scraper/pathfinder-driven jobs.
    # Importing the package self-registers all policies in tools.harvest.REGISTRY.
    # max_workers=1 because each harvest already drives a long sequential
    # per-URL LLM loop on a single local CPU model; running 2 concurrently
    # just thrashes the model pool and starves chat / cron jobs (research,
    # daily_digest, pa_topic_research) that share the same LLM slot.
    from tools.harvest import run_harvest
    tool_queue.register("harvest_run", HandlerConfig(
        handler=lambda p: run_harvest(p["run_id"]),
        max_workers=1, priority_default=4, source="harvest",
    ))
    tool_queue.register("summarise_page", HandlerConfig(
        handler=summarise_page_job,
        max_workers=1, priority_default=4, source="summariser",
    ))
    tool_queue.register("graph_extract", HandlerConfig(
        handler=_handle_graph_extract, max_workers=1, priority_default=3,
    ))
    tool_queue.register("scrape_page", HandlerConfig(
        handler=scrape_page_job,
        max_workers=1, priority_default=4, source="scraper",
    ))
    tool_queue.register("pathfinder_extract", HandlerConfig(
        handler=pathfinder_extract_job,
        max_workers=1, priority_default=4, source="pathfinder",
    ))
    tool_queue.register("extract_relationships", HandlerConfig(
        handler=extract_relationships_job,
        max_workers=1, priority_default=5, source="relationships",
    ))
    tool_queue.register("discover_agent_run", HandlerConfig(
        handler=discover_agent_job,
        max_workers=1, priority_default=5, source="discover_agent",
    ))
    tool_queue.register("daily_digest", HandlerConfig(
        handler=daily_digest_job,
        max_workers=1, priority_default=5, source="daily_digest",
    ))
    tool_queue.register("seed_feedback", HandlerConfig(
        handler=seed_feedback_job,
        max_workers=1, priority_default=5, source="seed_feedback",
    ))
    tool_queue.register("corpus_maintenance", HandlerConfig(
        handler=corpus_maintenance_job,
        max_workers=1, priority_default=5, source="corpus_maintenance",
    ))
    tool_queue.register("insight_produce", HandlerConfig(
        handler=insight_produce_job,
        max_workers=1, priority_default=5, source="insight",
    ))
    tool_queue.register("graph_resolve_entities", HandlerConfig(
        handler=graph_resolve_entities_job,
        max_workers=1, priority_default=5, source="graph_maintenance",
    ))
    tool_queue.register("graph_maintenance", HandlerConfig(
        handler=graph_maintenance_job,
        max_workers=1, priority_default=5, source="graph_maintenance",
    ))
    tool_queue.register("pa_topic_research", HandlerConfig(
        handler=pa_topic_research_job,
        max_workers=1, priority_default=5, source="pa",
    ))
    tool_queue.register("simulation_run", HandlerConfig(
        handler=run_simulation_job,
        max_workers=1, priority_default=4, source="simulation",
    ))
    _set_instance(tool_queue)
    app.state.tool_queue = tool_queue
    tool_queue.start()
    _log.info("tool job queue running")

    # Periodic dispatchers. Each one enqueues at most one job per tick and is
    # guarded by an inflight check. The single chat-idle gate in tool_queue
    # decides when a job actually runs — no startup delay needed here.
    try:
        from infra.config import get_feature
        from tools.enrichment.dispatcher import (
            jumpstart_discover_agent,
            jumpstart_pathfinder,
            jumpstart_scraper,
        )
        from tools.digest.dispatcher import jumpstart_daily_digest
        from tools.seed_feedback.dispatcher import jumpstart_seed_feedback
        from tools.corpus_maintenance.dispatcher import jumpstart_corpus_maintenance
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger

        scrape_seconds = int(get_feature("scraper", "dispatch_interval_seconds", 60))
        sched.add_job(
            jumpstart_scraper,
            IntervalTrigger(seconds=max(15, scrape_seconds)),
            id="enrichment_scrape_dispatcher",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        pathfinder_seconds = int(get_feature("pathfinder", "dispatch_interval_seconds", 120))
        sched.add_job(
            jumpstart_pathfinder,
            IntervalTrigger(seconds=max(30, pathfinder_seconds)),
            id="pathfinder_dispatcher",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        discover_minutes = int(get_feature("discover_agent", "run_interval_minutes", 20))
        sched.add_job(
            jumpstart_discover_agent,
            IntervalTrigger(minutes=max(1, discover_minutes)),
            id="discover_agent_dispatcher",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        _log.info(
            "enrichment dispatchers scheduled  scrape=%ds pathfinder=%ds discover=%dm",
            scrape_seconds, pathfinder_seconds, discover_minutes,
        )

        if get_feature("daily_digest", "enabled", True):
            digest_hour = int(get_feature("daily_digest", "cron_hour", 7))
            digest_minute = int(get_feature("daily_digest", "cron_minute", 0))
            sched.add_job(
                jumpstart_daily_digest,
                CronTrigger(hour=digest_hour, minute=digest_minute),
                id="daily_digest_dispatcher",
                max_instances=1,
                coalesce=True,
                replace_existing=True,
            )
            _log.info("daily_digest dispatcher scheduled  %02d:%02d UTC", digest_hour, digest_minute)

        from tools.research.research_planner import reap_stale_plans
        reap_minutes = int(get_feature("research", "reap_interval_minutes", 30) or 30)
        sched.add_job(
            reap_stale_plans,
            IntervalTrigger(minutes=max(5, reap_minutes)),
            id="research_plan_reaper",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        _log.info("research plan reaper scheduled  every=%dm", reap_minutes)

        if get_feature("seed_feedback", "enabled", True):
            seed_hours = int(get_feature("seed_feedback", "run_interval_hours", 6))
            sched.add_job(
                jumpstart_seed_feedback,
                IntervalTrigger(hours=max(1, seed_hours)),
                id="seed_feedback_dispatcher",
                max_instances=1,
                coalesce=True,
                replace_existing=True,
            )
            _log.info("seed_feedback dispatcher scheduled  every=%dh", seed_hours)

        if get_feature("graph_maintenance", "enabled", True):
            er_hours = int(get_feature("graph_maintenance", "entity_resolution_interval_hours", 24))
            gm_hours = int(get_feature("graph_maintenance", "maintenance_interval_hours", 168))
            from tools.graph_maintenance.dispatcher import (
                jumpstart_entity_resolution,
                jumpstart_graph_maintenance,
            )
            sched.add_job(
                jumpstart_entity_resolution,
                IntervalTrigger(hours=max(1, er_hours)),
                id="graph_entity_resolution_dispatcher",
                max_instances=1, coalesce=True, replace_existing=True,
            )
            sched.add_job(
                jumpstart_graph_maintenance,
                IntervalTrigger(hours=max(1, gm_hours)),
                id="graph_maintenance_dispatcher",
                max_instances=1, coalesce=True, replace_existing=True,
            )
            _log.info("graph maintenance scheduled  entity_res=%dh maintenance=%dh",
                      er_hours, gm_hours)

        if get_feature("insights", "enabled", True):
            insight_tick_minutes = int(get_feature("insights", "tick_minutes", 10))
            from tools.insight.dispatcher import jumpstart_insights
            sched.add_job(
                jumpstart_insights,
                IntervalTrigger(minutes=max(1, insight_tick_minutes)),
                id="insight_dispatcher",
                max_instances=1,
                coalesce=True,
                replace_existing=True,
            )
            _log.info("insight dispatcher scheduled  tick=%dm", insight_tick_minutes)

        if get_feature("corpus_maintenance", "enabled", True):
            maint_hours = int(get_feature("corpus_maintenance", "run_interval_hours", 12))
            sched.add_job(
                jumpstart_corpus_maintenance,
                IntervalTrigger(hours=max(1, maint_hours)),
                id="corpus_maintenance_dispatcher",
                max_instances=1,
                coalesce=True,
                replace_existing=True,
            )
            _log.info("corpus_maintenance dispatcher scheduled  every=%dh", maint_hours)
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
