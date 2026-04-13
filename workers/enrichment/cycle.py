from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

from config import (
    CATEGORY_COLLECTIONS,
    ENRICHMENT_LOG_RETENTION_DAYS,
    ENRICHMENT_TOKEN_BUDGET,
    is_feature_enabled,
)
from workers.enrichment.db import EnrichmentDB

_log = logging.getLogger("enrichment_agent.cycle")

_last_runs: dict[str | None, dict] = {}


def seed_enrichment_jobs(enrichment_agent_id: int) -> dict:
    """Seed individual scrape+summarise jobs into the tool queue for an
    enrichment agent.  Each due source becomes a pipeline job (scrape →
    summarise) in the tool_jobs table.  The queue's backoff and priority
    system handles scheduling.

    Returns a summary dict with counts.
    """
    if not is_feature_enabled("enrichment"):
        _log.info("enrichment disabled via config, skipping seed")
        return {"seeded": 0, "skipped": 0, "error": "disabled"}

    t0 = time.time()
    cycle_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _log.info("seed_enrichment_jobs cycle=%s agent=%d", cycle_id, enrichment_agent_id)

    try:
        db = EnrichmentDB()
    except Exception:
        _log.error("DB init failed", exc_info=True)
        return {"seeded": 0, "error": "db_init_failed"}

    agent_config = db.get_enrichment_agent(enrichment_agent_id) or {}
    if not agent_config:
        _log.error("enrichment agent %d not found", enrichment_agent_id)
        return {"seeded": 0, "error": "agent_not_found"}

    from workers.tool_queue import get_tool_queue
    tq = get_tool_queue()
    if not tq:
        _log.error("tool queue not available for enrichment seeding")
        return {"seeded": 0, "error": "queue_unavailable"}

    db.log_event(cycle_id, "cycle_start", message=f"agent={enrichment_agent_id} (seeding)")

    seeded = 0
    skipped = 0

    try:
        orgs = db.list_orgs()
    except Exception as e:
        db.log_event(cycle_id, "org_list_failed", message=str(e)[:500])
        orgs = [{"Id": 1}]

    for org in orgs:
        org_id = int(org.get("Id") or org.get("id") or 1)
        if agent_config.get("org_id") and int(agent_config["org_id"]) != org_id:
            continue

        try:
            sources = db.list_due_sources(org_id, enrichment_agent_id=enrichment_agent_id)
        except Exception as e:
            db.log_event(cycle_id, "sources_query_failed", org_id=org_id, message=str(e)[:500])
            continue

        if not sources:
            db.log_event(cycle_id, "no_sources_due", org_id=org_id,
                         message="list_due_sources returned 0 rows")
            continue

        for source in sources:
            source_url = source.get("url") or ""
            target_id = source.get("Id")
            consecutive_fails = int(source.get("consecutive_failures") or 0)

            if consecutive_fails >= 5:
                _log.debug("skipping permanently failed source %s", source_url[:60])
                skipped += 1
                continue

            if not source_url:
                skipped += 1
                continue

            category = (source.get("category") or "documentation").lower()
            collection = CATEGORY_COLLECTIONS.get(category, "scraped_documentation")

            # Submit a scrape→summarise pipeline to the tool queue.
            # The scrape handler runs scrape_page(); the summarise handler
            # uses enrichment_summarise config and stores to ChromaDB.
            tq.submit_pipeline(
                url=source_url,
                org_id=org_id,
                collection=collection,
                source="enrichment",
                priority=4,
                metadata={
                    "title": source.get("name") or source_url,
                    "snippet": "",
                    "scrape_target_id": target_id,
                    "category": category,
                    "cycle_id": cycle_id,
                },
                summarise_function="enrichment_summarise",
            )
            seeded += 1
            _log.debug("seeded source %s url=%s", target_id, source_url[:60])

    elapsed = round(time.time() - t0, 1)
    _log.info("seed_enrichment_jobs done  agent=%d seeded=%d skipped=%d elapsed=%.1fs",
              enrichment_agent_id, seeded, skipped, elapsed)
    db.log_event(cycle_id, "cycle_end", message=f"seeded={seeded} skipped={skipped}")

    _last_runs[enrichment_agent_id] = {
        "cycle_id": cycle_id,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "seeded": seeded,
        "skipped": skipped,
    }

    return {"seeded": seeded, "skipped": skipped, "cycle_id": cycle_id}


def run_log_cleanup() -> None:
    try:
        db = EnrichmentDB()
        deleted = db.purge_old_logs(ENRICHMENT_LOG_RETENTION_DAYS)
        _log.info("log cleanup deleted %d rows", deleted)
    except Exception:
        _log.error("log cleanup failed", exc_info=True)


def get_last_run(enrichment_agent_id: int | None = None) -> dict | None:
    return _last_runs.get(enrichment_agent_id)


def sources_due_count(enrichment_agent_id: int | None = None) -> int:
    try:
        db = EnrichmentDB()
        total = 0
        for org in db.list_orgs():
            org_id = int(org.get("Id") or org.get("id") or 1)
            total += len(db.list_due_sources(org_id, enrichment_agent_id=enrichment_agent_id))
        return total
    except Exception:
        return 0
