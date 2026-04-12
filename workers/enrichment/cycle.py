from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

from config import (
    ENRICHMENT_LOG_RETENTION_DAYS,
    ENRICHMENT_TOKEN_BUDGET,
    PROACTIVE_BUDGET_THRESHOLD,
    is_feature_enabled,
)
from graph import get_sparse_concepts
from workers.crawler import apply_polite_delay
from workers.enrichment.db import EnrichmentDB
from workers.enrichment.proactive import _proactive_search
from workers.enrichment.processing import _process_source

_log = logging.getLogger("enrichment_agent.cycle")

_last_runs: dict[str | None, dict] = {}


def run_enrichment_cycle(enrichment_agent_id: int | None = None) -> None:
    if not is_feature_enabled("enrichment"):
        _log.info("enrichment disabled via config, skipping cycle")
        return

    cycle_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = f"agent={enrichment_agent_id}" if enrichment_agent_id else "all"
    _log.info("cycle %s starting (%s)", cycle_id, label)
    started = time.time()
    try:
        db = EnrichmentDB()
    except Exception:
        _log.error("DB init failed", exc_info=True)
        return

    agent_config: dict = {}
    if enrichment_agent_id is not None:
        agent_config = db.get_enrichment_agent(enrichment_agent_id) or {}
        if not agent_config:
            _log.error("enrichment agent %d not found", enrichment_agent_id)
            return

    token_budget = int(agent_config.get("token_budget") or ENRICHMENT_TOKEN_BUDGET)

    if db.has_running_inferences():
        _log.info("cycle %s deferred — active agent_runs", cycle_id)
        db.log_event(cycle_id, "deferred", message="active agent_runs")
        return

    # Defer if any chat jobs are currently active — don't compete for CPU.
    from workers.jobs import STORE
    active_jobs = sum(1 for j in STORE._jobs.values() if not j.done)
    if active_jobs > 0:
        _log.info("cycle %s deferred — %d active chat jobs", cycle_id, active_jobs)
        db.log_event(cycle_id, "deferred", message=f"{active_jobs} active chat jobs")
        return

    db.log_event(cycle_id, "cycle_start", message=label)
    tokens_used = 0

    try:
        orgs = db.list_orgs()
    except Exception as e:
        db.log_event(cycle_id, "org_list_failed", message=str(e)[:500])
        orgs = [{"Id": 1}]

    if not orgs:
        db.log_event(cycle_id, "no_orgs", message="list_orgs returned empty")

    for org in orgs:
        org_id = int(org.get("Id") or org.get("id") or 1)
        if enrichment_agent_id and agent_config.get("org_id") and int(agent_config["org_id"]) != org_id:
            continue
        try:
            sources = db.list_due_sources(org_id, enrichment_agent_id=enrichment_agent_id)
        except Exception as e:
            db.log_event(
                cycle_id, "sources_query_failed",
                org_id=org_id, message=str(e)[:500],
            )
            continue

        if not sources:
            db.log_event(
                cycle_id, "no_sources_due",
                org_id=org_id,
                message="list_due_sources returned 0 rows",
            )

        # sparse concepts computed ONCE per org per cycle, shared with expand_frontier + _proactive_search
        try:
            sparse_concepts = get_sparse_concepts(org_id, limit=10)
        except Exception:
            _log.debug("sparse concept fetch failed org=%s", org_id, exc_info=True)
            sparse_concepts = []
        if sparse_concepts:
            _log.info(
                "cycle %s org=%s sparse concepts: %s",
                cycle_id, org_id, ", ".join(sparse_concepts[:10]),
            )

        for source in sources:
            remaining = token_budget - tokens_used
            if remaining <= 0:
                db.log_event(cycle_id, "budget_exhausted", org_id=org_id)
                break

            source_url = source.get("url") or ""
            consecutive_fails = int(source.get("consecutive_failures") or 0)
            if consecutive_fails >= 5:
                _log.info("skipping permanently failed source %s (consecutive_failures=%d)", source_url[:60], consecutive_fails)
                continue

            try:
                domain = urlparse(source_url).netloc.lower()
            except Exception:
                domain = ""
            apply_polite_delay(domain)

            try:
                tokens_used += _process_source(
                    source, org_id, cycle_id, db, remaining,
                    sparse_concepts=sparse_concepts,
                )
            except Exception as e:
                db.log_event(
                    cycle_id, "source_error", org_id=org_id,
                    scrape_target_id=source.get("Id"),
                    source_url=source_url,
                    message=str(e)[:500],
                )

        remaining = token_budget - tokens_used
        if sources and remaining > PROACTIVE_BUDGET_THRESHOLD:
            try:
                tokens_used += _proactive_search(
                    org_id, cycle_id, db, remaining,
                    sparse_concepts=sparse_concepts,
                )
            except Exception:
                _log.error("proactive_search failed", exc_info=True)

    elapsed = round(time.time() - started, 1)
    _log.info("cycle %s done  tokens=%d %.1fs (%s)", cycle_id, tokens_used, elapsed, label)
    db.log_event(
        cycle_id, "cycle_end",
        tokens_used=tokens_used,
        duration_seconds=elapsed,
    )

    _last_runs[enrichment_agent_id] = {
        "cycle_id": cycle_id,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "tokens_used": tokens_used,
        "duration_seconds": elapsed,
    }


def run_log_cleanup() -> None:
    try:
        db = EnrichmentDB()
        deleted = db.purge_old_logs(ENRICHMENT_LOG_RETENTION_DAYS)
        _log.info("log cleanup deleted %d rows", deleted)
    except Exception as e:
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
