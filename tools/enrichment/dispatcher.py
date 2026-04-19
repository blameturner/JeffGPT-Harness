from __future__ import annotations

import logging

from infra.config import get_feature, is_feature_enabled
from infra.nocodb_client import NocodbClient

_log = logging.getLogger("enrichment.dispatcher")


def _count_inflight(client: NocodbClient, job_type: str) -> int:
    """Count tool_jobs of a given type with status queued/running.
    NocoDB v1's where parser doesn't reliably support nested-paren ~or, so we
    do two single-status queries and sum."""
    total = 0
    for status in ("queued", "running"):
        try:
            data = client._get("tool_jobs", params={
                "where": f"(type,eq,{job_type})~and(status,eq,{status})",
                "limit": 50,
            })
            total += len(data.get("list", []))
        except Exception:
            pass
    return total


def _default_org_id(client: NocodbClient) -> int | None:
    """Pick an active org_id for jumpstart tool_jobs so they're filterable in the UI.

    Returns None when no tenant context exists yet; callers should skip dispatch
    instead of submitting jobs with synthetic org ids.
    """
    for table in ("discovery", "scrape_targets"):
        try:
            rows = client._get(table, params={
                "limit": 1,
                "fields": "org_id",
                "sort": "-CreatedAt",
            }).get("list", [])
            if rows:
                org = int(rows[0].get("org_id") or 0)
                if org:
                    return org
        except Exception:
            continue
    return None


def jumpstart_scraper(org_id: int | None = None) -> dict:
    """Scheduler hook: every `scraper.dispatch_interval_minutes`, queue ONE
    scrape_target batch job — but only if the previous one has finished. The
    inflight check ensures the scheduler can't pile up runs if the scrape takes
    longer than the cron interval."""
    if not is_feature_enabled("scraper"):
        return {"status": "disabled"}

    from workers.tool_queue import get_tool_queue
    tq = get_tool_queue()
    if not tq:
        return {"status": "no_queue"}

    client = NocodbClient()
    inflight = _count_inflight(client, "scrape_target")
    if inflight > 0:
        return {"status": "already_running", "inflight": inflight}

    if org_id is not None:
        org_id = int(org_id)
    if not org_id:
        org_id = _default_org_id(client)
    if not org_id:
        return {"status": "no_org_context"}
    try:
        job_id = tq.submit("scrape_target", {}, source="scraper_jumpstart", priority=4, org_id=org_id)
    except Exception:
        _log.warning("scraper jumpstart submit failed", exc_info=True)
        return {"status": "submit_failed"}
    _log.info("scraper jumpstart queued job=%s org_id=%d", job_id, org_id)
    return {"status": "kicked", "queued": 1, "org_id": org_id}


def jumpstart_pathfinder(org_id: int | None = None) -> dict:
    """Scheduler hook: every `pathfinder.recrawl_interval_minutes`, ensure there is
    exactly ONE pathfinder_crawl in the queue. Skip if one is already inflight.
    This cron is now the SOLE driver of pathfinder_crawl — self-chaining was
    removed from the handler because it produced a runaway loop when chat was
    idle (no backoff gate tripped and jobs ran back-to-back every ~1.5 s)."""
    if not get_feature("pathfinder", "enabled", True):
        return {"status": "disabled"}

    from workers.tool_queue import get_tool_queue
    tq = get_tool_queue()
    if not tq:
        return {"status": "no_queue"}

    client = NocodbClient()
    inflight = _count_inflight(client, "pathfinder_crawl")
    if inflight > 0:
        return {"status": "already_running", "inflight": inflight}

    if org_id is not None:
        org_id = int(org_id)
    if not org_id:
        org_id = _default_org_id(client)
    if not org_id:
        return {"status": "no_org_context"}
    try:
        job_id = tq.submit("pathfinder_crawl", {}, source="pathfinder_jumpstart", priority=5, org_id=org_id)
    except Exception:
        _log.warning("pathfinder jumpstart submit failed", exc_info=True)
        return {"status": "submit_failed"}
    _log.info("pathfinder jumpstart queued job=%s org_id=%d", job_id, org_id)
    return {"status": "kicked", "queued": 1, "org_id": org_id}


# Back-compat shims for the old function names
def enqueue_due_scrape_targets() -> dict:
    return jumpstart_scraper()


def enqueue_due_pathfinder_recrawls() -> dict:
    return jumpstart_pathfinder()


def jumpstart_discover_agent(org_id: int | None = None) -> dict:
    """Scheduler hook: every ``discover_agent.run_interval_minutes``, queue ONE
    discover_agent_run job — but only if none is already inflight.  The agent
    has its own cooldown gate inside the handler so double-triggers are safe."""
    if not get_feature("discover_agent", "enabled", True):
        return {"status": "disabled"}

    from workers.tool_queue import get_tool_queue
    tq = get_tool_queue()
    if not tq:
        return {"status": "no_queue"}

    client = NocodbClient()
    inflight = _count_inflight(client, "discover_agent_run")
    if inflight > 0:
        return {"status": "already_running", "inflight": inflight}

    if org_id is not None:
        org_id = int(org_id)
    if not org_id:
        org_id = _default_org_id(client)
    if not org_id:
        return {"status": "no_org_context"}
    try:
        job_id = tq.submit(
            "discover_agent_run", {"org_id": org_id},
            source="discover_agent_jumpstart", priority=5, org_id=org_id,
        )
    except Exception:
        _log.warning("discover_agent jumpstart submit failed", exc_info=True)
        return {"status": "submit_failed"}
    _log.info("discover_agent jumpstart queued job=%s org_id=%d", job_id, org_id)
    return {"status": "kicked", "queued": 1, "org_id": org_id}

