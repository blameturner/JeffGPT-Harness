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


def _default_org_id(client: NocodbClient) -> int:
    """Pick an active org_id for jumpstart tool_jobs so they're filterable in the
    UI. Falls back to 1 (the default tenant) if no data exists yet."""
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
    return 1


def jumpstart_scraper() -> dict:
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

    org_id = _default_org_id(client)
    try:
        job_id = tq.submit("scrape_target", {}, source="scraper_jumpstart", priority=4, org_id=org_id)
    except Exception:
        _log.warning("scraper jumpstart submit failed", exc_info=True)
        return {"status": "submit_failed"}
    _log.info("scraper jumpstart queued job=%s org_id=%d", job_id, org_id)
    return {"status": "kicked", "queued": 1, "org_id": org_id}


def jumpstart_pathfinder() -> dict:
    """Scheduler hook: every `pathfinder.recrawl_interval_minutes`, ensure there is
    exactly ONE pathfinder_crawl in the queue. Skip if one is already inflight
    (the handler self-chains, so normally there's always one queued — this cron is
    the safety net if the chain breaks)."""
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

    org_id = _default_org_id(client)
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
