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


def jumpstart_scraper() -> dict:
    """Scheduler hook: ensure at least one scrape_target job is in flight.
    Each scrape_target_job self-chains to the next, so this just kicks off the loop
    if the chain has fully drained (or never started)."""
    if not is_feature_enabled("scraper"):
        return {"status": "disabled"}

    from workers.tool_queue import get_tool_queue
    tq = get_tool_queue()
    if not tq:
        return {"status": "no_queue"}

    client = NocodbClient()
    inflight = _count_inflight(client, "scrape_target")
    if inflight > 0:
        _log.debug("scraper jumpstart: chain already running  inflight=%d", inflight)
        return {"status": "already_running", "inflight": inflight}

    target_workers = int(get_feature("scraper", "parallel_chains", 1))
    queued = 0
    for _ in range(max(1, target_workers)):
        try:
            tq.submit("scrape_target", {}, source="scraper_jumpstart", priority=5)
            queued += 1
        except Exception:
            _log.warning("scraper jumpstart submit failed", exc_info=True)
    _log.info("scraper jumpstart queued=%d (chain was empty)", queued)
    return {"status": "kicked", "queued": queued}


def jumpstart_pathfinder() -> dict:
    """Scheduler hook: ensure pathfinder_crawl chain is running."""
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

    try:
        tq.submit("pathfinder_crawl", {}, source="pathfinder_jumpstart", priority=4)
    except Exception:
        _log.warning("pathfinder jumpstart submit failed", exc_info=True)
        return {"status": "submit_failed"}
    _log.info("pathfinder jumpstart kicked")
    return {"status": "kicked", "queued": 1}


# Back-compat shims for the old function names
def enqueue_due_scrape_targets() -> dict:
    return jumpstart_scraper()


def enqueue_due_pathfinder_recrawls() -> dict:
    return jumpstart_pathfinder()
