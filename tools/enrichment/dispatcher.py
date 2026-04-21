
from __future__ import annotations

import logging

from infra.config import (
    NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS,
    get_feature,
    is_feature_enabled,
)
from infra.nocodb_client import NocodbClient
from tools._org import count_inflight as _count_inflight, default_org_id as _default_org_id

_log = logging.getLogger("enrichment.dispatcher")


def jumpstart_scraper(org_id: int | None = None) -> dict:
    """Enqueue one `scrape_page` job for the oldest due scrape_target."""
    if not is_feature_enabled("scraper"):
        return {"status": "disabled"}

    from tools.enrichment.scraper import fetch_due_target
    from workers.tool_queue import get_tool_queue

    tq = get_tool_queue()
    if not tq:
        return {"status": "no_queue"}

    client = NocodbClient()
    inflight = _count_inflight(client, "scrape_page")
    if inflight > 0:
        return {"status": "already_running", "inflight": inflight}

    if org_id is not None:
        org_id = int(org_id)
    if not org_id:
        org_id = _default_org_id(client)
    if not org_id:
        return {"status": "no_org_context"}

    target = fetch_due_target(client, org_id=org_id)
    if not target:
        return {"status": "idle"}
    target_id = int(target.get("Id") or 0)
    if not target_id:
        return {"status": "idle"}

    try:
        job_id = tq.submit(
            "scrape_page",
            {"target_id": target_id, "org_id": org_id},
            source="scraper_jumpstart",
            priority=4,
            org_id=org_id,
        )
    except Exception:
        _log.warning("scraper jumpstart submit failed", exc_info=True)
        return {"status": "submit_failed"}
    _log.info("scraper jumpstart queued job=%s target_id=%d org_id=%d", job_id, target_id, org_id)
    return {"status": "kicked", "target_id": target_id, "org_id": org_id}


def _oldest_approved_suggestion(client: NocodbClient, org_id: int) -> dict | None:
    try:
        rows = client._get(NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS, params={
            "where": f"(status,eq,approved)~and(org_id,eq,{org_id})",
            "sort": "CreatedAt",
            "limit": 1,
        }).get("list", [])
        return rows[0] if rows else None
    except Exception:
        return None


def jumpstart_pathfinder(org_id: int | None = None) -> dict:
    """Enqueue one `pathfinder_extract` job for the oldest approved suggestion."""
    if not get_feature("pathfinder", "enabled", True):
        return {"status": "disabled"}

    from workers.tool_queue import get_tool_queue

    tq = get_tool_queue()
    if not tq:
        return {"status": "no_queue"}

    client = NocodbClient()
    inflight = _count_inflight(client, "pathfinder_extract")
    if inflight > 0:
        return {"status": "already_running", "inflight": inflight}

    if org_id is not None:
        org_id = int(org_id)
    if not org_id:
        org_id = _default_org_id(client)
    if not org_id:
        return {"status": "no_org_context"}

    row = _oldest_approved_suggestion(client, org_id)
    if not row:
        return {"status": "idle"}
    suggested_id = int(row.get("Id") or 0)
    if not suggested_id:
        return {"status": "idle"}

    try:
        job_id = tq.submit(
            "pathfinder_extract",
            {"suggested_id": suggested_id, "org_id": org_id},
            source="pathfinder_jumpstart",
            priority=4,
            org_id=org_id,
        )
    except Exception:
        _log.warning("pathfinder jumpstart submit failed", exc_info=True)
        return {"status": "submit_failed"}
    _log.info("pathfinder jumpstart queued job=%s suggested_id=%d", job_id, suggested_id)
    return {"status": "kicked", "suggested_id": suggested_id, "org_id": org_id}


def jumpstart_discover_agent(org_id: int | None = None) -> dict:
    """Enqueue one `discover_agent_run` job per tick if none inflight."""
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
            "discover_agent_run",
            {"org_id": org_id},
            source="discover_agent_jumpstart",
            priority=5,
            org_id=org_id,
        )
    except Exception:
        _log.warning("discover_agent jumpstart submit failed", exc_info=True)
        return {"status": "submit_failed"}
    _log.info("discover_agent jumpstart queued job=%s org_id=%d", job_id, org_id)
    return {"status": "kicked", "org_id": org_id}
