"""APScheduler ticks for graph maintenance jobs.

Lightweight — just enqueues onto the tool queue. Heavy lifting is in
``tools.graph_maintenance.agent``.
"""
from __future__ import annotations

import logging

from infra.config import is_feature_enabled
from tools._org import default_org_id, resolve_org_id

_log = logging.getLogger("graph.maintenance.dispatcher")


def _submit(job_type: str, org_id: int | None = None) -> dict:
    if not is_feature_enabled("graph_maintenance"):
        return {"status": "disabled"}
    try:
        from infra.nocodb_client import NocodbClient
        from workers.tool_queue import get_tool_queue
        tq = get_tool_queue()
    except Exception:
        _log.warning("graph maintenance dispatcher: imports failed", exc_info=True)
        return {"status": "no_queue"}
    if not tq:
        return {"status": "no_queue"}

    try:
        org = resolve_org_id(org_id or default_org_id(NocodbClient()))
    except Exception:
        _log.warning("graph maintenance: org resolution failed", exc_info=True)
        return {"status": "no_org"}

    try:
        jid = tq.submit(job_type, {"org_id": org}, source="graph_maintenance", org_id=org)
        _log.info("%s enqueued  org=%d job=%s", job_type, org, jid)
        return {"status": "queued", "job_id": jid, "org_id": org}
    except Exception as e:
        _log.warning("%s submit failed  err=%s", job_type, e, exc_info=True)
        return {"status": "submit_failed", "error": str(e)}


def jumpstart_entity_resolution(org_id: int | None = None) -> dict:
    return _submit("graph_resolve_entities", org_id)


def jumpstart_graph_maintenance(org_id: int | None = None) -> dict:
    return _submit("graph_maintenance", org_id)
