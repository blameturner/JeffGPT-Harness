"""Cron trigger — fires assignments instead of POSTing to /run.

Drop-in replacement for the cron callback in scheduler.py. The original
_run_agent_job hits HTTP /run; this writes an `assignments` row and lets the
runtime pick it up.
"""
from __future__ import annotations

import logging

from infra.nocodb_client import NocodbClient
from tools._org import resolve_org_id
from workers.user_agents.runtime import enqueue_assignment

_log = logging.getLogger("agents.triggers.cron")


def fire_scheduled_assignment(agent_name: str, org_id: int, task: str, product: str = "") -> dict | None:
    """Resolve agent_name+org → agent_id, enqueue an assignment.

    Returns the inserted row, or None if the agent isn't found.
    """
    client = NocodbClient()
    org_id = resolve_org_id(org_id)
    rows = client._get_paginated("agents", params={
        "where": f"(name,eq,{agent_name})~and(org_id,eq,{org_id})",
        "limit": 1,
    })
    if not rows:
        _log.warning("scheduled fire: agent not found  name=%s org=%d", agent_name, org_id)
        return None
    agent = rows[0]
    if agent.get("active") is False:
        _log.info("scheduled fire skipped (inactive)  name=%s", agent_name)
        return None
    return enqueue_assignment(
        agent_id=int(agent["Id"]),
        task=task or agent.get("brief") or "",
        source="cron",
        source_meta={"product": product} if product else {},
        org_id=org_id,
        client=client,
    )
