"""Read-only view over APScheduler state joined with `agent_schedules` rows.

Used by the home dashboard's schedules widget: for each active cron-registered
agent, returns the next-run time plus the task description and product
metadata from NocoDB.
"""
from __future__ import annotations

import logging
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler

from infra.nocodb_client import NocodbClient
from scheduler import AGENT_JOB_PREFIX
from tools._org import resolve_org_id

_log = logging.getLogger("home.schedules")


def _index_by_id(rows: list[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for r in rows:
        rid = r.get("Id")
        if rid is not None:
            out[int(rid)] = r
    return out


def get_next_runs(sched: BackgroundScheduler | None, org_id: int | None = None) -> list[dict]:
    """Return one row per scheduled agent with its next_run_time."""
    if sched is None or not getattr(sched, "running", False):
        return []

    try:
        client = NocodbClient()
        if "agent_schedules" in client.tables:
            rows = client._get_paginated("agent_schedules", params={"limit": 500})
        else:
            rows = []
    except Exception:
        _log.warning("agent_schedules fetch failed", exc_info=True)
        rows = []
    meta_by_id = _index_by_id(rows)

    out: list[dict[str, Any]] = []
    for job in sched.get_jobs():
        if not job.id.startswith(AGENT_JOB_PREFIX):
            continue
        try:
            row_id = int(job.id[len(AGENT_JOB_PREFIX):])
        except ValueError:
            continue
        meta = meta_by_id.get(row_id, {})
        meta_org = resolve_org_id(meta.get("org_id"))
        if org_id is not None and meta_org != int(org_id):
            continue
        out.append({
            "id": row_id,
            "agent_name": meta.get("agent_name") or "",
            "task_description": meta.get("task_description") or "",
            "product": meta.get("product") or "",
            "cron_expression": meta.get("cron_expression") or "",
            "timezone": meta.get("timezone") or "Australia/Sydney",
            "org_id": meta_org,
            "active": bool(meta.get("active")),
            "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
        })
    out.sort(key=lambda r: (r["next_run_time"] or "9999", r["agent_name"]))
    return out


def get_schedule_meta(schedule_id: int) -> dict | None:
    client = NocodbClient()
    if "agent_schedules" not in client.tables:
        return None
    try:
        data = client._get("agent_schedules", params={
            "where": f"(Id,eq,{schedule_id})",
            "limit": 1,
        })
    except Exception:
        _log.warning("agent_schedules get failed  id=%d", schedule_id, exc_info=True)
        return None
    rows = data.get("list", [])
    return rows[0] if rows else None
