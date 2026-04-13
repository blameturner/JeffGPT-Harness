from __future__ import annotations

import logging
import threading
from typing import Any

import requests

_log = logging.getLogger("scheduler")
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from config import NOCODB_BASE_ID, NOCODB_TOKEN, NOCODB_URL
from workers.enrichment.cycle import run_log_cleanup, seed_enrichment_jobs
from workers.enrichment.db import EnrichmentDB

_scheduler: BackgroundScheduler | None = None
AGENT_JOB_PREFIX = "agent_schedule_"


def _fetch_agent_schedules() -> list[dict]:
    _log.debug("fetching agent_schedules from nocodb")
    try:
        meta = requests.get(
            f"{NOCODB_URL}/api/v1/db/meta/projects/{NOCODB_BASE_ID}/tables",
            headers={"xc-token": NOCODB_TOKEN},
            timeout=10,
        )
        meta.raise_for_status()
        tables = {t["title"]: t["id"] for t in meta.json()["list"]}
        if "agent_schedules" not in tables:
            return []
        data = requests.get(
            f"{NOCODB_URL}/api/v1/db/data/noco/{NOCODB_BASE_ID}/{tables['agent_schedules']}",
            headers={"xc-token": NOCODB_TOKEN},
            params={"where": "(active,eq,1)", "limit": 500},
            timeout=10,
        )
        data.raise_for_status()
        return data.json().get("list", [])
    except Exception as e:
        _log.error("fetch agent_schedules failed: %s", e)
        return []


def _run_agent_job(agent_name: str, org_id: int, task: str, product: str) -> None:
    def _call() -> None:
        _log.info("scheduled agent firing  agent=%s org=%d", agent_name, org_id)
        try:
            r = requests.post(
                "http://localhost:3800/run",
                json={
                    "agent_name": agent_name,
                    "org_id": org_id,
                    "task": task,
                    "product": product,
                },
                timeout=600,
            )
            if r.status_code >= 400:
                _log.error("scheduled agent %s failed: %d %s", agent_name, r.status_code, r.text[:200])
            else:
                _log.info("scheduled agent %s completed: %d", agent_name, r.status_code)
        except Exception as e:
            _log.error("scheduled agent %s error: %s", agent_name, e, exc_info=True)

    threading.Thread(target=_call, daemon=True).start()


def _register_agent_schedules(sched: BackgroundScheduler) -> int:
    # Drop previously-registered agent jobs so reload is idempotent.
    for job in list(sched.get_jobs()):
        if job.id.startswith(AGENT_JOB_PREFIX):
            sched.remove_job(job.id)

    count = 0
    for row in _fetch_agent_schedules():
        try:
            cron_expr = (row.get("cron_expression") or "").strip()
            agent_name = row.get("agent_name")
            org_id = int(row.get("org_id") or 0)
            if not cron_expr or not agent_name or not org_id:
                continue
            tz = row.get("timezone") or "Australia/Sydney"
            trigger = CronTrigger.from_crontab(cron_expr, timezone=tz)
            sched.add_job(
                _run_agent_job,
                trigger,
                id=f"{AGENT_JOB_PREFIX}{row['Id']}",
                args=[
                    agent_name,
                    org_id,
                    row.get("task_description") or "",
                    row.get("product") or "",
                ],
                max_instances=1,
                coalesce=True,
                replace_existing=True,
            )
            count += 1
        except Exception as e:
            _log.warning("agent_schedule row %s invalid: %s", row.get("Id"), e)
    return count


ENRICHMENT_JOB_PREFIX = "enrichment_agent_"


def _register_enrichment_agents(sched: BackgroundScheduler) -> int:
    """Register enrichment agents as queue seeders.

    Each enrichment agent's cron fires ``seed_enrichment_jobs(agent_id)``
    which creates individual scrape→summarise jobs in the tool_jobs queue.
    The queue's backoff system ensures no model contention with active
    chat/code sessions.
    """
    for job in list(sched.get_jobs()):
        if job.id.startswith(ENRICHMENT_JOB_PREFIX):
            sched.remove_job(job.id)
    count = 0
    try:
        db = EnrichmentDB()
        agents = db.list_enrichment_agents()
    except Exception:
        _log.warning("could not load enrichment_agents table, skipping")
        return 0
    for agent in agents:
        try:
            cron = (agent.get("cron_expression") or "").strip()
            agent_id = agent["Id"]
            if not cron:
                continue
            tz = agent.get("timezone") or "Australia/Sydney"
            trigger = CronTrigger.from_crontab(cron, timezone=tz)
            sched.add_job(
                seed_enrichment_jobs,
                trigger,
                id=f"{ENRICHMENT_JOB_PREFIX}{agent_id}",
                args=[agent_id],
                max_instances=1,
                coalesce=True,
                replace_existing=True,
            )
            count += 1
        except Exception:
            _log.warning("enrichment_agent row %s invalid", agent.get("Id"), exc_info=True)
    return count


def start_scheduler() -> BackgroundScheduler:
    global _scheduler
    sched = BackgroundScheduler(timezone="UTC")

    # Log cleanup is pure DB maintenance — no model calls, safe to keep.
    sched.add_job(
        run_log_cleanup,
        CronTrigger(hour=2, minute=0),
        id="enrichment_log_cleanup",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )

    # NOTE: No general enrichment cycle.  Only agent-specific cycles are
    # registered below.  Each fires seed_enrichment_jobs() which creates
    # tool_jobs entries — the queue handles the actual work.
    #
    # NOTE: batch_summarise is retired.  Per-turn RWKV summarisation in
    # workers/chat/history.py handles conversation compression inline.

    sched.start()
    registered = _register_agent_schedules(sched)
    enrichment_registered = _register_enrichment_agents(sched)
    _log.info("registered %d agent schedules, %d enrichment agents", registered, enrichment_registered)
    _scheduler = sched
    return sched


def reload_agent_schedules() -> dict[str, Any]:
    _log.info("reloading agent schedules")
    if _scheduler is None:
        _log.warning("reload requested but scheduler not running")
        return {"ok": False, "error": "scheduler not running"}
    agent_count = _register_agent_schedules(_scheduler)
    enrichment_count = _register_enrichment_agents(_scheduler)
    _log.info("reload complete  agents=%d enrichment=%d", agent_count, enrichment_count)
    return {"ok": True, "agent_schedules": agent_count, "enrichment_agents": enrichment_count}
