from __future__ import annotations

import logging
import threading
from typing import Any

import requests

_log = logging.getLogger("scheduler")
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger


_scheduler: BackgroundScheduler | None = None
AGENT_JOB_PREFIX = "agent_schedule_"


def _fetch_agent_schedules() -> list[dict]:
    _log.debug("fetching agent_schedules from nocodb")
    try:
        from infra.nocodb_client import NocodbClient
        client = NocodbClient()
        if "agent_schedules" not in client.tables:
            return []
        return client._get_paginated("agent_schedules", params={
            "where": "(active,eq,1)",
            "limit": 500,
        })
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
    # clear existing agent jobs so reload is idempotent
    for job in list(sched.get_jobs()):
        if job.id.startswith(AGENT_JOB_PREFIX):
            sched.remove_job(job.id)

    count = 0
    for row in _fetch_agent_schedules():
        try:
            cron_expr = (row.get("cron_expression") or "").strip()
            agent_name = row.get("agent_name")
            from tools._org import resolve_org_id
            org_id = resolve_org_id(row.get("org_id"))
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


def run_pa_for_org(org_id: int, force: bool = False) -> dict:
    """Run one PA pass for a single org. Returns a status dict describing
    what happened — used by the scheduler tick and the manual
    ``POST /home/pa/run`` endpoint.

    When ``force=True`` the 4h proactive gap is bypassed (the frontend's
    "I'm here now" button). Per-kind cooldowns still apply so the user
    never sees the exact same move type twice in rapid succession.
    """
    out: dict = {"status": "ok", "org_id": int(org_id), "surfaced": False, "kind": "", "reason": ""}
    try:
        from infra.config import is_feature_enabled
        if not is_feature_enabled("pa"):
            out["status"] = "skipped"
            out["reason"] = "feature_disabled"
            return out
        if int(org_id) <= 0:
            out["status"] = "error"
            out["reason"] = "invalid_org_id"
            return out

        from infra.nocodb_client import NocodbClient
        from shared.pa.memory import decay_warm_topics, MOVE_MODE_PROACTIVE
        from shared.pa.picker import pick_proactive_move, record_surface
        from shared.home_conversation import get_or_create_home_conversation

        try:
            decay_warm_topics(int(org_id))
        except Exception:
            pass

        candidate = pick_proactive_move(int(org_id), ignore_global_gap=force)
        if not candidate:
            out["reason"] = "no_move"
            return out

        client = NocodbClient()
        try:
            convo = get_or_create_home_conversation(int(org_id))
        except Exception:
            out["status"] = "error"
            out["reason"] = "no_home_conversation"
            return out

        record_surface(int(org_id), candidate, mode=MOVE_MODE_PROACTIVE, question_id=None)

        text = candidate.text.strip()
        why = (candidate.why or "").strip()[:200]
        content = f"{text}\n\n_why: {why}_" if why else text
        try:
            msg = client.add_message(
                conversation_id=int(convo["Id"]),
                role="assistant",
                content=content,
                org_id=int(org_id),
                model="pa",
                tokens_input=0,
                tokens_output=0,
            )
        except Exception:
            _log.warning("pa surface write failed  org=%d", org_id, exc_info=True)
            out["status"] = "error"
            out["reason"] = "message_write_failed"
            return out

        out["surfaced"] = True
        out["kind"] = candidate.kind
        out["text"] = text
        out["why"] = why
        out["message_id"] = (msg or {}).get("Id")
        _log.info("pa surface  org=%d kind=%s force=%s", org_id, candidate.kind, force)
        return out
    except Exception as e:
        _log.warning("run_pa_for_org failed  org=%d", org_id, exc_info=True)
        out["status"] = "error"
        out["reason"] = type(e).__name__
        return out


def _pa_tick() -> None:
    """Periodic tick: enumerates orgs with a home conversation and calls
    ``run_pa_for_org`` for each. Failures are swallowed."""
    try:
        from infra.config import is_feature_enabled
        if not is_feature_enabled("pa"):
            return
        from infra.nocodb_client import NocodbClient
        client = NocodbClient()
        if "conversations" not in client.tables:
            return
        try:
            rows = client._get_paginated("conversations", params={
                "where": "(kind,eq,home)",
                "limit": 200,
            })
        except Exception:
            rows = []
        if not rows:
            try:
                rows = client._get_paginated("conversations", params={
                    "where": "(title,eq,Home — ongoing)",
                    "limit": 200,
                })
            except Exception:
                rows = []
        org_ids = sorted({int(r.get("org_id") or 0) for r in rows if r.get("org_id")})
        for org_id in org_ids:
            if org_id <= 0:
                continue
            run_pa_for_org(org_id, force=False)
    except Exception:
        _log.warning("pa tick failed", exc_info=True)


def start_scheduler() -> BackgroundScheduler:
    global _scheduler
    sched = BackgroundScheduler(timezone="UTC")
    sched.start()
    registered = _register_agent_schedules(sched)
    _log.info("registered %d agent schedules", registered)
    try:
        sched.add_job(
            _pa_tick,
            IntervalTrigger(minutes=20),
            id="pa_proactive_tick",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        _log.info("pa proactive tick scheduled (20 min interval)")
    except Exception:
        _log.warning("pa tick registration failed", exc_info=True)
    _scheduler = sched
    return sched


def trigger_agent_job(agent_name: str, org_id: int, task: str = "", product: str = "") -> None:
    """Public entry point for firing a scheduled agent immediately (off-schedule)."""
    _log.info("manual trigger  agent=%s org=%d", agent_name, org_id)
    _run_agent_job(agent_name, org_id, task, product)


def reload_agent_schedules() -> dict[str, Any]:
    _log.info("reloading agent schedules")
    if _scheduler is None:
        _log.warning("reload requested but scheduler not running")
        return {"ok": False, "error": "scheduler not running"}
    agent_count = _register_agent_schedules(_scheduler)
    _log.info("reload complete  agents=%d", agent_count)
    return {"ok": True, "agent_schedules": agent_count}
