"""Admin / control-plane API.

One coherent surface for the frontend to render:

  * GET   /admin/runtime              — per-subsystem live status (last run,
                                         in-flight, errors_24h, next scheduled,
                                         feature.enabled). Plus huey + queue.
  * GET   /admin/config               — sectioned feature config (allowlisted)
  * GET   /admin/config/{section}     — one section
  * PATCH /admin/config/{section}     — replace one section in config.json on
                                         disk; reloads PLATFORM in process
  * POST  /admin/trigger/{subsystem}  — uniform trigger surface; returns job_id

The map of subsystem → tool-queue job type lives in ``TRIGGER_MAP`` below.
``CONFIG_SECTIONS`` whitelists which feature sections can be read/edited.

Status aggregation is computed from the ``tool_jobs`` table — a single
paginated read filtered to the last 24h, then bucketed in Python. Cheap and
honest: no parallel "subsystem state" table to drift.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel


# In-process TTL cache for /admin/runtime. The Console polls this every few
# seconds from multiple panels; without coalescing we'd do a 500-row NocoDB
# read per panel per second. 3 s is short enough to feel live, long enough
# to collapse a burst of polls into a single read.
_RUNTIME_CACHE_TTL_S = 3.0
_runtime_cache: dict[Any, tuple[float, dict]] = {}
_runtime_cache_lock = threading.Lock()


def _runtime_cache_get(key) -> dict | None:
    with _runtime_cache_lock:
        hit = _runtime_cache.get(key)
        if hit and (time.time() - hit[0]) < _RUNTIME_CACHE_TTL_S:
            return hit[1]
    return None


def _runtime_cache_set(key, value: dict) -> None:
    with _runtime_cache_lock:
        _runtime_cache[key] = (time.time(), value)
        # Cap entries — operators can switch orgs but we don't want unbounded growth.
        if len(_runtime_cache) > 32:
            oldest = sorted(_runtime_cache.items(), key=lambda kv: kv[1][0])[:8]
            for k, _ in oldest:
                _runtime_cache.pop(k, None)

from infra.config import (
    PLATFORM,
    get_feature,
    is_feature_enabled,
    write_feature_section,
)
from infra.huey_runtime import get_huey, get_huey_health, is_huey_consumer_running
from infra.config import HUEY_CONSUMER_WORKERS, HUEY_ENABLED
from infra.nocodb_client import NocodbClient
from tools._org import resolve_org_id
from workers.tool_queue import NOCODB_TABLE as TOOL_JOBS_TABLE, get_tool_queue

_log = logging.getLogger("main.admin")

router = APIRouter(prefix="/admin", tags=["admin"])


# ── subsystem catalogue ───────────────────────────────────────────────────

# subsystem_id → {label, job_types, feature_section, scheduler_job_ids,
#                  trigger_job_type, trigger_default_payload}
#
# - ``job_types``: every tool-queue type that belongs to this subsystem
#   (used for status aggregation; one subsystem may dispatch multiple types).
# - ``feature_section``: the key under config.features to surface. None if
#   the subsystem has no toggleable section.
# - ``scheduler_job_ids``: APScheduler job IDs whose next_run_time should be
#   reported. Empty list if not scheduler-driven.
# - ``trigger_job_type``: the queue type used by POST /admin/trigger.
#   None means trigger isn't supported (e.g. harvest needs a policy + seed —
#   use /harvest/run/{policy} instead).
SUBSYSTEMS: dict[str, dict] = {
    "graph": {
        "label": "Graph",
        "job_types": ["graph_extract", "graph_resolve_entities", "graph_maintenance"],
        "feature_section": "graph_maintenance",
        "scheduler_job_ids": [
            "graph_entity_resolution_dispatcher",
            "graph_maintenance_dispatcher",
        ],
        "trigger_job_type": "graph_maintenance",
    },
    "entity_resolution": {
        "label": "Entity resolution",
        "job_types": ["graph_resolve_entities"],
        "feature_section": "graph_maintenance",
        "scheduler_job_ids": ["graph_entity_resolution_dispatcher"],
        "trigger_job_type": "graph_resolve_entities",
    },
    "research": {
        "label": "Research",
        "job_types": ["research_planner", "research_agent", "research_review", "research_op"],
        "feature_section": "research",
        "scheduler_job_ids": ["research_plan_reaper"],
        "trigger_job_type": "research_op",  # body must include {plan_id, kind, params}
    },
    "harvest": {
        "label": "Harvest",
        "job_types": ["harvest_run", "harvest_finalise", "scrape_page",
                      "pathfinder_extract", "summarise_page"],
        "feature_section": "harvest",
        "scheduler_job_ids": [
            "enrichment_scrape_dispatcher",
            "pathfinder_dispatcher",
            "discover_agent_dispatcher",
        ],
        "trigger_job_type": None,  # use POST /harvest/run/{policy}
    },
    "discover_agent": {
        "label": "Discover agent",
        "job_types": ["discover_agent_run"],
        "feature_section": "discover_agent",
        "scheduler_job_ids": ["discover_agent_dispatcher"],
        "trigger_job_type": "discover_agent_run",
    },
    "daily_digest": {
        "label": "Daily digest",
        "job_types": ["daily_digest"],
        "feature_section": "daily_digest",
        "scheduler_job_ids": ["daily_digest_dispatcher"],
        "trigger_job_type": "daily_digest",
    },
    "seed_feedback": {
        "label": "Seed feedback",
        "job_types": ["seed_feedback"],
        "feature_section": "seed_feedback",
        "scheduler_job_ids": ["seed_feedback_dispatcher"],
        "trigger_job_type": "seed_feedback",
    },
    "corpus_maintenance": {
        "label": "Corpus maintenance",
        "job_types": ["corpus_maintenance"],
        "feature_section": "corpus_maintenance",
        "scheduler_job_ids": ["corpus_maintenance_dispatcher"],
        "trigger_job_type": "corpus_maintenance",
    },
    "insights": {
        "label": "Insights",
        "job_types": ["insight_produce"],
        "feature_section": "insights",
        "scheduler_job_ids": ["insight_dispatcher"],
        "trigger_job_type": "insight_produce",
    },
    "pa": {
        "label": "Personal Assistant",
        "job_types": ["pa_topic_research"],
        "feature_section": "pa",
        "scheduler_job_ids": [],
        "trigger_job_type": "pa_topic_research",
    },
    "simulation": {
        "label": "Simulation Lab",
        "job_types": ["simulation_run"],
        "feature_section": None,
        "scheduler_job_ids": [],
        "trigger_job_type": None,  # use POST /simulations
    },
}

# Whitelist of feature sections that can be read/written via /admin/config.
# Sections containing ``models`` blocks (chat, code, web_search, …) are
# excluded — those need careful per-key validation we don't do here.
CONFIG_SECTIONS = {
    "research", "graph_maintenance", "harvest", "pathfinder", "discover_agent",
    "daily_digest", "seed_feedback", "corpus_maintenance", "insights", "pa",
    "research_seeder", "anchored_asks", "surfacing", "tool_queue",
    "graph_extract", "home", "daily_brief", "scraper",
}


# ── helpers ───────────────────────────────────────────────────────────────

def _parse_iso(value) -> datetime | None:
    if value in (None, ""):
        return None
    s = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        try:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _scheduler_next_runs(request: Request, job_ids: list[str]) -> dict[str, str | None]:
    sched = getattr(request.app.state, "scheduler", None)
    out: dict[str, str | None] = {}
    if not sched:
        return out
    for jid in job_ids:
        job = sched.get_job(jid)
        if job and job.next_run_time:
            out[jid] = job.next_run_time.isoformat()
        else:
            out[jid] = None
    return out


def _aggregate_jobs(rows: list[dict], job_types: list[str], cutoff_24h: datetime) -> dict:
    """Bucket tool_jobs rows for one subsystem."""
    agg = {
        "in_flight": 0, "queued": 0, "running": 0,
        "completed_24h": 0, "failed_24h": 0,
        "last_run": None, "last_status": None, "last_error": "",
        "last_job_id": None, "total_24h": 0,
    }
    type_set = set(job_types)
    # Sort by CreatedAt desc so the first match per subsystem is the latest.
    for r in rows:
        if r.get("type") not in type_set:
            continue
        st = r.get("status") or ""
        if st == "queued":
            agg["queued"] += 1
            agg["in_flight"] += 1
        elif st == "running":
            agg["running"] += 1
            agg["in_flight"] += 1
        ts = _parse_iso(r.get("completed_at") or r.get("started_at") or r.get("CreatedAt"))
        if ts and ts >= cutoff_24h:
            agg["total_24h"] += 1
            if st == "completed":
                agg["completed_24h"] += 1
            elif st == "failed":
                agg["failed_24h"] += 1
        # capture the most-recent terminal run
        if st in ("completed", "failed", "cancelled") and agg["last_run"] is None:
            agg["last_run"] = r.get("completed_at") or r.get("started_at") or r.get("CreatedAt")
            agg["last_status"] = st
            agg["last_error"] = (r.get("error") or "")[:200]
            agg["last_job_id"] = r.get("job_id")
    return agg


def _fetch_recent_jobs(client: NocodbClient, limit: int = 500) -> list[dict]:
    if TOOL_JOBS_TABLE not in client.tables:
        return []
    return client._get_paginated(TOOL_JOBS_TABLE, params={
        "sort": "-CreatedAt", "limit": limit,
    })


# ── runtime aggregator ────────────────────────────────────────────────────

@router.get("/runtime")
def runtime(request: Request, org_id: int | None = None):
    """One-shot snapshot for the frontend dashboard.

    Returns:
      * subsystems: list[{id, label, enabled, in_flight, queued, running,
                          completed_24h, failed_24h, last_run, last_status,
                          last_error, last_job_id, next_scheduled_runs,
                          feature_section, trigger_supported}]
      * huey: consumer/health snapshot
      * queue: counts + idle gate
      * scheduler: running flag + per-job next_run map

    Cached for ``_RUNTIME_CACHE_TTL_S`` seconds keyed on (org_id) — the
    Console polls this from multiple panels and tabs; without the cache
    each poll triggers a 500-row NocoDB read. Per-org cache so org A's
    poll doesn't return org B's snapshot.
    """
    cache_key = ("runtime", int(org_id) if org_id is not None else 0)
    cached = _runtime_cache_get(cache_key)
    if cached is not None:
        return cached
    client = NocodbClient()
    rows = _fetch_recent_jobs(client)
    if org_id is not None:
        rows = [r for r in rows if int(r.get("org_id") or 0) == int(org_id)]
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

    subsystems: list[dict] = []
    for sub_id, meta in SUBSYSTEMS.items():
        agg = _aggregate_jobs(rows, meta["job_types"], cutoff)
        feature_section = meta["feature_section"]
        enabled = (
            is_feature_enabled(feature_section)
            if feature_section else None
        )
        next_runs = _scheduler_next_runs(request, meta["scheduler_job_ids"])
        subsystems.append({
            "id": sub_id,
            "label": meta["label"],
            "enabled": enabled,
            "feature_section": feature_section,
            "trigger_supported": meta["trigger_job_type"] is not None,
            "trigger_job_type": meta["trigger_job_type"],
            "job_types": meta["job_types"],
            "in_flight": agg["in_flight"],
            "queued": agg["queued"],
            "running": agg["running"],
            "completed_24h": agg["completed_24h"],
            "failed_24h": agg["failed_24h"],
            "total_24h": agg["total_24h"],
            "last_run": agg["last_run"],
            "last_status": agg["last_status"],
            "last_error": agg["last_error"],
            "last_job_id": agg["last_job_id"],
            "next_scheduled_runs": next_runs,
        })

    q = get_tool_queue()
    queue_status = q.status() if q else {"error": "tool queue not initialised"}

    h = get_huey()
    pending: Any = "?"
    scheduled: Any = "?"
    if h is not None:
        try:
            pending = h.pending_count()
        except Exception:
            pending = "error"
        try:
            scheduled = h.scheduled_count()
        except Exception:
            scheduled = "error"

    huey_block = {
        "enabled": bool(HUEY_ENABLED),
        "consumer_running": is_huey_consumer_running(),
        "workers": int(HUEY_CONSUMER_WORKERS or 1),
        "pending_count": pending,
        "scheduled_count": scheduled,
        "health": get_huey_health(),
    }

    sched = getattr(request.app.state, "scheduler", None)
    sched_block = {
        "running": bool(sched and sched.running),
        "jobs": [
            {"id": j.id,
             "next_run": j.next_run_time.isoformat() if j.next_run_time else None}
            for j in (sched.get_jobs() if sched else [])
        ],
    }

    payload = {
        "subsystems": subsystems,
        "huey": huey_block,
        "queue": queue_status,
        "scheduler": sched_block,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
    _runtime_cache_set(cache_key, payload)
    return payload


# ── config read/write ─────────────────────────────────────────────────────

@router.get("/chat-active")
def chat_active_status():
    """Snapshot of the chat-active gate. Surfaces ``count`` and how long the
    oldest active turn has been live so the user can spot a leaked counter
    (count > 0 with no real chat happening) and click reset."""
    from workers.tool_queue import chat_active_state
    return chat_active_state()


class ChatActiveReset(BaseModel):
    reason: str = "manual reset from admin UI"


@router.post("/chat-active/reset")
def chat_active_reset(body: ChatActiveReset | None = None):
    """Force the chat-active counter to zero. Use when the gate is wedged
    after a chat handler crashed without decrementing — symptom is every
    background worker stuck at ``_block_while_chat_active`` and no LLM
    progress despite jobs marked running."""
    from workers.tool_queue import reset_chat_active
    reason = (body.reason if body else "manual") or "manual"
    prev = reset_chat_active(reason=reason)
    return {"reset": True, "previous_count": prev, "reason": reason}


class ToolJobCancelRequest(BaseModel):
    reason: str = "user terminated"


@router.post("/tool-jobs/{job_id}/cancel")
def admin_cancel_tool_job(job_id: str, body: ToolJobCancelRequest | None = None):
    """Cooperative cancel for a running tool job. The handler is expected
    to honour ``ToolJobQueue.is_cancelled(job_id)`` between phases and abort
    cleanly; an LLM call already in flight runs to completion (we never
    kill threads). Use this when a job is stuck or no longer wanted."""
    from workers.tool_queue import get_tool_queue
    q = get_tool_queue()
    if q is None:
        raise HTTPException(status_code=503, detail="tool queue not initialised")
    reason = (body.reason if body else "user terminated") or "user terminated"
    ok = q.cancel_running(job_id, reason=reason)
    if not ok:
        raise HTTPException(status_code=404, detail="job not found or not cancellable")
    return {"cancelled": True, "job_id": job_id, "reason": reason}


@router.get("/config")
def list_config():
    """Return all whitelisted sections (current values from PLATFORM)."""
    features = PLATFORM.get("features", {})
    return {
        "sections": sorted(CONFIG_SECTIONS),
        "values": {k: features.get(k) for k in sorted(CONFIG_SECTIONS) if k in features},
    }


@router.get("/config/{section}")
def get_config_section(section: str):
    if section not in CONFIG_SECTIONS:
        raise HTTPException(status_code=404, detail=f"section '{section}' not editable")
    val = PLATFORM.get("features", {}).get(section)
    if val is None:
        raise HTTPException(status_code=404, detail=f"section '{section}' not present in config.json")
    return {"section": section, "value": val}


class ConfigPatch(BaseModel):
    value: dict


@router.patch("/config/{section}")
def patch_config_section(section: str, body: ConfigPatch):
    """Replace a feature section. Body is the FULL replacement object — to
    avoid partial-merge ambiguity. Frontend should GET, mutate, then PATCH.
    """
    if section not in CONFIG_SECTIONS:
        raise HTTPException(status_code=400, detail=f"section '{section}' not editable")
    if not isinstance(body.value, dict):
        raise HTTPException(status_code=400, detail="value must be an object")
    # Preserve any nested 'models' block — it's the per-section model registry
    # and editing it through the generic PATCH would be too lax. If the
    # incoming body omits 'models', re-attach the existing one.
    existing = PLATFORM.get("features", {}).get(section, {})
    new_value = dict(body.value)
    if isinstance(existing, dict) and "models" in existing and "models" not in new_value:
        new_value["models"] = existing["models"]
    errors = _validate_against_schema(section, new_value)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})
    try:
        write_feature_section(section, new_value)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"write failed: {e}")
    _append_history(section, existing if isinstance(existing, dict) else None,
                    new_value, source="patch")
    return {"section": section, "value": new_value}


# ── unified trigger ───────────────────────────────────────────────────────

class TriggerRequest(BaseModel):
    payload: dict = {}
    org_id: int | None = None
    bypass_idle: bool = False
    priority: int | None = None


@router.post("/trigger/{subsystem_id}")
def trigger_subsystem(subsystem_id: str, body: TriggerRequest | None = None):
    """Queue a one-shot run for a subsystem. Returns the job_id immediately —
    poll /tool-queue/jobs/{job_id} or watch /tool-queue/events for progress.

    For subsystems whose job needs structured params (e.g. research_op needs
    {plan_id, kind, params}; pa_topic_research needs {topic_id}), pass them
    in the request body's ``payload`` field. Other subsystems accept an
    empty body.
    """
    body = body or TriggerRequest()
    meta = SUBSYSTEMS.get(subsystem_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"unknown subsystem: {subsystem_id}")
    job_type = meta["trigger_job_type"]
    if not job_type:
        raise HTTPException(
            status_code=400,
            detail=(f"subsystem '{subsystem_id}' has no uniform trigger; "
                    f"use the dedicated endpoint instead"),
        )

    tq = get_tool_queue()
    if tq is None:
        raise HTTPException(status_code=503, detail="tool_queue_unavailable")

    org_id = int(body.org_id or resolve_org_id(0) or 1)
    payload = dict(body.payload or {})
    errors = _validate_trigger_payload(subsystem_id, payload)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})
    if body.bypass_idle:
        payload["force_bypass_idle"] = True
    payload.setdefault("org_id", org_id)

    try:
        job_id = tq.submit(
            job_type, payload,
            source=f"admin:{subsystem_id}",
            org_id=org_id,
            priority=body.priority,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"submit failed: {e}")
    _log.info("admin trigger  subsystem=%s  job=%s  type=%s  bypass_idle=%s",
              subsystem_id, job_id, job_type, body.bypass_idle)
    return {
        "status": "queued",
        "subsystem": subsystem_id,
        "job_type": job_type,
        "job_id": job_id,
        "bypass_idle": body.bypass_idle,
    }


# ── per-section config schemas ────────────────────────────────────────────
#
# Hand-curated overrides for fields where type/range matters more than the
# generic auto-schema can express (e.g. cron hour 0-23, percentages, enums).
# Anything not listed here is auto-derived from the current value's type.
_SCHEMA_OVERRIDES: dict[str, dict[str, dict]] = {
    "daily_digest": {
        "enabled":     {"type": "boolean", "default": True},
        "cron_hour":   {"type": "number", "min": 0, "max": 23, "step": 1},
        "cron_minute": {"type": "number", "min": 0, "max": 59, "step": 1},
    },
    "graph_maintenance": {
        "enabled":                          {"type": "boolean", "default": True},
        "entity_resolution_interval_hours": {"type": "number", "min": 1, "max": 720, "step": 1},
        "maintenance_interval_hours":       {"type": "number", "min": 1, "max": 720, "step": 1},
    },
    "seed_feedback": {
        "enabled":              {"type": "boolean", "default": True},
        "run_interval_hours":   {"type": "number", "min": 1, "max": 168, "step": 1},
        "min_relevance":        {"type": "enum", "options": ["rejected", "low", "medium", "high"]},
    },
    "research": {
        "max_queries":                     {"type": "number", "min": 1, "max": 50, "step": 1},
        "web_search_per_query_timeout_s":  {"type": "number", "min": 30, "max": 1800, "step": 10},
        "planner_timeout_s":               {"type": "number", "min": 60, "max": 3600, "step": 30},
        "synthesis_timeout_s":             {"type": "number", "min": 60, "max": 3600, "step": 30},
        "critic_timeout_s":                {"type": "number", "min": 60, "max": 3600, "step": 30},
        "reviewer_timeout_s":              {"type": "number", "min": 60, "max": 3600, "step": 30},
        "section_timeout_s":               {"type": "number", "min": 60, "max": 3600, "step": 30},
        "reap_interval_minutes":           {"type": "number", "min": 5, "max": 720, "step": 5},
    },
    "tool_queue": {
        "background_chat_idle_seconds": {"type": "number", "min": 0, "max": 7200, "step": 5},
        "default_max_attempts":         {"type": "number", "min": 1, "max": 5, "step": 1},
        "default_retry_backoff_seconds":{"type": "number", "min": 0, "max": 300, "step": 1},
    },
    "harvest": {
        "enabled": {"type": "boolean", "default": True},
    },
    "discover_agent": {
        "run_interval_minutes": {"type": "number", "min": 1, "max": 1440, "step": 1},
    },
    "insights": {
        "enabled":      {"type": "boolean", "default": True},
        "tick_minutes": {"type": "number", "min": 1, "max": 240, "step": 1},
    },
    "corpus_maintenance": {
        "enabled":            {"type": "boolean", "default": True},
        "run_interval_hours": {"type": "number", "min": 1, "max": 168, "step": 1},
    },
    "scraper": {
        "dispatch_interval_seconds": {"type": "number", "min": 15, "max": 3600, "step": 5},
    },
    "pathfinder": {
        "dispatch_interval_seconds": {"type": "number", "min": 30, "max": 3600, "step": 5},
    },
}


def _auto_field(value: Any) -> dict:
    if isinstance(value, bool):  # check before int — bool is a subclass
        return {"type": "boolean"}
    if isinstance(value, (int, float)):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        return {"type": "array"}
    if isinstance(value, dict):
        return {"type": "object"}
    return {"type": "string"}


def _build_section_schema(section: str, current: dict | None) -> dict:
    """Combine auto-detected types with curated overrides. The frontend uses
    this to render typed form fields instead of a generic JSON pane.

    'models' sub-block is reported but marked read-only — model-config edits
    have to go through PATCH-with-models-preserved or model-aware tooling.
    """
    current = current or {}
    overrides = _SCHEMA_OVERRIDES.get(section, {})
    fields: list[dict] = []
    seen: set[str] = set()
    # Curated overrides first (fixed display order)
    for name, spec in overrides.items():
        s = dict(spec)
        s["name"] = name
        s["current"] = current.get(name, s.get("default"))
        fields.append(s)
        seen.add(name)
    # Then any remaining keys from the live value
    for name, val in current.items():
        if name in seen or name == "models":
            continue
        f = _auto_field(val)
        f["name"] = name
        f["current"] = val
        fields.append(f)
    has_models = isinstance(current.get("models"), dict)
    return {
        "section": section,
        "fields": fields,
        "models_present": has_models,
        "models_keys": sorted(current.get("models", {}).keys()) if has_models else [],
        "models_editable_via": "use the models registry — not editable via /admin/config",
    }


def _validate_against_schema(section: str, value: dict) -> list[str]:
    """Return a list of error strings; empty list = OK."""
    errors: list[str] = []
    if not isinstance(value, dict):
        return ["section value must be an object"]
    for name, spec in _SCHEMA_OVERRIDES.get(section, {}).items():
        if name not in value:
            continue  # optional — only enforce if present
        v = value[name]
        t = spec.get("type")
        if t == "boolean" and not isinstance(v, bool):
            errors.append(f"{name}: must be boolean")
        elif t == "number":
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                errors.append(f"{name}: must be number")
            else:
                if "min" in spec and v < spec["min"]:
                    errors.append(f"{name}: must be ≥ {spec['min']}")
                if "max" in spec and v > spec["max"]:
                    errors.append(f"{name}: must be ≤ {spec['max']}")
        elif t == "string" and not isinstance(v, str):
            errors.append(f"{name}: must be string")
        elif t == "enum":
            options = spec.get("options") or []
            if v not in options:
                errors.append(f"{name}: must be one of {options}")
    return errors


@router.get("/config/{section}/schema")
def get_section_schema(section: str):
    if section not in CONFIG_SECTIONS:
        raise HTTPException(status_code=404, detail=f"section '{section}' not editable")
    current = PLATFORM.get("features", {}).get(section)
    if current is None:
        # Section doesn't exist yet — return overrides-only schema so the
        # frontend can render a creation form.
        return _build_section_schema(section, {})
    if not isinstance(current, dict):
        raise HTTPException(status_code=400, detail=f"section '{section}' is not an object")
    return _build_section_schema(section, current)


# ── per-trigger payload schemas ───────────────────────────────────────────
#
# Required and optional params for /admin/trigger/{subsystem_id}. Used both
# for validation (POST) and for rendering a form (GET schema).
_TRIGGER_SCHEMAS: dict[str, dict] = {
    "research": {
        "description": "Run a post-build research op on an existing plan.",
        "required": [
            {"name": "plan_id", "type": "number", "description": "research_plans Id"},
            {"name": "kind",    "type": "enum",
             "options": ["fact_check", "expand_section", "add_section",
                         "counter_arguments", "add_fresh_sources", "refresh_recency",
                         "reframe", "resize", "slide_deck", "email_tldr", "qa_pack",
                         "action_plan", "citation_audit", "chat_with_paper"]},
        ],
        "optional": [
            {"name": "params", "type": "object",
             "description": "Op-specific params (e.g. {section_title, target_words})"},
        ],
    },
    "pa": {
        "description": "Trigger a PA topic-research run.",
        "required": [],
        "optional": [
            {"name": "topic_id",   "type": "number"},
            {"name": "topic_text", "type": "string"},
        ],
    },
    "graph": {"description": "Run graph maintenance.", "required": [], "optional": []},
    "entity_resolution": {"description": "Resolve entity duplicates.", "required": [], "optional": []},
    "discover_agent": {"description": "Run the discover agent.", "required": [], "optional": []},
    "daily_digest": {"description": "Build today's digest now.", "required": [], "optional": []},
    "seed_feedback": {"description": "Run the seed-feedback signals.", "required": [], "optional": []},
    "corpus_maintenance": {"description": "Run corpus maintenance.", "required": [], "optional": []},
    "insights": {"description": "Produce insights.", "required": [], "optional": []},
}


def _validate_trigger_payload(subsystem_id: str, payload: dict) -> list[str]:
    schema = _TRIGGER_SCHEMAS.get(subsystem_id)
    if schema is None:
        return []  # no schema → permissive (handler will fail loud if wrong)
    errors: list[str] = []
    for spec in schema.get("required", []):
        name = spec["name"]
        if name not in payload:
            errors.append(f"{name}: required")
            continue
        v = payload[name]
        t = spec.get("type")
        if t == "number" and not isinstance(v, (int, float)):
            errors.append(f"{name}: must be number")
        elif t == "string" and not isinstance(v, str):
            errors.append(f"{name}: must be string")
        elif t == "enum" and v not in (spec.get("options") or []):
            errors.append(f"{name}: must be one of {spec.get('options')}")
        elif t == "object" and not isinstance(v, dict):
            errors.append(f"{name}: must be object")
    for spec in schema.get("optional", []):
        name = spec["name"]
        if name not in payload:
            continue
        v = payload[name]
        t = spec.get("type")
        if t == "number" and not isinstance(v, (int, float)):
            errors.append(f"{name}: must be number")
        elif t == "string" and not isinstance(v, str):
            errors.append(f"{name}: must be string")
        elif t == "object" and not isinstance(v, dict):
            errors.append(f"{name}: must be object")
    return errors


@router.get("/trigger/{subsystem_id}/schema")
def get_trigger_schema(subsystem_id: str):
    meta = SUBSYSTEMS.get(subsystem_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"unknown subsystem: {subsystem_id}")
    schema = _TRIGGER_SCHEMAS.get(subsystem_id, {
        "description": "No structured params required.",
        "required": [],
        "optional": [],
    })
    return {
        "subsystem": subsystem_id,
        "label": meta["label"],
        "trigger_supported": meta["trigger_job_type"] is not None,
        "trigger_job_type": meta["trigger_job_type"],
        **schema,
    }


# ── subsystem enable / disable ────────────────────────────────────────────

@router.post("/subsystems/{subsystem_id}/enable")
def enable_subsystem(subsystem_id: str):
    return _toggle_subsystem(subsystem_id, True)


@router.post("/subsystems/{subsystem_id}/disable")
def disable_subsystem(subsystem_id: str):
    return _toggle_subsystem(subsystem_id, False)


def _toggle_subsystem(subsystem_id: str, enabled: bool) -> dict:
    meta = SUBSYSTEMS.get(subsystem_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"unknown subsystem: {subsystem_id}")
    section = meta["feature_section"]
    if not section or section not in CONFIG_SECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"subsystem '{subsystem_id}' has no editable feature section",
        )
    existing = PLATFORM.get("features", {}).get(section)
    if not isinstance(existing, dict):
        # Section existed as a bare bool — promote to dict so we can carry
        # other settings later.
        new_value: dict = {"enabled": enabled}
    else:
        new_value = dict(existing)
        new_value["enabled"] = enabled
    prev = existing if isinstance(existing, (dict, bool)) else None
    try:
        write_feature_section(section, new_value)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"write failed: {e}")
    _append_history(section, prev if isinstance(prev, dict) else {"enabled": bool(prev)},
                    new_value, source=f"toggle:{subsystem_id}")
    return {
        "subsystem": subsystem_id,
        "section": section,
        "enabled": enabled,
        "value": new_value,
    }


# ── config history ────────────────────────────────────────────────────────
#
# JSONL append-only audit log next to config.json. One entry per PATCH or
# enable/disable. Cheap and trivially tail-readable.

_HISTORY_PATH = Path(__file__).parent.parent.parent / "config_history.jsonl"


def _append_history(section: str, prev_value: Any, new_value: Any, source: str) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "section": section,
        "source": source,
        "prev": prev_value,
        "new": new_value,
    }
    try:
        with open(_HISTORY_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        _log.warning("config history append failed", exc_info=True)


@router.get("/config/{section}/history")
def get_config_history(section: str, limit: int = 50):
    if section not in CONFIG_SECTIONS:
        raise HTTPException(status_code=404, detail=f"section '{section}' not editable")
    limit = max(1, min(int(limit), 500))
    if not _HISTORY_PATH.exists():
        return {"section": section, "history": []}
    try:
        with open(_HISTORY_PATH) as f:
            lines = f.readlines()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history read failed: {e}")
    out: list[dict] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if entry.get("section") != section:
            continue
        out.append(entry)
        if len(out) >= limit:
            break
    return {"section": section, "history": out}


@router.get("/config-history")
def get_full_config_history(limit: int = 100):
    """All sections, newest first. Useful for an audit log page."""
    limit = max(1, min(int(limit), 1000))
    if not _HISTORY_PATH.exists():
        return {"history": []}
    try:
        with open(_HISTORY_PATH) as f:
            lines = f.readlines()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history read failed: {e}")
    out: list[dict] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
        if len(out) >= limit:
            break
    return {"history": out}


# ── catalog ───────────────────────────────────────────────────────────────

@router.get("/subsystems")
def list_subsystems():
    """Static catalog used by the frontend to render subsystem cards before
    the first /admin/runtime poll completes."""
    return {
        "subsystems": [
            {
                "id": sub_id,
                "label": meta["label"],
                "feature_section": meta["feature_section"],
                "trigger_supported": meta["trigger_job_type"] is not None,
                "trigger_job_type": meta["trigger_job_type"],
                "job_types": meta["job_types"],
                "scheduler_job_ids": meta["scheduler_job_ids"],
            }
            for sub_id, meta in SUBSYSTEMS.items()
        ],
        "config_sections": sorted(CONFIG_SECTIONS),
    }
