import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from pydantic import BaseModel

from infra.config import NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS, get_feature
from infra.nocodb_client import NocodbClient
from workers.tool_queue import ToolJob, get_tool_queue
from tools._org import resolve_org_id

_log = logging.getLogger("main.enrichment")

router = APIRouter()


class PathfinderRequest(BaseModel):
    seed_url: str
    org_id: int


class ResearchRequest(BaseModel):
    topic: str
    org_id: int


class ResearchAgentRequest(BaseModel):
    plan_id: int



@router.get("/discovery/suggestions")
def discovery_suggestions_list(org_id: int, status: str | None = "pending", limit: int = 50):
    limit = min(max(1, limit), 500)
    client = NocodbClient()
    parts = [f"(org_id,eq,{org_id})"]
    if status:
        parts.append(f"(status,eq,{status})")
    params: dict = {
        "where": "~and".join(parts),
        "sort": "-CreatedAt",
        "limit": limit,
    }
    rows = client._get_paginated(NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS, params=params)
    return {"status": "ok", "rows": rows}


@router.post("/discovery/suggestions/{suggested_id}/approve")
def discovery_suggestion_approve(suggested_id: int, org_id: int):
    """User approves a suggested URL. Mark status=approved and enqueue
    pathfinder_extract immediately (bypass idle gate — this is user-initiated)."""
    client = NocodbClient()
    row = _get_single_row(client, NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS, suggested_id, org_id=org_id)
    if not row:
        return {"status": "not_found", "suggested_id": suggested_id}

    org_id = resolve_org_id(row.get("org_id"))
    if org_id <= 0:
        return {"status": "failed", "error": "missing_org_id"}

    try:
        client._patch(NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS, suggested_id, {
            "Id": suggested_id,
            "status": "approved",
            "error_message": "",
            "reviewed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        })
    except Exception:
        _log.warning("suggestion approve patch failed  id=%d", suggested_id, exc_info=True)
        return {"status": "failed", "error": "patch_failed"}

    tq = get_tool_queue()
    if not tq:
        return {"status": "failed", "error": "tool_queue_unavailable"}

    job_id = tq.submit(
        "pathfinder_extract",
        {"suggested_id": suggested_id, "org_id": org_id, "bypass_idle": True},
        source="discovery_suggestions_api",
        priority=3,
        org_id=org_id,
    )
    return {"status": "queued", "suggested_id": suggested_id, "job_id": job_id, "org_id": org_id}


@router.post("/discovery/suggestions/{suggested_id}/reject")
def discovery_suggestion_reject(suggested_id: int, org_id: int, reason: str | None = None):
    client = NocodbClient()
    row = _get_single_row(client, NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS, suggested_id, org_id=org_id)
    if not row:
        return {"status": "not_found", "suggested_id": suggested_id}
    try:
        client._patch(NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS, suggested_id, {
            "Id": suggested_id,
            "status": "rejected",
            "error_message": (reason or "")[:500],
            "reviewed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        })
    except Exception:
        return {"status": "failed", "error": "patch_failed"}
    return {"status": "ok", "suggested_id": suggested_id}


# ── Manual seed entry (bypasses discovery, goes straight to pathfinder) ───────

@router.post("/pathfinder/discover")
def pathfinder_discover(req: PathfinderRequest):
    """User directly submits a seed URL. We create a suggested_scrape_targets
    row pre-approved by the user and queue pathfinder_extract."""
    from tools.enrichment.pathfinder import _normalize

    norm_url = _normalize(req.seed_url)
    if not norm_url:
        return {"status": "failed", "error": "invalid_url", "raw": req.seed_url}
    if req.org_id <= 0:
        return {"status": "failed", "error": "invalid_org_id"}

    client = NocodbClient()
    try:
        row = client._post(NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS, {
            "org_id": req.org_id,
            "url": norm_url,
            "title": norm_url,
            "query": "manual_entry",
            "relevance": "high",
            "score": 100,
            "reason": "user-submitted seed",
            "status": "approved",
        })
        suggested_id = row.get("Id")
    except Exception:
        _log.warning("pathfinder discover insert failed  url=%s", norm_url[:80], exc_info=True)
        return {"status": "failed", "error": "insert_failed"}

    tq = get_tool_queue()
    if not tq:
        return {"status": "failed", "error": "tool_queue_unavailable"}

    job_id = tq.submit(
        "pathfinder_extract",
        {"suggested_id": suggested_id, "org_id": req.org_id, "bypass_idle": True},
        source="pathfinder_api",
        priority=3,
        org_id=req.org_id,
    )
    return {"status": "queued", "suggested_id": suggested_id, "job_id": job_id, "url": norm_url}


# ── Scraper control ───────────────────────────────────────────────────────────

@router.get("/scraper/start")
@router.post("/scraper/start")
def scraper_start(org_id: int | None = None):
    from tools.enrichment.dispatcher import jumpstart_scraper
    return jumpstart_scraper(org_id=org_id)


@router.post("/scrape-targets/{target_id}/run-now")
def scrape_target_run_now(target_id: int, org_id: int):
    tq = get_tool_queue()
    if not tq:
        return {"status": "failed", "error": "tool_queue_unavailable"}

    row = _get_single_row(NocodbClient(), "scrape_targets", target_id, org_id=org_id)
    if not row:
        return {"status": "not_found", "target_id": target_id}

    org_id = resolve_org_id(row.get("org_id"))
    if org_id <= 0:
        return {"status": "failed", "error": "missing_org_id", "target_id": target_id}

    job_id = tq.submit(
        "scrape_page",
        {"target_id": target_id, "org_id": org_id, "bypass_idle": True},
        source="scrape_target_api",
        priority=3,
        org_id=org_id,
    )
    return {"status": "queued", "target_id": target_id, "job_id": job_id, "org_id": org_id}


@router.get("/pathfinder/start")
@router.post("/pathfinder/start")
def pathfinder_start(org_id: int | None = None):
    from tools.enrichment.dispatcher import jumpstart_pathfinder
    return jumpstart_pathfinder(org_id=org_id)


@router.get("/discover-agent/start")
@router.post("/discover-agent/start")
def discover_agent_start(org_id: int | None = None):
    from tools.enrichment.dispatcher import jumpstart_discover_agent
    return jumpstart_discover_agent(org_id=org_id)


# ── Previews ──────────────────────────────────────────────────────────────────

@router.post("/pathfinder/fetch-next")
def pathfinder_fetch_next(org_id: int | None = None):
    from tools.enrichment.pathfinder import preview_next_approved
    row = preview_next_approved(org_id=org_id)
    if not row:
        return {"status": "empty", "row": None}
    return {"status": "ok", "row": row}


# ── Research (unchanged) ──────────────────────────────────────────────────────

@router.post("/research/create-plan")
def research_create_plan(req: ResearchRequest):
    from tools.research.research_planner import create_research_plan
    result = create_research_plan(req.topic, req.org_id)
    return {"status": result.get("status"), **result}


@router.post("/research/get-next")
def research_get_next():
    from tools.research.research_planner import get_next_plan
    row = get_next_plan()
    if not row:
        return {"status": "empty", "row": None}
    return {"status": "ok", "row": row}


@router.post("/research/complete")
def research_complete(plan_id: int):
    from tools.research.research_planner import complete_plan
    complete_plan(plan_id)
    return {"status": "ok", "plan_id": plan_id}


@router.post("/research/agent/run")
def research_agent_run(req: ResearchAgentRequest):
    tq = get_tool_queue()
    if not tq:
        return {"status": "failed", "error": "tool_queue_unavailable"}

    org_id = 1
    try:
        row = NocodbClient()._get("research_plans", params={"where": f"(Id,eq,{req.plan_id})", "limit": 1})
        plan = row.get("list", [])[0] if row.get("list") else None
        org_id = resolve_org_id((plan or {}).get("org_id"))
    except Exception:
        _log.warning("research_agent_run org lookup failed  plan_id=%d", req.plan_id, exc_info=True)

    if org_id <= 0:
        org_id = 1

    job_id = tq.submit(
        "research_agent",
        {"plan_id": req.plan_id, "org_id": org_id},
        source="enrichment_api",
        priority=3,
        org_id=org_id,
    )
    return {"status": "queued", "job_id": job_id}


@router.post("/research/agent/next")
def research_agent_next():
    from tools.research.agent import get_next_research
    row = get_next_research()
    if not row:
        return {"status": "empty", "row": None}
    return {"status": "ok", "row": row}


# ── Listing / dashboard helpers ───────────────────────────────────────────────

def _get_single_row(client: NocodbClient, table: str, row_id: int, org_id: int | None = None) -> dict | None:
    try:
        where = f"(Id,eq,{row_id})"
        if org_id is not None:
            where = f"{where}~and(org_id,eq,{int(org_id)})"
        rows = client._get(table, params={
            "where": where,
            "limit": 1,
        }).get("list", [])
        return rows[0] if rows else None
    except Exception:
        return None


def _recent_tool_jobs_for_org(client: NocodbClient, org_id: int, limit: int = 20) -> list[dict]:
    try:
        rows = client._get("tool_jobs", params={
            "where": f"(org_id,eq,{org_id})",
            "sort": "-CreatedAt",
            "limit": limit,
        }).get("list", [])
        return [ToolJob.from_row(r).to_api(verbose=True) for r in rows]
    except Exception:
        _log.warning("recent tool_jobs query failed  org_id=%d", org_id, exc_info=True)
        return []


def _scheduler_next_run(request: Request | None, job_id: str) -> str | None:
    if request is None:
        return None
    sched = getattr(request.app.state, "scheduler", None)
    if sched is None:
        return None
    try:
        job = sched.get_job(job_id)
        return job.next_run_time.isoformat() if job and job.next_run_time else None
    except Exception:
        return None


def _last_tool_job_snapshot(client: NocodbClient, org_id: int, job_type: str) -> dict | None:
    try:
        rows = client._get("tool_jobs", params={
            "where": f"(org_id,eq,{org_id})~and(type,eq,{job_type})",
            "sort": "-CreatedAt",
            "limit": 1,
        }).get("list", [])
        return ToolJob.from_row(rows[0]).to_api(verbose=True) if rows else None
    except Exception:
        return None


def build_enrichment_runtime_snapshot(request: Request | None, org_id: int, client: NocodbClient | None = None) -> dict:
    client = client or NocodbClient()
    from tools.enrichment.scraper import fetch_due_target
    from tools.enrichment.pathfinder import preview_next_approved

    try:
        next_scrape = fetch_due_target(client, org_id=org_id)
    except Exception:
        next_scrape = None

    try:
        next_pathfinder = preview_next_approved(org_id=org_id)
    except Exception:
        next_pathfinder = None

    return {
        "config": {
            "background_chat_idle_seconds": int(
                get_feature("tool_queue", "background_chat_idle_seconds", 30) or 30
            ),
            "scraper_dispatch_interval_seconds": int(
                get_feature("scraper", "dispatch_interval_seconds", 60) or 60
            ),
            "pathfinder_dispatch_interval_seconds": int(
                get_feature("pathfinder", "dispatch_interval_seconds", 120) or 120
            ),
            "discover_agent_run_interval_minutes": int(
                get_feature("discover_agent", "run_interval_minutes", 20) or 20
            ),
        },
        "schedule": {
            "next_scraper_dispatch": _scheduler_next_run(request, "enrichment_scrape_dispatcher"),
            "next_pathfinder_dispatch": _scheduler_next_run(request, "pathfinder_dispatcher"),
            "next_discover_agent_dispatch": _scheduler_next_run(request, "discover_agent_dispatcher"),
        },
        "last_jobs": {
            "scrape_page": _last_tool_job_snapshot(client, org_id, "scrape_page"),
            "pathfinder_extract": _last_tool_job_snapshot(client, org_id, "pathfinder_extract"),
            "discover_agent_run": _last_tool_job_snapshot(client, org_id, "discover_agent_run"),
            "summarise_page": _last_tool_job_snapshot(client, org_id, "summarise_page"),
            "extract_relationships": _last_tool_job_snapshot(client, org_id, "extract_relationships"),
        },
        "next_candidates": {
            "pathfinder": next_pathfinder,
            "scraper": next_scrape,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


@router.get("/discovery/list")
def discovery_list(org_id: int, status: str | None = None, limit: int = 50):
    """Legacy listing of the `discovery` table (kept for backwards compat). The
    new flow uses /discovery/suggestions."""
    limit = min(max(1, limit), 500)
    client = NocodbClient()
    parts = [f"(org_id,eq,{org_id})"]
    if status:
        parts.append(f"(status,eq,{status})")
    params: dict = {"limit": limit, "sort": "-CreatedAt", "where": "~and".join(parts)}
    try:
        rows = client._get_paginated("discovery", params=params)
    except Exception:
        rows = []
    return {"status": "ok", "rows": rows}


@router.get("/research-plans/list")
def research_plans_list(org_id: int, status: str | None = None, limit: int = 50):
    limit = min(max(1, limit), 500)
    client = NocodbClient()
    params: dict = {"limit": limit}
    if status:
        params["where"] = f"(status,eq,{status})~and(org_id,eq,{org_id})"
    else:
        params["where"] = f"(org_id,eq,{org_id})"
    rows = client._get_paginated("research_plans", params=params)
    return {"status": "ok", "rows": rows}


@router.get("/research-plans/{plan_id}")
def research_plan_get(plan_id: int, org_id: int):
    client = NocodbClient()
    data = client._get("research_plans", params={
        "where": f"(Id,eq,{plan_id})~and(org_id,eq,{org_id})",
        "limit": 1,
    })
    rows = data.get("list", [])
    if not rows:
        return {"status": "not_found", "row": None}
    return {"status": "ok", "row": rows[0]}


@router.get("/scrape-targets/list")
def scrape_targets_list(org_id: int, status: str | None = None, active_only: bool = True, limit: int = 100):
    limit = min(max(1, limit), 500)
    client = NocodbClient()
    parts = [f"(org_id,eq,{org_id})"]
    if active_only:
        parts.append("(active,eq,1)")
    if status:
        parts.append(f"(status,eq,{status})")
    params: dict = {
        "where": "~and".join(parts),
        "limit": limit,
        "sort": "-CreatedAt",
    }
    rows = client._get_paginated("scrape_targets", params=params)
    return {"status": "ok", "rows": rows}


@router.get("/discovery/{row_id}")
def discovery_get(row_id: int, org_id: int):
    client = NocodbClient()
    row = _get_single_row(client, "discovery", row_id, org_id=org_id)
    if not row:
        return {"status": "not_found", "row": None}
    return {"status": "ok", "row": row}


@router.get("/scrape-targets/{target_id}")
def scrape_target_get(target_id: int, org_id: int):
    client = NocodbClient()
    row = _get_single_row(client, "scrape_targets", target_id, org_id=org_id)
    if not row:
        return {"status": "not_found", "row": None}
    return {"status": "ok", "row": row}


@router.get("/dashboard")
def enrichment_dashboard(request: Request, org_id: int, limit: int = 20):
    limit = min(max(1, limit), 100)
    client = NocodbClient()
    try:
        suggestion_rows = client._get_paginated(NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS, params={
            "where": f"(org_id,eq,{org_id})",
            "sort": "-CreatedAt",
            "limit": limit,
        })
    except Exception:
        suggestion_rows = []
    scrape_target_rows = client._get_paginated("scrape_targets", params={
        "where": f"(org_id,eq,{org_id})",
        "sort": "-CreatedAt",
        "limit": limit,
    })
    queue_jobs = _recent_tool_jobs_for_org(client, org_id, limit=limit)
    return {
        "status": "ok",
        "org_id": org_id,
        "pipeline": build_enrichment_runtime_snapshot(request, org_id, client=client),
        "suggestions": {
            "count": len(suggestion_rows),
            "rows": suggestion_rows,
        },
        "scrape_targets": {
            "count": len(scrape_target_rows),
            "rows": scrape_target_rows,
        },
        "queue_jobs": {
            "count": len(queue_jobs),
            "rows": queue_jobs,
        },
    }
