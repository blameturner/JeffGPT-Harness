import logging

from fastapi import APIRouter
from pydantic import BaseModel

from infra.nocodb_client import NocodbClient
from workers.tool_queue import ToolJob, get_tool_queue

_log = logging.getLogger("main.enrichment")

router = APIRouter()


class PathfinderRequest(BaseModel):
    seed_url: str
    org_id: int
    max_depth: int = 3
    max_pages: int = 200
    same_host_only: bool = True


class ScraperRequest(BaseModel):
    org_id: int
    batch_size: int = 10


class ResearchRequest(BaseModel):
    topic: str
    org_id: int


class ResearchAgentRequest(BaseModel):
    plan_id: int


@router.post("/pathfinder/discover")
def pathfinder_discover(req: PathfinderRequest):
    """UI submits a seed URL: upsert as a discovery root and queue a pathfinder_crawl
    job. Resets status to 'discovered' on re-submission so the scheduler can pick it
    up on the next cycle even if the in-memory job is lost to a restart."""
    from tools.enrichment.pathfinder import upsert_discovery_root, _normalize

    # Normalize and validate the URL before touching the DB.
    # _normalize strips tracking params, www, trailing slashes and returns ""
    # for non-HTTP(S) schemes — matching exactly what discover() does internally.
    norm_url = _normalize(req.seed_url)
    if not norm_url:
        return {"status": "failed", "error": "invalid_url", "raw": req.seed_url}
    if req.org_id <= 0:
        return {"status": "failed", "error": "invalid_org_id"}

    client = NocodbClient()
    # reset_status=True: if the URL was previously crawled, mark it 'discovered'
    # again so both the direct job AND the scheduler fallback can process it.
    discovery_id = upsert_discovery_root(client, norm_url, req.org_id, reset_status=True)
    if not discovery_id:
        return {"status": "failed", "error": "discovery_upsert_failed"}

    tq = get_tool_queue()
    if not tq:
        # fallback: run synchronously (degraded)
        from tools.enrichment.pathfinder import discover
        return {"status": "ok", **discover(norm_url, req.org_id)}

    # Priority 3: user-initiated crawls run ahead of background enrichment (4/5)
    # but after interactive planned_search/research (2/3).
    job_id = tq.submit(
        "pathfinder_crawl",
        {"discovery_id": discovery_id},
        source="pathfinder_api",
        priority=3,
        org_id=req.org_id,
    )
    return {"status": "queued", "discovery_id": discovery_id, "job_id": job_id, "url": norm_url}


@router.post("/scraper/start")
def scraper_start():
    """UI button: jumpstart the scraper chain if it's idle."""
    from tools.enrichment.dispatcher import jumpstart_scraper
    return jumpstart_scraper()


@router.post("/pathfinder/start")
def pathfinder_start():
    """UI button: jumpstart the pathfinder chain if it's idle."""
    from tools.enrichment.dispatcher import jumpstart_pathfinder
    return jumpstart_pathfinder()


@router.post("/discover-agent/start")
def discover_agent_start():
    """UI button: immediately queue one discover_agent_run (bypasses scheduler interval,
    still subject to the handler's cooldown gate)."""
    from tools.enrichment.dispatcher import jumpstart_discover_agent
    return jumpstart_discover_agent()


@router.post("/pathfinder/fetch-next")
def pathfinder_fetch_next():
    from tools.enrichment.pathfinder import fetch_next
    
    row = fetch_next()
    if not row:
        return {"status": "empty", "row": None}
    return {"status": "ok", "row": row}


@router.post("/pathfinder/mark-processed")
def pathfinder_mark_processed(url_id: int):
    from tools.enrichment.pathfinder import mark_processed
    
    mark_processed(url_id)
    return {"status": "ok", "url_id": url_id}


@router.post("/scraper/run")
def scraper_run(req: ScraperRequest):
    from tools.enrichment.scraper import run_scraper
    
    result = run_scraper(req.batch_size)
    return {"status": "ok", **result}


@router.post("/scraper/scrape-next")
def scraper_scrape_next():
    from tools.enrichment.scraper import scrape_next
    
    row = scrape_next()
    if not row:
        return {"status": "empty", "row": None}
    return {"status": "ok", "row": row}


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
        from tools.research.agent import run_research_agent
        result = run_research_agent(req.plan_id)
        return {"status": result.get("status"), **result}

    org_id = 0
    try:
        row = NocodbClient()._get("research_plans", params={"where": f"(Id,eq,{req.plan_id})", "limit": 1})
        plan = row.get("list", [])[0] if row.get("list") else None
        org_id = int((plan or {}).get("org_id") or 0)
    except Exception:
        _log.warning("research_agent_run org lookup failed  plan_id=%d", req.plan_id, exc_info=True)

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


# Passing an explicit `fields` on NocoDB list GETs stops the m2m link expansion
# (which otherwise generates a massive UNION ALL query per linked table — one
# sub-SELECT per row — and tanks list-endpoint latency once tables grow).
_DISCOVERY_LIST_FIELDS = (
    "Id,org_id,url,url_hash,source_url,depth,domain,score,status,"
    "error_message,created_at,processed_at,CreatedAt"
)
_RESEARCH_PLAN_LIST_FIELDS = (
    "Id,org_id,topic,hypotheses,sub_topics,queries,schema,iterations,"
    "max_iterations,confidence_score,confidence_threshold,gap_report,"
    "paper_content,status,error_message,created_at,completed_at,CreatedAt"
)
_SCRAPE_TARGET_LIST_FIELDS = (
    "Id,org_id,url,name,category,active,frequency_hours,depth,discovered_from,"
    "auto_crawled,use_playwright,status,last_scraped_at,next_crawl_at,"
    "consecutive_failures,consecutive_unchanged,content_hash,chunk_count,"
    "last_scrape_error,CreatedAt"
)


def _get_single_row(client: NocodbClient, table: str, row_id: int) -> dict | None:
    try:
        rows = client._get(table, params={
            "where": f"(Id,eq,{row_id})",
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


@router.get("/discovery/list")
def discovery_list(org_id: int, status: str | None = None, limit: int = 50):
    limit = min(max(1, limit), 500)
    client = NocodbClient()
    params: dict = {"limit": limit, "sort": "-CreatedAt"}
    if status:
        params["where"] = f"(status,eq,{status})~and(org_id,eq,{org_id})"
    else:
        params["where"] = f"(org_id,eq,{org_id})"

    rows = client._get_paginated("discovery", params=params)
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
def research_plan_get(plan_id: int):
    """Fetch a single research plan (including paper_content) for the report UI."""
    client = NocodbClient()
    data = client._get("research_plans", params={
        "where": f"(Id,eq,{plan_id})",
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
def discovery_get(row_id: int):
    client = NocodbClient()
    row = _get_single_row(client, "discovery", row_id)
    if not row:
        return {"status": "not_found", "row": None}
    return {"status": "ok", "row": row}


@router.get("/scrape-targets/{target_id}")
def scrape_target_get(target_id: int):
    client = NocodbClient()
    row = _get_single_row(client, "scrape_targets", target_id)
    if not row:
        return {"status": "not_found", "row": None}
    return {"status": "ok", "row": row}


@router.get("/dashboard")
def enrichment_dashboard(org_id: int, limit: int = 20):
    """Combined enrichment view for frontend dashboards.

    Returns recent discovery rows, scrape targets, and verbose tool_jobs for the
    org without using `fields=` filters that previously caused NocoDB errors.
    """
    limit = min(max(1, limit), 100)
    client = NocodbClient()
    discovery_rows = client._get_paginated("discovery", params={
        "where": f"(org_id,eq,{org_id})",
        "sort": "-CreatedAt",
        "limit": limit,
    })
    scrape_target_rows = client._get_paginated("scrape_targets", params={
        "where": f"(org_id,eq,{org_id})",
        "sort": "-CreatedAt",
        "limit": limit,
    })
    queue_jobs = _recent_tool_jobs_for_org(client, org_id, limit=limit)
    return {
        "status": "ok",
        "org_id": org_id,
        "discovery": {
            "count": len(discovery_rows),
            "rows": discovery_rows,
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
