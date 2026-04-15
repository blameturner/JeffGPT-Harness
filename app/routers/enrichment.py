import logging

from fastapi import APIRouter
from pydantic import BaseModel

from infra.nocodb_client import NocodbClient
from workers.tool_queue import get_tool_queue

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
    job. The handler chains itself so pathfinder keeps walking through discovery rows."""
    from tools.enrichment.pathfinder import upsert_discovery_root

    client = NocodbClient()
    discovery_id = upsert_discovery_root(client, req.seed_url, req.org_id)
    if not discovery_id:
        return {"status": "failed", "error": "discovery_upsert_failed"}

    tq = get_tool_queue()
    if not tq:
        # fallback: run synchronously (degraded)
        from tools.enrichment.pathfinder import discover
        return {"status": "ok", **discover(req.seed_url, req.org_id)}

    job_id = tq.submit(
        "pathfinder_crawl",
        {"discovery_id": discovery_id},
        source="pathfinder_api",
        priority=4,
        org_id=req.org_id,
    )
    return {"status": "queued", "discovery_id": discovery_id, "job_id": job_id}


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
    
    job_id = tq.submit(
        "research_agent",
        {"plan_id": req.plan_id},
        source="enrichment_api",
        priority=3
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


@router.get("/discovery/list")
def discovery_list(org_id: int, status: str | None = None, limit: int = 50):
    client = NocodbClient()
    params: dict = {"limit": limit, "fields": _DISCOVERY_LIST_FIELDS}
    if status:
        params["where"] = f"(status,eq,{status})~and(org_id,eq,{org_id})"
    else:
        params["where"] = f"(org_id,eq,{org_id})"

    data = client._get("discovery", params=params)
    return {"status": "ok", "rows": data.get("list", [])}


@router.get("/research-plans/list")
def research_plans_list(org_id: int, status: str | None = None, limit: int = 50):
    # No `fields=` on research_plans — NocoDB v1 returns 404 for the whole request if
    # any listed column doesn't exist, and the schema's been in flux. research_plans
    # has no m2m links so returning all columns has no perf cost.
    client = NocodbClient()
    params: dict = {"limit": limit}
    if status:
        params["where"] = f"(status,eq,{status})~and(org_id,eq,{org_id})"
    else:
        params["where"] = f"(org_id,eq,{org_id})"

    data = client._get("research_plans", params=params)
    return {"status": "ok", "rows": data.get("list", [])}


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
        "fields": _SCRAPE_TARGET_LIST_FIELDS,
    }
    data = client._get("scrape_targets", params=params)
    return {"status": "ok", "rows": data.get("list", [])}