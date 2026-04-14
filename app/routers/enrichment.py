import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from infra.nocodb_client import NocodbClient

_log = logging.getLogger("main.enrichment")

router = APIRouter()


class PathfinderRequest(BaseModel):
    seed_url: str
    org_id: int
    max_depth: int = 3


class ScraperRequest(BaseModel):
    org_id: int
    batch_size: int = 10


class ResearchRequest(BaseModel):
    topic: str
    org_id: int


@router.post("/pathfinder/discover")
def pathfinder_discover(req: PathfinderRequest):
    from tools.enrichment.pathfinder import discover
    
    result = discover(req.seed_url, req.org_id, req.max_depth)
    return {"status": "ok", **result}


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


@router.get("/discovery/list")
def discovery_list(org_id: int, status: str | None = None, limit: int = 50):
    client = NocodbClient()
    params = {"limit": limit}
    if status:
        params["where"] = f"(status,eq,{status})~and(org_id,eq,{org_id})"
    else:
        params["where"] = f"(org_id,eq,{org_id})"
    
    data = client._get("discovery", params=params)
    return {"status": "ok", "rows": data.get("list", [])}


@router.get("/research-plans/list")
def research_plans_list(org_id: int, status: str | None = None, limit: int = 50):
    client = NocodbClient()
    params = {"limit": limit}
    if status:
        params["where"] = f"(status,eq,{status})~and(org_id,eq,{org_id})"
    else:
        params["where"] = f"(org_id,eq,{org_id})"
    
    data = client._get("research_plans", params=params)
    return {"status": "ok", "rows": data.get("list", [])}