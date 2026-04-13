import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

_log = logging.getLogger("main.enrichment")

router = APIRouter()


class EnrichmentAgentCreate(BaseModel):
    org_id: int
    name: str
    description: str | None = None
    category: str | None = None
    token_budget: int = 50000
    cron_expression: str | None = None
    timezone: str = "Australia/Sydney"
    active: bool = True


class EnrichmentAgentUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    category: str | None = None
    token_budget: int | None = None
    cron_expression: str | None = None
    timezone: str | None = None
    active: bool | None = None


class SourceCreate(BaseModel):
    org_id: int
    url: str
    name: str
    category: str = "documentation"
    frequency_hours: float = 24
    active: bool = True
    enrichment_agent_id: int | None = None
    use_playwright: bool = False


class SourceUpdate(BaseModel):
    name: str | None = None
    url: str | None = None
    category: str | None = None
    frequency_hours: float | None = None
    active: bool | None = None
    enrichment_agent_id: int | None = None
    use_playwright: bool | None = None


class SuggestionUpdate(BaseModel):
    status: str | None = None


class SuggestionApprove(BaseModel):
    enrichment_agent_id: int | None = None


class BulkSuggestionAction(BaseModel):
    parent_target: int
    enrichment_agent_id: int | None = None


@router.get("/enrichment/agents")
def list_enrichment_agents(org_id: int | None = None):
    from workers.enrichment.db import EnrichmentDB
    try:
        db = EnrichmentDB()
        return {"agents": db.list_enrichment_agents(org_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enrichment/agents")
def create_enrichment_agent(body: EnrichmentAgentCreate):
    from workers.enrichment.db import EnrichmentDB
    try:
        db = EnrichmentDB()
        agent = db.create_enrichment_agent(body.model_dump())
        return agent
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/enrichment/agents/{agent_id}")
def update_enrichment_agent(agent_id: int, body: EnrichmentAgentUpdate, request: Request):
    from workers.enrichment.db import EnrichmentDB
    from scheduler import reload_agent_schedules
    try:
        db = EnrichmentDB()
        updates = {k: v for k, v in body.model_dump().items() if v is not None}
        if not updates:
            return db.get_enrichment_agent(agent_id)
        result = db.update_enrichment_agent(agent_id, updates)
        # Reload scheduler if cron or timezone changed so new schedule takes effect immediately.
        if "cron_expression" in updates or "timezone" in updates or "active" in updates:
            reload_agent_schedules()
            _log.info("scheduler reloaded after enrichment agent %d update", agent_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enrichment/agents/{agent_id}/trigger")
def trigger_enrichment_agent(agent_id: int):
    import threading
    from workers.enrichment.cycle import seed_enrichment_jobs
    threading.Thread(target=seed_enrichment_jobs, args=[agent_id], daemon=True).start()
    return {"status": "triggered", "agent_id": agent_id}


@router.get("/enrichment/agents/{agent_id}/status")
def enrichment_agent_status(agent_id: int, request: Request):
    from workers.enrichment.cycle import get_last_run, sources_due_count
    sched = getattr(request.app.state, "scheduler", None)
    next_run = None
    if sched:
        job = sched.get_job(f"enrichment_agent_{agent_id}")
        if job and job.next_run_time:
            next_run = job.next_run_time.isoformat()
    return {
        "agent_id": agent_id,
        "next_run": next_run,
        "last_run": get_last_run(agent_id),
        "sources_due": sources_due_count(agent_id),
    }


@router.get("/enrichment/sources")
def list_sources(org_id: int, agent_id: int | None = None, active_only: bool = False):
    from workers.enrichment.db import EnrichmentDB
    try:
        db = EnrichmentDB()
        return {"sources": db.list_sources(org_id, enrichment_agent_id=agent_id, active_only=active_only)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enrichment/sources")
def create_source(body: SourceCreate):
    from workers.enrichment.db import EnrichmentDB
    try:
        db = EnrichmentDB()
        return db.create_source(body.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enrichment/sources/{source_id}")
def get_source(source_id: int):
    from workers.enrichment.db import EnrichmentDB
    try:
        db = EnrichmentDB()
        source = db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        return source
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/enrichment/sources/{source_id}")
def update_source(source_id: int, body: SourceUpdate):
    from workers.enrichment.db import EnrichmentDB
    try:
        db = EnrichmentDB()
        updates = {k: v for k, v in body.model_dump().items() if v is not None}
        if not updates:
            return db.get_source(source_id)
        db.update_scrape_target(source_id, **updates)
        return db.get_source(source_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/enrichment/sources/{source_id}")
def delete_source(source_id: int):
    from workers.enrichment.db import EnrichmentDB
    try:
        db = EnrichmentDB()
        db.delete_source(source_id)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enrichment/sources/{source_id}/flush")
def flush_source(source_id: int):
    from workers.enrichment.db import EnrichmentDB
    try:
        db = EnrichmentDB()
        return db.flush_source(source_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enrichment/sources/{source_id}/log")
def source_log(source_id: int, limit: int = 50):
    from workers.enrichment.db import EnrichmentDB
    try:
        db = EnrichmentDB()
        return {"log": db.list_log(scrape_target_id=source_id, limit=limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enrichment/log")
def enrichment_log(org_id: int | None = None, limit: int = 100):
    from workers.enrichment.db import EnrichmentDB
    try:
        db = EnrichmentDB()
        return {"log": db.list_log(org_id=org_id, limit=limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enrichment/suggestions")
def list_suggestions(org_id: int, status: str | None = None):
    from workers.enrichment.db import EnrichmentDB
    try:
        db = EnrichmentDB()
        return {"suggestions": db.list_suggestions(org_id, status=status)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enrichment/suggestions/{suggestion_id}")
def get_suggestion(suggestion_id: int):
    from workers.enrichment.db import EnrichmentDB
    try:
        db = EnrichmentDB()
        suggestion = db.get_suggestion(suggestion_id)
        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        return suggestion
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/enrichment/suggestions/{suggestion_id}")
def update_suggestion(suggestion_id: int, body: SuggestionUpdate):
    from workers.enrichment.db import EnrichmentDB
    try:
        db = EnrichmentDB()
        updates = {k: v for k, v in body.model_dump().items() if v is not None}
        if not updates:
            return db.get_suggestion(suggestion_id)
        return db.update_suggestion(suggestion_id, updates)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enrichment/suggestions/{suggestion_id}/approve")
def approve_suggestion(suggestion_id: int, body: SuggestionApprove | None = None):
    from workers.enrichment.db import EnrichmentDB
    try:
        db = EnrichmentDB()
        suggestion = db.get_suggestion(suggestion_id)
        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        org_id = int(suggestion.get("org_id") or 1)
        agent_id = body.enrichment_agent_id if body else None
        source = db.approve_suggestion(suggestion_id, org_id, enrichment_agent_id=agent_id)
        return {"ok": True, "source": source}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enrichment/suggestions/{suggestion_id}/reject")
def reject_suggestion(suggestion_id: int):
    from workers.enrichment.db import EnrichmentDB
    try:
        db = EnrichmentDB()
        return db.update_suggestion(suggestion_id, {"status": "rejected"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enrichment/suggestions/approve-by-parent")
def approve_suggestions_by_parent(body: BulkSuggestionAction):
    from workers.enrichment.db import EnrichmentDB
    from config import NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS
    try:
        db = EnrichmentDB()
        suggestions = db._get(
            NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS,
            params={
                "where": f"(parent_target,eq,{body.parent_target})~and(status,eq,pending)",
                "limit": 200,
            },
        ).get("list", [])
        if not suggestions:
            return {"ok": True, "approved": 0, "sources": []}
        sources = []
        for s in suggestions:
            org_id = int(s.get("org_id") or 1)
            source = db.approve_suggestion(s["Id"], org_id, enrichment_agent_id=body.enrichment_agent_id)
            sources.append(source)
        return {"ok": True, "approved": len(sources), "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enrichment/suggestions/reject-by-parent")
def reject_suggestions_by_parent(body: BulkSuggestionAction):
    from workers.enrichment.db import EnrichmentDB
    from config import NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS
    try:
        db = EnrichmentDB()
        suggestions = db._get(
            NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS,
            params={
                "where": f"(parent_target,eq,{body.parent_target})~and(status,eq,pending)",
                "limit": 200,
            },
        ).get("list", [])
        for s in suggestions:
            db.update_suggestion(s["Id"], {"status": "rejected"})
        return {"ok": True, "rejected": len(suggestions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enrichment/scrape-report")
def scrape_report(org_id: int):
    from workers.enrichment.db import EnrichmentDB
    try:
        db = EnrichmentDB()
        targets = db.list_all_targets(org_id)
        total = len(targets)
        by_status: dict[str, int] = {}
        by_error: dict[str, int] = {}
        failed_urls: list[dict] = []
        for t in targets:
            s = t.get("status") or "unknown"
            by_status[s] = by_status.get(s, 0) + 1
            if s in ("error", "rejected"):
                err = t.get("last_scrape_error") or "unknown"
                by_error[err] = by_error.get(err, 0) + 1
                fails = int(t.get("consecutive_failures") or 0)
                if fails >= 3:
                    failed_urls.append({
                        "url": t.get("url", "")[:100],
                        "consecutive_failures": fails,
                        "error": err,
                    })
        return {
            "total": total,
            "by_status": by_status,
            "by_error": by_error,
            "top_failures": sorted(by_error.items(), key=lambda x: -x[1])[:20],
            "persistent_failures": sorted(failed_urls, key=lambda x: -x["consecutive_failures"])[:20],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
