import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
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



def _domain_of(url: str) -> str:
    from urllib.parse import urlparse
    try:
        host = (urlparse(url or "").hostname or "").lower()
    except Exception:
        host = ""
    return host[4:] if host.startswith("www.") else host


def _canonicalise_url(url: str) -> str:
    """Strip well-known tracking params + fragment so `?utm=…` variants
    collapse onto the same scrape target. Used by the dedupe helper.
    """
    from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
    if not url:
        return ""
    DROP = {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src", "ref_url",
        "yclid", "_ga", "_gl", "igshid", "spm",
    }
    try:
        p = urlparse(url)
        kept = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if k not in DROP]
        return urlunparse(p._replace(query=urlencode(kept), fragment="")).rstrip("/")
    except Exception:
        return url


@router.get("/sources/health")
def sources_health(org_id: int, limit: int = 50):
    """Per-domain scrape health for the org.

    Groups `scrape_targets` rows by host and reports success/failure counts,
    average chunk yield, last-error, and per-domain freshness. Drives the
    'source health' panel.
    """
    db = NocodbClient()
    try:
        rows = db._get_paginated("scrape_targets", params={
            "where": f"(org_id,eq,{int(org_id)})",
            "limit": 5000,
        })
    except Exception:
        _log.warning("sources/health: scrape_targets fetch failed  org=%d", org_id, exc_info=True)
        rows = []

    by_domain: dict[str, dict] = {}
    for r in rows:
        url = (r.get("url") or "").strip()
        if not url:
            continue
        host = _domain_of(url) or "unknown"
        d = by_domain.setdefault(host, {
            "domain": host,
            "targets": 0,
            "active_targets": 0,
            "ok": 0,
            "errors": 0,
            "rejected": 0,
            "never_scraped": 0,
            "consecutive_failures_total": 0,
            "chunks_total": 0,
            "last_scraped_at": None,
            "last_error": None,
            "errors_recent": [],
        })
        d["targets"] += 1
        if int(r.get("active") or 0):
            d["active_targets"] += 1
        status = (r.get("status") or "").strip()
        if status == "ok":
            d["ok"] += 1
        elif status == "error":
            d["errors"] += 1
            err = (r.get("last_scrape_error") or "").strip()
            if err:
                d["errors_recent"].append({"url": url, "error": err[:160]})
        elif status == "rejected":
            d["rejected"] += 1
        elif not status:
            d["never_scraped"] += 1
        d["consecutive_failures_total"] += int(r.get("consecutive_failures") or 0)
        d["chunks_total"] += int(r.get("chunk_count") or 0)
        ts = r.get("last_scraped_at")
        if ts and (d["last_scraped_at"] is None or str(ts) > str(d["last_scraped_at"])):
            d["last_scraped_at"] = ts
        err = (r.get("last_scrape_error") or "").strip()
        if err and (d["last_error"] is None or status == "error"):
            d["last_error"] = err[:240]

    out = []
    for d in by_domain.values():
        d["errors_recent"] = d["errors_recent"][:5]
        finished = d["ok"] + d["errors"]
        d["success_rate"] = round(d["ok"] / finished, 3) if finished else None
        out.append(d)

    out.sort(key=lambda x: (-(x.get("targets") or 0), x.get("domain") or ""))
    return {
        "org_id": org_id,
        "domains": out[: max(1, min(int(limit), 500))],
        "total_domains": len(by_domain),
        "total_targets": sum(d["targets"] for d in by_domain.values()),
    }


class ScrapeBumpRequest(BaseModel):
    org_id: int
    query: str
    limit: int = 10


@router.post("/scrape-targets/bump")
def scrape_targets_bump(body: ScrapeBumpRequest):
    """Activity-aware priority bump: when the user starts asking about a
    topic, find pending scrape_targets whose URL or name contains any
    keyword from the query and flip their `next_crawl_at` to now so the
    next dispatch cycle picks them up.
    """
    q = (body.query or "").strip().lower()
    if not q:
        return {"status": "noop", "matched": 0}

    import re
    tokens = [t for t in re.findall(r"[a-z0-9]{3,}", q) if t not in {"the", "and", "for", "with"}][:6]
    if not tokens:
        return {"status": "noop", "matched": 0}

    db = NocodbClient()
    try:
        rows = db._get_paginated("scrape_targets", params={
            "where": f"(org_id,eq,{int(body.org_id)})~and(active,eq,1)",
            "limit": 1000,
        })
    except Exception:
        return {"status": "error", "matched": 0}

    matched = 0
    cap = max(1, min(int(body.limit), 50))
    now_iso = datetime.now(timezone.utc).isoformat()
    for r in rows:
        if matched >= cap:
            break
        hay = ((r.get("url") or "") + " " + (r.get("name") or "")).lower()
        if any(t in hay for t in tokens):
            try:
                db._patch("scrape_targets", r["Id"], {
                    "Id": r["Id"],
                    "next_crawl_at": now_iso,
                })
                matched += 1
            except Exception:
                _log.debug("scrape bump patch failed  id=%s", r.get("Id"), exc_info=True)
    return {"status": "ok", "matched": matched, "tokens": tokens}


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


# ── New suggestion API (matches frontend paths) ───────────────────────────────
#
# These wrap the existing /discovery/suggestions endpoints with the shapes the
# Enrichment UI calls. Keeping the older paths around for any cron / curl
# users; the bodies dispatch to the same logic.

class SuggestionDecisionBody(BaseModel):
    decision: str  # "approve" | "reject"
    reason: str | None = None


@router.get("/suggestions/pending")
def suggestions_pending(org_id: int, limit: int = 50):
    """Pending suggestions (the Enrichment inbox). Sourced from
    ``suggested_scrape_targets`` where status='pending'.

    Includes rows seeded by the harvest runner (look for ``query`` starting
    with ``harvest:`` or ``reason`` containing 'harvest run #N') so a single
    inbox surfaces both flows.

    Returns the UI-shaped ``{suggestions: [...]}`` payload used by Live, plus
    the legacy ``{status, rows}`` keys for any older caller still wired to
    the discovery endpoint shape."""
    legacy = discovery_suggestions_list(org_id=org_id, status="pending", limit=limit)
    rows = legacy.get("rows") or []
    suggestions = [_suggestion_for_ui(r) for r in rows]
    return {
        "status": "ok",
        "suggestions": suggestions,
        # Legacy keys kept for any existing caller.
        "rows": rows,
    }


def _suggestion_for_ui(row: dict) -> dict:
    """Map a suggested_scrape_targets row to the shape the Live UI expects.

    The UI's ``EnrichmentSuggestion`` type wants ``{id, kind, title, summary,
    created_at, source, status, evidence_count}``; the DB row gives us
    ``{Id, url, title, query, relevance, score, reason, status, CreatedAt}``.
    """
    # `score` (0-100 relevance) and `evidence_count` (number of supporting
    # docs) are different signals — don't fake one from the other. Keep
    # score as a real field; leave evidence_count out so the UI shows '—'
    # instead of a misleading number.
    return {
        "id": str(row.get("Id")),
        "kind": "scrape_target",
        "title": row.get("title") or row.get("url"),
        "summary": row.get("reason") or row.get("query") or "",
        "created_at": row.get("CreatedAt") or "",
        "source": _suggestion_source_tag(row),
        "status": row.get("status") or "pending",
        "score": row.get("score"),
    }


@router.get("/suggestions/preview/{suggested_id}")
def suggestion_preview(suggested_id: int, org_id: int):
    """Suggestion detail. Returns the row plus a normalised preview snippet
    derived from existing columns — the UI doesn't need a full HEAD/GET cycle
    here; it just wants something to show next to the approve/reject buttons.
    """
    client = NocodbClient()
    row = _get_single_row(client, NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS, suggested_id, org_id=org_id)
    if not row:
        return {"status": "not_found", "row": None, "preview": None}
    preview = {
        "url": row.get("url"),
        "title": row.get("title") or row.get("url"),
        "reason": row.get("reason"),
        "query": row.get("query"),
        "relevance": row.get("relevance"),
        "score": row.get("score"),
        "status": row.get("status"),
        "source": _suggestion_source_tag(row),
    }
    return {"status": "ok", "row": row, "preview": preview}


@router.post("/suggestions/{suggested_id}/decision")
def suggestion_decision(suggested_id: int, body: SuggestionDecisionBody, org_id: int):
    decision = (body.decision or "").strip().lower()
    if decision == "approve":
        return discovery_suggestion_approve(suggested_id, org_id=org_id)
    if decision == "reject":
        return discovery_suggestion_reject(suggested_id, org_id=org_id, reason=body.reason)
    if decision == "defer":
        # No queue side-effects — just shelve the row so it stops showing in
        # the pending inbox until a human revisits it.
        client = NocodbClient()
        try:
            client._patch(NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS, suggested_id, {
                "Id": suggested_id,
                "status": "deferred",
                "reviewed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            })
        except Exception:
            return {"status": "failed", "error": "patch_failed"}
        return {"status": "ok", "suggested_id": suggested_id, "decision": "defer"}
    return {"status": "failed", "error": "decision must be 'approve', 'reject', or 'defer'"}


def _suggestion_source_tag(row: dict) -> str:
    """Identify whether this suggestion came from harvest, the discover_agent,
    or a manual user submission — useful for the Enrichment UI badge."""
    q = str(row.get("query") or "")
    reason = str(row.get("reason") or "")
    if q.startswith("harvest:") or "harvest run" in reason:
        return "harvest"
    if q == "manual_entry" or "user-submitted" in reason:
        return "manual"
    return "discover_agent"


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


@router.delete("/research/plans/{plan_id}")
def research_delete_plan(plan_id: int, org_id: int | None = None):
    """Hard-delete a research plan + cancel any running tool_jobs that
    reference it. The Chroma corpus and any insight rows linked to the
    plan are LEFT in place — that work is durable and could be useful
    even if the plan itself is gone. Caller can fan-out separately if
    they want a deeper purge.
    """
    from infra.nocodb_client import NocodbClient
    from workers.tool_queue import get_tool_queue

    client = NocodbClient()
    rows = client._get("research_plans", params={
        "where": f"(Id,eq,{plan_id})",
        "limit": 1,
    }).get("list", [])
    if not rows:
        raise HTTPException(status_code=404, detail=f"plan {plan_id} not found")
    row = rows[0]
    if org_id is not None and int(row.get("org_id") or 0) != int(org_id):
        raise HTTPException(status_code=403, detail="plan belongs to a different org")

    # Best-effort: cancel any tool_jobs that target this plan_id. We don't
    # block on the results — research_agent / research_planner handlers
    # check is_cancelled() between phases and abort cleanly.
    cancelled_jobs: list[str] = []
    try:
        q = get_tool_queue()
        if q is not None:
            tool_rows = client._get_paginated("tool_jobs", params={
                "where": "(type,in,research_agent,research_planner,research_review,research_op)"
                         "~and(status,in,queued,running)",
                "limit": 200,
            })
            import json as _json
            for tr in tool_rows:
                raw = tr.get("payload_json") or tr.get("payload") or "{}"
                try:
                    payload = _json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    continue
                if int(payload.get("plan_id") or 0) == int(plan_id):
                    job_id = tr.get("job_id")
                    if job_id and q.cancel_running(job_id, reason=f"plan {plan_id} deleted"):
                        cancelled_jobs.append(job_id)
    except Exception:
        _log.warning("research_delete_plan: cancel tool_jobs failed  plan=%d", plan_id, exc_info=True)

    try:
        client._delete("research_plans", plan_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"delete failed: {e}")

    return {
        "status": "deleted",
        "plan_id": plan_id,
        "cancelled_jobs": cancelled_jobs,
    }


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


class ResearchOpRequest(BaseModel):
    params: dict = {}


@router.post("/research/{plan_id}/ops/{kind}")
def research_op_invoke(plan_id: int, kind: str, body: ResearchOpRequest | None = None):
    """Generic post-build operation invoker.

    `kind` is one of the keys in ``ASYNC_OPS`` or ``SYNC_OPS`` in
    ``tools.research.operations``. Sync ops run inline; async ops are queued.
    The frontend POSTs JSON like ``{"params": {"section_title": "..."}}``.
    """
    from fastapi import HTTPException
    from tools.research.operations import ASYNC_OPS, SYNC_OPS, run_research_op

    params = (body.params if body else {}) or {}

    if kind in SYNC_OPS:
        return run_research_op({"plan_id": plan_id, "kind": kind, "params": params})

    if kind not in ASYNC_OPS:
        raise HTTPException(status_code=400, detail=f"unknown op kind: {kind}")

    tq = get_tool_queue()
    if not tq:
        raise HTTPException(status_code=503, detail="tool_queue_unavailable")

    org_id = 1
    try:
        row = NocodbClient()._get("research_plans", params={"where": f"(Id,eq,{plan_id})", "limit": 1})
        plan = row.get("list", [])[0] if row.get("list") else None
        if plan:
            org_id = resolve_org_id(plan.get("org_id"))
    except Exception:
        _log.warning("research_op org lookup failed  plan_id=%d  kind=%s", plan_id, kind, exc_info=True)
    if org_id <= 0:
        org_id = 1

    job_id = tq.submit(
        "research_op",
        {"plan_id": plan_id, "kind": kind, "params": params},
        source=f"research_op_{kind}",
        priority=3,
        org_id=org_id,
    )
    return {"status": "queued", "plan_id": plan_id, "kind": kind, "job_id": job_id}


@router.get("/research/{plan_id}/artifacts")
def research_artifacts(plan_id: int):
    """Return all stashed artifacts for a plan (slide_deck, email_tldr,
    qa_pack, action_plan, fact_check, citation_audit, …).

    Artifacts live inside the existing ``schema`` JSON column under the
    reserved ``_artifacts`` key — no dedicated DB column required.
    """
    import json as _json
    from fastapi import HTTPException
    row = NocodbClient()._get("research_plans", params={"where": f"(Id,eq,{plan_id})", "limit": 1})
    plan = row.get("list", [])[0] if row.get("list") else None
    if not plan:
        raise HTTPException(status_code=404, detail="plan not found")
    raw_schema = plan.get("schema") or "{}"
    try:
        schema = _json.loads(raw_schema) if isinstance(raw_schema, str) else (raw_schema or {})
    except (_json.JSONDecodeError, TypeError):
        schema = {}
    arts = schema.get("_artifacts") if isinstance(schema, dict) else {}
    return {"plan_id": plan_id, "artifacts": arts if isinstance(arts, dict) else {}}


@router.get("/research/doc-types")
def research_doc_types():
    """Catalog of supported document types — frontend uses this to drive the
    new-paper picker and the reframe selector."""
    from tools.research.agent import DOC_TYPES, DEFAULT_DOC_TYPE
    return {
        "default": DEFAULT_DOC_TYPE,
        "types": [
            {"key": k, "opener": v["opener"], "closer": v["closer"], "tone": v["tone"]}
            for k, v in DOC_TYPES.items()
        ],
    }


@router.post("/research/{plan_id}/start")
def research_start(plan_id: int):
    """Invoke a deferred (hidden) research plan — clears its hidden type and
    queues the planner job that will then run the agent inline."""
    from tools.research.research_planner import start_research_plan
    return start_research_plan(plan_id)


class ResearchReviewRequest(BaseModel):
    instructions: str = ""


@router.post("/research/{plan_id}/review")
def research_review(plan_id: int, body: ResearchReviewRequest | None = None):
    """Trigger an explicit review pass on an already-completed research paper.

    A reviewer model reads the paper (with the user's optional ``instructions``
    folded in) and emits per-section revision notes; the writer rebuilds the
    affected sections and the new paper replaces the old. Runs async via the
    tool queue so a long review never blocks the request.
    """
    from fastapi import HTTPException
    tq = get_tool_queue()
    if not tq:
        raise HTTPException(status_code=503, detail="tool_queue_unavailable")

    instructions = (body.instructions if body else "") or ""

    org_id = 1
    try:
        row = NocodbClient()._get("research_plans", params={"where": f"(Id,eq,{plan_id})", "limit": 1})
        plan = row.get("list", [])[0] if row.get("list") else None
        if plan:
            org_id = resolve_org_id(plan.get("org_id"))
    except Exception:
        _log.warning("research_review org lookup failed plan_id=%d", plan_id, exc_info=True)
    if org_id <= 0:
        org_id = 1

    job_id = tq.submit(
        "research_review",
        {"plan_id": plan_id, "org_id": org_id, "instructions": instructions},
        source="research_review",
        priority=3,
        org_id=org_id,
    )
    return {"status": "queued", "plan_id": plan_id, "job_id": job_id}


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
                get_feature("tool_queue", "background_chat_idle_seconds", 1800) or 1800
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
def research_plans_list(org_id: int, status: str | None = None,
                        limit: int = 50, include_hidden: bool = False,
                        parent_insight_id: int | None = None):
    """List research plans for the org.

    Plans tagged ``type='pending insight'`` (auto-spawned follow-ups created
    from an insight) are excluded by default — they clutter the main feed.
    Pass ``include_hidden=true`` to surface them, or ``parent_insight_id=N``
    to scope to a single insight (in which case hidden plans are always
    included since that's the whole point of the view).
    """
    limit = min(max(1, limit), 500)
    client = NocodbClient()
    parts = [f"(org_id,eq,{org_id})"]
    if status:
        parts.append(f"(status,eq,{status})")
    if parent_insight_id is not None:
        parts.append(f"(parent_insight_id,eq,{int(parent_insight_id)})")
    # Fetch a generous slice so the post-filter for hidden rows still
    # honours `limit` after exclusion. The list grows to ~hundreds, not
    # tens of thousands, so this isn't expensive.
    fetch_limit = limit if (include_hidden or parent_insight_id is not None) else min(500, limit * 3)
    params: dict = {"limit": fetch_limit, "where": "~and".join(parts)}
    rows = client._get_paginated("research_plans", params=params)
    if parent_insight_id is None and not include_hidden:
        # Filter Python-side instead of via NocoDB's where clause:
        # `(type,neq,X)` excludes rows where type IS NULL in some NocoDB
        # builds (NULL semantics are inconsistent across versions). A
        # post-filter is cheap and unambiguous.
        rows = [r for r in rows if (r.get("type") or "") != "pending insight"]
        rows = rows[:limit]
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
