"""Home dashboard API.

The frontend (separate repo) paints one page from ``GET /home/overview`` and
then calls follow-ups for chat, question answers, schedule triggers, and
widget data. Streaming responses piggyback on the existing
``/stream/{job_id}`` SSE endpoint exposed by the agents router.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from infra.config import (
    NOCODB_TABLE_ASSISTANT_QUESTIONS,
    NOCODB_TABLE_DAILY_DIGESTS,
    get_feature,
    is_feature_enabled,
)
from infra.nocodb_client import NocodbClient
from infra.scheduler_introspect import get_next_runs, get_schedule_meta
from shared import digest_reader, home_questions
from shared.home_conversation import get_or_create_home_conversation, home_conversation_summary
from shared.jobs import STORE, run_in_background
from workers.chat.home import run_home_turn

_log = logging.getLogger("main.home")

router = APIRouter(prefix="/home", tags=["home"])


# ---- schemas -----------------------------------------------------------------

class HomeChatRequest(BaseModel):
    org_id: int
    model: str = "chat"
    message: str
    response_style: str | None = None
    search_mode: Literal["disabled", "basic", "standard"] = "basic"
    search_consent_confirmed: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


class AnswerRequest(BaseModel):
    org_id: int
    selected_option: str = ""
    answer_text: str = ""
    model: str = "chat"
    response_style: str | None = None

    @property
    def has_content(self) -> bool:
        return bool(self.selected_option or self.answer_text)


class DismissRequest(BaseModel):
    org_id: int
    reason: str = ""


class RetractRequest(BaseModel):
    org_id: int


class RunNowRequest(BaseModel):
    task: str | None = None
    product: str | None = None


class DigestFeedbackRequest(BaseModel):
    org_id: int
    signal: Literal["up", "down"]
    domain: str = ""
    note: str = ""


class SearchRequest(BaseModel):
    org_id: int
    query: str
    collection: str = "agent_outputs"
    n_results: int = 8


# ---- rate limit --------------------------------------------------------------

_RATE_WINDOW_S = 60.0
_RATE_MAX_CALLS = 30
_rate_buckets: dict[int, deque] = {}
_rate_lock = threading.Lock()


def _rate_check(org_id: int) -> None:
    now = time.time()
    with _rate_lock:
        q = _rate_buckets.setdefault(org_id, deque())
        while q and now - q[0] > _RATE_WINDOW_S:
            q.popleft()
        if len(q) >= _RATE_MAX_CALLS:
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        q.append(now)


# ---- widget helper -----------------------------------------------------------

def _widget(name: str) -> dict:
    cfg = get_feature("home", name, {})
    enabled = bool(cfg.get("enabled")) if isinstance(cfg, dict) else False
    return {"enabled": enabled, "message": "" if enabled else "Not configured", "data": None}


# ---- endpoints ---------------------------------------------------------------

@router.get("/overview")
def home_overview(org_id: int, request: Request):
    if not is_feature_enabled("home"):
        raise HTTPException(status_code=404, detail="home feature disabled")

    client = NocodbClient()
    out: dict = {"org_id": org_id}

    try:
        # Overview ships metadata only — full markdown via /home/digest.
        out["digest"] = digest_reader.as_payload(
            digest_reader.latest_digest(client, org_id),
            include_markdown=False,
        )
    except Exception:
        _log.warning("overview: digest panel failed  org=%d", org_id, exc_info=True)
        out["digest"] = None

    try:
        out["pending_questions"] = home_questions.list_pending(org_id, limit=5)
    except Exception:
        _log.warning("overview: questions panel failed  org=%d", org_id, exc_info=True)
        out["pending_questions"] = []

    try:
        from shared import insights as insights_mod
        out["recent_insights"] = insights_mod.list_recent(org_id, limit=3)
    except Exception:
        _log.warning("overview: insights panel unavailable  org=%d", org_id, exc_info=True)
        out["recent_insights"] = []

    try:
        out["home_conversation"] = home_conversation_summary(org_id)
    except Exception:
        _log.warning("overview: conversation panel failed  org=%d", org_id, exc_info=True)
        out["home_conversation"] = None

    try:
        sched = getattr(request.app.state, "scheduler", None)
        out["schedules"] = get_next_runs(sched, org_id=org_id)
    except Exception:
        _log.warning("overview: schedules panel failed  org=%d", org_id, exc_info=True)
        out["schedules"] = []

    out["widgets"] = {
        "email": _widget("email"),
        "calendar": _widget("calendar"),
        "graph": _widget("graph"),
    }
    return out


@router.get("/digest")
def home_digest(org_id: int, date: str | None = None):
    client = NocodbClient()
    row = digest_reader.digest_for_date(client, org_id, date) if date else digest_reader.latest_digest(client, org_id)
    if not row:
        raise HTTPException(status_code=404, detail="no digest available")
    return digest_reader.as_payload(row)


@router.get("/digests")
def home_digests(org_id: int, limit: int = 7):
    client = NocodbClient()
    rows = digest_reader.list_digests(client, org_id, limit=min(max(1, limit), 30))
    # Metadata only — don't load markdown for every row.
    return {"digests": [digest_reader.as_payload(r, include_markdown=False) for r in rows]}


class DigestRunRequest(BaseModel):
    org_id: int


@router.post("/digest/run")
def run_digest_now(body: DigestRunRequest):
    """Manually kick the daily digest (bypasses the morning cron)."""
    try:
        from workers.tool_queue import get_tool_queue
        tq = get_tool_queue()
    except Exception:
        raise HTTPException(status_code=503, detail="tool queue unavailable")
    if not tq:
        raise HTTPException(status_code=503, detail="tool queue unavailable")
    job_id = tq.submit(
        "daily_digest",
        {"org_id": body.org_id},
        source="home_manual",
        org_id=body.org_id,
    )
    return {"status": "queued", "tool_job_id": job_id}


@router.post("/digest/{digest_id}/feedback")
def digest_feedback(digest_id: int, body: DigestFeedbackRequest):
    client = NocodbClient()
    if "digest_feedback" not in client.tables:
        raise HTTPException(status_code=503, detail="digest_feedback table not provisioned")
    try:
        row = client._post("digest_feedback", {
            "digest_id": digest_id,
            "org_id": body.org_id,
            "signal": body.signal,
            "domain": body.domain or "",
            "note": body.note or "",
        })
        return {"status": "ok", "id": row.get("Id")}
    except Exception as e:
        _log.warning("digest_feedback write failed  id=%d", digest_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
def home_chat(request: HomeChatRequest):
    _rate_check(request.org_id)
    _log.info(
        "POST /home/chat  org=%d model=%s chars=%d",
        request.org_id, request.model, len(request.message),
    )
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    job = STORE.create()

    def _worker(j):
        run_home_turn(
            j,
            org_id=request.org_id,
            model=request.model,
            message=request.message,
            response_style=request.response_style,
            search_mode=request.search_mode,
            search_consent_confirmed=request.search_consent_confirmed,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

    run_in_background(job, _worker)
    return {"job_id": job.id}


@router.get("/questions")
def list_questions(org_id: int, status: str = "pending", limit: int = 20):
    if status != "pending":
        raise HTTPException(status_code=400, detail="only status=pending supported in v1")
    return {"questions": home_questions.list_pending(org_id, limit=limit)}


@router.post("/questions/{question_id}/answer")
def answer_question(question_id: int, body: AnswerRequest):
    if not body.has_content:
        raise HTTPException(status_code=400, detail="selected_option or answer_text required")

    question = home_questions.get_question(question_id)
    if not question:
        raise HTTPException(status_code=404, detail="question not found")
    if int(question.get("org_id") or 0) != int(body.org_id):
        raise HTTPException(status_code=403, detail="question belongs to a different org")
    if question.get("status") != home_questions.STATUS_PENDING:
        raise HTTPException(status_code=409, detail=f"question already {question.get('status')}")

    composed = home_questions.render_answer(question, body.selected_option, body.answer_text)
    job = STORE.create()

    def _worker(j):
        run_home_turn(
            j,
            org_id=body.org_id,
            model=body.model,
            message=composed,
            answer_question_id=question_id,
            answer_selected_option=body.selected_option,
            answer_free_text=body.answer_text,
            response_style=body.response_style,
            lightweight=True,
        )

    run_in_background(job, _worker)
    return {"job_id": job.id}


@router.post("/questions/{question_id}/dismiss")
def dismiss_question(question_id: int, body: DismissRequest):
    question = home_questions.get_question(question_id)
    if not question:
        raise HTTPException(status_code=404, detail="question not found")
    if int(question.get("org_id") or 0) != int(body.org_id):
        raise HTTPException(status_code=403, detail="question belongs to a different org")
    if question.get("status") != home_questions.STATUS_PENDING:
        raise HTTPException(status_code=409, detail=f"question already {question.get('status')}")
    home_questions.mark_dismissed(question_id, reason=body.reason)
    return {"status": "dismissed"}


@router.post("/questions/{question_id}/retract")
def retract_answer(question_id: int, body: RetractRequest):
    question = home_questions.get_question(question_id)
    if not question:
        raise HTTPException(status_code=404, detail="question not found")
    if int(question.get("org_id") or 0) != int(body.org_id):
        raise HTTPException(status_code=403, detail="question belongs to a different org")
    if question.get("status") != home_questions.STATUS_ANSWERED:
        raise HTTPException(status_code=409, detail=f"question is {question.get('status')}")
    home_questions.mark_pending(question_id)
    return {"status": "pending"}


@router.get("/schedules")
def list_schedules(org_id: int, request: Request):
    sched = getattr(request.app.state, "scheduler", None)
    return {"schedules": get_next_runs(sched, org_id=org_id)}


@router.post("/schedules/{schedule_id}/run-now")
def trigger_schedule(schedule_id: int, body: RunNowRequest):
    meta = get_schedule_meta(schedule_id)
    if not meta:
        raise HTTPException(status_code=404, detail="schedule not found")

    from scheduler import trigger_agent_job
    from tools._org import resolve_org_id

    agent_name = (meta.get("agent_name") or "").strip()
    if not agent_name:
        raise HTTPException(status_code=400, detail="schedule has no agent_name")
    org_id = resolve_org_id(meta.get("org_id"))
    task = body.task if body.task is not None else (meta.get("task_description") or "")
    product = body.product if body.product is not None else (meta.get("product") or "")

    trigger_agent_job(agent_name, org_id, task, product)
    return {"status": "dispatched", "agent_name": agent_name, "org_id": org_id}


@router.get("/widgets/email")
def widget_email():
    return _widget("email")


@router.get("/widgets/calendar")
def widget_calendar():
    return _widget("calendar")


@router.get("/widgets/graph")
def widget_graph(org_id: int, limit: int = 10):
    """Top entities + sparse concepts + top reinforced edges from Falkor.

    Uses the provenance fields (hits, last_seen, aliases) added by F1/F2 so
    the UI can show what the system has actually learned vs one-off
    extractions.
    """
    try:
        from infra.graph import get_graph, get_sparse_concepts
    except Exception:
        _log.warning("widget/graph: import failed", exc_info=True)
        return {"enabled": False, "message": "graph unavailable", "data": None}

    data: dict = {"top_entities": [], "sparse_concepts": [], "top_edges": []}
    try:
        graph = get_graph(org_id)
        result = graph.query(
            "MATCH (n) OPTIONAL MATCH (n)-[r]-() "
            "WITH n, count(r) AS deg, coalesce(n.aliases, []) AS aliases "
            "WHERE deg > 0 "
            "RETURN labels(n)[0], n.name, deg, aliases "
            "ORDER BY deg DESC LIMIT $limit",
            {"limit": int(limit)},
        )
        data["top_entities"] = [
            {"type": row[0], "name": row[1], "degree": row[2], "aliases": list(row[3] or [])}
            for row in result.result_set if row and row[1]
        ]
    except Exception:
        _log.warning("widget/graph top entities failed  org=%d", org_id, exc_info=True)

    try:
        data["sparse_concepts"] = get_sparse_concepts(org_id, limit=limit, max_degree=2)
    except Exception:
        _log.warning("widget/graph sparse failed  org=%d", org_id, exc_info=True)

    try:
        graph = get_graph(org_id)
        result = graph.query(
            "MATCH (a)-[r]->(b) "
            "RETURN a.name, type(r), b.name, "
            "  coalesce(r.hits,1) AS hits, coalesce(r.weight,1.0) AS weight, "
            "  r.last_seen AS last_seen "
            "ORDER BY hits DESC, weight DESC LIMIT $limit",
            {"limit": int(limit)},
        )
        data["top_edges"] = [
            {
                "from": row[0], "relationship": row[1], "to": row[2],
                "hits": int(row[3] or 1), "weight": float(row[4] or 1.0),
                "last_seen": row[5],
            }
            for row in result.result_set if row
        ]
    except Exception:
        _log.warning("widget/graph edges failed  org=%d", org_id, exc_info=True)

    return {"enabled": True, "message": "", "data": data}


class GraphJobRequest(BaseModel):
    org_id: int


@router.post("/graph/resolve-entities")
def run_entity_resolution(body: GraphJobRequest):
    """Manually run the alias-merge job (normally runs daily)."""
    try:
        from workers.tool_queue import get_tool_queue
        tq = get_tool_queue()
    except Exception:
        raise HTTPException(status_code=503, detail="tool queue unavailable")
    if not tq:
        raise HTTPException(status_code=503, detail="tool queue unavailable")
    job_id = tq.submit("graph_resolve_entities", {"org_id": body.org_id},
                       source="home_manual", org_id=body.org_id)
    return {"status": "queued", "tool_job_id": job_id}


@router.post("/graph/maintenance")
def run_graph_maintenance(body: GraphJobRequest):
    """Manually run the weekly maintenance pass (co-occurrence + decay + prune)."""
    try:
        from workers.tool_queue import get_tool_queue
        tq = get_tool_queue()
    except Exception:
        raise HTTPException(status_code=503, detail="tool queue unavailable")
    if not tq:
        raise HTTPException(status_code=503, detail="tool queue unavailable")
    job_id = tq.submit("graph_maintenance", {"org_id": body.org_id},
                       source="home_manual", org_id=body.org_id)
    return {"status": "queued", "tool_job_id": job_id}


@router.get("/widgets/activity")
def widget_activity(org_id: int, limit: int = 10):
    """Recent agent_runs for the org — a 'what has the system been doing' panel."""
    client = NocodbClient()
    try:
        rows = client._get_paginated("agent_runs", params={
            "where": f"(org_id,eq,{org_id})",
            "sort": "-CreatedAt",
            "limit": min(max(1, limit), 50),
        })
    except Exception:
        _log.warning("widget/activity fetch failed  org=%d", org_id, exc_info=True)
        return {"enabled": True, "message": "", "data": {"runs": []}}

    runs = []
    for r in rows:
        runs.append({
            "id": r.get("Id"),
            "agent_name": r.get("agent_name"),
            "status": r.get("status"),
            "summary": (r.get("summary") or "")[:500],
            "duration_seconds": r.get("duration_seconds"),
            "tokens_total": (int(r.get("tokens_input") or 0) + int(r.get("tokens_output") or 0)),
            "created_at": r.get("CreatedAt"),
        })
    return {"enabled": True, "message": "", "data": {"runs": runs}}


@router.post("/search")
def memory_search(body: SearchRequest):
    """RAG search over a Chroma collection — 'what do we already know about X'."""
    _rate_check(body.org_id)
    from infra.memory import recall
    try:
        hits = recall(body.query, body.org_id, collection_name=body.collection,
                      n_results=min(max(1, body.n_results), 25))
    except Exception as e:
        _log.warning("memory search failed  org=%d q=%s", body.org_id, body.query[:80], exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    return {"query": body.query, "collection": body.collection, "hits": hits}


@router.get("/feed")
def home_feed(org_id: int, limit: int = 20):
    """Unified chronological timeline: digests, insights, questions, notable runs."""
    from shared.home_feed import build_feed
    try:
        items = build_feed(org_id, limit=min(max(1, limit), 100))
    except Exception as e:
        _log.warning("feed build failed  org=%d", org_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    return {"items": items}


@router.post("/briefing")
def home_briefing(org_id: int, request: Request):
    """One-shot 'brief me' summary synthesised from today's feed + pending questions."""
    _rate_check(org_id)
    job = STORE.create()

    def _worker(j):
        from workers.chat.home import run_home_turn
        from shared.home_feed import build_feed
        feed = build_feed(org_id, limit=25)
        feed_lines = []
        for it in feed:
            feed_lines.append(f"- [{it.get('kind')}] {it.get('title')}  ({it.get('created_at')})")
            snippet = (it.get("snippet") or "").strip()
            if snippet:
                feed_lines.append(f"  {snippet[:300]}")
        pending = home_questions.list_pending(org_id, limit=10)
        q_lines = [f"- ({q['id']}) {q['question_text']}" for q in pending]
        message = (
            "Brief me on the state of things. Use the FEED and OPEN QUESTIONS "
            "below to write a tight status: what's new, what I should look at, "
            "what's pending from me. Keep it under 200 words, bulleted.\n\n"
            f"FEED ({len(feed)} items):\n" + "\n".join(feed_lines) + "\n\n"
            f"OPEN QUESTIONS ({len(pending)}):\n" + ("\n".join(q_lines) or "(none)")
        )
        run_home_turn(
            j,
            org_id=org_id,
            model="chat",
            message=message,
            search_mode="disabled",
            max_tokens=600,
        )

    run_in_background(job, _worker)
    return {"job_id": job.id}


@router.get("/conversation/export", response_class=PlainTextResponse)
def export_conversation(org_id: int):
    """Download the org's rolling home conversation as markdown."""
    convo = get_or_create_home_conversation(org_id)
    client = NocodbClient()
    msgs = client.list_messages(convo["Id"], org_id=org_id)
    lines = [f"# {convo.get('title') or 'Home conversation'}", ""]
    for m in msgs:
        role = m.get("role") or "?"
        ts = m.get("CreatedAt") or ""
        lines.append(f"## {role}  _{ts}_")
        lines.append("")
        lines.append((m.get("content") or "").strip())
        lines.append("")
    return "\n".join(lines)


@router.get("/insights")
def list_insights(org_id: int, limit: int = 10):
    from shared import insights as insights_mod
    return {"insights": insights_mod.list_recent(org_id, limit=min(max(1, limit), 50))}


@router.get("/insights/{insight_id}")
def get_insight(insight_id: int):
    from shared import insights as insights_mod
    row = insights_mod.get(insight_id)
    if not row:
        raise HTTPException(status_code=404, detail="insight not found")
    return row


class InsightTriggerRequest(BaseModel):
    org_id: int
    topic_hint: str | None = None


@router.post("/insights/produce")
def produce_insight_now(body: InsightTriggerRequest):
    """Manually kick the insight producer (bypasses the activity-aware gate)."""
    try:
        from workers.tool_queue import get_tool_queue
        tq = get_tool_queue()
    except Exception:
        raise HTTPException(status_code=503, detail="tool queue unavailable")
    if not tq:
        raise HTTPException(status_code=503, detail="tool queue unavailable")
    from shared import insights as insights_mod
    payload = {"org_id": body.org_id, "trigger": insights_mod.TRIGGER_MANUAL}
    if body.topic_hint:
        payload["topic_hint"] = body.topic_hint
    job_id = tq.submit("insight_produce", payload, source="home_manual", org_id=body.org_id)
    return {"status": "queued", "tool_job_id": job_id}


@router.get("/health")
def home_health(request: Request, org_id: int = 1):
    """Full dependency check. Returns `ok: true` only if every required piece
    is wired. `blockers` lists everything that would make the dashboard fail
    silently or look empty."""
    client = NocodbClient()
    sched = getattr(request.app.state, "scheduler", None)
    scheduler_running = bool(sched and getattr(sched, "running", False))

    required_tables = [
        "daily_digests",
        "assistant_questions",
        "insights",
        "conversations",
        "messages",
        "agent_runs",
        "research_plans",
        "scrape_targets",
        "suggested_scrape_targets",
    ]
    optional_tables = ["digest_feedback", "agent_schedules"]
    tables = {t: (t in client.tables) for t in required_tables + optional_tables}

    required_features = ["home", "daily_digest", "insights", "research", "graph_maintenance"]
    features = {f: is_feature_enabled(f) for f in required_features}

    required_models = [
        "chat",
        "insight_topic_picker",
        "insight_synthesis",
        "insight_ack",
        "research_planner",
        "research_agent",
        "daily_digest",
        "relationships",
        "graph_alias_judge",
    ]
    from infra.config import PLATFORM
    configured_models = set((PLATFORM.get("models") or {}).keys())
    models = {m: (m in configured_models) for m in required_models}

    # Huey / tool queue
    queue_running = False
    try:
        from workers.tool_queue import get_tool_queue
        tq = get_tool_queue()
        queue_running = bool(tq and tq.status().get("running"))
    except Exception:
        _log.warning("/home/health: tool queue probe failed", exc_info=True)

    # Idle time
    try:
        from workers.tool_queue import seconds_since_chat
        secs = seconds_since_chat()
        seconds_since_chat_v = None if secs == float("inf") else int(secs)
    except Exception:
        seconds_since_chat_v = None

    # Last digest / last insight ages
    last_digest_at: str | None = None
    last_insight_at: str | None = None
    try:
        row = digest_reader.latest_digest(client, org_id)
        if row:
            last_digest_at = row.get("CreatedAt") or row.get("digest_date")
    except Exception:
        _log.warning("/home/health: last digest probe failed", exc_info=True)
    try:
        from shared import insights as insights_mod
        last_insight_at = insights_mod.latest_created_at(org_id)
    except Exception:
        _log.warning("/home/health: last insight probe failed", exc_info=True)

    # Blocker roll-up
    blockers: list[str] = []
    if not scheduler_running:
        blockers.append("apscheduler not running")
    if not queue_running:
        blockers.append("tool queue (Huey) not running")
    for t, present in tables.items():
        if t in required_tables and not present:
            blockers.append(f"missing NocoDB table: {t}")
    for f, enabled in features.items():
        if not enabled:
            blockers.append(f"feature disabled: {f}")
    for m, present in models.items():
        if not present:
            blockers.append(f"model not configured in config.json: {m}")

    return {
        "ok": not blockers,
        "blockers": blockers,
        "scheduler_running": scheduler_running,
        "queue_running": queue_running,
        "tables": tables,
        "features": features,
        "models": models,
        "seconds_since_chat": seconds_since_chat_v,
        "last_digest_at": last_digest_at,
        "last_insight_at": last_insight_at,
    }
