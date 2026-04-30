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


class LoopResolveRequest(BaseModel):
    org_id: int
    note: str = ""


class LoopDropRequest(BaseModel):
    org_id: int
    reason: str = ""


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
        {"org_id": body.org_id, "bypass_idle": True},
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


# ---- memory: ask & collections ----------------------------------------------

class MemoryAskRequest(BaseModel):
    org_id: int
    query: str
    collections: list[str] | None = None  # default: search a sensible bundle
    n_results: int = 6
    max_tokens: int = 500


_DEFAULT_ASK_COLLECTIONS = [
    "agent_outputs", "research", "research_knowledge",
    "discovery", "chat_entity_mentions",
]


@router.post("/memory/ask")
def memory_ask(body: MemoryAskRequest):
    """Synthesised answer over Chroma memory.

    Pulls top hits from one or more collections, then runs an LLM synthesis
    that cites the snippets it used. This is the 'Ask Memory' feature: the
    user asks a natural-language question and gets a grounded answer plus
    the sources it leaned on.
    """
    _rate_check(body.org_id)
    q = (body.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query is required")

    from infra.memory import recall
    cols = [c for c in (body.collections or _DEFAULT_ASK_COLLECTIONS) if c]
    n = min(max(1, body.n_results), 20)

    bundles: list[dict] = []
    for c in cols:
        try:
            hits = recall(q, body.org_id, collection_name=c, n_results=n)
        except Exception:
            _log.warning("memory_ask: recall failed  collection=%s", c, exc_info=True)
            continue
        for h in hits:
            bundles.append({"collection": c, **h})

    bundles.sort(key=lambda h: float(h.get("distance") or 1.0))
    bundles = bundles[: n * 2]

    if not bundles:
        return {
            "query": q,
            "answer": "I have nothing in memory that matches that yet.",
            "sources": [],
            "collections_searched": cols,
        }

    snippets: list[str] = []
    sources: list[dict] = []
    for i, h in enumerate(bundles, start=1):
        text = (h.get("text") or "").strip().replace("\n\n", "\n")
        if not text:
            continue
        snippets.append(f"[{i}] ({h.get('collection')}) {text[:600]}")
        sources.append({
            "id": i,
            "chunk_id": h.get("id"),
            "collection": h.get("collection"),
            "distance": h.get("distance"),
            "metadata": h.get("metadata") or {},
            "snippet": text[:300],
        })

    prompt = (
        "You are answering a question using ONLY the memory snippets below. "
        "Be direct. If the snippets don't actually answer the question, say so plainly. "
        "Cite the snippets you used inline like [1], [3]. Don't invent sources.\n\n"
        f"QUESTION: {q}\n\n"
        "MEMORY SNIPPETS:\n"
        + "\n\n".join(snippets)
        + "\n\nANSWER:"
    )

    try:
        from shared.models import model_call
        answer, _tokens = model_call("chat", prompt, max_tokens=int(body.max_tokens))
    except Exception:
        _log.warning("memory_ask: synthesis failed", exc_info=True)
        answer = ""

    if not answer:
        # Graceful degradation: hand back the raw snippets so the user sees
        # *something* even if the model is unreachable.
        answer = "Synthesis unavailable — showing raw matches:\n\n" + "\n\n".join(
            s[:400] for s in snippets[:5]
        )

    return {
        "query": q,
        "answer": answer.strip(),
        "sources": sources,
        "collections_searched": cols,
    }


@router.delete("/memory/items/{chunk_id}")
def memory_forget(chunk_id: str, org_id: int, collection: str):
    """Delete a single chunk by id from a scoped collection. Used by the
    'remove this snippet' affordance on memory_ask source cards.

    Uses query params so it composes cleanly with HTTP DELETE — browsers,
    proxies, and many HTTP libraries don't support DELETE with a body.
    """
    from infra.memory import forget, get_chunk
    existing = get_chunk(chunk_id, org_id, collection)
    if not existing:
        raise HTTPException(status_code=404, detail="chunk not found")
    ok = forget(chunk_id, org_id, collection)
    if not ok:
        raise HTTPException(status_code=500, detail="forget failed")
    return {"status": "deleted", "chunk_id": chunk_id, "collection": collection}


@router.get("/memory/health")
def memory_health(org_id: int):
    """Per-collection counts + provenance breakdown + freshness window.
    Drives the Memory > Health page on the dashboard."""
    from infra.memory import collection_health
    try:
        return collection_health(org_id)
    except Exception as e:
        _log.warning("memory/health failed  org=%d", org_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/collections")
def memory_collections(org_id: int):
    """List the Chroma collections that exist for this org and how full they
    are. Frontend uses this to populate the 'Search in...' picker."""
    try:
        from infra.memory import client
    except Exception:
        raise HTTPException(status_code=503, detail="memory backend unavailable")
    prefix = f"org_{org_id}_"
    out: list[dict] = []
    try:
        for c in client.list_collections():
            name = c.name
            if not name.startswith(prefix):
                continue
            try:
                count = c.count()
            except Exception:
                count = None
            out.append({
                "name": name[len(prefix):],
                "scoped_name": name,
                "records": count,
            })
    except Exception:
        _log.warning("memory/collections list failed  org=%d", org_id, exc_info=True)
        raise HTTPException(status_code=500, detail="collection list failed")
    out.sort(key=lambda x: x["name"])
    return {"org_id": org_id, "collections": out}


# ---- graph: search, entity, ask ---------------------------------------------

class GraphSearchRequest(BaseModel):
    org_id: int
    query: str
    max_hops: int = 1
    edge_limit: int = 50


class GraphAskRequest(BaseModel):
    org_id: int
    query: str
    max_hops: int = 1
    edge_limit: int = 60
    max_tokens: int = 500


def _graph_search_core(org_id: int, query: str, max_hops: int, edge_limit: int) -> dict:
    from shared.graph_recall import _load_entity_names, _match_entities
    try:
        known = _load_entity_names(org_id)
    except Exception:
        known = set()

    matches = _match_entities(query, known) if known else []

    # Substring fallback when fuzzy/word match returns nothing — handles the
    # case where entity cache hasn't warmed yet or the user's query doesn't
    # contain a known name verbatim.
    if not matches:
        try:
            from infra.graph import list_entities_for_resolution
            entries = list_entities_for_resolution(org_id, limit=200, min_degree=1, timeout_ms=4000)
        except Exception:
            entries = []
        low = query.lower()
        for e in entries:
            n = (e.get("name") or "").strip()
            if n and (n.lower() in low or low in n.lower()):
                matches.append(n)
                if len(matches) >= 5:
                    break

    edges: list[dict] = []
    if matches:
        try:
            from infra.graph import get_weighted_neighbourhood
            edges = get_weighted_neighbourhood(
                org_id, matches,
                max_hops=max(1, min(int(max_hops), 2)),
                edge_limit=max(1, min(int(edge_limit), 200)),
            )
        except Exception:
            _log.warning("graph search: neighbourhood failed  org=%d", org_id, exc_info=True)
            edges = []

    entities: dict[str, dict] = {}
    for e in edges:
        # Augment each edge with a confidence score (deferred import; the
        # helper is defined further down to avoid forward refs at module
        # parse time).
        try:
            e["confidence"] = _edge_confidence(e.get("hits", 1), e.get("weight", 1.0), e.get("last_seen"))
        except NameError:
            # _edge_confidence not yet defined at runtime — set neutral score
            e["confidence"] = 0.5
        for side in ("from", "to"):
            name = e.get(side)
            if not name:
                continue
            entities.setdefault(name, {
                "name": name,
                "type": e.get(f"{side}_type"),
                "edges_in": 0,
                "edges_out": 0,
            })
        if e.get("from") in entities:
            entities[e["from"]]["edges_out"] += 1
        if e.get("to") in entities:
            entities[e["to"]]["edges_in"] += 1

    return {
        "query": query,
        "matched_entities": matches,
        "entities": list(entities.values()),
        "edges": edges,
    }


@router.post("/graph/search")
def graph_search(body: GraphSearchRequest):
    """Search the knowledge graph by free-text query.

    Matches the query against known entity names (word + substring), pulls
    their weighted neighbourhood, and returns a structured graph slice the
    frontend can render as a list or as a small force-directed view. This is
    the 'Search Knowledge Graph' feature.
    """
    _rate_check(body.org_id)
    q = (body.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query is required")
    return _graph_search_core(body.org_id, q, body.max_hops, body.edge_limit)


@router.get("/graph/entity")
def graph_entity(org_id: int, name: str, max_hops: int = 1, edge_limit: int = 80):
    """Full neighbourhood for a single entity, alias-aware."""
    nm = (name or "").strip()
    if not nm:
        raise HTTPException(status_code=400, detail="name is required")
    try:
        from infra.graph import get_weighted_neighbourhood, get_graph
    except Exception:
        raise HTTPException(status_code=503, detail="graph unavailable")

    try:
        edges = get_weighted_neighbourhood(
            org_id, [nm],
            max_hops=max(1, min(int(max_hops), 2)),
            edge_limit=max(1, min(int(edge_limit), 300)),
        )
    except Exception:
        _log.warning("graph/entity neighbourhood failed  name=%s", nm, exc_info=True)
        edges = []

    node_meta: dict = {"name": nm, "type": None, "aliases": [], "degree": 0}
    try:
        graph = get_graph(org_id)
        result = graph.query(
            "MATCH (n) WHERE n.name = $name OR $name IN coalesce(n.aliases, []) "
            "OPTIONAL MATCH (n)-[r]-() "
            "RETURN labels(n)[0], n.name, coalesce(n.aliases, []), count(r) "
            "LIMIT 1",
            {"name": nm},
        )
        for row in result.result_set:
            node_meta = {
                "name": row[1] or nm,
                "type": row[0],
                "aliases": list(row[2] or []),
                "degree": int(row[3] or 0),
            }
            break
    except Exception:
        _log.debug("graph/entity meta lookup failed  name=%s", nm, exc_info=True)

    return {"entity": node_meta, "edges": edges}


@router.post("/graph/ask")
def graph_ask(body: GraphAskRequest):
    """Search the graph then synthesise a natural-language answer.

    Combines :func:`graph_search` with a chat-model synthesis pass so the
    frontend can show a sentence-shaped answer alongside the graph view.
    """
    _rate_check(body.org_id)
    q = (body.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query is required")

    slice_ = _graph_search_core(body.org_id, q, body.max_hops, body.edge_limit)
    edges = slice_["edges"]

    if not edges:
        return {
            **slice_,
            "answer": "Nothing in the knowledge graph matches that yet.",
        }

    edge_lines = [
        f"- ({e['from']}:{e['from_type']}) -[{e['relationship']} hits={e['hits']}]-> "
        f"({e['to']}:{e['to_type']})"
        for e in edges[:60]
    ]
    prompt = (
        "Answer the question using ONLY the knowledge-graph triples below. "
        "Triples are facts the system has extracted from prior conversations and research. "
        "Be specific. If the triples don't actually answer the question, say so. "
        "Don't invent relationships.\n\n"
        f"QUESTION: {q}\n\n"
        f"MATCHED ENTITIES: {', '.join(slice_['matched_entities']) or '(none)'}\n\n"
        "TRIPLES:\n" + "\n".join(edge_lines) + "\n\nANSWER:"
    )

    try:
        from shared.models import model_call
        answer, _tok = model_call("chat", prompt, max_tokens=int(body.max_tokens))
    except Exception:
        _log.warning("graph_ask: synthesis failed", exc_info=True)
        answer = ""

    if not answer:
        answer = "Synthesis unavailable. Top edges:\n" + "\n".join(edge_lines[:10])

    return {**slice_, "answer": answer.strip()}


# ---- graph: path, timeline, diff, export, manual merge ----------------------

class GraphPathRequest(BaseModel):
    org_id: int
    from_name: str
    to_name: str
    max_hops: int = 4


def _edge_confidence(hits: int, weight: float, last_seen: str | None) -> float:
    """Combine reinforcement + recency into a single 0–1 score.

    - hits saturates at ~10 (log scale)
    - weight is already cumulative; cap at 5 then normalise
    - recency: full credit within 7d, decays to 0 at 90d
    """
    import math
    from datetime import datetime, timezone

    h = min(math.log1p(max(0, int(hits or 0))) / math.log1p(10), 1.0)
    w = min(float(weight or 1.0) / 5.0, 1.0)
    r = 0.5  # default if last_seen missing
    if last_seen:
        try:
            dt = datetime.fromisoformat(str(last_seen).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)
            if age_days <= 7:
                r = 1.0
            elif age_days >= 90:
                r = 0.0
            else:
                r = 1.0 - (age_days - 7) / 83.0
        except Exception:
            pass
    # Weight: 50% reinforcement, 30% strength, 20% recency
    return round(0.5 * h + 0.3 * w + 0.2 * r, 3)


@router.post("/graph/path")
def graph_path(body: GraphPathRequest):
    """Shortest weighted path between two entities — 'how is X connected to Y'.

    Uses FalkorDB's built-in shortestPath. Alias-aware on both endpoints.
    Returns the chain of nodes + edges, or an empty list if no path
    exists within ``max_hops``.
    """
    _rate_check(body.org_id)
    src = (body.from_name or "").strip()
    dst = (body.to_name or "").strip()
    if not src or not dst:
        raise HTTPException(status_code=400, detail="from_name and to_name are required")

    try:
        from infra.graph import get_graph
    except Exception:
        raise HTTPException(status_code=503, detail="graph unavailable")

    hops = max(1, min(int(body.max_hops), 6))
    try:
        graph = get_graph(body.org_id)
        result = graph.query(
            "MATCH (a), (b) "
            "WHERE (a.name = $src OR $src IN coalesce(a.aliases, [])) "
            "  AND (b.name = $dst OR $dst IN coalesce(b.aliases, [])) "
            f"MATCH p = shortestPath((a)-[*..{hops}]-(b)) "
            "RETURN [n IN nodes(p) | [labels(n)[0], n.name]] AS path_nodes, "
            "       [r IN relationships(p) | [type(r), coalesce(r.hits,1), coalesce(r.weight,1.0), r.last_seen]] AS path_edges "
            "LIMIT 1",
            {"src": src, "dst": dst},
        )
    except Exception:
        _log.warning("graph/path failed  src=%s dst=%s", src, dst, exc_info=True)
        return {"from": src, "to": dst, "path": [], "edges": []}

    rows = result.result_set if result else []
    if not rows or not rows[0]:
        return {"from": src, "to": dst, "path": [], "edges": []}

    raw_nodes = rows[0][0] or []
    raw_edges = rows[0][1] or []
    nodes = [{"type": (n[0] or None), "name": n[1]} for n in raw_nodes if n and n[1]]
    edges: list[dict] = []
    for i, e in enumerate(raw_edges):
        if not e:
            continue
        rel, hits, weight, last_seen = e
        edges.append({
            "step": i,
            "from": nodes[i]["name"] if i < len(nodes) else None,
            "to": nodes[i + 1]["name"] if i + 1 < len(nodes) else None,
            "relationship": rel,
            "hits": int(hits or 1),
            "weight": float(weight or 1.0),
            "last_seen": last_seen,
            "confidence": _edge_confidence(int(hits or 1), float(weight or 1.0), last_seen),
        })

    return {"from": src, "to": dst, "path": nodes, "edges": edges, "hops": len(edges)}


@router.get("/graph/entity/{name}/timeline")
def graph_entity_timeline(name: str, org_id: int):
    """Per-edge first/last seen timeline for one entity.

    Surfaces 'this entity went quiet 30 days ago' and 'this is a brand-new
    relationship' patterns in the UI.
    """
    nm = (name or "").strip()
    if not nm:
        raise HTTPException(status_code=400, detail="name is required")
    try:
        from infra.graph import get_graph
    except Exception:
        raise HTTPException(status_code=503, detail="graph unavailable")
    try:
        graph = get_graph(org_id)
        result = graph.query(
            "MATCH (a)-[r]-(b) "
            "WHERE a.name = $name OR $name IN coalesce(a.aliases, []) "
            "RETURN type(r), b.name, labels(b)[0], "
            "  coalesce(r.hits,1), coalesce(r.weight,1.0), "
            "  r.first_seen, r.last_seen "
            "ORDER BY r.last_seen DESC NULLS LAST",
            {"name": nm},
        )
    except Exception:
        _log.warning("graph/entity/timeline failed  name=%s", nm, exc_info=True)
        return {"entity": nm, "events": []}

    events: list[dict] = []
    for row in (result.result_set if result else []):
        if not row:
            continue
        rel, other, other_type, hits, weight, first_seen, last_seen = row
        events.append({
            "relationship": rel,
            "to": other,
            "to_type": other_type,
            "hits": int(hits or 1),
            "weight": float(weight or 1.0),
            "first_seen": first_seen,
            "last_seen": last_seen,
            "confidence": _edge_confidence(int(hits or 1), float(weight or 1.0), last_seen),
        })

    return {"entity": nm, "events": events, "edge_count": len(events)}


@router.get("/graph/diff")
def graph_diff(org_id: int, since_days: int = 7):
    """What changed in the graph in the last N days.

    Returns new entities (`first_seen >= since`), refreshed entities
    (`last_seen >= since` but `first_seen < since`), and new/refreshed
    edges. Becomes a 'this week in your graph' digest section.
    """
    from datetime import datetime, timedelta, timezone
    days = max(1, min(int(since_days), 90))
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    try:
        from infra.graph import get_graph
    except Exception:
        raise HTTPException(status_code=503, detail="graph unavailable")

    new_edges: list[dict] = []
    refreshed_edges: list[dict] = []
    try:
        graph = get_graph(org_id)
        # New edges: first_seen >= since
        result = graph.query(
            "MATCH (a)-[r]->(b) "
            "WHERE r.first_seen >= $since "
            "RETURN a.name, type(r), b.name, coalesce(r.hits,1), coalesce(r.weight,1.0), "
            "  r.first_seen, r.last_seen "
            "ORDER BY r.first_seen DESC LIMIT 100",
            {"since": since},
        )
        for row in (result.result_set if result else []):
            if not row:
                continue
            new_edges.append({
                "from": row[0], "relationship": row[1], "to": row[2],
                "hits": int(row[3] or 1), "weight": float(row[4] or 1.0),
                "first_seen": row[5], "last_seen": row[6],
            })
        # Refreshed edges: last_seen >= since AND first_seen < since
        result = graph.query(
            "MATCH (a)-[r]->(b) "
            "WHERE r.last_seen >= $since AND r.first_seen < $since "
            "RETURN a.name, type(r), b.name, coalesce(r.hits,1), coalesce(r.weight,1.0), "
            "  r.first_seen, r.last_seen "
            "ORDER BY r.last_seen DESC LIMIT 100",
            {"since": since},
        )
        for row in (result.result_set if result else []):
            if not row:
                continue
            refreshed_edges.append({
                "from": row[0], "relationship": row[1], "to": row[2],
                "hits": int(row[3] or 1), "weight": float(row[4] or 1.0),
                "first_seen": row[5], "last_seen": row[6],
            })
    except Exception:
        _log.warning("graph/diff failed  org=%d", org_id, exc_info=True)

    new_entities = sorted({e["from"] for e in new_edges} | {e["to"] for e in new_edges})
    refreshed_entities = sorted(
        ({e["from"] for e in refreshed_edges} | {e["to"] for e in refreshed_edges})
        - set(new_entities)
    )
    return {
        "since": since,
        "since_days": days,
        "new_entities": new_entities,
        "refreshed_entities": refreshed_entities,
        "new_edges": new_edges,
        "refreshed_edges": refreshed_edges,
    }


@router.get("/graph/export")
def graph_export(org_id: int, seed: str, hops: int = 2, format: str = "cytoscape"):
    """Export a subgraph rooted at ``seed`` for client-side visualisation.

    ``format=cytoscape`` returns Cytoscape's elements format (nodes + edges).
    ``format=raw`` returns the same data as a generic ``{nodes, edges}``.
    """
    fmt = (format or "cytoscape").lower()
    if fmt not in ("cytoscape", "raw"):
        raise HTTPException(status_code=400, detail="format must be cytoscape or raw")
    nm = (seed or "").strip()
    if not nm:
        raise HTTPException(status_code=400, detail="seed is required")
    try:
        from infra.graph import get_weighted_neighbourhood
        edges = get_weighted_neighbourhood(
            org_id, [nm],
            max_hops=max(1, min(int(hops), 3)),
            edge_limit=300,
        )
    except Exception:
        edges = []

    nodes_seen: dict[str, dict] = {}
    edge_list: list[dict] = []
    for e in edges:
        for side in ("from", "to"):
            n = e.get(side)
            if n and n not in nodes_seen:
                nodes_seen[n] = {"id": n, "label": n, "type": e.get(f"{side}_type")}
        edge_list.append({
            "id": f"{e['from']}|{e['relationship']}|{e['to']}",
            "source": e["from"],
            "target": e["to"],
            "label": e["relationship"],
            "hits": e["hits"],
            "weight": e["weight"],
            "confidence": _edge_confidence(e["hits"], e["weight"], e.get("last_seen")),
        })

    if fmt == "cytoscape":
        elements = (
            [{"data": n} for n in nodes_seen.values()] +
            [{"data": e} for e in edge_list]
        )
        return {"format": "cytoscape", "elements": elements}
    return {"format": "raw", "nodes": list(nodes_seen.values()), "edges": edge_list}


class GraphMergeRequest(BaseModel):
    org_id: int
    label: str
    canonical: str
    alias: str


@router.post("/graph/entity/merge")
def graph_entity_merge(body: GraphMergeRequest):
    """Manually merge two entity nodes — fixes alias-resolver misses.

    Wraps the existing :func:`infra.graph.merge_alias` helper. Both nodes
    must share the same label (e.g. both `Person` or both `Concept`).
    """
    try:
        from infra.graph import merge_alias
        result = merge_alias(
            body.org_id, body.label, body.canonical.strip(), body.alias.strip(),
        )
    except Exception as e:
        _log.warning("graph/entity/merge failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    if result.get("status") != "ok":
        raise HTTPException(status_code=409, detail=result.get("error") or result.get("status"))
    # Drop the entity-name cache so the chat path picks up the merge immediately
    try:
        from shared.graph_recall import invalidate_entity_cache
        invalidate_entity_cache(body.org_id)
    except Exception:
        pass
    return result


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


class InsightResearchRequest(BaseModel):
    focus: str
    org_id: int | None = None


@router.post("/insights/{insight_id}/research")
def insight_deep_dive(insight_id: int, body: InsightResearchRequest):
    """Kick off a follow-up research plan scoped to this insight. When the
    plan completes, its paper is appended to the insight body."""
    focus = (body.focus or "").strip()
    if not focus:
        raise HTTPException(status_code=400, detail="focus is required")
    from shared import insights as insights_mod
    insight = insights_mod.get(insight_id)
    if not insight:
        raise HTTPException(status_code=404, detail="insight not found")
    org_id = int(body.org_id or insight.get("org_id") or 0)
    base_topic = (insight.get("topic") or insight.get("title") or "").strip()
    topic = f"{base_topic}: {focus}" if base_topic else focus

    from tools.research.research_planner import create_research_plan
    result = create_research_plan(
        topic,
        org_id=org_id,
        parent_insight_id=insight_id,
        focus=focus,
    )
    status = result.get("status") if isinstance(result, dict) else None
    if status not in ("queued", "disabled"):
        raise HTTPException(status_code=500, detail=result.get("error") or "failed to queue plan")
    return {
        "status": status,
        "insight_id": insight_id,
        "plan_id": result.get("plan_id"),
        "tool_job_id": result.get("job_id"),
    }


@router.get("/insights/{insight_id}/research")
def insight_research_list(insight_id: int):
    """List research plans linked to this insight (primary + follow-ups)."""
    client = NocodbClient()
    if "research_plans" not in client.tables:
        raise HTTPException(status_code=503, detail="research_plans table not provisioned")
    try:
        rows = client._get_paginated("research_plans", params={
            "where": f"(parent_insight_id,eq,{insight_id})",
            "sort": "-CreatedAt",
            "limit": 50,
        })
    except Exception:
        _log.warning("insight research list failed  id=%d", insight_id, exc_info=True)
        rows = []
    return {
        "insight_id": insight_id,
        "plans": [
            {
                "plan_id": r.get("Id"),
                "topic": r.get("topic"),
                "focus": r.get("focus"),
                "status": r.get("status"),
                "confidence_score": r.get("confidence_score"),
                "completed_at": r.get("completed_at"),
                "created_at": r.get("CreatedAt"),
            }
            for r in rows
        ],
    }


# ---- recent conversations ---------------------------------------------------

@router.get("/conversations/recent")
def recent_conversations(org_id: int, limit: int = 10) -> dict:
    """Sidebar list: recent chat conversations (excluding the rolling home
    conversation). Each row carries enough metadata for the UI to render
    title, last-active time, and message count without a follow-up call.
    """
    db = NocodbClient()
    limit = min(max(1, limit), 50)
    try:
        rows = db.list_conversations(org_id, limit=limit)
    except Exception as e:
        _log.warning("recent_conversations failed  org=%d", org_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    out: list[dict] = []
    for r in rows:
        out.append({
            "id": r.get("Id"),
            "title": (r.get("title") or "Untitled").strip() or "Untitled",
            "kind": r.get("kind") or "chat",
            "model": r.get("model"),
            "default_response_style": r.get("default_response_style"),
            "created_at": r.get("CreatedAt"),
            "updated_at": r.get("UpdatedAt") or r.get("CreatedAt"),
        })
    return {"conversations": out}


# ---- stats summary ----------------------------------------------------------

@router.get("/stats/summary")
def stats_summary(org_id: int) -> dict:
    """Lightweight stats roll-up sized for dashboard polling.

    Returns three windows (24h / 7d / 30d) plus today's top models and
    agents. Source: ``agent_runs`` (LLM usage rows) — much cheaper than
    paging through every chat message like ``/stats/usage`` does.
    """
    from datetime import datetime, timedelta, timezone

    db = NocodbClient()
    now = datetime.now(timezone.utc)
    cutoffs = {
        "24h": now - timedelta(hours=24),
        "7d": now - timedelta(days=7),
        "30d": now - timedelta(days=30),
    }

    rows: list[dict] = []
    try:
        rows = db._get_paginated("agent_runs", params={
            "where": f"(org_id,eq,{org_id})",
            "sort": "-CreatedAt",
            "limit": 5000,
        })
    except Exception:
        _log.warning("stats/summary fetch failed  org=%d", org_id, exc_info=True)

    def _parse(ts):
        if not ts:
            return None
        s = str(ts).replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
        except Exception:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    def _window(start: datetime) -> dict:
        in_window = [r for r in rows if (_parse(r.get("CreatedAt")) or now) >= start]
        tok_in = sum(int(r.get("tokens_input") or 0) for r in in_window)
        tok_out = sum(int(r.get("tokens_output") or 0) for r in in_window)
        errors = sum(1 for r in in_window if r.get("status") == "failed")
        durations = [float(r.get("duration_seconds") or 0) for r in in_window if r.get("duration_seconds")]
        avg_dur = round(sum(durations) / len(durations), 2) if durations else 0.0
        return {
            "requests": len(in_window),
            "tokens_input": tok_in,
            "tokens_output": tok_out,
            "tokens_total": tok_in + tok_out,
            "errors": errors,
            "error_rate": round(errors / max(len(in_window), 1), 4),
            "avg_duration_seconds": avg_dur,
        }

    windows = {k: _window(start) for k, start in cutoffs.items()}

    last_24h_rows = [r for r in rows if (_parse(r.get("CreatedAt")) or now) >= cutoffs["24h"]]

    by_model: dict[str, dict] = {}
    for r in last_24h_rows:
        name = (r.get("model_name") or "unknown").strip() or "unknown"
        e = by_model.setdefault(name, {"model_name": name, "requests": 0, "tokens": 0, "errors": 0})
        e["requests"] += 1
        e["tokens"] += int(r.get("tokens_input") or 0) + int(r.get("tokens_output") or 0)
        if r.get("status") == "failed":
            e["errors"] += 1
    top_models = sorted(by_model.values(), key=lambda x: x["requests"], reverse=True)[:5]

    by_agent: dict[str, dict] = {}
    for r in last_24h_rows:
        name = (r.get("agent_name") or "unknown").strip() or "unknown"
        e = by_agent.setdefault(name, {"agent_name": name, "runs": 0, "successful": 0, "failed": 0})
        e["runs"] += 1
        st = r.get("status")
        # agent_runs has two writers (nocodb_client + workers/user_agents/runtime)
        # that disagree on terminal-success spelling; treat both as success.
        if st in ("complete", "completed"):
            e["successful"] += 1
        elif st == "failed":
            e["failed"] += 1
    top_agents = sorted(by_agent.values(), key=lambda x: x["runs"], reverse=True)[:5]

    return {
        "org_id": org_id,
        "as_of": now.isoformat(),
        "windows": windows,
        "top_models_24h": top_models,
        "top_agents_24h": top_agents,
        "sample_size": len(rows),
    }


# ---- queues summary ---------------------------------------------------------

@router.get("/queues/summary")
def queues_summary(request: Request, org_id: int, limit: int = 25) -> dict:
    """Dashboard-shaped view of the tool queue.

    Groups jobs by type (queued / running / completed-24h / failed-24h),
    surfaces the currently running set with task summary + age, and
    pulls the most recent failures and completions. The raw
    ``/tool-queue/jobs`` endpoint is fine for power users; this is what
    the home page should poll.
    """
    from datetime import datetime, timezone

    q = getattr(request.app.state, "tool_queue", None)
    if q is None:
        return {
            "queue_ready": False,
            "by_type": [],
            "currently_running": [],
            "recent_failures": [],
            "recent_completions": [],
            "totals": {"queued": 0, "running": 0, "completed_24h": 0, "failed_24h": 0},
        }

    try:
        recent = q.list_jobs(limit=max(50, min(limit * 4, 400)), org_id=org_id, verbose=False)
    except Exception:
        _log.warning("queues/summary list_jobs failed  org=%d", org_id, exc_info=True)
        recent = []

    now = datetime.now(timezone.utc)

    def _parse(ts):
        if not ts:
            return None
        s = str(ts).replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
        except Exception:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    def _age_seconds(ts) -> int | None:
        dt = _parse(ts)
        return int((now - dt).total_seconds()) if dt else None

    by_type: dict[str, dict] = {}
    totals = {"queued": 0, "running": 0, "completed_24h": 0, "failed_24h": 0}

    for j in recent:
        t = j.get("type") or "unknown"
        e = by_type.setdefault(t, {
            "type": t, "queued": 0, "running": 0,
            "completed_24h": 0, "failed_24h": 0,
        })
        st = j.get("status")
        if st == "queued":
            e["queued"] += 1; totals["queued"] += 1
        elif st == "running":
            e["running"] += 1; totals["running"] += 1
        else:
            ts = j.get("completed_at") or j.get("started_at")
            age = _age_seconds(ts)
            if age is not None and age <= 86400:
                if st == "completed":
                    e["completed_24h"] += 1; totals["completed_24h"] += 1
                elif st == "failed":
                    e["failed_24h"] += 1; totals["failed_24h"] += 1

    by_type_list = sorted(
        by_type.values(),
        key=lambda x: (x["running"] + x["queued"], x["completed_24h"]),
        reverse=True,
    )

    def _row(j: dict) -> dict:
        return {
            "job_id": j.get("job_id"),
            "type": j.get("type"),
            "status": j.get("status"),
            "task": j.get("task") or j.get("title") or j.get("url"),
            "source": j.get("source"),
            "priority": j.get("priority"),
            "started_at": j.get("started_at"),
            "completed_at": j.get("completed_at"),
            "age_seconds": _age_seconds(j.get("started_at") or j.get("completed_at")),
            "error": (j.get("error") or "")[:300] or None,
            "result_status": j.get("result_status"),
        }

    running = [j for j in recent if j.get("status") == "running"]
    running.sort(key=lambda j: j.get("started_at") or "", reverse=True)
    failures = [j for j in recent if j.get("status") == "failed"]
    failures.sort(key=lambda j: j.get("completed_at") or "", reverse=True)
    completions = [j for j in recent if j.get("status") == "completed"]
    completions.sort(key=lambda j: j.get("completed_at") or "", reverse=True)

    cap = min(max(1, limit), 100)
    return {
        "queue_ready": True,
        "as_of": now.isoformat(),
        "totals": totals,
        "by_type": by_type_list,
        "currently_running": [_row(j) for j in running[:cap]],
        "recent_failures": [_row(j) for j in failures[: min(10, cap)]],
        "recent_completions": [_row(j) for j in completions[: min(15, cap)]],
        "backoff": (q.status() or {}).get("backoff"),
    }


# ---- one-shot dashboard -----------------------------------------------------

@router.get("/dashboard")
def dashboard(request: Request, org_id: int) -> dict:
    """Single payload that paints the entire home page.

    Wraps ``/home/overview`` plus the new feature surfaces (insights,
    conversations, queues, stats, graph slice, memory, schedules,
    questions, PA snapshot). Intended as the dashboard's primary poll
    target — every panel below the fold reads from this one response,
    falling back to its dedicated endpoint only on user action (refresh
    one panel, drill in, etc).

    Every section is wrapped in its own try/except: a single broken
    dependency must not blank the whole page.
    """
    out: dict = {"org_id": org_id, "as_of": None}
    from datetime import datetime, timezone
    out["as_of"] = datetime.now(timezone.utc).isoformat()

    if not is_feature_enabled("home"):
        raise HTTPException(status_code=404, detail="home feature disabled")

    client = NocodbClient()

    # Digest meta
    try:
        out["digest"] = digest_reader.as_payload(
            digest_reader.latest_digest(client, org_id),
            include_markdown=False,
        )
    except Exception:
        _log.warning("dashboard: digest failed  org=%d", org_id, exc_info=True)
        out["digest"] = None

    # PA snapshot (loops + topics + facts + status header)
    pa: dict = {"feature_enabled": is_feature_enabled("pa")}
    if pa["feature_enabled"]:
        try:
            from shared.pa.memory import (
                list_open_loops, list_warm_topics, list_user_facts,
                last_move_at, MOVE_MODE_PROACTIVE,
            )
            from shared.pa.picker import PROACTIVE_MIN_GAP_HOURS

            pa["loops"] = list_open_loops(org_id, status="open", limit=10) or []
            pa["topics"] = list_warm_topics(org_id, limit=8, min_warmth=0.1) or []
            facts_rows = list_user_facts(org_id, kind=None, limit=50) or []
            pa["facts"] = [r for r in facts_rows if r.get("confidence") != "deleted"][:20]

            last_dt = None
            try:
                last_dt = last_move_at(org_id, mode=MOVE_MODE_PROACTIVE)
            except Exception:
                last_dt = None
            gap_ready = True
            seconds_until_ready = 0
            if last_dt is not None:
                elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
                gap_s = PROACTIVE_MIN_GAP_HOURS * 3600
                if elapsed < gap_s:
                    gap_ready = False
                    seconds_until_ready = int(gap_s - elapsed)
            pa["status"] = {
                "last_proactive_at": last_dt.isoformat() if last_dt else None,
                "gap_ready": gap_ready,
                "seconds_until_auto_ready": seconds_until_ready,
            }
        except Exception:
            _log.warning("dashboard: pa snapshot failed  org=%d", org_id, exc_info=True)
    out["pa"] = pa

    # Pending questions
    try:
        out["pending_questions"] = home_questions.list_pending(org_id, limit=5)
    except Exception:
        _log.warning("dashboard: questions failed", exc_info=True)
        out["pending_questions"] = []

    # Insights
    try:
        from shared import insights as insights_mod
        out["insights"] = insights_mod.list_recent(org_id, limit=6)
    except Exception:
        _log.warning("dashboard: insights failed", exc_info=True)
        out["insights"] = []

    # Recent conversations (excluding the rolling home convo)
    try:
        rows = client.list_conversations(org_id, limit=8)
        out["conversations"] = [
            {
                "id": r.get("Id"),
                "title": (r.get("title") or "Untitled").strip() or "Untitled",
                "kind": r.get("kind") or "chat",
                "model": r.get("model"),
                "updated_at": r.get("UpdatedAt") or r.get("CreatedAt"),
            }
            for r in rows
        ]
    except Exception:
        _log.warning("dashboard: conversations failed", exc_info=True)
        out["conversations"] = []

    # Home conversation summary
    try:
        out["home_conversation"] = home_conversation_summary(org_id)
    except Exception:
        _log.warning("dashboard: home conversation summary failed", exc_info=True)
        out["home_conversation"] = None

    # Mixed feed
    try:
        from shared.home_feed import build_feed
        out["feed"] = build_feed(org_id, limit=10)
    except Exception:
        _log.warning("dashboard: feed failed", exc_info=True)
        out["feed"] = []

    # Queues — call the in-process helper directly (saves an HTTP hop)
    try:
        out["queues"] = queues_summary(request, org_id, limit=10)
    except Exception:
        _log.warning("dashboard: queues failed", exc_info=True)
        out["queues"] = None

    # Lightweight stats
    try:
        out["stats"] = stats_summary(org_id)
    except Exception:
        _log.warning("dashboard: stats failed", exc_info=True)
        out["stats"] = None

    # Graph snapshot (compact)
    try:
        out["graph"] = widget_graph(org_id, limit=8)
    except Exception:
        _log.warning("dashboard: graph snapshot failed", exc_info=True)
        out["graph"] = {"enabled": False, "data": None}

    # Memory snapshot (compact collection list)
    try:
        out["memory"] = memory_collections(org_id)
    except HTTPException:
        out["memory"] = {"org_id": org_id, "collections": []}
    except Exception:
        _log.warning("dashboard: memory snapshot failed", exc_info=True)
        out["memory"] = {"org_id": org_id, "collections": []}

    # Schedules
    try:
        sched = getattr(request.app.state, "scheduler", None)
        out["schedules"] = get_next_runs(sched, org_id=org_id)[:8]
    except Exception:
        _log.warning("dashboard: schedules failed", exc_info=True)
        out["schedules"] = []

    return out


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


# ---- PA memory layer ---------------------------------------------------------

@router.get("/loops")
def list_loops(org_id: int, status: str | None = "open", limit: int = 50) -> dict:
    if not is_feature_enabled("pa"):
        return {"loops": [], "feature_enabled": False}
    from shared.pa.memory import list_open_loops
    effective_status: str | None = None if status == "all" else status
    try:
        loops = list_open_loops(org_id, status=effective_status, limit=min(max(1, limit), 500))
    except Exception as e:
        _log.warning("list_loops failed  org=%d status=%s", org_id, status, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    return {"loops": loops, "feature_enabled": True}


@router.post("/loops/{loop_id}/resolve")
def resolve_open_loop(loop_id: int, body: LoopResolveRequest) -> dict:
    if not is_feature_enabled("pa"):
        raise HTTPException(status_code=404, detail="pa feature disabled")
    from shared.pa.memory import list_open_loops, resolve_loop
    try:
        rows = list_open_loops(body.org_id, status=None, limit=500)
    except Exception as e:
        _log.warning("resolve_loop lookup failed  org=%d id=%d", body.org_id, loop_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    if not any(int(r.get("Id") or r.get("id") or 0) == int(loop_id) for r in rows):
        raise HTTPException(status_code=404, detail="loop not found")
    try:
        resolve_loop(loop_id, note=body.note)
    except Exception as e:
        _log.warning("resolve_loop failed  id=%d", loop_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "resolved", "loop_id": loop_id}


@router.post("/loops/{loop_id}/drop")
def drop_open_loop(loop_id: int, body: LoopDropRequest) -> dict:
    if not is_feature_enabled("pa"):
        raise HTTPException(status_code=404, detail="pa feature disabled")
    from shared.pa.memory import drop_loop, list_open_loops
    try:
        rows = list_open_loops(body.org_id, status=None, limit=500)
    except Exception as e:
        _log.warning("drop_loop lookup failed  org=%d id=%d", body.org_id, loop_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    if not any(int(r.get("Id") or r.get("id") or 0) == int(loop_id) for r in rows):
        raise HTTPException(status_code=404, detail="loop not found")
    try:
        drop_loop(loop_id, reason=body.reason)
    except Exception as e:
        _log.warning("drop_loop failed  id=%d", loop_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "dropped", "loop_id": loop_id}


@router.get("/topics")
def list_topics(org_id: int, limit: int = 10) -> dict:
    if not is_feature_enabled("pa"):
        return {"topics": [], "feature_enabled": False}
    from shared.pa.memory import list_warm_topics, topic_sources
    try:
        topics = list_warm_topics(org_id, limit=min(max(1, limit), 100), min_warmth=0.1)
    except Exception as e:
        _log.warning("list_topics failed  org=%d", org_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    out: list[dict] = []
    for t in topics:
        try:
            sources = topic_sources(t)
        except Exception:
            _log.warning("topic_sources decode failed  org=%d", org_id, exc_info=True)
            sources = []
        item = dict(t)
        item["sources"] = sources
        out.append(item)
    return {"topics": out}


@router.get("/facts")
def list_facts(org_id: int, kind: str | None = None, limit: int = 100) -> dict:
    if not is_feature_enabled("pa"):
        return {"facts": [], "feature_enabled": False}
    from shared.pa.memory import list_user_facts
    try:
        rows = list_user_facts(org_id, kind=kind, limit=min(max(1, limit), 500))
    except Exception as e:
        _log.warning("list_facts failed  org=%d kind=%s", org_id, kind, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    facts = [r for r in rows if (r.get("confidence") != "deleted")]
    return {"facts": facts}


@router.delete("/facts/{fact_id}")
def delete_fact(fact_id: int, org_id: int) -> dict:
    if not is_feature_enabled("pa"):
        raise HTTPException(status_code=404, detail="pa feature disabled")
    from shared.pa.memory import delete_user_fact, list_user_facts
    try:
        rows = list_user_facts(org_id, kind=None, limit=500)
    except Exception as e:
        _log.warning("delete_fact lookup failed  org=%d id=%d", org_id, fact_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    match = None
    for r in rows:
        rid = int(r.get("Id") or r.get("id") or 0)
        if rid == int(fact_id):
            match = r
            break
    if not match:
        raise HTTPException(status_code=404, detail="fact not found")
    if match.get("confidence") == "deleted":
        raise HTTPException(status_code=409, detail="fact already deleted")
    try:
        delete_user_fact(fact_id)
    except Exception as e:
        _log.warning("delete_user_fact failed  id=%d", fact_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "deleted", "fact_id": fact_id}


class PARunRequest(BaseModel):
    org_id: int
    force: bool = True


@router.post("/pa/run")
def pa_run_now(body: PARunRequest) -> dict:
    """Manually trigger the PA for this org. Skips the 4h global gap by
    default (the "I'm back at my desk" button). Per-kind cooldowns still
    apply, so repeated presses won't spam the same move type.
    """
    if not is_feature_enabled("pa"):
        raise HTTPException(status_code=404, detail="pa feature disabled")
    try:
        from scheduler import run_pa_for_org
        result = run_pa_for_org(int(body.org_id), force=bool(body.force))
    except Exception as e:
        _log.warning("pa_run_now failed  org=%d", body.org_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    return result


@router.get("/pa/snapshot")
def pa_snapshot(org_id: int, loops_limit: int = 20, topics_limit: int = 10, facts_limit: int = 50) -> dict:
    """One call that returns everything the PA panel needs: open loops, warm
    topics, and user facts. Lets the frontend paint the whole side panel
    without 3 round trips. Fields fall back to empty lists when PA is off
    or its tables are unprovisioned."""
    out: dict = {
        "feature_enabled": is_feature_enabled("pa"),
        "loops": [],
        "topics": [],
        "facts": [],
    }
    if not out["feature_enabled"]:
        return out
    try:
        from shared.pa.memory import list_open_loops, list_warm_topics, list_user_facts
    except Exception:
        return out
    try:
        out["loops"] = list_open_loops(org_id, status="open", limit=min(max(1, loops_limit), 200)) or []
    except Exception:
        _log.debug("pa_snapshot: loops failed  org=%d", org_id, exc_info=True)
    try:
        out["topics"] = list_warm_topics(org_id, limit=min(max(1, topics_limit), 50), min_warmth=0.1) or []
    except Exception:
        _log.debug("pa_snapshot: topics failed  org=%d", org_id, exc_info=True)
    try:
        rows = list_user_facts(org_id, kind=None, limit=min(max(1, facts_limit), 200)) or []
        out["facts"] = [r for r in rows if r.get("confidence") != "deleted"]
    except Exception:
        _log.debug("pa_snapshot: facts failed  org=%d", org_id, exc_info=True)
    return out


@router.get("/pa/status")
def pa_status(org_id: int) -> dict:
    """Snapshot of PA state for the Home page header: last proactive
    surface time, whether the 4h auto gap is clear, warm-topic count,
    open-loop count. Cheap read-only call."""
    if not is_feature_enabled("pa"):
        return {"enabled": False}
    try:
        from shared.pa.memory import (
            last_move_at, list_warm_topics, list_open_loops,
            MOVE_MODE_PROACTIVE,
        )
        from shared.pa.picker import PROACTIVE_MIN_GAP_HOURS
    except Exception:
        return {"enabled": False}

    last_dt = None
    try:
        last_dt = last_move_at(org_id, mode=MOVE_MODE_PROACTIVE)
    except Exception:
        last_dt = None

    gap_ready = True
    seconds_until_ready = 0
    if last_dt is not None:
        try:
            from datetime import datetime, timezone
            elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
            gap_s = PROACTIVE_MIN_GAP_HOURS * 3600
            if elapsed < gap_s:
                gap_ready = False
                seconds_until_ready = int(gap_s - elapsed)
        except Exception:
            pass

    try:
        warm_count = len(list_warm_topics(org_id, limit=50, min_warmth=0.1))
    except Exception:
        warm_count = 0
    try:
        loops_open = list_open_loops(org_id, status="open", limit=200) or []
        loops_nudged = list_open_loops(org_id, status="nudged", limit=200) or []
        open_count = len(loops_open) + len(loops_nudged)
    except Exception:
        open_count = 0

    return {
        "enabled": True,
        "last_proactive_at": last_dt.isoformat() if last_dt else None,
        "gap_ready": gap_ready,
        "seconds_until_auto_ready": seconds_until_ready,
        "warm_topics": warm_count,
        "open_loops": open_count,
    }
