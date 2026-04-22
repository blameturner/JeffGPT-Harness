"""Home dashboard API.

The frontend (separate repo) paints one page from ``GET /home/overview`` and
then issues follow-up calls for chat, question answers, schedule triggers,
and widget refreshes. Streaming responses piggyback on the existing
``/stream/{job_id}`` SSE endpoint exposed by the agents router.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from infra.config import NOCODB_TABLE_DAILY_DIGESTS, get_feature, is_feature_enabled
from infra.nocodb_client import NocodbClient
from infra.scheduler_introspect import get_next_runs, get_schedule_meta
from shared import home_questions
from shared.home_conversation import home_conversation_summary
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
    answer_question_id: int | None = None
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
    search_mode: Literal["disabled", "basic", "standard"] = "basic"

    @property
    def has_content(self) -> bool:
        return bool(self.selected_option or self.answer_text)


class RunNowRequest(BaseModel):
    task: str | None = None
    product: str | None = None


# ---- helpers -----------------------------------------------------------------

def _latest_digest_row(client: NocodbClient, org_id: int) -> dict | None:
    if NOCODB_TABLE_DAILY_DIGESTS not in client.tables:
        return None
    try:
        rows = client._get_paginated(NOCODB_TABLE_DAILY_DIGESTS, params={
            "where": f"(org_id,eq,{org_id})",
            "sort": "-digest_date",
            "limit": 1,
        })
    except Exception:
        _log.debug("daily_digests fetch failed  org=%d", org_id, exc_info=True)
        return None
    return rows[0] if rows else None


def _digest_for_date(client: NocodbClient, org_id: int, date: str) -> dict | None:
    if NOCODB_TABLE_DAILY_DIGESTS not in client.tables:
        return None
    try:
        data = client._get(NOCODB_TABLE_DAILY_DIGESTS, params={
            "where": f"(org_id,eq,{org_id})~and(digest_date,eq,{date})",
            "limit": 1,
        })
    except Exception:
        _log.debug("daily_digest date fetch failed  org=%d date=%s", org_id, date, exc_info=True)
        return None
    rows = data.get("list", [])
    return rows[0] if rows else None


def _read_markdown(row: dict | None) -> str:
    if not row:
        return ""
    path = (row.get("markdown_path") or "").strip()
    if not path:
        return ""
    try:
        p = Path(path).expanduser()
        if p.is_file():
            return p.read_text(encoding="utf-8")
    except Exception:
        _log.debug("digest markdown read failed  path=%s", path, exc_info=True)
    return ""


def _digest_payload(row: dict | None) -> dict | None:
    if not row:
        return None
    return {
        "id": row.get("Id"),
        "date": row.get("digest_date"),
        "markdown": _read_markdown(row),
        "path": row.get("markdown_path"),
        "cluster_count": row.get("cluster_count"),
        "source_count": row.get("source_count"),
        "created_at": row.get("CreatedAt"),
    }


def _widget(name: str) -> dict:
    cfg = get_feature("home", name, {})
    enabled = bool(cfg.get("enabled")) if isinstance(cfg, dict) else False
    return {
        "enabled": enabled,
        "message": "" if enabled else "Not configured",
        "data": None,
    }


# ---- endpoints ---------------------------------------------------------------

@router.get("/overview")
def home_overview(org_id: int, request: Request):
    if not is_feature_enabled("home"):
        raise HTTPException(status_code=404, detail="home feature disabled")

    client = NocodbClient()
    out: dict = {"org_id": org_id}

    try:
        out["digest"] = _digest_payload(_latest_digest_row(client, org_id))
    except Exception:
        _log.warning("overview: digest panel failed  org=%d", org_id, exc_info=True)
        out["digest"] = None

    try:
        out["pending_questions"] = home_questions.list_pending(org_id)
    except Exception:
        _log.warning("overview: questions panel failed  org=%d", org_id, exc_info=True)
        out["pending_questions"] = []

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
    row = _digest_for_date(client, org_id, date) if date else _latest_digest_row(client, org_id)
    if not row:
        raise HTTPException(status_code=404, detail="no digest available")
    return _digest_payload(row)


@router.post("/chat")
def home_chat(request: HomeChatRequest):
    _log.info(
        "POST /home/chat  org=%d model=%s q=%s chars=%d",
        request.org_id, request.model, request.answer_question_id, len(request.message),
    )
    if not request.message.strip() and not request.answer_question_id:
        raise HTTPException(status_code=400, detail="message is required")

    job = STORE.create()

    def _worker(j):
        run_home_turn(
            j,
            org_id=request.org_id,
            model=request.model,
            message=request.message,
            answer_question_id=request.answer_question_id,
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

    composed = _compose_answer_for_chat(question, body.selected_option, body.answer_text)
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
            search_mode=body.search_mode,
        )

    run_in_background(job, _worker)
    return {"job_id": job.id}


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
    return {
        "status": "dispatched",
        "agent_name": agent_name,
        "org_id": org_id,
    }


@router.get("/widgets/email")
def widget_email():
    return _widget("email")


@router.get("/widgets/calendar")
def widget_calendar():
    return _widget("calendar")


@router.get("/widgets/graph")
def widget_graph():
    return _widget("graph")


# ---- internals ---------------------------------------------------------------

def _compose_answer_for_chat(question: dict, selected_option: str, answer_text: str) -> str:
    labels_by_value = {
        str(o.get("value") or ""): str(o.get("label") or "")
        for o in (question.get("suggested_options") or [])
    }
    label = labels_by_value.get(selected_option, selected_option)
    lines = [f"(Answering: '{question.get('question_text') or ''}')"]
    if selected_option:
        lines.append(f"Selected: {label}")
    if answer_text:
        lines.append(answer_text)
    return "\n".join(lines)
