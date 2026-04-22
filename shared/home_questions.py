"""Assistant-initiated questions surfaced to the user on the home dashboard.

Producer (e.g. a graph-discovery cron) calls ``queue_question(...)`` to push
a structured question with optional quick-reply options and a follow-up
action. When the user answers via the dashboard, ``mark_answered(...)``
persists the answer and ``dispatch_followup(...)`` kicks off the follow-up
tool-queue job (e.g. research a topic the user said "yes" to).

The NocoDB `assistant_questions` table is provisioned manually in the
NocoDB UI (project convention — see README / `daily_digests` precedent).
All reads/writes no-op silently if the table is absent so the dashboard
keeps rendering while the table is being created.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from infra.config import NOCODB_TABLE_ASSISTANT_QUESTIONS
from infra.nocodb_client import NocodbClient

_log = logging.getLogger("home.questions")

STATUS_PENDING = "pending"
STATUS_ANSWERED = "answered"
STATUS_DISMISSED = "dismissed"


def _table_present(client: NocodbClient) -> bool:
    if NOCODB_TABLE_ASSISTANT_QUESTIONS in client.tables:
        return True
    _log.info("%s table absent — question API returning inert results",
              NOCODB_TABLE_ASSISTANT_QUESTIONS)
    return False


def _decode_options(raw: Any) -> list[dict]:
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError):
        return []


def _hydrate(row: dict) -> dict:
    return {
        "id": row.get("Id"),
        "org_id": row.get("org_id"),
        "question_text": row.get("question_text") or "",
        "suggested_options": _decode_options(row.get("suggested_options")),
        "context_ref": row.get("context_ref") or "",
        "followup_action": row.get("followup_action") or "",
        "status": row.get("status") or STATUS_PENDING,
        "answer_selected_option": row.get("answer_selected_option") or "",
        "answer_text": row.get("answer_text") or "",
        "conversation_id": row.get("conversation_id"),
        "message_id": row.get("message_id"),
        "created_at": row.get("CreatedAt") or row.get("created_at"),
        "answered_at": row.get("answered_at"),
    }


def queue_question(
    org_id: int,
    question_text: str,
    suggested_options: list[dict] | None = None,
    context_ref: str = "",
    followup_action: str = "",
) -> int | None:
    """Insert a pending question. Returns the row id, or None if the table
    is absent (so producers don't crash during bootstrap)."""
    client = NocodbClient()
    if not _table_present(client):
        return None
    payload = {
        "org_id": int(org_id),
        "question_text": question_text,
        "suggested_options": json.dumps(suggested_options or []),
        "context_ref": context_ref or "",
        "followup_action": followup_action or "",
        "status": STATUS_PENDING,
    }
    try:
        row = client._post(NOCODB_TABLE_ASSISTANT_QUESTIONS, payload)
        qid = row.get("Id")
        _log.info("queued question  org=%d id=%s context=%s", org_id, qid, context_ref[:60])
        return qid
    except Exception:
        _log.warning("queue_question failed  org=%d", org_id, exc_info=True)
        return None


def list_pending(org_id: int, limit: int = 20) -> list[dict]:
    client = NocodbClient()
    if not _table_present(client):
        return []
    try:
        rows = client._get_paginated(NOCODB_TABLE_ASSISTANT_QUESTIONS, params={
            "where": f"(org_id,eq,{org_id})~and(status,eq,{STATUS_PENDING})",
            "sort": "-CreatedAt",
            "limit": limit,
        })
    except Exception:
        _log.warning("list_pending failed  org=%d", org_id, exc_info=True)
        return []
    return [_hydrate(r) for r in rows]


def get_question(question_id: int) -> dict | None:
    client = NocodbClient()
    if not _table_present(client):
        return None
    try:
        data = client._get(NOCODB_TABLE_ASSISTANT_QUESTIONS, params={
            "where": f"(Id,eq,{question_id})",
            "limit": 1,
        })
    except Exception:
        _log.warning("get_question failed  id=%s", question_id, exc_info=True)
        return None
    rows = data.get("list", [])
    return _hydrate(rows[0]) if rows else None


def mark_answered(
    question_id: int,
    selected_option: str = "",
    answer_text: str = "",
    conversation_id: int | None = None,
    message_id: int | None = None,
) -> None:
    client = NocodbClient()
    if not _table_present(client):
        return
    payload: dict[str, Any] = {
        "status": STATUS_ANSWERED,
        "answer_selected_option": selected_option or "",
        "answer_text": answer_text or "",
        "answered_at": datetime.now(timezone.utc).isoformat(),
    }
    if conversation_id is not None:
        payload["conversation_id"] = int(conversation_id)
    if message_id is not None:
        payload["message_id"] = int(message_id)
    try:
        client._patch(NOCODB_TABLE_ASSISTANT_QUESTIONS, int(question_id), payload)
        _log.info("question answered  id=%s option=%s", question_id, selected_option[:40])
    except Exception:
        _log.warning("mark_answered failed  id=%s", question_id, exc_info=True)


def mark_dismissed(question_id: int, reason: str = "") -> None:
    client = NocodbClient()
    if not _table_present(client):
        return
    try:
        client._patch(NOCODB_TABLE_ASSISTANT_QUESTIONS, int(question_id), {
            "status": STATUS_DISMISSED,
            "answer_text": reason or "",
            "answered_at": datetime.now(timezone.utc).isoformat(),
        })
        _log.info("question dismissed  id=%s", question_id)
    except Exception:
        _log.warning("mark_dismissed failed  id=%s", question_id, exc_info=True)


def dispatch_followup(action: str, org_id: int, question_id: int | None = None) -> dict:
    """Interpret a `followup_action` string and enqueue the matching tool job.

    Supported schemes (v1):
      - ``enqueue:research:<entity>``  → research_agent job
      - ``enqueue:scrape:<url>``       → scrape_page job
      - ``""`` / unknown               → no-op
    """
    if not action:
        return {"status": "noop"}

    try:
        scheme, _, rest = action.partition(":")
    except Exception:
        return {"status": "invalid", "action": action}

    if scheme != "enqueue" or not rest:
        _log.info("followup unhandled  action=%s", action)
        return {"status": "unhandled", "action": action}

    job_type, _, arg = rest.partition(":")
    if not job_type or not arg:
        return {"status": "invalid", "action": action}

    try:
        from workers.tool_queue import get_tool_queue
    except Exception:
        _log.warning("tool_queue import failed; followup skipped")
        return {"status": "queue_unavailable"}
    tq = get_tool_queue()
    if not tq:
        return {"status": "queue_unavailable"}

    if job_type == "research":
        payload = {"topic": arg, "org_id": org_id, "source_question_id": question_id}
        target_type = "research_planner"
    elif job_type == "scrape":
        payload = {"url": arg, "org_id": org_id, "source_question_id": question_id}
        target_type = "scrape_page"
    else:
        _log.info("followup unknown job_type=%s", job_type)
        return {"status": "unknown_job_type", "job_type": job_type}

    try:
        job_id = tq.submit(target_type, payload, source=f"home_question:{question_id}", org_id=org_id)
        _log.info("followup dispatched  action=%s job_id=%s", action, job_id)
        return {"status": "dispatched", "job_type": target_type, "job_id": job_id}
    except Exception as e:
        _log.warning("followup submit failed  action=%s err=%s", action, e, exc_info=True)
        return {"status": "submit_failed", "error": str(e)}
