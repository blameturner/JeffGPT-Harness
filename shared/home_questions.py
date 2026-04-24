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


def exists_by_context(org_id: int, context_ref: str) -> bool:
    """Return True if a pending or answered question with this exact
    ``context_ref`` already exists. Producers use this to avoid spamming
    duplicates when the same upstream signal fires repeatedly."""
    if not context_ref:
        return False
    client = NocodbClient()
    if not _table_present(client):
        return False
    try:
        rows = client._get_paginated(NOCODB_TABLE_ASSISTANT_QUESTIONS, params={
            "where": f"(org_id,eq,{org_id})~and(context_ref,eq,{context_ref})",
            "limit": 1,
        })
        return bool(rows)
    except Exception:
        return False


def queue_question_deduped(
    org_id: int,
    question_text: str,
    context_ref: str,
    suggested_options: list[dict] | None = None,
    followup_action: str = "",
) -> int | None:
    """``queue_question`` wrapper that no-ops if a question with the same
    ``context_ref`` already exists for the org."""
    if exists_by_context(org_id, context_ref):
        return None
    return queue_question(
        org_id=org_id,
        question_text=question_text,
        suggested_options=suggested_options,
        context_ref=context_ref,
        followup_action=followup_action,
    )


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


def mark_pending(question_id: int) -> None:
    """Revert an answered question to pending (undo / retract)."""
    client = NocodbClient()
    if not _table_present(client):
        return
    try:
        client._patch(NOCODB_TABLE_ASSISTANT_QUESTIONS, int(question_id), {
            "status": STATUS_PENDING,
            "answer_selected_option": "",
            "answer_text": "",
            "answered_at": None,
        })
        _log.info("question retracted  id=%s", question_id)
    except Exception:
        _log.warning("mark_pending failed  id=%s", question_id, exc_info=True)


def render_answer(question: dict, selected_option: str, answer_text: str) -> str:
    """Compose the user's answer as a chat message, keeping the structured
    selection visible to the model."""
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


def render_provenance(context_ref: str) -> str:
    """Turn a ``context_ref`` like ``graph_discovery:entity=Postgres`` into a
    human-readable 'Because …' phrase for the UI. Unknown schemes echo back."""
    if not context_ref:
        return ""
    try:
        scheme, _, rest = context_ref.partition(":")
    except Exception:
        return context_ref
    if scheme == "graph_discovery" and rest:
        parts = dict(p.split("=", 1) for p in rest.split(",") if "=" in p)
        entity = parts.get("entity") or parts.get("topic")
        if entity:
            return f"Because you've been working with {entity}"
    if scheme == "chat_idle" and rest:
        return "Because the system had idle time to dig deeper"
    if scheme == "manual":
        return "You queued this manually"
    return context_ref


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
    """Interpret a ``followup_action`` string and enqueue the matching job.

    Supported schemes (v1):
      - ``enqueue:research:<topic>``   → creates a research_plans row via
                                          ``create_research_plan`` which
                                          itself enqueues the planner
      - ``enqueue:scrape:<url>``       → writes a ``suggested_scrape_targets``
                                          row that the enrichment pipeline
                                          picks up on its next dispatcher tick
      - ``""`` / unknown               → no-op
    """
    if not action:
        return {"status": "noop"}

    scheme, _, rest = action.partition(":")
    if scheme != "enqueue" or not rest:
        _log.warning("followup unhandled action  action=%s", action)
        return {"status": "unhandled", "action": action}

    job_type, _, arg = rest.partition(":")
    if not job_type or not arg:
        _log.warning("followup malformed action  action=%s", action)
        return {"status": "invalid", "action": action}

    if job_type == "research":
        try:
            from tools.research.research_planner import create_research_plan
        except Exception:
            _log.error("followup research: create_research_plan import failed", exc_info=True)
            return {"status": "import_failed"}
        try:
            result = create_research_plan(arg, org_id=org_id)
        except Exception as e:
            _log.error("followup research: create_research_plan failed  topic=%s err=%s", arg, e, exc_info=True)
            return {"status": "submit_failed", "error": str(e)[:200]}
        if result.get("status") != "queued":
            _log.warning("followup research not queued  topic=%s result=%s", arg, result)
            return {"status": result.get("status") or "failed", "detail": result}
        _log.info("followup research queued  topic=%s plan_id=%s",
                  arg, result.get("plan_id"))
        return {
            "status": "dispatched",
            "job_type": "research_planner",
            "plan_id": result.get("plan_id"),
            "tool_job_id": result.get("job_id"),
        }

    if job_type == "scrape":
        from infra.nocodb_client import NocodbClient
        client = NocodbClient()
        if "suggested_scrape_targets" not in client.tables:
            _log.error("followup scrape: suggested_scrape_targets table missing")
            return {"status": "table_missing", "table": "suggested_scrape_targets"}
        try:
            row = client._post("suggested_scrape_targets", {
                "url": arg,
                "org_id": int(org_id),
                "status": "new",
                "source": f"home_question:{question_id}" if question_id else "home_question",
            })
            _log.info("followup scrape suggested  url=%s row_id=%s", arg, row.get("Id"))
            return {"status": "dispatched", "job_type": "suggested_scrape_targets",
                    "row_id": row.get("Id")}
        except Exception as e:
            _log.error("followup scrape: write failed  url=%s err=%s", arg, e, exc_info=True)
            return {"status": "submit_failed", "error": str(e)[:200]}

    _log.warning("followup unknown job_type  job_type=%s", job_type)
    return {"status": "unknown_job_type", "job_type": job_type}
