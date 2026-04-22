"""Home-dashboard chat adapter.

Thin wrapper over :class:`workers.chat.agent.ChatAgent` that:

1. Targets the org's rolling "home" conversation instead of creating a new one.
2. Prepends today's digest to the system prompt so replies like "tell me more
   about the first cluster" have the digest in scope without the user needing
   to paste it.
3. When ``answer_question_id`` is supplied, hydrates the pending
   ``assistant_questions`` row, annotates the user turn with the structured
   answer, and marks the question answered + dispatches its follow-up action
   after the chat job finishes.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from infra.config import NOCODB_TABLE_DAILY_DIGESTS, get_feature
from infra.nocodb_client import NocodbClient
from shared import home_questions
from shared.home_conversation import get_or_create_home_conversation
from shared.jobs import STORE
from workers.chat.agent import ChatAgent

_log = logging.getLogger("home.chat")


def _latest_digest(client: NocodbClient, org_id: int) -> dict | None:
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


def _read_digest_markdown(row: dict | None) -> str:
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


def _build_digest_preface(org_id: int) -> str:
    """Return a system-prompt preface containing today's digest (truncated),
    or empty string if no digest is available."""
    client = NocodbClient()
    row = _latest_digest(client, org_id)
    markdown = _read_digest_markdown(row)
    if not markdown:
        return ""
    cap = int(get_feature("home", "digest_preface_chars", 2000))
    snippet = markdown[:cap]
    date_str = (row or {}).get("digest_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return (
        "You are replying inside the user's HOME dashboard. The user may be "
        "responding to the daily digest below. Ground your answer in it when "
        "relevant; otherwise treat the reply as ordinary conversation.\n\n"
        f"--- DAILY DIGEST ({date_str}) ---\n{snippet}\n--- END DIGEST ---"
    )


def run_home_turn(
    job,
    org_id: int,
    model: str,
    message: str,
    answer_question_id: int | None = None,
    answer_selected_option: str = "",
    answer_free_text: str = "",
    response_style: str | None = None,
    search_mode: str = "basic",
    search_consent_confirmed: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> None:
    """Run a single home-conversation turn. Matches the shape of
    :meth:`ChatAgent.run_job` so it can be invoked via ``run_in_background``."""
    convo = get_or_create_home_conversation(org_id, model=model)

    question: dict[str, Any] | None = None
    if answer_question_id:
        question = home_questions.get_question(int(answer_question_id))
        if not question:
            STORE.append(job, {
                "type": "error",
                "message": f"question {answer_question_id} not found",
            })
            return

    system_preface = _build_digest_preface(org_id)

    agent = ChatAgent(
        model=model,
        org_id=org_id,
        search_enabled=search_mode != "disabled",
    )
    agent._search_mode = search_mode

    _log.info(
        "home turn  org=%d conv=%s question=%s digest_preface=%d",
        org_id, convo.get("Id"), answer_question_id, len(system_preface),
    )

    try:
        agent.run_job(
            job,
            user_message=message,
            conversation_id=convo.get("Id"),
            system=system_preface or None,
            temperature=temperature,
            max_tokens=max_tokens,
            rag_enabled=None,
            rag_collection=None,
            knowledge_enabled=None,
            search_consent_confirmed=search_consent_confirmed,
            response_style=response_style,
        )
    finally:
        if question:
            try:
                home_questions.mark_answered(
                    question_id=int(question["id"]),
                    selected_option=answer_selected_option,
                    answer_text=answer_free_text,
                    conversation_id=convo.get("Id"),
                )
                followup = (question.get("followup_action") or "").strip()
                if followup:
                    result = home_questions.dispatch_followup(
                        followup,
                        org_id=org_id,
                        question_id=int(question["id"]),
                    )
                    STORE.append(job, {
                        "type": "status",
                        "phase": "followup",
                        "message": f"followup: {result.get('status')}",
                        "detail": result,
                    })
            except Exception:
                _log.warning("post-answer bookkeeping failed  q=%s", question.get("id"), exc_info=True)
