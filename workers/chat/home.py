"""Home-dashboard chat adapter.

Thin wrapper over :class:`workers.chat.agent.ChatAgent` that:

1. Targets the org's rolling "home" conversation instead of creating a new one.
2. Prepends today's digest to the system prompt so replies like "tell me more
   about the first cluster" have the digest in scope without the user needing
   to paste it. The preface is cached per-org with a short TTL so chat turns
   don't repeatedly hit NocoDB + the filesystem.
3. When ``answer_question_id`` is supplied, hydrates the pending
   ``assistant_questions`` row, then marks the question answered + dispatches
   its follow-up action after the chat job finishes. Answer flows use a
   *lightweight* path (no web search, bounded tokens) because the real work
   is dispatched via ``followup_action``; the chat turn is just the human-
   readable acknowledgement.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

from infra.config import get_feature
from infra.nocodb_client import NocodbClient
from shared import digest_reader, home_questions
from shared.home_conversation import get_or_create_home_conversation
from shared.jobs import STORE
from workers.chat.agent import ChatAgent

_log = logging.getLogger("home.chat")

_PREFACE_CACHE_TTL_S = 300
_preface_cache: dict[int, tuple[float, str]] = {}
_preface_lock = threading.Lock()


def _build_digest_preface(org_id: int) -> str:
    now = time.time()
    with _preface_lock:
        hit = _preface_cache.get(org_id)
        if hit and now - hit[0] < _PREFACE_CACHE_TTL_S:
            return hit[1]

    client = NocodbClient()
    row = digest_reader.latest_digest(client, org_id)
    markdown, _ = digest_reader.read_markdown(row)
    if not markdown:
        preface = ""
    else:
        cap = int(get_feature("home", "digest_preface_chars", 2000))
        snippet = markdown[:cap]
        date_str = (row or {}).get("digest_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        preface = (
            "You are replying inside the user's HOME dashboard. The user may be "
            "responding to the daily digest below. Ground your answer in it when "
            "relevant; otherwise treat the reply as ordinary conversation.\n\n"
            f"--- DAILY DIGEST ({date_str}) ---\n{snippet}\n--- END DIGEST ---"
        )

    with _preface_lock:
        _preface_cache[org_id] = (now, preface)
    return preface


def invalidate_digest_preface(org_id: int | None = None) -> None:
    """Called when a new digest lands so the next chat turn sees it."""
    with _preface_lock:
        if org_id is None:
            _preface_cache.clear()
        else:
            _preface_cache.pop(org_id, None)


_ACK_SYSTEM = (
    "You are acknowledging the user's answer to a structured question on their "
    "HOME dashboard. The backend has already dispatched any follow-up work. "
    "Reply in ONE short sentence confirming you got their answer and naming "
    "the follow-up that will happen, if any. Do not restate the question. "
    "Do not offer to do more work."
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
    lightweight: bool = False,
) -> None:
    """Run a single home-conversation turn.

    Set ``lightweight=True`` for answer-acknowledgement flows: disables web
    search, bounds max_tokens, and switches to a terse system prompt.
    """
    convo = get_or_create_home_conversation(org_id, model=model)

    question: dict[str, Any] | None = None
    if answer_question_id:
        question = home_questions.get_question(int(answer_question_id))
        if not question:
            STORE.append(job, {"type": "error", "message": f"question {answer_question_id} not found"})
            return

    if lightweight:
        search_mode = "disabled"
        if max_tokens is None:
            max_tokens = 200
        system_preface = _ACK_SYSTEM
    else:
        system_preface = _build_digest_preface(org_id)

    agent = ChatAgent(
        model=model,
        org_id=org_id,
        search_enabled=search_mode != "disabled",
    )
    agent._search_mode = search_mode

    _log.info(
        "home turn  org=%d conv=%s question=%s lightweight=%s preface=%d",
        org_id, convo.get("Id"), answer_question_id, lightweight, len(system_preface),
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
