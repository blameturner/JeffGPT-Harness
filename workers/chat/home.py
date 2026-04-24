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

from infra.config import get_feature, is_feature_enabled
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


def _build_pa_context(org_id: int) -> str:
    """Inject top warm topics + open loops so the assistant stays oriented
    across turns without the user re-briefing it."""
    if not is_feature_enabled("pa"):
        return ""
    try:
        from shared.pa.memory import list_warm_topics, list_open_loops
    except Exception:
        return ""
    lines: list[str] = []
    try:
        topics = list_warm_topics(org_id, limit=3, min_warmth=0.3)
    except Exception:
        topics = []
    if topics:
        lines.append("CURRENT WARM TOPICS (what the user has been on):")
        for t in topics:
            phrase = (t.get("entity_or_phrase") or "").strip()
            brief = (t.get("background_brief") or "").strip()
            if brief:
                lines.append(f"- {phrase}: {brief[:400]}")
            else:
                lines.append(f"- {phrase}")
    try:
        loops = list_open_loops(org_id, status=None, limit=8) or []
    except Exception:
        loops = []
    open_loops = [lp for lp in loops if lp.get("status") in ("open", "nudged")]
    if open_loops:
        lines.append("")
        lines.append("OPEN LOOPS (things the user said they'd do — reference naturally if relevant, do NOT pivot to these):")
        for lp in open_loops[:5]:
            lines.append(f"- {(lp.get('text') or '')[:120]}")
    if not lines:
        return ""
    lines.append("")
    lines.append("Use this context only when the user's message connects to it. Do not steer to these topics unprompted.")
    return "\n".join(lines)


def _latest_assistant_reply(org_id: int, conversation_id: int) -> tuple[str, int | None]:
    """Fetch the most recent assistant message for a conversation. Called
    synchronously from run_home_turn after agent.run_job has committed."""
    try:
        client = NocodbClient()
        msgs = client.list_messages(int(conversation_id), org_id=org_id) or []
    except Exception:
        return "", None
    for m in reversed(msgs):
        if m.get("role") == "assistant" and (m.get("content") or "").strip():
            return m.get("content") or "", m.get("Id")
    return "", None


def _run_extractor_async(
    org_id: int,
    user_message: str,
    assistant_reply: str,
    source_message_id: int | None,
) -> None:
    """Runs the PA post-turn extractor in a daemon thread so the response
    path is unblocked. Best-effort only — the reply text is handed in by the
    caller to avoid any race with DB persistence."""
    if not is_feature_enabled("pa"):
        return
    if not (assistant_reply or "").strip():
        return

    def _worker():
        try:
            from shared.pa.extractor import extract_and_persist
            result = extract_and_persist(
                org_id=org_id,
                user_message=user_message,
                assistant_reply=assistant_reply,
                source_message_id=source_message_id,
            )
            _log.info(
                "pa extractor  org=%d loops=+%d/-%d facts=%d topics=%d",
                org_id,
                result.get("loops_created", 0),
                result.get("loops_resolved", 0),
                result.get("facts_written", 0),
                result.get("topics_boosted", 0),
            )
            _queue_topic_research(org_id, result.get("new_topic_ids") or [])
        except Exception:
            _log.warning("pa extractor thread failed  org=%d", org_id, exc_info=True)

    threading.Thread(target=_worker, daemon=True, name=f"pa-extractor-{org_id}").start()


def _queue_topic_research(org_id: int, topic_ids: list[int]) -> None:
    if not topic_ids:
        return
    try:
        from workers.tool_queue import get_tool_queue
        tq = get_tool_queue()
    except Exception:
        return
    if not tq:
        return
    for tid in topic_ids:
        try:
            tq.submit(
                "pa_topic_research",
                {"org_id": int(org_id), "topic_id": int(tid)},
                source="pa_extractor",
                org_id=int(org_id),
            )
        except Exception:
            _log.debug("pa_topic_research enqueue failed  topic=%s", tid, exc_info=True)


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
        preface = _build_digest_preface(org_id)
        pa_ctx = _build_pa_context(org_id)
        if pa_ctx and preface:
            system_preface = f"{preface}\n\n{pa_ctx}"
        else:
            system_preface = pa_ctx or preface

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
        if not lightweight and convo.get("Id"):
            # Read reply synchronously (post-commit) so the extractor
            # thread never races the DB write.
            reply, source_msg_id = _latest_assistant_reply(org_id, int(convo["Id"]))
            _run_extractor_async(org_id, message, reply, source_msg_id)
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
