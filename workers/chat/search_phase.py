from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from tools.search.intent import (
    CHAT_INTENT_CHITCHAT,
    CHAT_INTENT_CONTEXTUAL,
    SEARCH_POLICY_NONE,
    classify_message_intent,
)
from tools.search.orchestrator import run_web_search

_log = logging.getLogger("chat.search_phase")


@dataclass
class SearchPhaseResult:
    search_context: str = ""
    search_sources: list[dict] = field(default_factory=list)
    search_confidence: str = "none"
    search_status: str = "not_used"
    search_note: str = ""
    search_errored: bool = False
    intent_dict: dict | None = None
    consent_required: bool = False
    consent_reason: str = ""


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def run_search_phase(
    user_message: str,
    history: list[dict],
    convo: dict,
    conversation_id: int,
    org_id: int,
    search_enabled: bool,
    search_consent_declined: bool,
    emit: Callable[[dict], None],
    span: Callable[[str, float], None],
) -> SearchPhaseResult:
    result = SearchPhaseResult()

    if not search_consent_declined:
        _t_classifier = time.perf_counter()
        try:
            result.intent_dict = classify_message_intent(user_message, history=history)
        except Exception:
            _log.warning("classify_message_intent failed", exc_info=True)
            result.intent_dict = None
        span("intent_classify_ms", _t_classifier)

        _grounding = convo.get("contextual_grounding_enabled")
        grounding_ok = _truthy(_grounding if _grounding is not None else True)
        if (
            result.intent_dict
            and result.intent_dict.get("intent") == CHAT_INTENT_CONTEXTUAL
            and not grounding_ok
        ):
            _log.info(
                "contextual grounding disabled  conv=%s — downgrading to chitchat",
                conversation_id,
            )
            result.intent_dict["intent"] = CHAT_INTENT_CHITCHAT
            result.intent_dict["search_policy"] = SEARCH_POLICY_NONE
            result.intent_dict["response_template"] = "chitchat_casual"

        if result.intent_dict:
            emit({
                "type": "intent_classified",
                "route": result.intent_dict.get("route"),
                "intent": result.intent_dict.get("intent"),
                "secondary_intent": result.intent_dict.get("secondary_intent"),
                "entities": result.intent_dict.get("entities") or [],
                "confidence": result.intent_dict.get("confidence"),
            })

    policy = (result.intent_dict or {}).get("search_policy", SEARCH_POLICY_NONE)
    confidence = (result.intent_dict or {}).get("confidence", "low")
    intent_label = (result.intent_dict or {}).get("intent") or CHAT_INTENT_CHITCHAT

    should_auto_search = (
        result.intent_dict is not None
        and policy != SEARCH_POLICY_NONE
        and (search_enabled or confidence == "high")
    )

    needs_consent = (
        result.intent_dict is not None
        and policy != SEARCH_POLICY_NONE
        and not search_enabled
        and confidence in ("medium", "low")
        and not search_consent_declined
    )

    if should_auto_search:
        _log.info(
            "dispatching search  conv=%s intent=%s policy=%s",
            conversation_id,
            result.intent_dict.get("intent"),
            result.intent_dict.get("search_policy"),
        )
        emit({"type": "searching"})
        _t_exec = time.perf_counter()
        try:
            result.search_context, result.search_sources, result.search_confidence = run_web_search(
                user_message,
                org_id,
                intent_dict=result.intent_dict,
                history=history,
            )
        except Exception:
            _log.error("web search failed", exc_info=True)
            result.search_context, result.search_sources, result.search_confidence = "", [], "failed"
            result.search_errored = True
        span("search_execute_ms", _t_exec)
        emit({
            "type": "search_complete",
            "source_count": len(result.search_sources),
            "sources": result.search_sources,
            "ok": bool(result.search_sources),
            "confidence": result.search_confidence,
        })
        if result.search_errored:
            result.search_status = "error"
        elif result.search_sources:
            result.search_status = "used"
        elif result.search_confidence == "deferred":
            result.search_status = "deferred"
            emit({
                "type": "search_deferred",
                "entities": (result.intent_dict or {}).get("entities") or [],
            })
        elif result.search_confidence == "failed":
            result.search_status = "failed"
        else:
            result.search_status = "no_results"
        if result.search_status in ("no_results",):
            result.search_note = (
                "SEARCH STATUS: A live web search was performed but the "
                "search engine returned no results at all"
                + (" (the search backend errored)." if result.search_errored
                   else " for this query.")
                + " Tell the user you searched but found nothing, then "
                "answer from your own knowledge with a recency caveat. "
                "Do NOT say you cannot search — you can and did, "
                "it just returned empty this time."
            )
    elif needs_consent:
        result.search_status = "consent_required"
        result.consent_reason = (
            f"this looks like a {intent_label.replace('_', ' ')} — "
            "may benefit from a web search"
        )
        _log.info(
            "search consent required  conv=%s intent=%s confidence=%s",
            conversation_id, intent_label, confidence,
        )
        result.consent_required = True
    elif search_consent_declined:
        result.search_status = "declined"
        result.search_note = (
            "SEARCH STATUS: The user declined a live web search for this "
            "question. Answer from general knowledge and explicitly flag "
            "that anything time-sensitive may be out of date. Do NOT "
            "claim you lack the ability to search — the user chose not "
            "to allow it this turn."
        )

    return result
