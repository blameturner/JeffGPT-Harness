from __future__ import annotations

import logging
import threading

import requests

_log = logging.getLogger("chat.persistence")


def _log_db_exception(context: str, conversation_id: int, exc: BaseException) -> None:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        _log.error(
            "%s  conv=%s status=%s body=%s",
            context, conversation_id, exc.response.status_code, exc.response.text[:2000],
            exc_info=True,
        )
    else:
        _log.error("%s  conv=%s", context, conversation_id, exc_info=True)


def schedule_status_processing_write(db, conversation_id: int) -> None:
    def _bg():
        try:
            db.update_conversation(conversation_id, {"status": "processing"})
        except Exception:
            _log.warning("status update to processing failed  conv=%s", conversation_id)

    threading.Thread(target=_bg, daemon=True).start()


def schedule_user_message_write(
    db,
    conversation_id: int,
    org_id: int,
    user_message: str,
    model: str,
    style_key: str,
) -> threading.Event:
    # nocodb server-stamps created_at on POST arrival — caller must wait before posting assistant msg or ordering flips
    written = threading.Event()

    def _bg():
        try:
            db.add_message(
                conversation_id=conversation_id,
                org_id=org_id,
                role="user",
                content=user_message,
                model=model,
                response_style=style_key,
            )
        except Exception as e:
            _log_db_exception("user message persist failed", conversation_id, e)
        finally:
            written.set()

    threading.Thread(target=_bg, daemon=True).start()
    return written


def persist_assistant_message(
    db,
    conversation_id: int,
    org_id: int,
    output: str,
    final_model: str,
    tokens_input: int,
    tokens_output: int,
    style_key: str,
    search_sources: list[dict],
    search_status: str,
    search_confidence: str,
    search_context: str,
    intent_dict: dict | None,
) -> bool:
    intent_meta: dict = {}
    if intent_dict:
        # dual-write: nested classification.* for frontend, flat intent/intent_entities for legacy consumers
        classification = {
            "route": intent_dict.get("route"),
            "intent": intent_dict.get("intent"),
            "secondary_intent": intent_dict.get("secondary_intent"),
            "entities": intent_dict.get("entities") or [],
            "confidence": intent_dict.get("confidence"),
        }
        intent_meta = {
            "classification": classification,
            "intent": intent_dict.get("intent"),
            "intent_entities": intent_dict.get("entities") or [],
            "search_queries": (
                [s.get("url") for s in search_sources if s.get("url")]
                if search_sources
                else []
            ),
        }
        if search_status == "failed":
            intent_meta["search_status_reason"] = "reformulation_exhausted"
        elif search_status == "deferred":
            intent_meta["search_status_reason"] = "contextual_latency_cap"

    try:
        msg_row = db.add_message(
            conversation_id=conversation_id,
            org_id=org_id,
            role="assistant",
            content=output,
            model=str(final_model),
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            response_style=style_key,
            search_used=bool(search_sources) or search_status in ("used", "no_results", "error"),
            search_status=search_status,
            search_confidence=search_confidence,
            search_source_count=len(search_sources),
            search_context_text=search_context,
            **intent_meta,
        )
        _log.info("persisted assistant message  conv=%s chars=%d", conversation_id, len(output))
    except Exception as e:
        _log_db_exception("assistant message persist failed", conversation_id, e)
        return False

    if search_sources and msg_row.get("Id"):
        try:
            db.add_message_search_sources(
                message_id=msg_row["Id"],
                conversation_id=conversation_id,
                org_id=org_id,
                sources=search_sources,
            )
        except Exception as e:
            _log_db_exception("search sources persist failed", conversation_id, e)

    return True
