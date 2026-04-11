from __future__ import annotations

import logging
import threading

_log = logging.getLogger("chat.persistence")


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
    # Caller must wait on the returned Event before POSTing the assistant
    # message — NocoDB server-stamps created_at per POST arrival, so reversed
    # arrival order would invert list_messages on the next turn.
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
        except Exception:
            _log.error("user message persist failed", exc_info=True)
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
) -> None:
    intent_meta: dict = {}
    if intent_dict:
        # NocoDB JSON columns — pass native lists, not json.dumps'd strings.
        intent_meta = {
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
        if search_sources and msg_row.get("Id"):
            try:
                db.add_message_search_sources(
                    message_id=msg_row["Id"],
                    conversation_id=conversation_id,
                    org_id=org_id,
                    sources=search_sources,
                )
            except Exception:
                _log.error("search sources persist failed  conv=%s", conversation_id, exc_info=True)
    except Exception:
        _log.error("assistant message persist failed  conv=%s", conversation_id, exc_info=True)
