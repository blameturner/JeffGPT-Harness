from __future__ import annotations

import logging

_log = logging.getLogger("chat.graph")


def extract_and_write_graph(
    user_text: str,
    assistant_text: str,
    conversation_id: int,
    org_id: int,
) -> None:
    _log.info("graph extraction starting  conv=%d", conversation_id)
    try:
        from workers.enrichment_agent import _extract_relationships
    except Exception as e:
        _log.warning("graph extraction skipped — import failed: %s", e)
        return

    combined = (
        f"USER TURN:\n{user_text.strip()}\n\n"
        f"ASSISTANT REPLY:\n{assistant_text.strip()}"
    )

    try:
        written, tokens = _extract_relationships(combined, org_id)
        _log.info(
            "graph extraction done  conv=%d written=%d tokens=%d",
            conversation_id, written, tokens,
        )
    except Exception:
        _log.warning("graph extraction failed", exc_info=True)
