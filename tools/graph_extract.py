"""graph_extract tool handler — extracts knowledge graph relationships from a chat turn."""

import logging

_log = logging.getLogger("tools.graph_extract")


def _handle_graph_extract(payload: dict) -> dict:
    """Extract knowledge graph relationships from a chat turn."""
    from workers.chat.graph import extract_and_write_graph

    user_text = payload.get("user_text") or ""
    assistant_text = payload.get("assistant_text") or ""
    conversation_id = payload.get("conversation_id") or 0
    org_id = int(payload.get("org_id") or 0)

    if not user_text and not assistant_text:
        _log.info("queue graph_extract: skipped — empty input  org=%d", org_id)
        return {"written": 0, "error": "empty input"}

    _log.info("queue graph_extract: starting  org=%d  text_chars=%d",
              org_id, len(user_text) + len(assistant_text))
    extract_and_write_graph(user_text, assistant_text, conversation_id, org_id)
    _log.info("queue graph_extract: complete  org=%d", org_id)
    return {"status": "ok"}
