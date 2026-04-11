from __future__ import annotations

import logging

from config import BASE_SYSTEM_PROMPT
from workers.web_search import build_temporal_context

_log = logging.getLogger("chat.payload")


def build_chat_payload(
    history: list[dict],
    user_message: str,
    style_prompt: str,
    system: str | None,
    search_context: str,
    search_note: str,
    rag_context: str,
) -> list[dict]:
    payload: list[dict] = []
    payload.append({"role": "system", "content": BASE_SYSTEM_PROMPT})
    payload.append({"role": "system", "content": build_temporal_context()})
    if system:
        payload.append({"role": "system", "content": system})
    if search_context:
        payload.append({"role": "system", "content": search_context})
    if search_note:
        payload.append({"role": "system", "content": search_note})
    if rag_context:
        payload.append({
            "role": "system",
            "content": (
                "The following context was retrieved from this "
                "conversation's memory. Use it where relevant.\n\n"
                f"{rag_context}"
            ),
        })
    # style last so formatting instructions sit closest to history
    payload.append({"role": "system", "content": style_prompt})
    payload.extend(history)
    payload.append({"role": "user", "content": user_message})
    return payload
