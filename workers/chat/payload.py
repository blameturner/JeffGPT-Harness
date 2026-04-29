from __future__ import annotations

import logging

from infra.config import BASE_SYSTEM_PROMPT
from shared.temporal import build_temporal_context

_log = logging.getLogger("chat.payload")


_SEARCH_RESULTS_PROMPT = """\
The following are LIVE web search results retrieved just now for this conversation. \
Use these sources to inform your answer. Cite specific facts from them. \
Do NOT claim you cannot search the web — you already did, and these are the results. \
Do NOT output tool calls, function calls, or any special tokens. \
Just write your answer in plain text using the results below.

"""


def build_chat_payload(
    history: list[dict],
    user_message: str,
    style_prompt: str,
    system: str | None,
    search_context: str,
    search_note: str,
    rag_context: str,
    search_status: str = "",
    graph_context: str = "",
    chat_memory: str = "",
    system_note: str = "",
) -> list[dict]:
    payload: list[dict] = []
    payload.append({"role": "system", "content": BASE_SYSTEM_PROMPT})
    payload.append({"role": "system", "content": build_temporal_context()})
    if system_note and system_note.strip():
        payload.append({"role": "system", "content": f"[chat_system_note] {system_note.strip()}"})
    if chat_memory:
        payload.append({"role": "system", "content": chat_memory})
    if system:
        payload.append({"role": "system", "content": system})

    if search_context:
        payload.append({"role": "system", "content": _SEARCH_RESULTS_PROMPT + search_context})
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
    if graph_context:
        payload.append({
            "role": "system",
            "content": (
                "The following facts come from the user's knowledge graph. "
                "Treat them as established context; cite them when relevant. "
                "Format edges as `(A) -[REL]-> (B)` when referring back.\n\n"
                f"{graph_context}"
            ),
        })
    # style last so formatting instructions sit closest to history
    payload.append({"role": "system", "content": style_prompt})
    payload.extend(history)
    payload.append({"role": "user", "content": user_message})
    return payload
