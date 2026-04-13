from __future__ import annotations

import logging

from config import BASE_SYSTEM_PROMPT
from workers.search.temporal import build_temporal_context

_log = logging.getLogger("chat.payload")


def build_chat_payload(
    history: list[dict],
    user_message: str,
    style_prompt: str,
    system: str | None,
    search_context: str,
    search_note: str,
    rag_context: str,
    search_status: str = "",
) -> list[dict]:
    payload: list[dict] = []
    payload.append({"role": "system", "content": BASE_SYSTEM_PROMPT})
    payload.append({"role": "system", "content": build_temporal_context()})
    if system:
        payload.append({"role": "system", "content": system})
    if search_context and search_status == "awaiting_approval":
        payload.append({"role": "system", "content": (
            "You have prepared a search/research plan for the user's question. "
            "Present this plan clearly and ask the user to approve it or suggest changes. "
            "Do NOT answer the question yet — you have not analysed any sources. "
            "Just present the plan below and ask for confirmation.\n\n"
            + search_context
        )})
    elif search_context and search_status == "queued":
        payload.append({"role": "system", "content": (
            "You have queued a deep search or research job for the user's question. "
            "Tell the user their request is being processed and results will be "
            "delivered to this conversation when ready. Do NOT attempt to answer "
            "the question — the analysis has not completed yet.\n\n"
            + search_context
        )})
    elif search_context:
        payload.append({"role": "system", "content": (
            "The following are LIVE web search results retrieved just now for "
            "this conversation. Use these sources to inform your answer. "
            "Cite specific facts from them. Do NOT claim you cannot search "
            "the web — you already did, and these are the results.\n\n"
            + search_context
        )})
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
