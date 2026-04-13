from __future__ import annotations

import logging

from config import BASE_SYSTEM_PROMPT
from workers.search.temporal import build_temporal_context

_log = logging.getLogger("chat.payload")


_PLAN_APPROVAL_PROMPT = """\
CRITICAL INSTRUCTION — READ CAREFULLY.

You are in PLAN PRESENTATION mode. Your ONLY job right now is to present the \
search/research plan below to the user and ask them to approve or revise it.

RULES YOU MUST FOLLOW:
1. Present the plan clearly — show the queries and what will be searched.
2. Ask the user: "Would you like me to proceed with this plan, or would you like to change anything?"
3. Do NOT answer the user's original question.
4. Do NOT attempt to research, analyse, or provide information about the topic.
5. Do NOT use any knowledge you have about the topic.
6. Do NOT speculate, hypothesise, or offer opinions on the topic.
7. Your ENTIRE response must be about the plan itself — nothing else.

If you answer the user's question instead of presenting the plan, you have FAILED your task.

THE PLAN:
"""

_QUEUED_PROMPT = """\
CRITICAL INSTRUCTION — READ CAREFULLY.

A deep search or research job has been approved and is now running in the background. \
Your ONLY job is to confirm this to the user.

RULES YOU MUST FOLLOW:
1. Tell the user their search/research has been approved and is now running.
2. Tell them results will be delivered to this conversation when ready.
3. Do NOT answer the user's original question.
4. Do NOT attempt to provide information about the topic.
5. Do NOT speculate about what the results might show.
6. Keep your response to 2-3 sentences maximum.

JOB DETAILS:
"""

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
) -> list[dict]:
    payload: list[dict] = []
    payload.append({"role": "system", "content": BASE_SYSTEM_PROMPT})
    payload.append({"role": "system", "content": build_temporal_context()})
    if system:
        payload.append({"role": "system", "content": system})

    if search_context and search_status == "awaiting_approval":
        # Plan presentation — override style prompt to prevent the model
        # from treating this as a normal chat turn.
        payload.append({"role": "system", "content": _PLAN_APPROVAL_PROMPT + search_context})
        # Skip style prompt and RAG — they can mislead the model into answering.
        payload.extend(history)
        payload.append({"role": "user", "content": user_message})
        return payload

    if search_context and search_status == "queued":
        payload.append({"role": "system", "content": _QUEUED_PROMPT + search_context})
        # Skip style prompt and RAG — same reason.
        payload.extend(history)
        payload.append({"role": "user", "content": user_message})
        return payload

    # Normal search results or no search
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
    # style last so formatting instructions sit closest to history
    payload.append({"role": "system", "content": style_prompt})
    payload.extend(history)
    payload.append({"role": "user", "content": user_message})
    return payload
