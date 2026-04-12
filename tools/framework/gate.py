"""
Heuristic gate: decides whether a user message *might* need any tool.

Zero model calls. Reuses the battle-tested regex patterns from
workers.search.heuristics (_explicit_search_intent, _definitely_no_search)
and adds rag_lookup + code_exec detection that the existing heuristics don't cover.

Design principle — false positives are cheap (planner gets invoked unnecessarily
and may return an empty plan). False negatives are expensive (user gets a stale
answer). So this gate is intentionally broad.
"""

from __future__ import annotations

import re

from workers.search.heuristics import _definitely_no_search, _explicit_search_intent


_RAG_LOOKUP = re.compile(
    r"\b(we (discuss|discussed|talked|spoke|covered|mentioned)|"
    r"(did|have) we (discuss|discussed|talk|talked|cover|covered)|"
    r"you (said|mentioned|told me|noted)|"
    r"our previous|last time|earlier (we|you|i)|remember when|"
    r"as (we|you) (said|noted|mentioned))\b",
    re.I,
)

_CODE_EXEC = re.compile(
    # "run this <optional adjective> (code|script|program|snippet)"
    r"\b(run|execute|eval)\b[^.?!]{0,40}\b(this|that|the)\b(\s+\w+){0,2}\s+(code|script|program|snippet)\b"
    # "write and run ..."
    r"|\bwrite and run\b"
    # "run a python script" / "execute bash code" etc.
    r"|\b(run|execute|eval)\s+(a|an|this|that|the|my|some|the following)?\s*"
    r"(python|bash|shell|node|javascript|js|typescript|ts|sql)\s+(script|code|snippet|program)?\b"
    # bare "run this python"
    r"|\b(run|execute)\s+(this|that|the)\s+(python|bash|shell|node|js|ts|sql)\b",
    re.I,
)

# Patterns the spec covers that _explicit_search_intent does NOT already hit.
# Includes bare financial/temporal terms not already covered by heuristics.py.
_EXTRA_SEARCH = re.compile(
    r"\b(202[5-9]|today|tonight|this (week|month|year))\b"
    r"|\b(news|forecast|temperature|weather)\b"
    r"|\b(latest|recent|current)\s+\w+"
    r"|\b(rate|rates|price|prices|cost|decision|update)\b",
    re.I,
)


# Follow-up patterns — short messages that probably continue a prior tool-using turn.
_FOLLOW_UP = re.compile(
    r"\b(more|another|also|what about|and|any other|any others|details|details on|elaborate|"
    r"expand|continue|tell me more|what else|go on)\b",
    re.I,
)


def gate_check(message: str, conversation_context: str = "") -> set[str]:
    """
    Return the set of tool names that *might* be needed.

    conversation_context is optional — a short string (usually the last
    assistant turn). If it contains markers of a prior tool-using turn
    (e.g. "[Tool results"), we drop the threshold for follow-up searches
    like "and what about X?" that otherwise wouldn't trigger.

    Examples:
        gate_check("thanks")                                    -> set()
        gate_check("what's the weather in Sydney?")             -> {"web_search"}
        gate_check("run this python script")                    -> {"code_exec"}
        gate_check("what did we discuss about auth?")           -> {"rag_lookup"}
        gate_check("and what about Melbourne?",
                   conversation_context="[Tool results — Sydney weather]")
                                                                -> {"web_search"}
    """
    msg = (message or "").strip()
    if not msg:
        return set()

    # Definitely-no-search still allows rag_lookup + code_exec on short conversational turns.
    if _definitely_no_search(msg):
        hints: set[str] = set()
        if _RAG_LOOKUP.search(msg):
            hints.add("rag_lookup")
        if _CODE_EXEC.search(msg) or "```" in msg:
            hints.add("code_exec")
        # Context-aware: follow-up on a prior search-using turn reopens web_search.
        if conversation_context and _FOLLOW_UP.search(msg):
            if "[Tool results" in conversation_context or "web_search" in conversation_context.lower():
                hints.add("web_search")
        return hints

    hints: set[str] = set()
    if _explicit_search_intent(msg)[0] or _EXTRA_SEARCH.search(msg):
        hints.add("web_search")
    if _RAG_LOOKUP.search(msg):
        hints.add("rag_lookup")
    if _CODE_EXEC.search(msg) or "```" in msg:
        hints.add("code_exec")

    # Context-aware follow-up: pulls in web_search for vague continuations.
    if "web_search" not in hints and conversation_context and _FOLLOW_UP.search(msg):
        if "[Tool results" in conversation_context or "web_search" in conversation_context.lower():
            hints.add("web_search")

    return hints
