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

import logging
import re

_log = logging.getLogger("tools.gate")

from workers.search.heuristics import _definitely_no_search, _explicit_search_intent


_RAG_LOOKUP = re.compile(
    r"\b(we (discuss|discussed|talked|spoke|covered|mentioned)|"
    r"(did|have) we (discuss|discussed|talk|talked|cover|covered)|"
    r"you (said|mentioned|told me|noted)|"
    r"our previous|last time|earlier (we|you|i)|remember when|"
    r"as (we|you) (said|noted|mentioned))\b",
    re.I,
)

## deep_search is NOT auto-detected — it's a UI toggle (search_mode="deep").
## The gate only handles web_search, rag_lookup, and code_exec hints.

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

# Broad search triggers — we want to search optimistically.  RWKV
# summarisation is fast and the planner can still return an empty plan,
# so false positives are cheap.
_EXTRA_SEARCH = re.compile(
    # Temporal / current events
    r"\b(202[4-9]|today|tonight|this (week|month|year)|right now|currently)\b"
    r"|\b(news|forecast|temperature|weather|election|announcement)\b"
    r"|\b(latest|recent|current|new|updated|upcoming)\s+\w+"
    r"|\b(rate|rates|price|prices|cost|decision|update|release|version)\b"
    # Knowledge / factual questions
    r"|\b(what is|what are|who is|who are|how does|how do|how to|why does|why do)\b"
    r"|\b(tell me about|explain|describe|define|meaning of|difference between)\b"
    r"|\b(best|top|recommended|popular|comparison|versus|vs)\b"
    # Entities that likely benefit from search
    r"|\b(company|product|framework|library|tool|platform|service|app|api)\b"
    r"|\b(country|city|government|policy|regulation|law|standard)\b",
    re.I,
)


# Follow-up patterns — short messages that probably continue a prior tool-using turn.
_FOLLOW_UP = re.compile(
    r"\b(more|another|also|what about|and|any other|any others|details|details on|elaborate|"
    r"expand|continue|tell me more|what else|go on)\b",
    re.I,
)

# Code-mode signals. These apply ONLY when the caller passes mode="code" so
# the chat flow's behaviour is unchanged. The patterns are intentionally
# broad because the code agent will be heavily tooled — false positives cost
# one planner call, false negatives cost stale API documentation.
_CODE_API_LOOKUP = re.compile(
    # "how do I X in Python", "what's the syntax for Y", "is there a Z function in <lib>"
    r"\b(how (do|to)|what(?:'s| is) the (syntax|api|signature)|is there (a|an)|does \w+ (have|support)|"
    r"which (method|function|class|api|flag))\b"
    # library / api reference style
    r"|\b(docs? for|documentation for|reference for|api reference|man page for)\b"
    # specific libraries the planner should search for
    r"|\b(asyncio|fastapi|pydantic|sqlalchemy|numpy|pandas|django|flask|react|pytorch|tensorflow|"
    r"httpx|requests|playwright|langchain|openai sdk|anthropic sdk)\b",
    re.I,
)

_CODE_ERROR_SIGNALS = re.compile(
    # python-style errors
    r"\b(TypeError|ValueError|KeyError|AttributeError|NameError|ImportError|RuntimeError|"
    r"IndexError|AssertionError|ZeroDivisionError|StopIteration|RecursionError|"
    r"ModuleNotFoundError|FileNotFoundError|PermissionError|UnicodeDecodeError)\b"
    # JS-style
    r"|\b(ReferenceError|SyntaxError|Uncaught|undefined is not a function|cannot read propert)\b"
    # generic
    r"|\bstack\s*trace\b|Traceback \(most recent call last\)"
    # command / shell errors
    r"|\bcommand not found\b|permission denied|segmentation fault",
    re.I,
)

_CODE_RUN_REQUEST = re.compile(
    # direct run requests ("run this", "execute this") already covered by _CODE_EXEC,
    # but also: "what's the output of", "what does this function return", "test this snippet"
    r"\b(what(?:'s| does| is) (the )?(output|result) of"
    # allow zero or more intervening nouns between this/that/it and the verb
    r"|what does (this|it|that) (?:\w+\s+){0,3}(print|return|output|give|produce|yield)"
    r"|test (this|it|that) (?:\w+\s+){0,2}(function|snippet|script|code))\b",
    re.I,
)


def gate_check(
    message: str,
    conversation_context: str = "",
    mode: str = "chat",
) -> set[str]:
    """
    Return the set of tool names that *might* be needed.

    conversation_context is optional — a short string (usually the last
    assistant turn). If it contains markers of a prior tool-using turn
    (e.g. "[Tool results"), we drop the threshold for follow-up searches
    like "and what about X?" that otherwise wouldn't trigger.

    mode defaults to "chat". Pass mode="code" from the code agent path to
    enable additional patterns for API lookups, error traces, and test/run
    phrasing that the chat-focused heuristics miss.

    Examples:
        gate_check("thanks")                                    -> set()
        gate_check("what's the weather in Sydney?")             -> {"web_search"}
        gate_check("run this python script")                    -> {"code_exec"}
        gate_check("what did we discuss about auth?")           -> {"rag_lookup"}
        gate_check("and what about Melbourne?",
                   conversation_context="[Tool results — Sydney weather]")
                                                                -> {"web_search"}
        gate_check("how do I use asyncio.gather?", mode="code") -> {"web_search"}
        gate_check("TypeError: cannot ...", mode="code")        -> {"web_search","rag_lookup"}
    """
    msg = (message or "").strip()
    if not msg:
        return set()

    is_code = (mode or "chat").lower() == "code"

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
        # Code-mode extras — API/doc lookups can appear in short turns too.
        if is_code:
            if _CODE_API_LOOKUP.search(msg):
                hints.add("web_search")
            if _CODE_ERROR_SIGNALS.search(msg):
                hints.add("web_search")
                hints.add("rag_lookup")
            if _CODE_RUN_REQUEST.search(msg):
                hints.add("code_exec")
        if hints:
            _log.info("gate  msg=%s hints=%s (no_search path)", msg[:80], sorted(hints))
        return hints

    hints: set[str] = set()
    if _explicit_search_intent(msg)[0] or _EXTRA_SEARCH.search(msg):
        hints.add("web_search")
    if _RAG_LOOKUP.search(msg):
        hints.add("rag_lookup")
    if _CODE_EXEC.search(msg) or "```" in msg:
        hints.add("code_exec")

    # Code-mode augments — broader coverage for API lookups and error traces.
    if is_code:
        if _CODE_API_LOOKUP.search(msg):
            hints.add("web_search")
        if _CODE_ERROR_SIGNALS.search(msg):
            hints.add("web_search")
            hints.add("rag_lookup")
        if _CODE_RUN_REQUEST.search(msg):
            hints.add("code_exec")

    # Context-aware follow-up: pulls in web_search for vague continuations.
    if "web_search" not in hints and conversation_context and _FOLLOW_UP.search(msg):
        if "[Tool results" in conversation_context or "web_search" in conversation_context.lower():
            hints.add("web_search")

    # Optimistic search: if the message is non-trivial and no search hint
    # was found yet, still suggest web_search.  The planner decides whether
    # to actually search — false positives are cheap (one planner call),
    # false negatives cost stale answers.
    if "web_search" not in hints and len(msg) > 30:
        hints.add("web_search")

    if hints:
        _log.info("gate  msg=%s hints=%s", msg[:80], sorted(hints))
    return hints
