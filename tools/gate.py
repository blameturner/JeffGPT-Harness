from __future__ import annotations

import logging
import re

_log = logging.getLogger("tools.gate")

from infra.config import is_feature_enabled
from tools.search.heuristics import _definitely_no_search, _explicit_search_intent


_RAG_LOOKUP = re.compile(
    r"\b(we (discuss|discussed|talked|spoke|covered|mentioned)|"
    r"(did|have) we (discuss|discussed|talk|talked|cover|covered)|"
    r"you (said|mentioned|told me|noted)|"
    r"our previous|last time|earlier (we|you|i)|remember when|"
    r"as (we|you) (said|noted|mentioned))\b",
    re.I,
)


_EXTRA_SEARCH = re.compile(
    r"\b(202[4-9]|today|tonight|this (week|month|year)|right now|currently)\b"
    r"|\b(news|forecast|temperature|weather|election|announcement)\b"
    r"|\b(latest|recent|current|new|updated|upcoming)\s+\w+"
    r"|\b(rate|rates|price|prices|cost|decision|update|release|version)\b"
    r"|\b(what is|what are|who is|who are|how does|how do|how to|why does|why do)\b"
    r"|\b(tell me about|explain|describe|define|meaning of|difference between)\b"
    r"|\b(best|top|recommended|popular|comparison|versus|vs)\b"
    r"|\b(company|product|framework|library|tool|platform|service|app|api)\b"
    r"|\b(country|city|government|policy|regulation|law|standard)\b",
    re.I,
)


_FOLLOW_UP = re.compile(
    r"\b(more|another|also|what about|and|any other|any others|details|details on|elaborate|"
    r"expand|continue|tell me more|what else|go on)\b",
    re.I,
)

_CODE_API_LOOKUP = re.compile(
    r"\b(how (do|to)|what(?:'s| is) the (syntax|api|signature)|is there (a|an)|does \w+ (have|support)|"
    r"which (method|function|class|api|flag))\b"
    r"|\b(docs? for|documentation for|reference for|api reference|man page for)\b"
    r"|\b(asyncio|fastapi|pydantic|sqlalchemy|numpy|pandas|django|flask|react|pytorch|tensorflow|"
    r"httpx|requests|playwright|langchain|openai sdk|anthropic sdk)\b",
    re.I,
)

_CODE_ERROR_SIGNALS = re.compile(
    r"\b(TypeError|ValueError|KeyError|AttributeError|NameError|ImportError|RuntimeError|"
    r"IndexError|AssertionError|ZeroDivisionError|StopIteration|RecursionError|"
    r"ModuleNotFoundError|FileNotFoundError|PermissionError|UnicodeDecodeError)\b"
    r"|\b(ReferenceError|SyntaxError|Uncaught|undefined is not a function|cannot read propert)\b"
    r"|\bstack\s*trace\b|Traceback \(most recent call last\)"
    r"|\bcommand not found\b|permission denied|segmentation fault",
    re.I,
)

_URL_IN_TEXT = re.compile(r"https?://[^\s<>()\[\]{}\"']+", re.I)


def gate_check(
    message: str,
    conversation_context: str = "",
    mode: str = "chat",
) -> set[str]:
    msg = (message or "").strip()
    if not msg:
        return set()

    is_code = (mode or "chat").lower() == "code"

    if _definitely_no_search(msg):
        hints: set[str] = set()
        if _URL_IN_TEXT.search(msg):
            hints.add("url_scraper")
        if _RAG_LOOKUP.search(msg):
            hints.add("rag_lookup")
        if conversation_context and _FOLLOW_UP.search(msg):
            if "[Tool results" in conversation_context or "web_search" in conversation_context.lower():
                hints.add("web_search")
        if is_code:
            if _CODE_API_LOOKUP.search(msg):
                hints.add("web_search")
            if _CODE_ERROR_SIGNALS.search(msg):
                hints.add("web_search")
                hints.add("rag_lookup")
        hints = {h for h in hints if is_feature_enabled(h)}
        if hints:
            _log.info("gate  msg=%s hints=%s (no_search path)", msg[:80], sorted(hints))
        return hints

    hints: set[str] = set()
    if _URL_IN_TEXT.search(msg):
        hints.add("url_scraper")
    if _explicit_search_intent(msg)[0] or _EXTRA_SEARCH.search(msg):
        hints.add("web_search")
    if _RAG_LOOKUP.search(msg):
        hints.add("rag_lookup")

    if is_code:
        if _CODE_API_LOOKUP.search(msg):
            hints.add("web_search")
        if _CODE_ERROR_SIGNALS.search(msg):
            hints.add("web_search")
            hints.add("rag_lookup")

    if "web_search" not in hints and conversation_context and _FOLLOW_UP.search(msg):
        if "[Tool results" in conversation_context or "web_search" in conversation_context.lower():
            hints.add("web_search")

    if "web_search" not in hints and len(msg) > 30:
        hints.add("web_search")

    hints = {h for h in hints if is_feature_enabled(h)}

    if hints:
        _log.info("gate  msg=%s hints=%s", msg[:80], sorted(hints))
    return hints
