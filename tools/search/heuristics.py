from __future__ import annotations

import json
import logging
import re

from shared.models import model_call

_log = logging.getLogger("web_search.heuristics")


_EXPLICIT_SEARCH_PATTERNS = [
    re.compile(r"\b(?:search|look up|google|find|lookup)\b.*\b(?:for|about|this|that|it|me)\b", re.I),
    re.compile(r"\b(?:search|look up|google|find|lookup)\s+(?:for\s+)?(?:me\s+)?[\"'].+[\"']", re.I),
    re.compile(r"\bwhat(?:'s| is| are)\b.*\b(?:today|tonight|this week|this weekend|right now|currently|latest|recent)\b", re.I),
    re.compile(r"\b(?:current|latest|recent|live|today'?s?|tonight'?s?)\b.*\b(?:price|score|weather|news|results?|standings?|fixtures?|schedule|status)\b", re.I),
    re.compile(r"\b(?:who won|who is winning|what happened|what's happening)\b", re.I),
    re.compile(r"\bcan you (?:search|look|find|check)\b", re.I),
    re.compile(r"\b(?:give me|get me|pull up|show me)\b.*\b(?:info|information|details|data|docs|documentation)\b", re.I),
    re.compile(r"\b(?:best|top|recommended)\b.*\b(?:resources?|sources?|articles?|guides?|tutorials?|docs)\b", re.I),
]


def _explicit_search_intent(message: str) -> tuple[bool, str]:
    for pat in _EXPLICIT_SEARCH_PATTERNS:
        if pat.search(message):
            return True, "explicit search request"
    return False, ""


_NO_SEARCH_TRIGGER_WORDS = re.compile(
    r"\b(today|tonight|this (?:week|weekend|month|year)|"
    r"right now|currently|latest|recent|news|price|score|"
    r"weather|stock|standings?|results?|fixtures?|schedule|"
    r"search|google|look up|lookup|find|check|who won|"
    r"what happened|what's happening|documentation|docs|"
    r"version|release|changelog|cve|vulnerability)\b",
    re.I,
)
_EXTERNAL_QUESTION = re.compile(
    r"\b(who|what|when|where|which|how many|how much)\b.*\?",
    re.I,
)
_CODE_PREFIXES = (
    "def ", "class ", "import ", "from ", "function ", "const ",
    "let ", "var ", "$ ", "# ", "SELECT ", "UPDATE ",
)


_DEFINITE_CHITCHAT = re.compile(
    r"^(?:hi|hey|hello|thanks|thank you|ok|okay|sure|yes|no|yep|nope|cool|"
    r"great|good|nice|cheers|bye|goodbye|lol|haha|hmm|ah|oh|wow|please|"
    r"sorry|np|ty|thx|k|got it|understood|ack|roger|noted|will do|on it)[\s!?.]*$",
    re.I,
)


def _definitely_no_search(message: str) -> bool:
    """Return True only for messages that DEFINITELY don't need search.

    Intentionally narrow — we'd rather search unnecessarily than miss
    relevant information.  Only blocks:
    - Empty messages
    - Code blocks / snippets
    - Very short chitchat (greetings, thanks, single-word acks)
    """
    msg = message.strip()
    if not msg:
        return True
    # Code blocks should not trigger search.
    if "```" in msg or msg.lstrip().startswith(_CODE_PREFIXES):
        return True
    # Definite chitchat — single-word/phrase acks, greetings, thanks.
    if len(msg) <= 80 and _DEFINITE_CHITCHAT.search(msg):
        return True
    # Everything else gets a chance at search.
    return False


def needs_web_search(message: str) -> tuple[bool, str, str]:
    msg = (message or "").strip()
    if not msg:
        return False, "", ""

    explicit, reason = _explicit_search_intent(msg)
    if explicit:
        _log.info("search intent detected (pattern): %s", reason)
        return True, reason, "high"

    if _definitely_no_search(msg):
        _log.info("search skip (heuristic): short conversational message")
        return False, "", ""

    prompt = (
        "Decide whether answering this user message would benefit from a web search. "
        "DEFAULT TO YES. It is almost always better to search and bring relevant "
        "context than to answer without it. Return ONLY a JSON object:\n"
        '{"needs_search": true|false, "confidence": "high"|"medium"|"low", '
        '"reason": "<15 words or fewer>"}\n\n'
        "needs_search=true (confidence high) when:\n"
        "- Current events, scores, prices, weather, news, schedules\n"
        "- References to 'today', 'this week', 'latest', 'recent', 'current'\n"
        "- Specific products, frameworks, tools, or technologies\n"
        "- Company or person's recent activity\n"
        "- Fact-checking or verification requests\n"
        "- Requests for documentation, resources, or guides\n"
        "- Any named entity (person, company, place, product, concept)\n"
        "- Explanations of real-world topics, processes, or systems\n\n"
        "needs_search=true (confidence medium) when:\n"
        "- The topic might have recent developments\n"
        "- Background context could improve the answer quality\n"
        "- Unclear whether the user needs current vs general info\n"
        "- A web search could add value even if not strictly necessary\n\n"
        "needs_search=false ONLY when:\n"
        "- Purely asking to write/edit code with no factual lookup needed\n"
        "- Casual chitchat with no information need\n"
        "- Continuing a task (e.g. 'make it shorter', 'try again')\n\n"
        f"Message: {msg[:500]}"
    )
    try:
        _log.info("heuristic classify start")
        raw, _tokens = model_call("tool_planner", prompt, max_tokens=80, temperature=0.0)
        if not raw:
            return False, "", ""
        match = re.search(r"\{.*?\}", raw, re.S)
        if not match:
            return False, "", ""
        data = json.loads(match.group(0))
        needs = bool(data.get("needs_search"))
        reason = str(data.get("reason") or "").strip()[:120]
        confidence = str(data.get("confidence") or "medium").lower()
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"
        if needs:
            _log.info("search intent detected (classifier): confidence=%s reason=%s", confidence, reason)
        return needs, reason, confidence if needs else ""
    except Exception as e:
        _log.warning("needs_web_search classifier failed: %s", e)
        return False, "", ""
