from __future__ import annotations

import json
import logging
import re

from workers.enrichment.models import model_call

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


def _definitely_no_search(message: str) -> bool:
    msg = message.strip()
    if not msg:
        return True
    if len(msg) > 400:
        return False
    if _NO_SEARCH_TRIGGER_WORDS.search(msg):
        return False
    if _EXTERNAL_QUESTION.search(msg):
        return False
    if "```" in msg or msg.lstrip().startswith(_CODE_PREFIXES):
        return True
    if len(msg) <= 200:
        return True
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
        "Lean towards YES — it's better to search and find nothing than to miss "
        "useful information. Return ONLY a JSON object:\n"
        '{"needs_search": true|false, "confidence": "high"|"medium"|"low", '
        '"reason": "<15 words or fewer>"}\n\n'
        "needs_search=true (confidence high) when:\n"
        "- Current events, scores, prices, weather, news, schedules\n"
        "- References to 'today', 'this week', 'latest', 'recent', 'current'\n"
        "- Specific products, frameworks, tools, or technologies\n"
        "- Company or person's recent activity\n"
        "- Fact-checking or verification requests\n"
        "- Requests for documentation, resources, or guides\n\n"
        "needs_search=true (confidence medium) when:\n"
        "- The topic might have recent developments\n"
        "- Unclear whether the user needs current vs general info\n"
        "- A web search could add value but isn't essential\n\n"
        "needs_search=false when:\n"
        "- Clearly asking for help writing code, prose, or analysis\n"
        "- Casual conversation or follow-up within an ongoing thread\n"
        "- Abstract or philosophical questions\n\n"
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
