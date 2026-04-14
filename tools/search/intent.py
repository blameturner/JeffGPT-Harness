from __future__ import annotations

import json
import logging
import re
import time

from infra.config import is_feature_enabled
from shared.temporal import build_prompt_date_header

_log = logging.getLogger("web_search.intent")


_CHITCHAT_TRIGGER_WORDS = re.compile(
    r"\b(today|tonight|this (?:week|weekend|month|year)|"
    r"right now|currently|latest|recent|news|price|score|weather|"
    r"stock|standings?|results?|fixtures?|schedule|"
    r"search|google|look up|lookup|find|check|who won|"
    r"what happened|what's happening|documentation|docs|"
    r"version|release|changelog|cve|vulnerability)\b",
    re.I,
)
_CHITCHAT_WH_QUESTION = re.compile(
    r"\b(who|what|when|where|which|how many|how much)\b.*\?",
    re.I,
)
_CHITCHAT_CODE_PREFIXES = (
    "def ", "class ", "import ", "from ", "function ", "const ",
    "let ", "var ", "$ ", "# ", "SELECT ", "UPDATE ",
)
_CHITCHAT_OPENER = re.compile(
    r"^\s*(hi|hello|hey|yo|good (?:morning|afternoon|evening)|howdy|"
    r"thanks|thank you|ty|tyvm|please|"
    r"ok|okay|cool|nice|great|awesome|sweet|sure|alright|fine|"
    r"lol|haha|hehe|hmm|mhm|ah|oh|wow|"
    r"yes|yeah|yep|yup|no|nah|nope)\b",
    re.I,
)


def _definitely_chitchat(message: str, history: list[dict] | None) -> bool:
    msg = message.strip()
    if not msg:
        return True
    if len(msg) > 80:
        return False
    if "```" in msg or msg.lstrip().startswith(_CHITCHAT_CODE_PREFIXES):
        return False
    if _CHITCHAT_TRIGGER_WORDS.search(msg):
        return False
    if _CHITCHAT_WH_QUESTION.search(msg):
        return False
    # require a positive opener — otherwise "can you explain..." misroutes as chitchat
    return bool(_CHITCHAT_OPENER.match(msg))


INTENT_ROUTE_CHAT = "chat"
INTENT_ROUTE_CODE = "code"
INTENT_ROUTE_TASK = "task"
INTENT_ROUTES = {INTENT_ROUTE_CHAT, INTENT_ROUTE_CODE, INTENT_ROUTE_TASK}

CHAT_INTENT_CHITCHAT = "chitchat"
CHAT_INTENT_CONTEXTUAL = "contextual_enrichment"
CHAT_INTENT_FACTUAL = "factual_lookup"
CHAT_INTENT_EXPLANATORY = "explanatory"
CHAT_INTENT_RECOMMENDATION = "recommendation"
CHAT_INTENT_COMPARISON = "comparison"
CHAT_INTENT_RESEARCH = "research_synthesis"
CHAT_INTENT_TROUBLESHOOTING = "troubleshooting"

CHAT_INTENTS = {
    CHAT_INTENT_CHITCHAT,
    CHAT_INTENT_CONTEXTUAL,
    CHAT_INTENT_FACTUAL,
    CHAT_INTENT_EXPLANATORY,
    CHAT_INTENT_RECOMMENDATION,
    CHAT_INTENT_COMPARISON,
    CHAT_INTENT_RESEARCH,
    CHAT_INTENT_TROUBLESHOOTING,
}

CODE_INTENT_EXPLAIN = "code_explain"
CODE_INTENT_REVIEW = "code_review"
CODE_INTENT_REFACTOR = "code_refactor"
CODE_INTENT_DEBUG = "code_debug"
CODE_INTENT_BUILD = "code_build"
CODE_INTENT_TEST = "code_test"
CODE_INTENT_OPTIMISE = "code_optimise"
CODE_INTENT_SECURITY = "code_security"
CODE_INTENT_LOOKUP = "code_lookup"

CODE_INTENTS = {
    CODE_INTENT_EXPLAIN,
    CODE_INTENT_REVIEW,
    CODE_INTENT_REFACTOR,
    CODE_INTENT_DEBUG,
    CODE_INTENT_BUILD,
    CODE_INTENT_TEST,
    CODE_INTENT_OPTIMISE,
    CODE_INTENT_SECURITY,
    CODE_INTENT_LOOKUP,
}

TASK_INTENT_REMEMBER = "task_remember"
TASK_INTENT_SCHEDULE = "task_schedule"
TASK_INTENT_SUMMARISE = "task_summarise_input"
TASK_INTENT_SEARCH_EXPLICIT = "task_search_explicit"

TASK_INTENTS = {
    TASK_INTENT_REMEMBER,
    TASK_INTENT_SCHEDULE,
    TASK_INTENT_SUMMARISE,
    TASK_INTENT_SEARCH_EXPLICIT,
}

ALL_INTENTS = CHAT_INTENTS | CODE_INTENTS | TASK_INTENTS

SEARCH_POLICY_NONE = "none"
SEARCH_POLICY_CONTEXTUAL = "contextual"
SEARCH_POLICY_FOCUSED = "focused"
SEARCH_POLICY_FULL = "full"

INTENT_SEARCH_POLICY: dict[str, str] = {
    CHAT_INTENT_CHITCHAT: SEARCH_POLICY_NONE,
    CHAT_INTENT_CONTEXTUAL: SEARCH_POLICY_CONTEXTUAL,
    CHAT_INTENT_FACTUAL: SEARCH_POLICY_FOCUSED,
    CHAT_INTENT_EXPLANATORY: SEARCH_POLICY_FOCUSED,
    CHAT_INTENT_RECOMMENDATION: SEARCH_POLICY_FOCUSED,
    CHAT_INTENT_COMPARISON: SEARCH_POLICY_FULL,
    CHAT_INTENT_RESEARCH: SEARCH_POLICY_FULL,
    CHAT_INTENT_TROUBLESHOOTING: SEARCH_POLICY_FOCUSED,
    CODE_INTENT_EXPLAIN: SEARCH_POLICY_NONE,
    CODE_INTENT_REVIEW: SEARCH_POLICY_NONE,
    CODE_INTENT_REFACTOR: SEARCH_POLICY_NONE,
    CODE_INTENT_DEBUG: SEARCH_POLICY_CONTEXTUAL,
    CODE_INTENT_BUILD: SEARCH_POLICY_CONTEXTUAL,
    CODE_INTENT_TEST: SEARCH_POLICY_NONE,
    CODE_INTENT_OPTIMISE: SEARCH_POLICY_NONE,
    CODE_INTENT_SECURITY: SEARCH_POLICY_NONE,
    CODE_INTENT_LOOKUP: SEARCH_POLICY_FOCUSED,
    TASK_INTENT_REMEMBER: SEARCH_POLICY_NONE,
    TASK_INTENT_SCHEDULE: SEARCH_POLICY_NONE,
    TASK_INTENT_SUMMARISE: SEARCH_POLICY_NONE,
    TASK_INTENT_SEARCH_EXPLICIT: SEARCH_POLICY_FULL,
}

INTENT_RESPONSE_TEMPLATE: dict[str, str] = {
    CHAT_INTENT_CHITCHAT: "chitchat_casual",
    CHAT_INTENT_CONTEXTUAL: "conversational_weave",
    CHAT_INTENT_FACTUAL: "direct_answer",
    CHAT_INTENT_EXPLANATORY: "explanatory",
    CHAT_INTENT_RECOMMENDATION: "recommendation",
    CHAT_INTENT_COMPARISON: "comparison",
    CHAT_INTENT_RESEARCH: "research_synthesis",
    CHAT_INTENT_TROUBLESHOOTING: "troubleshooting",
    CODE_INTENT_EXPLAIN: "code_explain",
    CODE_INTENT_REVIEW: "code_review",
    CODE_INTENT_REFACTOR: "code_refactor",
    CODE_INTENT_DEBUG: "code_debug",
    CODE_INTENT_BUILD: "code_build",
    CODE_INTENT_TEST: "code_test",
    CODE_INTENT_OPTIMISE: "code_optimise",
    CODE_INTENT_SECURITY: "code_security",
    CODE_INTENT_LOOKUP: "code_lookup",
    TASK_INTENT_REMEMBER: "task_confirm",
    TASK_INTENT_SCHEDULE: "task_confirm",
    TASK_INTENT_SUMMARISE: "task_confirm",
    TASK_INTENT_SEARCH_EXPLICIT: "research_synthesis",
}


def _format_history_for_classifier(history: list[dict] | None, limit: int = 3) -> str:
    if not history:
        return "(no prior turns)"
    turns = history[-limit:]
    lines: list[str] = []
    for t in turns:
        role = t.get("role") or "user"
        content = (t.get("content") or "").strip()
        if len(content) > 250:
            content = content[:247] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(no prior turns)"


_INTENT_CLASSIFIER_PROMPT = """{date_header}

You classify a user message in a chat session. Respond with ONE JSON object:
{{
  "route": "chat" | "code" | "task",
  "intent": "<enum value from the list below>",
  "secondary_intent": "<enum value or null>",
  "entities": ["..."],
  "location_hint": "<place name or null>",
  "time_sensitive": true | false,
  "temporal_anchor": "<phrase like 'this season' / 'last week' or null>",
  "confidence": "high" | "medium" | "low"
}}

Chat intents (route=chat):
- chitchat: emotional, opinion, small-talk, no real-world anchors
- contextual_enrichment: STATEMENT containing real-world anchors (team, event, place, product) without an explicit question — the assistant should ground the reply with a few facts
- factual_lookup: specific fact wanted
- explanatory: "how does X work?" / "explain Y"
- recommendation: "where should I..." / "what's a good..."
- comparison: "X vs Y" / "which of these"
- research_synthesis: multi-angle investigation of a non-trivial topic
- troubleshooting: "why is my bread dense?" / non-code diagnosis

Code intents (route=code, fires when message contains code, error messages, or explicitly discusses programming):
- code_explain, code_review, code_refactor, code_debug, code_build, code_test, code_optimise, code_security
- code_lookup: "how do I use X in Y?" / "what's the syntax for Z?" — search-essential API lookup

Task intents (route=task, explicit command):
- task_remember, task_schedule, task_summarise_input, task_search_explicit

Rules:
- Consider the last 3 turns of conversation for context; "make it faster" after "write me a scraper" = code_optimise.
- Extract entities verbatim (proper nouns, named events, branded products, specific people/places).
- time_sensitive=true when the answer would change based on the current date.
- Prefer contextual_enrichment over chitchat whenever the user mentions a real-world thing the assistant could be informed about, even if no question is asked.
- If you're not sure between two intents, pick the more specific one.

Examples:
---
History: (no prior turns)
Message: How's your day going?
{{"route":"chat","intent":"chitchat","secondary_intent":null,"entities":[],"location_hint":null,"time_sensitive":false,"temporal_anchor":null,"confidence":"high"}}
---
History: (no prior turns)
Message: Going to the Sydney Derby tonight. Wanderers have been terrible this season.
{{"route":"chat","intent":"contextual_enrichment","secondary_intent":null,"entities":["Sydney Derby","Western Sydney Wanderers"],"location_hint":"Sydney","time_sensitive":true,"temporal_anchor":"this season","confidence":"high"}}
---
History: (no prior turns)
Message: What is the A-League top scorer right now?
{{"route":"chat","intent":"factual_lookup","secondary_intent":null,"entities":["A-League"],"location_hint":null,"time_sensitive":true,"temporal_anchor":"right now","confidence":"high"}}
---
History: assistant: Here are some fine-dining picks in Sydney: Nour, Bentley, Firedoor.
Message: Analyse the menu for the recommended Nour
{{"route":"chat","intent":"factual_lookup","secondary_intent":"recommendation","entities":["Nour"],"location_hint":"Sydney","time_sensitive":false,"temporal_anchor":null,"confidence":"high"}}
---
History: (no prior turns)
Message: Write me a Python function that scrapes the live price of AAPL
{{"route":"code","intent":"code_build","secondary_intent":"factual_lookup","entities":["AAPL","Python"],"location_hint":null,"time_sensitive":false,"temporal_anchor":null,"confidence":"high"}}
---
History: (no prior turns)
Message: def foo(x): return x*2 — why does this return None when I call it with None?
{{"route":"code","intent":"code_debug","secondary_intent":null,"entities":["Python"],"location_hint":null,"time_sensitive":false,"temporal_anchor":null,"confidence":"high"}}
---
History: (no prior turns)
Message: How do I use asyncio.gather with a semaphore in Python?
{{"route":"code","intent":"code_lookup","secondary_intent":null,"entities":["asyncio","asyncio.gather","semaphore","Python"],"location_hint":null,"time_sensitive":false,"temporal_anchor":null,"confidence":"high"}}
---
History: (no prior turns)
Message: Remember that I prefer tabs over spaces
{{"route":"task","intent":"task_remember","secondary_intent":null,"entities":[],"location_hint":null,"time_sensitive":false,"temporal_anchor":null,"confidence":"high"}}
---

Recent history:
{history}

Current message:
{message}

Answer with ONE JSON object only, no prose.
Answer:"""


def _fallback_intent() -> dict:
    return {
        "route": INTENT_ROUTE_CHAT,
        "intent": CHAT_INTENT_CHITCHAT,
        "secondary_intent": None,
        "entities": [],
        "location_hint": None,
        "time_sensitive": False,
        "temporal_anchor": None,
        "confidence": "low",
        "search_policy": SEARCH_POLICY_NONE,
        "response_template": INTENT_RESPONSE_TEMPLATE[CHAT_INTENT_CHITCHAT],
        "classifier_raw": None,
    }


def _derive_policy(intent_dict: dict) -> dict:
    intent = intent_dict.get("intent") or CHAT_INTENT_CHITCHAT
    intent_dict["search_policy"] = INTENT_SEARCH_POLICY.get(intent, SEARCH_POLICY_NONE)
    intent_dict["response_template"] = INTENT_RESPONSE_TEMPLATE.get(
        intent, INTENT_RESPONSE_TEMPLATE[CHAT_INTENT_CHITCHAT]
    )
    return intent_dict


def classify_message_intent(
    message: str,
    history: list[dict] | None = None,
) -> dict:
    msg = (message or "").strip()
    if not msg:
        return _fallback_intent()

    if _definitely_chitchat(msg, history):
        _log.info("intent classifier: heuristic skip (definite chitchat)")
        result = _fallback_intent()
        result["confidence"] = "high"
        result["classifier_raw"] = "heuristic_skip"
        return result

    if not is_feature_enabled("intent_classifier"):
        _log.info("intent classifier disabled, using heuristic fallback")
        return _fallback_intent()

    from infra.config import get_function_config
    cfg = get_function_config("intent_classifier")

    prompt = _INTENT_CLASSIFIER_PROMPT.format(
        date_header=build_prompt_date_header(),
        history=_format_history_for_classifier(history),
        message=msg[:cfg.get("max_input_chars", 2000)],
    )

    started = time.time()
    try:
        from shared.models import model_call
        raw, _tokens = model_call("intent_classifier", prompt)
        if not raw:
            _log.warning("intent classifier: empty response, using fallback")
            return _fallback_intent()
    except Exception as e:
        _log.warning("intent classifier failed: %s", e)
        return _fallback_intent()

    elapsed = round(time.time() - started, 2)

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = cleaned.rstrip("`").strip()
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if not json_match:
        _log.warning("intent classifier: no JSON in response: %s", raw[:200])
        result = _fallback_intent()
        result["classifier_raw"] = raw[:500]
        return result

    try:
        parsed = json.loads(json_match.group(0))
    except json.JSONDecodeError as e:
        _log.warning("intent classifier: JSON parse failed (%s): %s", e, raw[:200])
        result = _fallback_intent()
        result["classifier_raw"] = raw[:500]
        return result

    route = parsed.get("route")
    if route not in INTENT_ROUTES:
        route = INTENT_ROUTE_CHAT

    intent = parsed.get("intent")
    if intent not in ALL_INTENTS:
        _log.debug("intent classifier: unknown intent %r, defaulting", intent)
        intent = CHAT_INTENT_CHITCHAT if route == INTENT_ROUTE_CHAT else None

    if route == INTENT_ROUTE_CHAT and intent not in CHAT_INTENTS:
        intent = CHAT_INTENT_CHITCHAT
    elif route == INTENT_ROUTE_CODE and intent not in CODE_INTENTS:
        intent = CODE_INTENT_EXPLAIN
    elif route == INTENT_ROUTE_TASK and intent not in TASK_INTENTS:
        intent = TASK_INTENT_REMEMBER

    secondary = parsed.get("secondary_intent")
    if secondary not in ALL_INTENTS:
        secondary = None

    entities = parsed.get("entities") or []
    if not isinstance(entities, list):
        entities = []
    entities = [str(e).strip() for e in entities if str(e).strip()][:15]

    location_hint = parsed.get("location_hint")
    if location_hint is not None:
        location_hint = str(location_hint).strip()[:80] or None

    time_sensitive = bool(parsed.get("time_sensitive", False))

    temporal_anchor = parsed.get("temporal_anchor")
    if temporal_anchor is not None:
        temporal_anchor = str(temporal_anchor).strip()[:80] or None

    confidence = parsed.get("confidence", "medium")
    if confidence not in ("high", "medium", "low"):
        confidence = "medium"

    result = {
        "route": route,
        "intent": intent,
        "secondary_intent": secondary,
        "entities": entities,
        "location_hint": location_hint,
        "time_sensitive": time_sensitive,
        "temporal_anchor": temporal_anchor,
        "confidence": confidence,
        "classifier_raw": None,
    }
    _derive_policy(result)

    _log.info(
        "intent classified  route=%s intent=%s secondary=%s entities=%d "
        "time_sensitive=%s policy=%s confidence=%s %.2fs",
        result["route"], result["intent"], result["secondary_intent"],
        len(result["entities"]), result["time_sensitive"],
        result["search_policy"], result["confidence"], elapsed,
    )
    return result
