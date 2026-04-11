from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from concurrent.futures import Future
from datetime import datetime
from typing import Iterable
from zoneinfo import ZoneInfo

from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from config import (
    CATEGORY_COLLECTIONS,
    CHAT_TIMEZONE,
    MAX_SUMMARY_INPUT_CHARS,
    MODELS,
    SEARXNG_URL,
)
from memory import remember

_log = logging.getLogger("web_search")


# --- Temporal context ------------------------------------------------------
#
# The base chat system prompt has no date. Without this, the model reasons
# from its training cutoff and claims things like "the 2025-26 season
# hasn't happened yet" when asked in April 2026. Every chat turn gets a
# fresh current-date system message built from this helper, and every
# classifier/query-gen/extraction prompt that cares about "today" /
# "this season" / "recent" / "latest" prepends a short date header.

def now_in_chat_tz() -> datetime:
    """Return wall-clock datetime in the configured chat timezone."""
    try:
        return datetime.now(ZoneInfo(CHAT_TIMEZONE))
    except Exception:
        # Invalid timezone string — fall back to UTC rather than crash.
        _log.warning("invalid CHAT_TIMEZONE=%r, falling back to UTC", CHAT_TIMEZONE)
        from datetime import timezone
        return datetime.now(timezone.utc)


def build_temporal_context(now: datetime | None = None) -> str:
    """Build the 'current date' system message injected into every chat turn.

    Example output::

        Current date and time: Saturday, 11 April 2026, 14:32 AEST.
        ISO: 2026-04-11T14:32:17+10:00.
        When the user says 'today', 'this week', 'this season', 'recent',
        or 'latest', resolve it relative to this date. Do NOT claim
        something is in the future if the date above shows it is in the
        past or present.
    """
    now = now or now_in_chat_tz()
    human = now.strftime("%A, %d %B %Y, %H:%M %Z").strip()
    return (
        f"Current date and time: {human}.\n"
        f"ISO: {now.isoformat()}.\n"
        "When the user says 'today', 'this week', 'this season', 'recent', "
        "or 'latest', resolve it relative to this date. Do NOT claim "
        "something is in the future if the date above shows it is in the "
        "past or present."
    )


def build_prompt_date_header(now: datetime | None = None) -> str:
    """Short one-line date header prepended to classifier/query-gen prompts.

    Keeps the model grounded without bloating prompts with the full
    temporal context used in the payload's system message. Returned
    without a trailing newline so callers can compose naturally.
    """
    now = now or now_in_chat_tz()
    return f"Today is {now.strftime('%A, %-d %B %Y')} ({now.strftime('%Y-%m-%d')})."

SCRAPE_BLOCKLIST = {
    "reddit.com",
    "news.com.au",
    "medium.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "facebook.com",
    "instagram.com",
    "nytimes.com",
    "wsj.com",
    "ft.com",
}


def _is_blocklisted(url: str) -> bool:
    try:
        host = (urlparse(url).netloc or "").lower()
    except Exception:
        return False
    if host.startswith("www."):
        host = host[4:]
    return any(host == d or host.endswith("." + d) for d in SCRAPE_BLOCKLIST)

MAX_SOURCES = 8
OVERFETCH_FACTOR = 3
PER_PAGE_CHAR_CAP = 20_000
SUMMARY_MAX_TOKENS = 500
SUMMARY_INPUT_CHAR_CAP = MAX_SUMMARY_INPUT_CHARS
SUMMARY_BATCH_SIZE = 3
SEARXNG_TIMEOUT = 10
SCRAPE_TIMEOUT = 15
FAST_TIMEOUT = 60

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
BROWSER_HEADERS = {
    "User-Agent": BROWSER_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

_INJECTION_PATTERNS = [
    re.compile(r"<\s*/?\s*(?:system|assistant|user|s|\|im_start\||\|im_end\|)[^>]*>", re.I),
    re.compile(r"ignore (?:all )?(?:previous|prior|above|earlier) (?:instructions|messages|context|prompts?)", re.I),
    re.compile(r"disregard (?:all )?(?:prior|previous|above|earlier) (?:instructions|messages|context|prompts?)", re.I),
    re.compile(r"(?:you are|act as|pretend (?:you are|to be)|roleplay as|behave as) (?:a |an )?[a-z ]{3,30}", re.I),
    re.compile(r"new (?:instructions?|rules?|persona|role):", re.I),
    re.compile(r"(?:system|admin|root|developer) (?:prompt|message|override|mode):", re.I),
    re.compile(r"(?:forget|override|bypass|skip) (?:all |your )?(?:previous |prior )?(?:instructions|rules|guidelines|safety)", re.I),
    re.compile(r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>", re.I),
    re.compile(r"human:|assistant:|###\s*(?:system|instruction|human|assistant)", re.I),
    re.compile(r"(?:do not|don'?t) (?:follow|obey|listen to) (?:your |the )?(?:previous|original|system)", re.I),
    re.compile(r"(?:reveal|show|print|output|repeat) (?:your |the )?(?:system ?prompt|instructions|rules)", re.I),
]


def _strip_injection_patterns(text: str) -> str:
    for pat in _INJECTION_PATTERNS:
        text = pat.sub("[redacted]", text)
    return text


def _is_safe_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if host in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]"):
        return False
    if host.startswith("10.") or host.startswith("192.168.") or host.startswith("169.254."):
        return False
    if host.startswith("172."):
        # RFC 1918: 172.16.0.0 – 172.31.255.255
        try:
            second_octet = int(host.split(".")[1])
            if 16 <= second_octet <= 31:
                return False
        except (IndexError, ValueError):
            return False
    if host.endswith(".local") or host.endswith(".internal"):
        return False
    return True


# Enrichment / crawl helpers must NEVER resolve to the reasoner model. These
# resolvers fall back only between tool <-> fast, and return (None, None) if
# neither is available so callers fail loudly rather than silently blasting
# the reasoner.
_ENRICHMENT_SAFE_ROLES = ("tool", "fast")


def _resolve_safe_model(preferred_role: str) -> tuple[str | None, str | None]:
    """Return ``(url, model_id)`` for ``preferred_role``, never the reasoner.

    Resolution order:

    1. Look for the exact role name (``tool`` or ``fast``)
    2. Fall back to the other role in the pair
    3. Fall back to any catalog entry whose role is NOT ``reasoner`` and
       whose URL is not the reasoner's URL. This keeps port-scan setups
       (where roles are derived from model IDs like ``Qwen2.5-3B-Instruct``)
       working while still refusing to route enrichment traffic to a
       reasoner.
    4. Return ``(None, None)`` if nothing safe is available.
    """
    assert preferred_role in _ENRICHMENT_SAFE_ROLES, f"unsafe role {preferred_role!r}"

    # Phase 1/2 — prefer named tool/fast roles
    chain = [preferred_role] + [r for r in _ENRICHMENT_SAFE_ROLES if r != preferred_role]
    for role in chain:
        entry = MODELS.get(role)
        if isinstance(entry, dict) and entry.get("url"):
            if role != preferred_role:
                _log.debug("no '%s' role in catalog, falling back to %s",
                           preferred_role, role)
            return entry.get("url"), entry.get("model_id") or role

    # Phase 3 — fall back to any non-reasoner entry. Dedup by URL so an
    # entry registered under both its role and its model_id doesn't get
    # iterated twice.
    reasoner_entry = MODELS.get("reasoner")
    reasoner_url = reasoner_entry.get("url") if isinstance(reasoner_entry, dict) else None
    seen_urls: set[str] = set()
    for v in MODELS.values():
        if not isinstance(v, dict):
            continue
        url = v.get("url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        if v.get("role") == "reasoner":
            continue
        if reasoner_url and url == reasoner_url:
            continue
        _log.debug(
            "no named tool/fast role for %s, falling back to %s",
            preferred_role, v.get("role"),
        )
        return url, v.get("model_id") or v.get("role")

    _log.error(
        "no safe enrichment model available  preferred=%s catalog_roles=%s",
        preferred_role,
        sorted({v.get("role") for v in MODELS.values() if isinstance(v, dict)}),
    )
    return None, None


def _fast_model() -> tuple[str | None, str | None]:
    return _resolve_safe_model("fast")


def _tool_model() -> tuple[str | None, str | None]:
    return _resolve_safe_model("tool")


# --- Intent taxonomy + classifier -----------------------------------------
#
# Every chat turn passes through classify_message_intent() before any
# search fires. The classifier returns a structured intent dict — route,
# sub-intent, entities, time_sensitive — and Python code maps that to
# search budget, response template, and style. Policy lives here in the
# lookup tables, not inside the LLM prompt, so it's reviewable.
#
# See the plan file for the full taxonomy rationale.

# Route enum — which agent handles the turn
INTENT_ROUTE_CHAT = "chat"
INTENT_ROUTE_CODE = "code"
INTENT_ROUTE_TASK = "task"
INTENT_ROUTES = {INTENT_ROUTE_CHAT, INTENT_ROUTE_CODE, INTENT_ROUTE_TASK}

# Chat sub-intents (route=chat)
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

# Code sub-intents (route=code) — aligned with workers/styles.py CODE_STYLES
CODE_INTENT_EXPLAIN = "code_explain"
CODE_INTENT_REVIEW = "code_review"
CODE_INTENT_REFACTOR = "code_refactor"
CODE_INTENT_DEBUG = "code_debug"
CODE_INTENT_BUILD = "code_build"
CODE_INTENT_TEST = "code_test"
CODE_INTENT_OPTIMISE = "code_optimise"
CODE_INTENT_SECURITY = "code_security"
CODE_INTENT_LOOKUP = "code_lookup"  # new — search-forward API/syntax lookup

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

# Task sub-intents (route=task)
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

# Search policy per intent — drives budget in run_web_search.
#
#   none       no search at all
#   contextual 1 query, 3 pages, 100-word extracts, 5s hard cap
#   focused    1-2 queries, 5 pages, standard extraction
#   full       2-3 queries, 8 pages, rerank + extraction pass
SEARCH_POLICY_NONE = "none"
SEARCH_POLICY_CONTEXTUAL = "contextual"
SEARCH_POLICY_FOCUSED = "focused"
SEARCH_POLICY_FULL = "full"

INTENT_SEARCH_POLICY: dict[str, str] = {
    # chat
    CHAT_INTENT_CHITCHAT: SEARCH_POLICY_NONE,
    CHAT_INTENT_CONTEXTUAL: SEARCH_POLICY_CONTEXTUAL,
    CHAT_INTENT_FACTUAL: SEARCH_POLICY_FOCUSED,
    CHAT_INTENT_EXPLANATORY: SEARCH_POLICY_FOCUSED,
    CHAT_INTENT_RECOMMENDATION: SEARCH_POLICY_FOCUSED,
    CHAT_INTENT_COMPARISON: SEARCH_POLICY_FULL,
    CHAT_INTENT_RESEARCH: SEARCH_POLICY_FULL,
    CHAT_INTENT_TROUBLESHOOTING: SEARCH_POLICY_FOCUSED,
    # code
    CODE_INTENT_EXPLAIN: SEARCH_POLICY_NONE,
    CODE_INTENT_REVIEW: SEARCH_POLICY_NONE,
    CODE_INTENT_REFACTOR: SEARCH_POLICY_NONE,
    CODE_INTENT_DEBUG: SEARCH_POLICY_CONTEXTUAL,
    CODE_INTENT_BUILD: SEARCH_POLICY_CONTEXTUAL,
    CODE_INTENT_TEST: SEARCH_POLICY_NONE,
    CODE_INTENT_OPTIMISE: SEARCH_POLICY_NONE,
    CODE_INTENT_SECURITY: SEARCH_POLICY_NONE,
    CODE_INTENT_LOOKUP: SEARCH_POLICY_FOCUSED,
    # task
    TASK_INTENT_REMEMBER: SEARCH_POLICY_NONE,
    TASK_INTENT_SCHEDULE: SEARCH_POLICY_NONE,
    TASK_INTENT_SUMMARISE: SEARCH_POLICY_NONE,
    TASK_INTENT_SEARCH_EXPLICIT: SEARCH_POLICY_FULL,
}

# Response-template key per intent — consumed in §5 to pick the
# search_context block. Keys will be added to workers.styles in §5.
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
    """Format the last N turns as a short transcript for the classifier.

    Each turn is trimmed to ~250 chars so the whole block stays small.
    """
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
    """Safe default used on classifier failure — pure chitchat, no search."""
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
    """Attach search_policy and response_template to an intent dict in place."""
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
    """Classify a chat message into a structured intent dict.

    Single tool-model call, conversation-aware, returns the shape
    documented in the plan §2. Fails closed to chitchat (safe default:
    no search, natural conversational reply).
    """
    msg = (message or "").strip()
    if not msg:
        return _fallback_intent()

    tool_url, tool_model = _tool_model()
    if not tool_url:
        _log.warning("intent classifier: no tool model available, using fallback")
        return _fallback_intent()

    prompt = _INTENT_CLASSIFIER_PROMPT.format(
        date_header=build_prompt_date_header(),
        history=_format_history_for_classifier(history),
        message=msg[:1500],
    )

    started = time.time()
    try:
        resp = httpx.post(
            f"{tool_url}/v1/chat/completions",
            json={
                "model": tool_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 200,
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        _log.warning("intent classifier call failed: %s", e)
        return _fallback_intent()

    elapsed = round(time.time() - started, 2)

    # Strip markdown fences if present, then pull out the JSON object.
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

    # Validate and normalise each field. Any invalid field → substitute
    # the safe default rather than trusting the model.
    route = parsed.get("route")
    if route not in INTENT_ROUTES:
        route = INTENT_ROUTE_CHAT

    intent = parsed.get("intent")
    if intent not in ALL_INTENTS:
        _log.debug("intent classifier: unknown intent %r, defaulting", intent)
        intent = CHAT_INTENT_CHITCHAT if route == INTENT_ROUTE_CHAT else None

    # Cross-check route + intent consistency: intent must belong to its route's enum.
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
        "classifier_raw": None,  # populated only on fallback for debugging
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


def searxng_search(query: str, max_results: int = MAX_SOURCES) -> list[dict]:
    search_url = f"{SEARXNG_URL}/search"
    _log.debug("searxng request  url=%s query=%s", search_url, query[:120])
    try:
        resp = httpx.get(
            search_url,
            params={"q": query, "format": "json"},
            timeout=SEARXNG_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        _log.error("searxng %d for '%s': %s", e.response.status_code, query, e.response.text[:300])
        return []
    except httpx.TimeoutException:
        _log.error("searxng timeout after %ds for '%s'", SEARXNG_TIMEOUT, query)
        return []
    except Exception:
        _log.error("searxng failed for '%s'", query, exc_info=True)
        return []

    results = data.get("results") or []
    out: list[dict] = []
    for r in results[: max_results * 2]:
        url = r.get("url")
        if not url:
            continue
        out.append({
            "title": (r.get("title") or "").strip()[:200],
            "url": url,
            "snippet": (r.get("content") or "").strip(),
        })
    _log.debug("searxng returned %d raw results, kept %d", len(results), len(out))
    return out


def _quote_entity(entity: str) -> str:
    """Wrap an entity in double quotes so SearxNG phrase-matches it.

    Multi-word entities need the quotes or they get tokenised and each
    word is searched independently — which is how we ended up with
    "Western Sydney Wanderers" losing to "2 on the go" in the earlier
    failure case. Single-word entities don't need quoting but we do it
    anyway for consistency.
    """
    e = entity.strip().strip('"\'')
    if not e:
        return ""
    return f'"{e}"'


def _current_year() -> int:
    return now_in_chat_tz().year


def generate_search_queries(intent_dict: dict, message: str | None = None) -> list[str]:
    """Build 1-3 search queries from an intent dict.

    Takes the classifier's output (§2) and produces template-based queries
    anchored on the extracted entities, not a freehand LLM interpretation
    of the raw user message. This is the fix for the "on the go" failure:
    the classifier identifies "Western Sydney Wanderers" as the entity,
    this function quotes it, and SearxNG phrase-matches on the proper
    noun instead of the incidental idiom.

    ``message`` is kept as a last-resort fallback when the intent dict
    has no entities (rare, mostly chitchat).
    """
    intent = intent_dict.get("intent") or CHAT_INTENT_CHITCHAT
    entities = intent_dict.get("entities") or []
    location = (intent_dict.get("location_hint") or "").strip()
    temporal = (intent_dict.get("temporal_anchor") or "").strip()
    time_sensitive = bool(intent_dict.get("time_sensitive"))
    year = _current_year()

    # Helper: quote the primary entity (first non-empty entity)
    quoted = [_quote_entity(e) for e in entities if _quote_entity(e)]
    primary = quoted[0] if quoted else ""
    all_quoted = " ".join(quoted[:3])

    def _time_suffix() -> str:
        if temporal:
            # Keep the user's temporal phrase + the year, so
            # "this season" + 2026 → the classifier's anchor plus
            # an absolute year the search engine can match against.
            return f"{temporal} {year}"
        if time_sensitive:
            return str(year)
        return ""

    queries: list[str] = []

    if intent == CHAT_INTENT_CONTEXTUAL and primary:
        # Tight single-query budget for contextual enrichment.
        q = primary
        if location:
            q += f" {location}"
        suffix = _time_suffix()
        if suffix:
            q += f" {suffix}"
        else:
            q += " current"
        queries = [q.strip()]

    elif intent == CHAT_INTENT_FACTUAL and quoted:
        base = all_quoted
        if location:
            base += f" {location}"
        suffix = _time_suffix()
        queries = [
            f"{base} {suffix}".strip() if suffix else base.strip(),
            f"{base} latest",
        ]

    elif intent == CHAT_INTENT_RECOMMENDATION:
        # Recommendations can fire even with no named entities — the
        # user's own phrasing becomes the query noun (e.g. "fine dining").
        noun = (message or "").strip()
        if quoted:
            base = all_quoted
            if location:
                base += f" {location}"
            queries = [f"{base} reviews", f"best {base}"]
        elif noun:
            loc = f" {location}" if location else ""
            queries = [
                f"best {noun}{loc}",
                f"{noun}{loc} reviews",
            ]

    elif intent == CHAT_INTENT_COMPARISON and len(quoted) >= 2:
        a, b = quoted[0], quoted[1]
        queries = [
            f"{a} vs {b}",
            f"{a} comparison",
            f"{b} comparison",
        ]

    elif intent == CHAT_INTENT_RESEARCH and quoted:
        base = all_quoted
        queries = [
            f"{base} research {year}" if time_sensitive else f"{base} research",
            f"{base} review article",
            f"{base} criticism",
        ]

    elif intent == CHAT_INTENT_EXPLANATORY and quoted:
        base = all_quoted
        queries = [
            f"how does {base} work",
            f"{base} explained",
        ]

    elif intent == CHAT_INTENT_TROUBLESHOOTING:
        noun = (message or "").strip()[:120]
        if quoted:
            queries = [f"{all_quoted} troubleshooting", f"{all_quoted} problem fix"]
        elif noun:
            queries = [f"{noun} troubleshooting", f"why {noun}"]

    elif intent == CODE_INTENT_LOOKUP and quoted:
        queries = [
            f"{all_quoted} documentation",
            f"{all_quoted} example",
        ]

    elif intent == CODE_INTENT_DEBUG and quoted:
        queries = [
            f"{quoted[0]} error",
            f"{quoted[0]} fix",
        ]

    elif intent == CODE_INTENT_BUILD and quoted:
        queries = [f"{all_quoted} example code"]

    elif intent == TASK_INTENT_SEARCH_EXPLICIT:
        # User explicitly asked for a search — trust the message phrasing
        # more, but still quote entities if present.
        noun = (message or "").strip()[:200]
        if quoted:
            queries = [f"{all_quoted} {noun}".strip()]
        elif noun:
            queries = [noun]

    # Final fallback: if nothing got generated (classifier failure mode,
    # or an intent with no entities), use the raw message lightly cleaned.
    if not queries and message:
        cleaned = re.sub(r"\s+", " ", message.strip())[:200]
        if cleaned:
            queries = [cleaned]

    # Dedupe while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            unique.append(q)

    _log.info(
        "queries generated  intent=%s entities=%d queries=%s",
        intent, len(entities), unique[:3],
    )
    return unique[:3]


def rerank_candidates(
    candidates: list[dict],
    intent_dict: dict,
    max_candidates: int = 15,
) -> list[tuple[dict, int]]:
    """Pre-scrape rerank: score SearxNG results 1-5 on title+snippet only.

    SearxNG ranks by keyword overlap, which is how "Analyse Data in Excel"
    ended up as the top result for "analyse the menu for Nour". One cheap
    tool-model call scores the top candidates before we spend any scrape
    budget on them. Anything scoring 1 is guaranteed garbage and gets
    dropped.

    Returns ``[(candidate, score), ...]`` sorted by score descending.
    Preserves SearxNG order as a stable secondary sort so ties break
    sensibly.
    """
    if not candidates:
        return []

    tool_url, tool_model = _tool_model()
    if not tool_url:
        # No model available — pass candidates through with neutral score.
        return [(c, 3) for c in candidates]

    head = candidates[:max_candidates]
    intent_label = intent_dict.get("intent") or "chitchat"
    entities = intent_dict.get("entities") or []
    entities_str = ", ".join(entities[:5]) if entities else "(none extracted)"

    listing = []
    for i, c in enumerate(head, start=1):
        title = (c.get("title") or c.get("url") or "").strip()[:120]
        url = (c.get("url") or "").strip()[:160]
        snippet = re.sub(r"\s+", " ", (c.get("snippet") or "").strip())[:200]
        listing.append(f"{i}. {title}\n   {url}\n   {snippet}")

    prompt = (
        f"{build_prompt_date_header()}\n\n"
        f"The user's intent is: {intent_label}\n"
        f"They're asking about: {entities_str}\n\n"
        "Rate each candidate 1-5 for how likely it is to actually contain "
        "what the user wants. Score 1 if the candidate matches a phrase "
        "from the query but is clearly about a different topic (e.g. an "
        "Excel tutorial for 'analyse the menu', or a dictionary entry "
        "for a common idiom). Score 5 only if the candidate is clearly "
        "on-topic and about the named entities.\n\n"
        "Return ONLY a JSON array of integers, one per candidate in the "
        "same order. Example: [5,4,1,3,2,1,1,4]\n\n"
        f"Candidates:\n" + "\n".join(listing) + "\n\nScores:"
    )

    started = time.time()
    try:
        resp = httpx.post(
            f"{tool_url}/v1/chat/completions",
            json={
                "model": tool_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 80,
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        _log.warning("rerank call failed: %s — passing candidates through", e)
        return [(c, 3) for c in candidates]

    elapsed = round(time.time() - started, 2)

    # Extract the JSON array of scores
    match = re.search(r"\[[\s\S]*?\]", raw)
    if not match:
        _log.warning("rerank: no JSON array in response: %s", raw[:200])
        return [(c, 3) for c in candidates]
    try:
        scores = json.loads(match.group(0))
    except json.JSONDecodeError:
        _log.warning("rerank: JSON parse failed: %s", raw[:200])
        return [(c, 3) for c in candidates]
    if not isinstance(scores, list):
        return [(c, 3) for c in candidates]

    # Pad/truncate scores to match candidate count, clamp to 1..5
    normalised: list[int] = []
    for i in range(len(head)):
        if i < len(scores):
            try:
                s = int(scores[i])
            except (TypeError, ValueError):
                s = 3
            s = max(1, min(5, s))
        else:
            s = 3
        normalised.append(s)

    # Any tail candidates beyond max_candidates get neutral score
    tail = candidates[max_candidates:]
    ranked = list(zip(head, normalised)) + [(c, 3) for c in tail]

    # Stable sort by score desc, preserving original SearxNG order for ties
    ranked.sort(key=lambda pair: -pair[1])

    _log.info(
        "rerank complete  candidates=%d scored=%d kept_hi=%d dropped_lo=%d %.2fs",
        len(candidates), len(head),
        sum(1 for _, s in ranked if s >= 4),
        sum(1 for _, s in ranked if s <= 1),
        elapsed,
    )
    return ranked


# --- No-results fallback and reformulation (§6) ---------------------------

def reformulate_query(
    original_query: str,
    intent_dict: dict,
) -> str | None:
    """One reformulation pass when a query returned zero useful results.

    Asks the tool model to produce a simpler query stripped of filler
    words but keeping the entities intact. Used as a last resort before
    declaring search failure. Returns ``None`` if reformulation itself
    fails or returns something identical to the original.
    """
    tool_url, tool_model = _tool_model()
    if not tool_url:
        return None

    entities = intent_dict.get("entities") or []
    intent = intent_dict.get("intent") or "unknown"
    entities_str = ", ".join(entities[:5]) if entities else "(none extracted)"

    prompt = (
        f"{build_prompt_date_header()}\n\n"
        f"The web search query \"{original_query}\" returned no useful "
        f"results. The user's intent is {intent} and the entities they "
        f"mentioned are: {entities_str}.\n\n"
        "Produce ONE simpler search query stripped of filler words, "
        "quoting proper nouns with double quotes, and focused on the "
        "entities. Return ONLY the query string — no prose, no quotes "
        "around the whole thing, no preamble.\n\nSimpler query:"
    )
    try:
        resp = httpx.post(
            f"{tool_url}/v1/chat/completions",
            json={
                "model": tool_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 60,
            },
            timeout=20,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        _log.warning("query reformulation failed: %s", e)
        return None

    # Strip quotes around the whole response and any markdown fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[^\n]*\n", "", cleaned).rstrip("`").strip()
    cleaned = cleaned.strip('"\'').strip()
    # Take only the first line — the model sometimes adds explanation
    cleaned = cleaned.splitlines()[0].strip() if cleaned else ""

    if not cleaned or cleaned.lower() == original_query.lower():
        _log.debug("reformulation returned empty or unchanged: %r", cleaned)
        return None

    _log.info("query reformulated  %r -> %r", original_query, cleaned)
    return cleaned[:200]


def build_failure_context(
    queries_tried: list[str],
    intent_dict: dict,
) -> str:
    """Build the search_context system message used when search fails
    entirely.

    Explicitly tells the chat model NOT to invent sources and to flag
    any general-knowledge fallback as such. This is the antidote to the
    "fine dining analysis framework" hallucination.
    """
    entities = intent_dict.get("entities") or []
    entities_str = ", ".join(entities[:5]) if entities else "(no specific entities extracted)"
    queries_str = " | ".join(f'"{q}"' for q in queries_tried[:3])
    return (
        f"SEARCH STATUS: failed. Tried {len(queries_tried)} "
        f"query/queries: {queries_str}. No relevant sources found for "
        f"entities {entities_str}.\n\n"
        "IMPORTANT rules for your reply:\n"
        "- Do NOT invent sources, URLs, or specific facts.\n"
        "- Do NOT fabricate a structured framework or analysis that "
        "looks like it came from research — the user will notice and "
        "lose trust.\n"
        "- Tell the user explicitly that you searched and couldn't "
        "find specific information about what they mentioned.\n"
        "- Optionally suggest ONE or TWO things they could search "
        "themselves (different phrasing, a specific site, etc.).\n"
        "- If general knowledge from your training is genuinely "
        "useful, you may share it — but mark it clearly as general "
        "knowledge (\"from what I know generally\"), never present it "
        "as search-derived fact.\n"
        "- Keep the reply short. Don't pad."
    )


def _nocodb_truthy(value) -> bool:
    """NocoDB checkboxes can return bool, int, or string representations."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _sanitise_url(url: str) -> str:
    """Strip fragments and dangerous characters from a URL before navigating."""
    parsed = urlparse(url)
    clean = parsed._replace(fragment="")
    return clean.geturl()


_STEALTH_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
window.chrome = {runtime: {}};
"""


# --- Playwright dedicated-worker-thread plumbing -----------------------------
#
# sync_playwright() is pinned to the thread that called .start(). Because the
# enrichment cycle runs in APScheduler/manual-trigger worker threads that come
# and go, the old "module-level singleton" design crashed with
# "cannot switch to a different thread (which happens to have exited)"
# on every call after the first.
#
# Fix: all playwright calls run on ONE dedicated worker thread, owned by this
# module. Callers submit (url, Future) to a queue and block on .result(). The
# worker thread is the only thread that ever touches sync_playwright, so the
# thread-affinity contract is never violated.

_PW_QUEUE_SENTINEL = object()
_pw_queue: queue.Queue | None = None
_pw_worker: threading.Thread | None = None
_pw_worker_lock = threading.Lock()
PLAYWRIGHT_FETCH_TIMEOUT = 60  # seconds — upper bound the caller will wait


def _playwright_worker_main() -> None:
    """Single long-lived thread that owns the playwright instance."""
    pw_instance = None
    browser = None

    def _launch_browser():
        nonlocal pw_instance, browser
        from playwright.sync_api import sync_playwright
        pw_instance = sync_playwright().start()
        browser = pw_instance.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-extensions",
                "--disable-gpu",
                "--disable-background-networking",
                "--disable-default-apps",
                "--disable-sync",
                "--disable-translate",
            ],
        )
        _log.info("playwright chromium launched (worker thread)")

    def _teardown_browser():
        nonlocal pw_instance, browser
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass
        if pw_instance is not None:
            try:
                pw_instance.stop()
            except Exception:
                pass
        browser = None
        pw_instance = None

    def _browser_alive() -> bool:
        if browser is None:
            return False
        try:
            _ = browser.contexts  # probe
            return True
        except Exception:
            return False

    while True:
        item = _pw_queue.get()
        if item is _PW_QUEUE_SENTINEL:
            _teardown_browser()
            return
        url, fut = item
        if fut.cancelled():
            continue
        started = time.time()
        try:
            if not _browser_alive():
                _teardown_browser()
                _launch_browser()
            assert browser is not None
            context = browser.new_context(
                user_agent=BROWSER_UA,
                viewport={"width": 1280, "height": 800},
                java_script_enabled=True,
                service_workers="block",
                permissions=[],
                accept_downloads=False,
                locale="en-US",
                timezone_id="America/New_York",
            )
            try:
                page = context.new_page()
                page.add_init_script(_STEALTH_SCRIPT)

                _allowed_types = {"document", "stylesheet", "font", "image", "script", "xhr", "fetch"}
                page.route("**/*", lambda route: (
                    route.continue_() if route.request.resource_type in _allowed_types
                    else route.abort()
                ))

                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                try:
                    page.wait_for_load_state("networkidle", timeout=5_000)
                except Exception:
                    pass  # DOM is loaded, proceed with what we have

                final_url = page.url
                if not _is_safe_url(final_url):
                    _log.warning("playwright_fetch blocked redirect to %s", final_url[:120])
                    fut.set_result("")
                    continue

                text = page.inner_text("body")
                text = _strip_injection_patterns(text)[:PER_PAGE_CHAR_CAP]
                elapsed = round(time.time() - started, 2)
                _log.info("playwright_fetch ok  url=%s chars=%d %.2fs", url[:120], len(text), elapsed)
                fut.set_result(text)
            finally:
                try:
                    context.close()
                except Exception:
                    pass
        except Exception as e:
            elapsed = round(time.time() - started, 2)
            _log.warning("playwright_fetch failed  url=%s error=%s %.2fs", url[:120], e, elapsed)
            # If the browser itself looks broken, tear it down so the next
            # iteration relaunches a fresh one.
            if not _browser_alive():
                _log.warning("playwright browser appears dead, resetting")
                _teardown_browser()
            fut.set_result("")


def _ensure_playwright_worker() -> None:
    global _pw_queue, _pw_worker
    if _pw_worker is not None and _pw_worker.is_alive():
        return
    with _pw_worker_lock:
        if _pw_worker is not None and _pw_worker.is_alive():
            return
        _pw_queue = queue.Queue()
        _pw_worker = threading.Thread(
            target=_playwright_worker_main,
            name="playwright-worker",
            daemon=True,
        )
        _pw_worker.start()
        _log.info("playwright worker thread started")


def playwright_fetch(url: str) -> str:
    """Fetch page text using in-process Playwright/Chromium.

    Thread-safe: all playwright work is dispatched to a single dedicated
    worker thread. The caller blocks on a Future until the result is ready.
    """
    if not _is_safe_url(url):
        _log.warning("playwright_fetch blocked unsafe url %s", url[:120])
        return ""
    url = _sanitise_url(url)
    _ensure_playwright_worker()
    assert _pw_queue is not None
    fut: Future = Future()
    _pw_queue.put((url, fut))
    try:
        return fut.result(timeout=PLAYWRIGHT_FETCH_TIMEOUT)
    except Exception as e:
        _log.warning("playwright_fetch wait failed  url=%s error=%s", url[:120], e)
        return ""


MAX_RESPONSE_BYTES = 5_000_000  # 5 MB — skip anything larger


def _scrape_with_httpx(url: str) -> str:
    """Plain httpx + BeautifulSoup scraper. Returns extracted text or empty string."""
    started = time.time()
    try:
        resp = httpx.get(
            url,
            timeout=SCRAPE_TIMEOUT,
            follow_redirects=True,
            headers=BROWSER_HEADERS,
        )
        resp.raise_for_status()
        if len(resp.content) > MAX_RESPONSE_BYTES:
            _log.warning("scrape skip  response too large (%d bytes) for %s", len(resp.content), url)
            return ""
        final_url = str(resp.url)
        if final_url != url and not _is_safe_url(final_url):
            _log.warning("scrape blocked redirect to unsafe url %s", final_url[:120])
            return ""
    except httpx.HTTPStatusError as e:
        _log.warning("scrape %d for %s (%s)", e.response.status_code, url, e.response.reason_phrase)
        return ""
    except httpx.TimeoutException:
        _log.warning("scrape timeout after %ds for %s", SCRAPE_TIMEOUT, url)
        return ""
    except Exception as e:
        _log.warning("scrape failed for %s: %s", url, e)
        return ""

    elapsed = round(time.time() - started, 2)
    content_type = (resp.headers.get("content-type") or "").lower()
    if content_type and "text/html" not in content_type and "text/plain" not in content_type:
        _log.debug("scrape skip  non-html content-type=%s for %s", content_type.split(";")[0], url)
        return ""
    _log.debug("scrape ok    %s  status=%d size=%d %.2fs", url, resp.status_code, len(resp.text), elapsed)

    try:
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        _log.warning("html parse failed for %s: %s", url, e)
        return ""

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _strip_injection_patterns(text)
    text = text[:PER_PAGE_CHAR_CAP]
    if not text:
        _log.warning("scrape empty after extraction for %s (raw html was %d chars)", url, len(resp.text))
    else:
        _log.debug("scrape extracted %d chars from %s", len(text), url)
    return text


def scrape_page(
    url: str,
    snippet: str = "",
    source: dict | None = None,
    meta_out: dict | None = None,
) -> str:
    """Fetch and extract page text. Falls back to `snippet` on any failure.

    If ``meta_out`` is provided, it is populated with diagnostic info:
        ``path``  — which fetch strategy won (``scraper`` / ``playwright_direct`` /
                    ``playwright_auto`` / ``fallback`` / ``blocked`` / ``unsafe``)
        ``chars`` — length of the returned text
    """
    def _set_meta(path: str, chars: int) -> None:
        if meta_out is not None:
            meta_out["path"] = path
            meta_out["chars"] = chars

    fallback = _strip_injection_patterns(snippet)[:PER_PAGE_CHAR_CAP] if snippet else ""

    if not _is_safe_url(url):
        _log.warning("scrape skip  unsafe url %s", url[:120])
        _set_meta("unsafe", len(fallback))
        return fallback

    if _is_blocklisted(url):
        _log.debug("scrape skip  blocklisted %s", url)
        _set_meta("blocked", len(fallback))
        return fallback

    # Playwright direct — skip scraper entirely
    if source and _nocodb_truthy(source.get("use_playwright")):
        text = playwright_fetch(url)
        _log.info("path=playwright_direct  url=%s ok=%s", url[:120], bool(text))
        out = text or fallback
        _set_meta("playwright_direct", len(out))
        return out

    # Plain scraper
    text = _scrape_with_httpx(url)

    if text and text != fallback:
        _log.info("path=scraper  url=%s chars=%d", url[:120], len(text))
        _set_meta("scraper", len(text))
        return text

    # Scraper returned nothing useful — try Playwright when called with a source (enrichment)
    if source is not None:
        pw_text = playwright_fetch(url)
        if pw_text:
            target_id = source.get("Id")
            if target_id:
                try:
                    from workers.enrichment_agent import EnrichmentDB
                    EnrichmentDB().update_scrape_target(target_id, use_playwright=True)
                    _log.info("auto-promoted to playwright_direct id=%s", target_id)
                except Exception as e:
                    _log.warning("playwright promotion failed id=%s: %s", target_id, e)
            _log.info("path=playwright_auto  url=%s chars=%d", url[:120], len(pw_text))
            _set_meta("playwright_auto", len(pw_text))
            return pw_text
        _log.info("path=playwright_auto  url=%s failed", url[:120])
        _set_meta("playwright_auto_failed", len(fallback))
        return fallback

    _set_meta("fallback", len(fallback))
    return fallback



# --- Per-intent extraction templates ---------------------------------------
#
# Each template is a short goal statement the model gets on top of a
# shared structured-output instruction. The varying piece is WHAT the
# model should preserve from the page — the output shape stays
# consistent (summary/relevance/source_type) so downstream code doesn't
# branch per intent.

_EXTRACTION_GOAL_BY_INTENT: dict[str, str] = {
    CHAT_INTENT_CONTEXTUAL: (
        "Extract the 3 most useful facts a friend who happened to know about "
        "this topic would mention in passing. Preserve specific numbers, "
        "names, dates. Keep it under 120 words. No preamble."
    ),
    CHAT_INTENT_FACTUAL: (
        "Extract the specific fact the user is asking about, along with its "
        "date, source authority, and any directly-relevant numbers. "
        "Preserve verbatim quotes for key claims. Max 200 words."
    ),
    CHAT_INTENT_EXPLANATORY: (
        "Extract the definition, the mechanism (how it works), and one "
        "concrete example from the page. Preserve technical terms verbatim. "
        "Max 300 words."
    ),
    CHAT_INTENT_RECOMMENDATION: (
        "Extract the specific items being recommended, their attributes "
        "(names, prices, locations, distinguishing features), and any "
        "verbatim positive or negative review quotes. Preserve names and "
        "numbers exactly. Max 300 words."
    ),
    CHAT_INTENT_COMPARISON: (
        "Extract the specific attributes of each option on the page "
        "(features, prices, versions, pros, cons). Preserve numbers and "
        "specific claims verbatim. Max 300 words."
    ),
    CHAT_INTENT_RESEARCH: (
        "Extract the methodology, findings, sample size, authors, dates, "
        "and any counter-findings or caveats. Preserve verbatim quotes for "
        "key claims. Max 400 words."
    ),
    CHAT_INTENT_TROUBLESHOOTING: (
        "Extract the symptoms, the root causes identified on this page, and "
        "the specific fixes or workarounds proposed. Preserve error messages "
        "and command lines verbatim. Max 300 words."
    ),
    CODE_INTENT_LOOKUP: (
        "Extract the API signature, one working code example verbatim, the "
        "version or release that this applies to, and any gotchas. Use "
        "fenced code blocks for code. Max 300 words."
    ),
    CODE_INTENT_DEBUG: (
        "Extract the root cause analysis, the specific fix, and any "
        "related error messages or version info. Preserve code verbatim. "
        "Max 250 words."
    ),
    CODE_INTENT_BUILD: (
        "Extract the usage example, required dependencies, and any setup "
        "steps. Preserve code verbatim in fenced blocks. Max 250 words."
    ),
}


def _extraction_goal_for(intent_dict: dict) -> str:
    """Return the per-intent extraction goal text, with a safe default."""
    intent = intent_dict.get("intent") or CHAT_INTENT_FACTUAL
    return _EXTRACTION_GOAL_BY_INTENT.get(
        intent,
        "Extract a factual summary of the page preserving key names, "
        "numbers, and dates. Max 250 words.",
    )


# Content types from the enricher's classifier enum that we accept for
# extraction. Imported lazily to avoid a circular enrichment_agent import.
_ACCEPTABLE_CONTENT_TYPES = {
    "REFERENCE", "ARTICLE", "ENCYCLOPEDIC", "FORUM", "PRODUCT",
}
# UNCLEAR is soft-accepted (extracted, but flagged).
_SOFT_CONTENT_TYPES = {"UNCLEAR"}
# The rest (NAVIGATION, BOILERPLATE, GENERATED, PAYWALL) are dropped.


def _extract_one_page(
    page_text: str,
    query: str,
    intent_dict: dict,
) -> dict | None:
    """Extract structured info from a single page using the fast model.

    Returns the assessment dict or None on rejection/failure. Pure function
    — no I/O other than the LLM call. Safe to run concurrently via fan_out.

    Thread-safety: the `_fast_call` helper uses module-level httpx.post;
    requests are stateless with no shared mutable state between threads.
    """
    if not page_text or not page_text.strip():
        return None

    # Deferred import — enrichment_agent pulls in heavy deps (FalkorDB,
    # NocoDB, crawler) that we don't want at web_search module-load time.
    try:
        from workers.enrichment_agent import (
            _fast_call,
            _heuristic_quality_gate,
            _classify_content_type,
        )
    except Exception as e:
        _log.warning("extraction helpers import failed: %s", e)
        return None

    # L2: heuristic quality gate — cheap, deterministic, drops obvious junk.
    passed, gate_reason, gate_metrics = _heuristic_quality_gate(page_text)
    if not passed:
        _log.debug(
            "extraction drop  gate=%s metrics=%s",
            gate_reason, gate_metrics,
        )
        return None

    # L3: content-type classifier — drops NAVIGATION / BOILERPLATE /
    # GENERATED / PAYWALL pages before spending extraction tokens.
    content_type, _classifier_raw, _classifier_tokens = _classify_content_type(page_text)
    if content_type is None:
        # Classifier couldn't parse — be lenient, treat as UNCLEAR.
        content_type = "UNCLEAR"
    if content_type not in _ACCEPTABLE_CONTENT_TYPES and content_type not in _SOFT_CONTENT_TYPES:
        _log.debug("extraction drop  content_type=%s", content_type)
        return None

    # L4: per-intent extraction via the fast model.
    goal = _extraction_goal_for(intent_dict)
    excerpt = page_text[:SUMMARY_INPUT_CHAR_CAP]
    prompt = (
        f"{build_prompt_date_header()}\n\n"
        f"You are extracting information from a web page for a user who asked:\n"
        f"'{query[:500]}'\n\n"
        f"GOAL: {goal}\n\n"
        "Return ONLY a JSON object with these fields:\n"
        "- summary: the extracted content per the GOAL above\n"
        "- relevance: \"high\" (directly answers the question), "
        "\"medium\" (related/useful), or \"low\" (tangential)\n"
        "- source_type: one of \"official_docs\", \"news_article\", "
        "\"blog_post\", \"research_paper\", \"forum\", \"product_page\", "
        "\"government\", \"unknown\"\n\n"
        "If the page is genuinely useless for the question, return: "
        '{"irrelevant": true}\n\n'
        f"PAGE:\n{excerpt}"
    )

    raw, tokens = _fast_call(prompt, max_tokens=SUMMARY_MAX_TOKENS, temperature=0.2)
    if not raw:
        _log.debug("extraction: empty response from fast model")
        return None

    # Parse the structured response. Tolerate markdown fences and prose
    # preambles that small models sometimes emit.
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned).rstrip("`").strip()
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if not json_match:
        _log.debug("extraction: no JSON in response: %s", raw[:200])
        return None
    try:
        data = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        _log.debug("extraction: JSON parse failed: %s", raw[:200])
        return None

    if data.get("irrelevant"):
        return None

    summary = str(data.get("summary") or "").strip()
    if not summary:
        return None

    return {
        "summary": summary,
        "relevance": str(data.get("relevance") or "unknown").lower(),
        "source_type": str(data.get("source_type") or "unknown").lower(),
        "content_type": content_type,
        "tokens": tokens,
    }


def _background_graph_write(text: str, org_id: int, url: str) -> None:
    """Fire-and-forget relationship extraction into FalkorDB.

    Runs the enricher's ironclad :func:`_extract_relationships` on a
    scraped web-search page so the knowledge graph grows from
    real-time user searches, not only from scheduled enrichment. All
    errors swallowed and logged — we never want graph writes to block
    or break a chat reply.
    """
    try:
        from workers.enrichment_agent import _extract_relationships
    except Exception as e:
        _log.debug("graph write skipped  import failed: %s", e)
        return
    try:
        written, tokens = _extract_relationships(text, org_id)
        _log.info(
            "search graph write  url=%s written=%d tokens=%d",
            url[:120], written, tokens,
        )
    except Exception:
        _log.warning("search graph write failed  url=%s", url[:120], exc_info=True)


def extract_from_pages(
    pages: list[dict],
    query: str,
    intent_dict: dict,
    org_id: int | None = None,
    fire_graph_writes: bool = True,
) -> list[dict | None]:
    """Per-page extraction pipeline for web search.

    Replaces the old ``summarise_pages_batch`` (single-call batched prompt
    via the tool model, timeout-prone, shallow). New design:

    1. **L2 quality gate** — reject too-short / low-diversity / repeat-heavy
       pages without spending any LLM tokens.
    2. **L3 content-type classifier** — reject NAVIGATION / BOILERPLATE /
       GENERATED / PAYWALL. Reuses the enricher's classifier.
    3. **L4 per-intent extraction** via ``_fast_call`` (Gemma E2B) with
       per-intent prompt templates. Summary field carries the intent-specific
       content while the shape stays uniform.
    4. **Fan-out** — pages extracted concurrently via
       :func:`workers.crawler.fan_out`, bounded by ``MODEL_PARALLEL_SLOTS``.
    5. **Background graph writes** — for each accepted page, fire a
       daemon-thread call to the enricher's ``_extract_relationships``
       so FalkorDB grows from search traffic too.

    Returns a list aligned with ``pages``. Each element is either an
    assessment dict with keys ``summary``, ``relevance``, ``source_type``,
    ``content_type``, ``tokens`` — or ``None`` if the page was rejected,
    failed extraction, or timed out.
    """
    if not pages:
        return []

    # Deferred import to avoid circular load
    try:
        from workers.crawler import fan_out
    except Exception:
        _log.warning("fan_out unavailable — falling back to sequential extraction")
        return [_extract_one_page(p.get("text", ""), query, intent_dict) for p in pages]

    # Build one callable per page. Each closure captures its own `text`.
    def _make_worker(page: dict):
        text = page.get("text", "")
        return lambda: _extract_one_page(text, query, intent_dict)

    workers = [_make_worker(p) for p in pages]
    started = time.time()
    results = fan_out(
        workers,
        label="extract_from_pages",
        max_workers=min(len(pages), 4),
    )
    elapsed = round(time.time() - started, 2)

    kept = sum(1 for r in results if r is not None)
    _log.info(
        "extract_from_pages  pages=%d accepted=%d dropped=%d %.2fs",
        len(pages), kept, len(pages) - kept, elapsed,
    )

    # Fire background graph writes for accepted pages. Only runs if we
    # have an org_id (the search path always does, the code path may
    # not). Daemon threads so they never block the caller.
    if fire_graph_writes and org_id is not None:
        for page, result in zip(pages, results):
            if result is None:
                continue
            url = page.get("result", {}).get("url", "") if isinstance(page.get("result"), dict) else ""
            text = page.get("text", "")
            if not text:
                continue
            threading.Thread(
                target=_background_graph_write,
                args=(text, org_id, url),
                daemon=True,
                name="search-graph-write",
            ).start()

    return results


# Legacy helpers (summarise_page, summarise_pages_batch, _parse_query_list)
# were removed as part of the web-search intent refactor. They have been
# replaced by the extract_from_pages pipeline (_fast_call, quality gate,
# content-type classifier, fan_out, per-intent extraction prompts).
# The old top-level tool-model batched-summarise path was the source of the
# "httpx.ReadTimeout at summarise_page:519" failure seen in production.


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


# Conservative "definitely no search" heuristic — trips only when a message
# is short, conversational, contains no recency/lookup trigger word, and is
# not shaped like a factual wh-question. Anything ambiguous falls through
# to the classifier LLM below.
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
    # Long messages are ambiguous — let the classifier decide.
    if len(msg) > 400:
        return False
    # Any recency/lookup trigger word → bail to classifier.
    if _NO_SEARCH_TRIGGER_WORDS.search(msg):
        return False
    # Wh-question shape → could be a factual lookup.
    if _EXTERNAL_QUESTION.search(msg):
        return False
    # Clearly code or a shell snippet → definitely no search.
    if "```" in msg or msg.lstrip().startswith(_CODE_PREFIXES):
        return True
    # Short conversational turn with no triggers and no external question.
    if len(msg) <= 200:
        return True
    return False


def needs_web_search(message: str) -> tuple[bool, str, str]:
    """Classify whether a message needs web search.

    Returns (needs_search, reason, confidence).
    confidence is "high" (auto-invoke), "medium" (consent prompt), or "" (no search).
    Defaults to (False, "", "") on any failure.
    """
    msg = (message or "").strip()
    if not msg:
        return False, "", ""

    explicit, reason = _explicit_search_intent(msg)
    if explicit:
        _log.info("search intent detected (pattern): %s", reason)
        return True, reason, "high"

    # Conservative skip: eliminate the classifier LLM call for clearly
    # non-search messages. Only trips when no trigger word, no wh-question,
    # and message is short/code. Anything ambiguous still hits the classifier.
    if _definitely_no_search(msg):
        _log.info("search skip (heuristic): short conversational message")
        return False, "", ""

    tool_url, tool_model = _tool_model()
    if not tool_url:
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
        resp = httpx.post(
            f"{tool_url}/v1/chat/completions",
            json={
                "model": tool_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 80,
            },
            timeout=20,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
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


def _dedupe(results: Iterable[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for r in results:
        url = r.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(r)
    return out


# --- Per-policy search budget --------------------------------------------

_POLICY_BUDGETS = {
    SEARCH_POLICY_CONTEXTUAL: {
        "max_candidates": 5,
        "rerank_max": 5,
        "max_scrape": 3,
        "rerank_drop_threshold": 3,
        "hard_cap_s": 5.0,
    },
    SEARCH_POLICY_FOCUSED: {
        "max_candidates": 10,
        "rerank_max": 10,
        "max_scrape": 5,
        "rerank_drop_threshold": 2,
        "hard_cap_s": 30.0,
    },
    SEARCH_POLICY_FULL: {
        "max_candidates": 15,
        "rerank_max": 15,
        "max_scrape": 8,
        "rerank_drop_threshold": 2,
        "hard_cap_s": 60.0,
    },
}


def _run_search_inner(
    query: str,
    org_id: int,
    intent_dict: dict,
    budget: dict,
) -> tuple[str, list[dict], str]:
    """Inner search pipeline. Called either directly or inside a
    time-bounded future for contextual_enrichment.

    Steps:
      1. Generate entity-aware queries from intent_dict (§3)
      2. SearxNG fetch + dedupe
      3. Pre-scrape rerank via tool model (§3)
      4. Reformulate once if everything scored poorly (§6)
      5. Scrape top N candidates (playwright worker thread handles
         parallelism internally)
      6. Per-page extraction with quality gate + content-type classifier
         + fast-model extraction + fan_out + background graph writes (§4)
      7. Write accepted pages to ChromaDB
      8. Build style-adaptive context block (§5)
      9. Return (context_block, sources, confidence)

    Never raises — all errors degrade to the "failed" confidence state.
    """
    from workers.styles import search_context_for

    queries_tried: list[str] = []
    queries = generate_search_queries(intent_dict, message=query)
    if not queries:
        _log.info("search skip  no queries generated")
        return "", [], "none"

    # 1) Fetch candidates from SearxNG
    raw_results: list[dict] = []
    for q in queries:
        queries_tried.append(q)
        raw_results.extend(searxng_search(q, max_results=budget["max_candidates"]))
        if len(raw_results) >= budget["max_candidates"]:
            break
    raw_results = _dedupe(raw_results)[: budget["max_candidates"]]

    # 2) Pre-scrape rerank — drop distractors before spending scrape tokens.
    #    This is the fix for "Analyse Data in Excel" being top-ranked.
    if raw_results:
        ranked = rerank_candidates(raw_results, intent_dict, max_candidates=budget["rerank_max"])
        kept = [c for c, score in ranked if score >= budget["rerank_drop_threshold"]]
    else:
        kept = []

    # 3) If rerank dropped everything, try one reformulation pass.
    if not kept and raw_results:
        _log.info("rerank dropped all candidates — attempting reformulation")
        reformulated = reformulate_query(queries[0], intent_dict)
        if reformulated:
            queries_tried.append(reformulated)
            raw_results = _dedupe(searxng_search(reformulated, max_results=budget["max_candidates"]))
            if raw_results:
                ranked = rerank_candidates(raw_results, intent_dict, max_candidates=budget["rerank_max"])
                kept = [c for c, score in ranked if score >= budget["rerank_drop_threshold"]]

    # 4) If SearxNG returned nothing at all, also try reformulation.
    if not raw_results:
        _log.info("searxng returned nothing — attempting reformulation")
        reformulated = reformulate_query(queries[0], intent_dict)
        if reformulated:
            queries_tried.append(reformulated)
            raw_results = _dedupe(searxng_search(reformulated, max_results=budget["max_candidates"]))
            if raw_results:
                ranked = rerank_candidates(raw_results, intent_dict, max_candidates=budget["rerank_max"])
                kept = [c for c, score in ranked if score >= budget["rerank_drop_threshold"]]

    # 5) If we still have nothing usable, return the failure context block.
    if not kept:
        _log.warning("search failed  queries_tried=%s", queries_tried)
        context_block = build_failure_context(queries_tried, intent_dict)
        return context_block, [], "failed"

    # 6) Scrape top N accepted candidates.
    to_scrape = kept[: budget["max_scrape"]]
    scraped_pages: list[dict] = []
    scrape_failures = 0
    for r in to_scrape:
        text = scrape_page(r["url"], snippet=r.get("snippet", ""))
        if not text:
            scrape_failures += 1
            continue
        scraped_pages.append({"result": r, "text": text})

    if not scraped_pages:
        _log.warning("scraping failed on all %d candidates", len(to_scrape))
        context_block = build_failure_context(queries_tried, intent_dict)
        return context_block, [], "failed"

    # 7) Per-page extraction (fast model, quality gate, content-type
    #    classifier, fan_out concurrency, background graph writes).
    assessments = extract_from_pages(
        scraped_pages,
        query,
        intent_dict,
        org_id=org_id,
        fire_graph_writes=True,
    )

    # 8) Assemble results; write accepted pages to chroma.
    results: list[dict] = []
    for page, assessed in zip(scraped_pages, assessments):
        if not assessed:
            continue
        result = page["result"]
        entry = {
            "title": result.get("title") or result["url"],
            "url": result["url"],
            "summary": assessed["summary"],
            "relevance": assessed.get("relevance", "unknown"),
            "source_type": assessed.get("source_type", "unknown"),
            "content_type": assessed.get("content_type", "UNCLEAR"),
        }
        results.append(entry)
        try:
            remember(
                text=f"{entry['title']}\n\n{entry['summary']}",
                metadata={
                    "url": result["url"],
                    "title": entry["title"],
                    "query": query,
                    "relevance": entry["relevance"],
                    "source_type": entry["source_type"],
                    "content_type": entry["content_type"],
                    "intent": intent_dict.get("intent") or "",
                    "fetched_at": time.time(),
                },
                org_id=org_id,
                collection_name="web_search",
            )
        except Exception:
            _log.error("chroma write failed for %s", result["url"], exc_info=True)

    if not results:
        _log.warning("extraction rejected all scraped pages")
        context_block = build_failure_context(queries_tried, intent_dict)
        return context_block, [], "failed"

    # 9) Sort by relevance and compute confidence.
    high = [r for r in results if r["relevance"] == "high"]
    medium = [r for r in results if r["relevance"] == "medium"]
    low = [r for r in results if r["relevance"] not in ("high", "medium")]

    if high:
        confidence = "high"
    elif medium:
        confidence = "medium"
    else:
        confidence = "low"

    sorted_results = high + medium + low

    # 10) Build style-adaptive context block from the intent's response
    #     template. Replaces the old fixed clinical block that forced
    #     numbered citations regardless of intent.
    template_key = intent_dict.get("response_template") or "direct_answer"
    template_body = search_context_for(template_key)

    fact_blocks: list[str] = []
    for i, entry in enumerate(sorted_results, start=1):
        # Source tag kept minimal — attribution is rendered by the UI,
        # so the model sees clean source_N markers for its own
        # internal bookkeeping but doesn't need to surface them.
        fact_blocks.append(
            f"SOURCE {i} — {entry['title']} ({entry['url']})\n"
            f"{entry['summary']}"
        )
    facts_section = "\n\n".join(fact_blocks)

    context_block = (
        f"{template_body}\n\n"
        f"---\nFACTS AVAILABLE FOR YOUR REPLY:\n\n{facts_section}"
    )

    sources = [
        {
            "index": i + 1,
            "title": e["title"],
            "url": e["url"],
            "relevance": e["relevance"],
            "source_type": e["source_type"],
            "content_type": e["content_type"],
            "snippet": e["summary"][:200],
        }
        for i, e in enumerate(sorted_results)
    ]

    _log.info(
        "search done  intent=%s policy=%s queries=%d candidates=%d kept=%d "
        "scraped=%d accepted=%d confidence=%s",
        intent_dict.get("intent"),
        intent_dict.get("search_policy"),
        len(queries_tried),
        len(raw_results),
        len(kept),
        len(scraped_pages),
        len(results),
        confidence,
    )

    # Background suggestion pass — same as before, never blocks the reply.
    summaries_for_suggest = [(e["title"], e["url"], e["summary"]) for e in results]
    threading.Thread(
        target=_suggest_sources_from_search,
        args=(summaries_for_suggest, query, org_id),
        daemon=True,
    ).start()

    return context_block, sources, confidence


def run_web_search(
    query: str,
    org_id: int,
    intent_dict: dict | None = None,
    history: list[dict] | None = None,
) -> tuple[str, list[dict], str]:
    """Unified search entry point.

    Dispatches on ``intent_dict["search_policy"]`` to the right budget.
    For ``contextual_enrichment`` wraps the search in a 5-second hard cap
    so the chat reply never waits longer than that for grounding.

    If ``intent_dict`` is not supplied, classifies the query on the fly
    (back-compat for callers that haven't been updated to §8 yet).

    Returns ``(context_block, sources, confidence)`` where confidence is:
      - "high"/"medium"/"low"  — results found, ranked
      - "none"                 — policy was none, no search attempted
      - "failed"               — search ran, no usable results after
                                 reformulation — context_block tells the
                                 model not to fabricate
      - "deferred"             — contextual enrichment exceeded its 5s cap
    """
    _log.debug("search start  query=%s org=%d", query[:100], org_id)

    if intent_dict is None:
        intent_dict = classify_message_intent(query, history=history)

    policy = intent_dict.get("search_policy") or SEARCH_POLICY_NONE
    if policy == SEARCH_POLICY_NONE:
        return "", [], "none"

    budget = _POLICY_BUDGETS.get(policy)
    if not budget:
        _log.warning("unknown search policy %r, defaulting to focused", policy)
        budget = _POLICY_BUDGETS[SEARCH_POLICY_FOCUSED]

    # Contextual enrichment has a hard latency cap — wrap the inner
    # pipeline in a future and time out the caller if the search takes
    # too long. The work continues in the background (the thread isn't
    # cancelled — Python can't cancel blocking httpx calls cleanly) but
    # the chat reply proceeds without waiting.
    if policy == SEARCH_POLICY_CONTEXTUAL:
        import concurrent.futures as _futures
        ex = _futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="contextual-search")
        try:
            fut = ex.submit(_run_search_inner, query, org_id, intent_dict, budget)
            try:
                return fut.result(timeout=budget["hard_cap_s"])
            except _futures.TimeoutError:
                _log.info(
                    "contextual enrichment deferred  query=%s cap=%.1fs",
                    query[:100], budget["hard_cap_s"],
                )
                return "", [], "deferred"
        finally:
            # Don't wait — background work can finish on its own.
            ex.shutdown(wait=False)

    # Non-contextual policies run inline.
    return _run_search_inner(query, org_id, intent_dict, budget)


def _suggest_sources_from_search(
    summaries: list[tuple[str, str, str]],
    query: str,
    org_id: int,
) -> None:
    """Evaluate web search results and suggest worthy ones as scrape targets."""
    if not summaries:
        return

    tool_url, tool_model = _tool_model()
    if not tool_url:
        return

    sources_text = "\n".join(
        f"{i+1}. {title} ({url})\n   {summary[:300]}"
        for i, (title, url, summary) in enumerate(summaries)
    )
    prompt = (
        "You are evaluating web search results to find sources worth ONGOING "
        "monitoring in a knowledge base. Most search results are NOT good "
        "monitoring targets — be very selective.\n\n"
        "A source MUST have ALL of:\n"
        "- INSTITUTIONAL AUTHORITY: maintained by a recognised organisation\n"
        "- REGULAR UPDATES: publishes new content on a recurring basis\n"
        "- ORIGINAL CONTENT: primary source, not aggregator or reposter\n"
        "- EDITORIAL STANDARDS: institutional accountability, not anonymous\n\n"
        "ALWAYS REJECT: social media, forums, Medium/Substack (unless institutional), "
        "paywalled sites, personal blogs, content farms, aggregators, YouTube, "
        "one-off news articles (suggest the publication's section page instead).\n\n"
        f"Search context: user asked about '{query}'\n\n"
        f"RESULTS:\n{sources_text}\n\n"
        "For each result worth monitoring long-term, return:\n"
        "- index: the result number\n"
        "- name: the organisation or publication\n"
        "- category: one of documentation, news, competitive, regulatory, "
        "  research, security, model_releases\n"
        "- authority: WHO maintains this and WHY they are credible\n"
        "- score: 1-10 (8+ = clearly authoritative, 7 = probably good)\n"
        "- suggested_url: the best URL to monitor (may differ from the search "
        "  result — prefer a section/feed page over a single article)\n\n"
        "Return a JSON array. If NONE are worth monitoring, return []. "
        "0 suggestions is the expected outcome for most searches."
    )

    try:
        resp = httpx.post(
            f"{tool_url}/v1/chat/completions",
            json={
                "model": tool_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 400,
            },
            timeout=FAST_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        _log.debug("search suggestion evaluation failed", exc_info=True)
        return

    try:
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip()).strip()
        items = json.loads(cleaned)
    except Exception:
        _log.debug("search suggestion unparseable: %s", raw[:200])
        return
    if not isinstance(items, list) or not items:
        return

    # Lazy import to avoid circular dependency (enrichment_agent imports from web_search)
    try:
        from workers.enrichment_agent import EnrichmentDB
        db = EnrichmentDB()
    except Exception:
        _log.debug("could not init EnrichmentDB for search suggestions")
        return

    cycle_id = f"websearch_{int(time.time())}"
    recorded = 0
    for item in items[:3]:
        try:
            score = int(item.get("score") or 0)
            if score < 7:
                continue
            category = str(item.get("category") or "").lower()
            if category not in CATEGORY_COLLECTIONS:
                continue
            idx = int(item.get("index", 0)) - 1
            if idx < 0 or idx >= len(summaries):
                continue

            url = str(item.get("suggested_url") or summaries[idx][1]).strip()
            if not url.startswith("http"):
                continue

            authority = str(item.get("authority") or "")
            name = str(item.get("name") or summaries[idx][0])
            confidence = "high" if score >= 8 else "medium"

            db.record_suggestion(
                org_id=org_id,
                url=url,
                name=name,
                category=category,
                reason=f"Web search: {authority}"[:500],
                confidence=confidence,
                confidence_score=score,
                suggested_by_url=summaries[idx][1],
                suggested_by_cycle=cycle_id,
            )
            recorded += 1
        except Exception:
            _log.debug("search suggestion record failed", exc_info=True)

    if recorded:
        _log.info("search suggestions recorded=%d from query='%s'", recorded, query[:80])
