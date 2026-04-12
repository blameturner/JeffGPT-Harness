from __future__ import annotations

import json
import logging
import re
import time

import httpx

from workers.search.intent import (
    CHAT_INTENT_CHITCHAT,
    CHAT_INTENT_COMPARISON,
    CHAT_INTENT_CONTEXTUAL,
    CHAT_INTENT_EXPLANATORY,
    CHAT_INTENT_FACTUAL,
    CHAT_INTENT_RECOMMENDATION,
    CHAT_INTENT_RESEARCH,
    CHAT_INTENT_TROUBLESHOOTING,
    CODE_INTENT_BUILD,
    CODE_INTENT_DEBUG,
    CODE_INTENT_LOOKUP,
    TASK_INTENT_SEARCH_EXPLICIT,
)
from config import thinking_skip_messages
from workers.search.models import _tool_model
from workers.search.temporal import build_prompt_date_header, now_in_chat_tz

_log = logging.getLogger("web_search.queries")


def _quote_entity(entity: str) -> str:
    # Multi-word entities need phrase quoting so SearxNG doesn't tokenise them
    # into independent words (that's how "Western Sydney Wanderers" lost to
    # "2 on the go" pre-fix).
    e = entity.strip().strip('"\'')
    if not e:
        return ""
    return f'"{e}"'


def _current_year() -> int:
    return now_in_chat_tz().year


def generate_search_queries(intent_dict: dict, message: str | None = None) -> list[str]:
    intent = intent_dict.get("intent") or CHAT_INTENT_CHITCHAT
    entities = intent_dict.get("entities") or []
    location = (intent_dict.get("location_hint") or "").strip()
    temporal = (intent_dict.get("temporal_anchor") or "").strip()
    time_sensitive = bool(intent_dict.get("time_sensitive"))
    year = _current_year()

    quoted = [_quote_entity(e) for e in entities if _quote_entity(e)]
    primary = quoted[0] if quoted else ""
    all_quoted = " ".join(quoted[:3])

    def _time_suffix() -> str:
        if temporal:
            return f"{temporal} {year}"
        if time_sensitive:
            return str(year)
        return ""

    queries: list[str] = []

    if intent == CHAT_INTENT_CONTEXTUAL and primary:
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
        noun = (message or "").strip()[:200]
        if quoted:
            queries = [f"{all_quoted} {noun}".strip()]
        elif noun:
            queries = [noun]

    if not queries and message:
        cleaned = re.sub(r"\s+", " ", message.strip())[:200]
        if cleaned:
            queries = [cleaned]

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
    if not candidates:
        return []

    tool_url, tool_model = _tool_model()
    if not tool_url:
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
        _log.info("rerank start  model=%s candidates=%d", tool_model, len(head))
        resp = httpx.post(
            f"{tool_url}/v1/chat/completions",
            json={
                "model": tool_model,
                "messages": thinking_skip_messages(tool_model, [
                    {"role": "user", "content": prompt},
                ]),
                "temperature": 0.0,
                "max_tokens": 80,
            },
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        _log.warning("rerank failed  model=%s: %s — passing candidates through", tool_model, e)
        return [(c, 3) for c in candidates]

    elapsed = round(time.time() - started, 2)

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

    tail = candidates[max_candidates:]
    ranked = list(zip(head, normalised)) + [(c, 3) for c in tail]

    # Stable sort by score desc preserves original SearxNG order for ties.
    ranked.sort(key=lambda pair: -pair[1])

    _log.info(
        "rerank complete  candidates=%d scored=%d kept_hi=%d dropped_lo=%d %.2fs",
        len(candidates), len(head),
        sum(1 for _, s in ranked if s >= 4),
        sum(1 for _, s in ranked if s <= 1),
        elapsed,
    )
    return ranked


def reformulate_query(
    original_query: str,
    intent_dict: dict,
) -> str | None:
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
        _log.info("query reformulate start  model=%s", tool_model)
        resp = httpx.post(
            f"{tool_url}/v1/chat/completions",
            json={
                "model": tool_model,
                "messages": thinking_skip_messages(tool_model, [
                    {"role": "user", "content": prompt},
                ]),
                "temperature": 0.0,
                "max_tokens": 60,
            },
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        _log.warning("query reformulation failed  model=%s: %s", tool_model, e)
        return None

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[^\n]*\n", "", cleaned).rstrip("`").strip()
    cleaned = cleaned.strip('"\'').strip()
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
