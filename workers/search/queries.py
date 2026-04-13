from __future__ import annotations

import json
import logging
import re
import time

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
from workers.enrichment.models import model_call
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
        _log.info("rerank start  candidates=%d", len(head))
        raw, _tokens = model_call("tool_planner", prompt, max_tokens=80, temperature=0.0)
        if not raw:
            return [(c, 3) for c in candidates]
    except Exception as e:
        _log.warning("rerank failed: %s — passing candidates through", e)
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


_STOPWORDS: frozenset[str] = frozenset(
    "a about above after again against all am an and any are aren't as at be "
    "because been before being below between both but by can can't cannot could "
    "couldn't did didn't do does doesn't doing don't down during each few for "
    "from further get got had hadn't has hasn't have haven't having he he'd "
    "he'll he's her here here's hers herself him himself his how how's i i'd "
    "i'll i'm i've if in into is isn't it it's its itself let's me more most "
    "mustn't my myself no nor not of off on once only or other ought our ours "
    "ourselves out over own please same shan't she she'd she'll she's should "
    "shouldn't so some such than that that's the their theirs them themselves "
    "then there there's these they they'd they'll they're they've this those "
    "through to too under until up upon very was wasn't we we'd we'll we're "
    "we've were weren't what what's when when's where where's which while who "
    "who's whom why why's will with won't would wouldn't you you'd you'll "
    "you're you've your yours yourself yourselves".split()
)

# Words that are part of question/instruction framing, not the topic itself.
# Separated from stopwords so topic-extraction can strip these without
# losing genuinely meaningful words like "compare" in a comparison query.
_FRAMING_WORDS: frozenset[str] = frozenset(
    "tell explain describe discuss define help show give list name "
    "know want need like think say said going go make made "
    "handle find look search check create run start stop try "
    "please kindly wondering curious looking".split()
)


def _extract_keywords(message: str) -> list[str]:
    """Extract meaningful keywords by removing stopwords and framing. Pure Python, sub-ms."""
    words = re.findall(r"[a-zA-Z0-9][\w\-.']*[a-zA-Z0-9]|[a-zA-Z0-9]", message)
    noise = _STOPWORDS | _FRAMING_WORDS
    return [w for w in words if w.lower() not in noise and len(w) > 1]


def _extract_phrases(message: str) -> list[str]:
    """Extract contiguous runs of non-stopword tokens as phrases.

    "tell me about the Linux kernel architecture" → ["Linux kernel architecture"]
    """
    words = re.findall(r"[a-zA-Z0-9][\w\-.']*[a-zA-Z0-9]|[a-zA-Z0-9]", message)
    noise = _STOPWORDS | _FRAMING_WORDS
    phrases: list[str] = []
    current: list[str] = []
    for w in words:
        if w.lower() in noise:
            if len(current) >= 2:
                phrases.append(" ".join(current))
            current = []
        else:
            current.append(w)
    if len(current) >= 2:
        phrases.append(" ".join(current))
    return phrases


# ---- Core topic extraction ----

# Regex patterns stripped from the START of the message to isolate the actual
# topic.  Order matters — longer/more specific patterns first.  Each pattern
# should consume the framing prefix so we're left with the user's real subject.
_QUESTION_PREAMBLES: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # Instruction preambles
        r"^(?:can|could|would) you (?:please |kindly )?(?:tell|show|explain|describe|find|look up|search|help(?: me)?(?: with)?|give me(?: info(?:rmation)? (?:on|about))?)\s*(?:me )?\s*",
        r"^(?:please |kindly )?(?:tell|show|explain|describe|find|look up|search|help(?: me)?(?: with)?|give me(?: info(?:rmation)? (?:on|about))?)\s*(?:me )?\s*",
        r"^i(?:'m| am) (?:looking for|curious about|wondering about|interested in|trying to (?:find|understand|learn(?: about)?))\s*",
        r"^i (?:want|need|would like) to (?:know|learn|understand|find out)(?: (?:more )?about)?\s*",
        r"^(?:do you know|have you heard)(?: anything)? about\s*",
        r"^(?:what do you know about|what can you tell me about)\s*",
        # Question words — only strip the framing, keep the topic
        r"^what (?:is|are|was|were|does|do|did|would|could|should)(?: (?:a|an|the))?\s*",
        r"^who (?:is|are|was|were)\s*",
        r"^where (?:is|are|was|were|can|do|does)\s*",
        r"^when (?:is|are|was|were|did|does|will)\s*",
        r"^why (?:is|are|was|were|did|does|do|don't|doesn't|won't|can't)\s*",
        r"^how (?:does|do|did|is|are|can|could|to|would|should)\s*",
        r"^(?:is|are|was|were|does|do|did|has|have|will|can|could|should|would) (?:the |a |an )?\s*",
    ]
]

# Leftover determiners / prepositions after stripping preambles.
_LEADING_FLUFF = re.compile(
    r"^(?:the |a |an |about |for |in |on |of |to |with |by |from |regarding |concerning |re: )+",
    re.IGNORECASE,
)

# Trailing punctuation / filler that adds no search value.
_TRAILING_FLUFF = re.compile(r"[?.!,;:]+$")


def _strip_preamble(text: str) -> str:
    """Remove question/instruction framing from the start of a message."""
    result = text
    for pat in _QUESTION_PREAMBLES:
        result = pat.sub("", result, count=1)
        if result != text:
            break
    result = _LEADING_FLUFF.sub("", result).strip()
    result = _TRAILING_FLUFF.sub("", result).strip()
    return result


def _detect_entities(text: str) -> list[str]:
    """Detect likely proper nouns / named entities from capitalisation patterns.

    Returns multi-word entities first (more specific), then single-word ones.
    Skips sentence-initial capitalisation by checking position context.
    """
    # Find capitalised word sequences (2+ words starting with uppercase)
    multi = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
    # Single capitalised words — but not sentence-initial ones
    words = text.split()
    singles: list[str] = []
    for i, w in enumerate(words):
        clean = re.sub(r"[^a-zA-Z]", "", w)
        if not clean or len(clean) < 2:
            continue
        if clean[0].isupper() and clean[1:].islower():
            # Skip if first word of message or after sentence-ending punctuation
            if i == 0:
                continue
            prev = words[i - 1] if i > 0 else ""
            if prev.endswith((".", "!", "?", ":")):
                continue
            if clean.lower() not in _STOPWORDS and clean.lower() not in _FRAMING_WORDS:
                singles.append(clean)

    # Also detect ALL-CAPS acronyms (2-6 chars)
    acronyms = re.findall(r"\b([A-Z]{2,6})\b", text)
    acronyms = [a for a in acronyms if a not in ("I", "A")]

    # Quoted terms the user explicitly marked as entities
    quoted = re.findall(r'"([^"]{2,60})"', text)
    quoted += re.findall(r"'([^']{2,60})'", text)

    # Combine: quoted first (user intent), then multi-word, then acronyms, then singles
    all_entities: list[str] = []
    seen: set[str] = set()
    for e in quoted + multi + acronyms + singles:
        key = e.lower()
        if key not in seen:
            seen.add(key)
            all_entities.append(e)
    return all_entities


def _detect_question_type(lower: str) -> str | None:
    """Classify the question type to tailor query shape."""
    if re.match(r"^(how (to|do|does|can|could|should))\b", lower):
        return "howto"
    if re.match(r"^(what (is|are|was|were))\b", lower):
        return "definition"
    if re.match(r"^(who (is|are|was|were))\b", lower):
        return "person"
    if re.match(r"^(where|what .* located|location of)\b", lower):
        return "location"
    if re.match(r"^(when|what (date|time|year))\b", lower):
        return "temporal"
    if re.match(r"^(why|what .* reason|what causes?)\b", lower):
        return "causal"
    if re.search(r"\b(vs\.?|versus|compared? to|difference between|better)\b", lower):
        return "comparison"
    if re.search(r"\b(best|top|recommend|suggestion|review)\b", lower):
        return "recommendation"
    if re.search(r"\b(error|bug|issue|problem|fix|broken|crash|fail|won't|doesn't work)\b", lower):
        return "troubleshooting"
    return None


def _is_time_sensitive(lower: str) -> bool:
    """Check if the query is about current/recent information."""
    return bool(re.search(
        r"\b(latest|recent|current|today|now|this year|this month|this week|"
        r"right now|currently|2025|2026|new|upcoming|just released|breaking)\b",
        lower,
    ))


def generate_broad_queries(message: str, *, max_queries: int = 10) -> list[str]:
    """Generate on-topic search queries from a user message.

    Zero model calls — heuristic extraction with focused strategies.
    Every query stays tightly anchored to the user's actual topic.

    Design principles:
      - Core topic is king: strip question framing aggressively, keep subject.
      - Entities get quoted: multi-word proper nouns stay together in SearXNG.
      - No drift strategies: never generate queries from isolated keyword
        pairs or generic reformulations ("what is X", "X explained") that
        can match unrelated domains.
      - Fewer, better queries beat many scattered ones.
    """
    cleaned = re.sub(r"\s+", " ", message.strip())
    if not cleaned:
        return []

    lower = cleaned.lower()
    year = _current_year()
    time_sensitive = _is_time_sensitive(lower)

    # ---- Phase 1: Extract the core topic ----
    core = _strip_preamble(cleaned)
    if not core:
        core = cleaned  # preamble stripping ate everything — use raw

    # ---- Phase 2: Structural analysis ----
    entities = _detect_entities(cleaned)
    keywords = _extract_keywords(cleaned)
    phrases = _extract_phrases(cleaned)
    qtype = _detect_question_type(lower)

    # ---- Phase 3: Build queries, most specific first ----
    queries: list[str] = []

    # --- Q1: Core topic (cleaned of framing) ---
    # This is usually the single best query.
    queries.append(core)

    # --- Q2: Core topic with entities quoted ---
    # Keeps multi-word names together so SearXNG treats them as phrases.
    if entities:
        quoted_core = core
        for entity in entities:
            if entity in quoted_core and f'"{entity}"' not in quoted_core:
                quoted_core = quoted_core.replace(entity, f'"{entity}"', 1)
        if quoted_core != core:
            queries.append(quoted_core)

    # --- Q3: Entity-focused query ---
    # When entities are detected, a query built around just the entities
    # is often the most precise.
    if entities:
        entity_q = " ".join(
            f'"{e}"' if " " in e else e
            for e in entities[:3]
        )
        # Add topic keywords that aren't part of any entity
        entity_lower_set = {e.lower() for e in entities}
        extra_kw = [
            k for k in keywords
            if k.lower() not in entity_lower_set
            and not any(k.lower() in e.lower() for e in entities)
        ]
        if extra_kw:
            entity_q += " " + " ".join(extra_kw[:3])
        queries.append(entity_q.strip())

    # --- Q4: Longest content phrase quoted ---
    # The longest non-stopword run is likely the most specific topic chunk.
    if phrases:
        longest = max(phrases, key=len)
        if len(longest.split()) >= 2:
            queries.append(f'"{longest}"')

    # --- Q5: Question-type specific reformulation ---
    topic_str = " ".join(keywords[:5]) if keywords else core
    if qtype == "howto":
        queries.append(f"how to {topic_str}")
    elif qtype == "comparison" and len(entities) >= 2:
        queries.append(f'"{entities[0]}" vs "{entities[1]}"')
    elif qtype == "comparison" and len(keywords) >= 2:
        queries.append(f"{keywords[0]} vs {keywords[1]}")
    elif qtype == "recommendation":
        queries.append(f"best {topic_str}")
    elif qtype == "troubleshooting":
        queries.append(f"{topic_str} solution fix")
    elif qtype == "person" and entities:
        queries.append(f'"{entities[0]}" biography')
    elif qtype == "definition":
        queries.append(f"{topic_str} definition meaning")

    # --- Q6: Time-sensitive variant ---
    if time_sensitive:
        queries.append(f"{core} {year}")

    # --- Q7: Keywords-only (concise, no filler) ---
    # Useful when the natural-language query has noise SearXNG can't handle.
    if keywords and len(keywords) >= 2:
        kw_q = " ".join(keywords[:6])
        queries.append(kw_q)

    # --- Q8: Additional entity phrases quoted individually ---
    # Each distinct phrase becomes its own query for coverage.
    for p in phrases:
        if len(p.split()) >= 2:
            queries.append(f'"{p}"')

    # --- Q9: Core with domain/context qualifier ---
    # If we detected a question about something that commonly has
    # ambiguous names, add a domain hint derived from the message.
    _DOMAIN_SIGNALS = {
        "health": r"\b(symptom|treatment|disease|diagnosis|medical|health|doctor|patient|medication|dose|side effect)\b",
        "programming": r"\b(code|programming|function|library|framework|api|debug|compile|syntax|runtime|package|npm|pip)\b",
        "sports": r"\b(team|player|match|game|score|season|league|championship|tournament|coach|roster)\b",
        "finance": r"\b(stock|market|invest|price|dividend|earnings|portfolio|trading|crypto|bitcoin|etf)\b",
        "legal": r"\b(law|legal|court|judge|statute|regulation|compliance|attorney|lawsuit|rights)\b",
        "science": r"\b(research|study|experiment|hypothesis|theory|journal|peer.review|evidence|data|findings)\b",
        "cooking": r"\b(recipe|cook|ingredient|bake|fry|roast|dish|cuisine|meal|flavor)\b",
    }
    for domain, pattern in _DOMAIN_SIGNALS.items():
        if re.search(pattern, lower):
            # Only add if domain word not already in core
            if domain not in core.lower():
                queries.append(f"{core} {domain}")
            break  # one domain hint is enough

    # ---- Phase 4: Deduplicate and rank ----
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        q = q.strip()
        if not q:
            continue
        key = q.lower()
        # Skip if it's a pure subset of an already-added query
        if key in seen:
            continue
        # Skip very short queries (likely just a single common word)
        if len(q) < 3:
            continue
        seen.add(key)
        unique.append(q)

    result = unique[:max_queries]
    _log.info(
        "broad_queries generated  unique=%d capped=%d queries=%s",
        len(unique), len(result), result,
    )
    return result


def reformulate_query(
    original_query: str,
    intent_dict: dict,
) -> str | None:
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
        _log.info("query reformulate start")
        raw, _tokens = model_call("tool_planner", prompt, max_tokens=60, temperature=0.0)
        if not raw:
            return None
    except Exception as e:
        _log.warning("query reformulation failed: %s", e)
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
