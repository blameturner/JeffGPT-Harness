from __future__ import annotations

import json
import logging
import re
import time

from tools.search.intent import (
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
from shared.models import model_call
from shared.temporal import build_prompt_date_header, now_in_chat_tz

_log = logging.getLogger("web_search.queries")


def _quote_entity(entity: str) -> str:
    # must phrase-quote multi-word entities or searxng tokenises them into independent words
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

    # stable sort preserves searxng order on score ties
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

# kept separate from stopwords so comparison queries can still use "compare"
_FRAMING_WORDS: frozenset[str] = frozenset(
    "tell explain describe discuss define help show give list name "
    "know want need like think say said going go make made "
    "handle find look search check create run start stop try "
    "please kindly wondering curious looking "
    "okay ok cool sure surely yes yeah right nice great awesome "
    "thanks thank hey hi hello well anyway actually basically".split()
)


def _extract_keywords(message: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9][\w\-.']*[a-zA-Z0-9]|[a-zA-Z0-9]", message)
    noise = _STOPWORDS | _FRAMING_WORDS
    return [w for w in words if w.lower() not in noise and len(w) > 1]


def _extract_phrases(message: str) -> list[str]:
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


# order matters: longer/more-specific patterns first so they eat the full preamble
_QUESTION_PREAMBLES: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^(?:can|could|would) you (?:please |kindly )?(?:tell|show|explain|describe|find|look up|search|help(?: me)?(?: with)?|give me(?: info(?:rmation)? (?:on|about))?)\s*(?:me )?\s*",
        r"^(?:please |kindly )?(?:tell|show|explain|describe|find|look up|search|help(?: me)?(?: with)?|give me(?: info(?:rmation)? (?:on|about))?)\s*(?:me )?\s*",
        r"^i(?:'m| am) (?:looking for|curious about|wondering about|interested in|trying to (?:find|understand|learn(?: about)?))\s*",
        r"^i (?:want|need|would like) to (?:know|learn|understand|find out)(?: (?:more )?about)?\s*",
        r"^(?:do you know|have you heard)(?: anything)? about\s*",
        r"^(?:what do you know about|what can you tell me about)\s*",
        r"^what (?:is|are|was|were|does|do|did|would|could|should)(?: (?:a|an|the))?\s*",
        r"^who (?:is|are|was|were)\s*",
        r"^where (?:is|are|was|were|can|do|does)\s*",
        r"^when (?:is|are|was|were|did|does|will)\s*",
        r"^why (?:is|are|was|were|did|does|do|don't|doesn't|won't|can't)\s*",
        r"^how (?:does|do|did|is|are|can|could|to|would|should)\s*",
        r"^(?:is|are|was|were|does|do|did|has|have|will|can|could|should|would) (?:the |a |an )?\s*",
    ]
]

_LEADING_FLUFF = re.compile(
    r"^(?:the |a |an |about |for |in |on |of |to |with |by |from |regarding |concerning |re: )+",
    re.IGNORECASE,
)

_TRAILING_FLUFF = re.compile(r"[?.!,;:]+$")


_FILLER_SENTENCE = re.compile(
    r"^(okay|ok|cool|sure|surely|yes|yeah|right|nice|great|awesome|thanks|"
    r"hey|hi|hello|well|anyway|actually|basically|alright|got it|sounds good|"
    r"i would like to work with you|i want to work with you|let's work on)"
    r"[^.!?]*[.!?]\s*",
    re.IGNORECASE,
)
_TRAILING_FILLER = re.compile(
    r"[.!?\s]+(surely|certainly|right|yeah|yes|please|thanks|"
    r"can you|could you|would you|I'?m sure|there must be|"
    r"there are more|you can find|let me know)[^.]*[.!?]?\s*$",
    re.IGNORECASE,
)


def _strip_preamble(text: str) -> str:
    result = text
    for _ in range(3):
        m = _FILLER_SENTENCE.match(result)
        if m:
            result = result[m.end():]
        else:
            break
    result = _TRAILING_FILLER.sub("", result).strip()
    for pat in _QUESTION_PREAMBLES:
        prev = result
        result = pat.sub("", result, count=1)
        if result != prev:
            break
    result = _LEADING_FLUFF.sub("", result).strip()
    result = _TRAILING_FLUFF.sub("", result).strip()
    return result


def _detect_entities(text: str) -> list[str]:
    multi = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
    words = text.split()
    singles: list[str] = []
    for i, w in enumerate(words):
        clean = re.sub(r"[^a-zA-Z]", "", w)
        if not clean or len(clean) < 2:
            continue
        if clean[0].isupper() and clean[1:].islower():
            # skip sentence-initial capitalisation — not a proper noun signal
            if i == 0:
                continue
            prev = words[i - 1] if i > 0 else ""
            if prev.endswith((".", "!", "?", ":")):
                continue
            if clean.lower() not in _STOPWORDS and clean.lower() not in _FRAMING_WORDS:
                singles.append(clean)

    acronyms = re.findall(r"\b([A-Z]{2,6})\b", text)
    acronyms = [a for a in acronyms if a not in ("I", "A")]

    quoted = re.findall(r'"([^"]{2,60})"', text)
    quoted += re.findall(r"'([^']{2,60})'", text)

    # order matters: user-quoted first, then multi-word, acronyms, singles
    all_entities: list[str] = []
    seen: set[str] = set()
    for e in quoted + multi + acronyms + singles:
        key = e.lower()
        if key not in seen:
            seen.add(key)
            all_entities.append(e)
    return all_entities


def _detect_question_type(lower: str) -> str | None:
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
    return bool(re.search(
        r"\b(latest|recent|current|today|now|this year|this month|this week|"
        r"right now|currently|2025|2026|new|upcoming|just released|breaking)\b",
        lower,
    ))


def generate_broad_queries(message: str, *, max_queries: int = 10, conversation_topics: list[str] | None = None) -> list[str]:
    cleaned = re.sub(r"\s+", " ", message.strip())
    if not cleaned:
        return []

    lower = cleaned.lower()
    year = _current_year()
    time_sensitive = _is_time_sensitive(lower)

    core = _strip_preamble(cleaned)
    if not core:
        core = cleaned

    entities = _detect_entities(cleaned)
    keywords = _extract_keywords(cleaned)
    phrases = _extract_phrases(cleaned)
    qtype = _detect_question_type(lower)

    # paragraph-length cores are too noisy for searxng; fall back to entity/keyword queries
    long_message = len(core) > 150

    queries: list[str] = []

    if not long_message:
        queries.append(core)

    if entities and not long_message:
        quoted_core = core
        for entity in entities:
            if entity in quoted_core and f'"{entity}"' not in quoted_core:
                quoted_core = quoted_core.replace(entity, f'"{entity}"', 1)
        if quoted_core != core:
            queries.append(quoted_core)

    if entities:
        entity_q = " ".join(
            f'"{e}"' if " " in e else e
            for e in entities[:3]
        )
        entity_lower_set = {e.lower() for e in entities}
        extra_kw = [
            k for k in keywords
            if k.lower() not in entity_lower_set
            and not any(k.lower() in e.lower() for e in entities)
        ]
        if extra_kw:
            entity_q += " " + " ".join(extra_kw[:3])
        queries.append(entity_q.strip())

    if phrases:
        longest = max(phrases, key=len)
        if len(longest.split()) >= 2:
            queries.append(f'"{longest}"')

    # skip qtype reformulation on long messages — it emits nonsense like "Hetzner vs Intel Xeon"
    topic_str = " ".join(keywords[:5]) if keywords else core
    if long_message:
        pass
    elif qtype == "howto":
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

    if time_sensitive:
        queries.append(f"{core} {year}")

    if keywords and len(keywords) >= 2:
        kw_q = " ".join(keywords[:6])
        queries.append(kw_q)

    for p in phrases:
        if len(p.split()) >= 2:
            queries.append(f'"{p}"')

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
            if domain not in core.lower():
                queries.append(f"{core} {domain}")
            break

    if conversation_topics:
        core_lower = core.lower()
        new_topics = [t for t in conversation_topics if t not in core_lower]
        if new_topics:
            topic_suffix = " ".join(new_topics[:3])
            queries.append(f"{core} {topic_suffix}")
            if keywords:
                kw_topic = " ".join(keywords[:4]) + " " + " ".join(new_topics[:2])
                queries.append(kw_topic)

    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        q = q.strip()
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
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


def _condense_long_message(message: str) -> str:
    """For messages over 200 chars, extract a condensed searchable form.

    Strips conversational preamble/context and returns just the core question
    or topic the user actually wants searched. Falls back to the first 200 chars
    of the cleaned message if the message is short enough.
    """
    if len(message) <= 200:
        return message
    # Take up to first 600 chars to keep prompt small, then strip to last sentence
    # that ends with '?' or use the stripped preamble of the whole thing
    head = message[:600]
    # Prefer the last sentence ending in '?' — that's usually the real question
    sentences = re.split(r"(?<=[.!?])\s+", head)
    questions = [s.strip() for s in sentences if "?" in s]
    if questions:
        return questions[-1][:300]
    # Otherwise return stripped preamble of full message, capped at 300 chars
    stripped = _strip_preamble(message[:400])
    return stripped[:300] if stripped else message[:200]


def generate_llm_queries(
    user_message: str,
    *,
    conversation_topics: list[str] | None = None,
    count_min: int = 10,
    count_max: int = 20,
) -> list[str]:
    """LLM-driven search query fan-out for the "standard" search mode.

    Asks the query-generator model for `count_max` diverse queries covering
    direct phrasing, alt phrasings, entity-focused, time-anchored, and
    comparison/context angles. Returns a deduped list truncated to `count_max`.

    Falls back to `generate_broad_queries` on any model failure, empty output,
    parse failure, or fewer than `count_min` usable queries. Never raises.
    """
    cleaned = re.sub(r"\s+", " ", (user_message or "").strip())
    if not cleaned:
        return []

    # For long conversational messages, condense to the actual searchable question
    # to avoid the LLM latching onto surface words (e.g. "rebuild" → rebuilding articles)
    search_focus = _condense_long_message(cleaned)

    # Also extract explicit tech/proper-noun entities from the full message
    # so the LLM knows to treat them as specific products/frameworks, not generic words
    all_entities = _detect_entities(cleaned)
    entities_hint = ""
    if all_entities:
        entities_hint = (
            f"Key named entities/technologies in the message (treat as specific proper nouns, "
            f"NOT generic words): {', '.join(all_entities[:10])}\n"
        )

    topics_line = ""
    if conversation_topics:
        topics_line = f"Prior conversation topics: {', '.join(list(conversation_topics)[:6])}\n"

    # For long messages, only send the condensed focus to the model — not the full
    # noisy message. Sending both lets the model latch onto surface words in the
    # full text and ignore the focus hint (especially with smaller models).
    is_long = len(cleaned) > 200
    message_line = (
        f"Search topic: {search_focus}\n"
        if is_long
        else f"User question: {cleaned}\n"
    )

    prompt = (
        f"{build_prompt_date_header()}\n\n"
        f"{topics_line}"
        f"{entities_hint}"
        f"{message_line}\n"
        f"Generate {count_max} diverse web search queries to find information that would help answer this. "
        f"Focus on the SPECIFIC technologies, products, and concepts mentioned — do NOT generate queries "
        f"about unrelated things that share common words.\n\n"
        "Cover:\n"
        "- the direct question phrased for a search engine\n"
        "- specific named technologies/frameworks/libraries mentioned (use exact names)\n"
        "- alternative phrasings (different keywords, same intent)\n"
        "- comparison or 'best practice' angles for any tech stack decisions\n"
        "- time-anchored queries when the topic is time-sensitive (add the year)\n\n"
        "Rules:\n"
        "- Each query: 3-10 words, optimised for searxng/Google.\n"
        "- Use exact product/technology names — never substitute similar-sounding unrelated words.\n"
        "- No duplicates. No meta-commentary.\n"
        "- If the question is conversational advice-seeking with no clear factual target, "
        "  generate queries about the specific technologies and patterns in the entities list.\n"
        "- Output ONLY a JSON array of strings. No markdown fences, no prose.\n\n"
        "Example output: [\"TanStack Start SSR architecture 2026\", \"Hono API gateway monorepo\", ...]"
    )

    t0 = time.time()
    try:
        raw, _tokens = model_call("web_search_query_generator", prompt)
    except Exception as e:
        _log.warning("llm_queries model_call failed: %s — falling back to broad queries", e)
        return generate_broad_queries(
            cleaned, max_queries=count_max, conversation_topics=conversation_topics,
        )

    if not raw:
        _log.warning("llm_queries empty response — falling back to broad queries")
        return generate_broad_queries(
            cleaned, max_queries=count_max, conversation_topics=conversation_topics,
        )

    # strip accidental markdown fencing before JSON parse
    body = raw.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[^\n]*\n", "", body).rstrip("`").strip()
    match = re.search(r"\[[\s\S]*\]", body)
    parsed: list[str] = []
    if match:
        try:
            arr = json.loads(match.group(0))
            if isinstance(arr, list):
                parsed = [str(q).strip() for q in arr if str(q).strip()]
        except Exception:
            parsed = []

    seen: set[str] = set()
    unique: list[str] = []
    for q in parsed:
        key = q.lower()
        if key in seen or len(q) < 3:
            continue
        seen.add(key)
        unique.append(q)

    if len(unique) < count_min:
        _log.warning(
            "llm_queries parsed %d < min %d — falling back to broad queries  elapsed=%.2fs",
            len(unique), count_min, time.time() - t0,
        )
        return generate_broad_queries(
            cleaned, max_queries=count_max, conversation_topics=conversation_topics,
        )

    result = unique[:count_max]
    _log.info(
        "llm_queries generated  parsed=%d kept=%d elapsed=%.2fs",
        len(unique), len(result), time.time() - t0,
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
