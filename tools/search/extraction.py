from __future__ import annotations

import json
import logging
import re
import threading
import time

from infra.config import get_function_config, is_feature_enabled
from tools.search.intent import (
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
)
from shared.temporal import build_prompt_date_header

_log = logging.getLogger("web_search.extraction")


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
    intent = intent_dict.get("intent") or CHAT_INTENT_FACTUAL
    return _EXTRACTION_GOAL_BY_INTENT.get(
        intent,
        "Extract a factual summary of the page preserving key names, "
        "numbers, and dates. Max 250 words.",
    )


_ACCEPTABLE_CONTENT_TYPES = {
    "REFERENCE", "ARTICLE", "ENCYCLOPEDIC", "FORUM", "PRODUCT",
}
_SOFT_CONTENT_TYPES = {"UNCLEAR"}


def _extract_one_page(
    page_text: str,
    query: str,
    intent_dict: dict,
) -> dict | None:
    if not page_text or not page_text.strip():
        return None

    # deferred import: shared.quality pulls FalkorDB at import time
    try:
        from shared.quality import (
            _classify_content_type_heuristic,
            _heuristic_quality_gate,
        )
    except Exception as e:
        _log.warning("extraction helpers import failed: %s", e)
        return None

    passed, gate_reason, gate_metrics = _heuristic_quality_gate(page_text)
    if not passed:
        _log.debug(
            "extraction drop  gate=%s metrics=%s",
            gate_reason, gate_metrics,
        )
        return None

    content_type, _classifier_signals = _classify_content_type_heuristic(page_text)
    if content_type not in _ACCEPTABLE_CONTENT_TYPES and content_type not in _SOFT_CONTENT_TYPES:
        _log.debug("extraction drop  content_type=%s", content_type)
        return None

    cfg = get_function_config("search_extraction")
    goal = _extraction_goal_for(intent_dict)
    excerpt = page_text[:cfg.get("max_input_chars", 12000)]
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

    from shared.models import model_call
    raw, tokens = model_call("search_extraction", prompt)
    if not raw:
        _log.debug("extraction: empty response from model")
        return None

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
    try:
        from shared.relationships import _extract_relationships
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


_PASSTHROUGH_CHARS_PER_PAGE = 800
_SENTENCE_MIN_CHARS = 40
_SENTENCE_MAX_CHARS = 400
_HEAD_FALLBACK_CHARS = 600
_STOPWORD_SET = frozenset(
    "a an the and or of for to in on at from by with about as is are was were "
    "be been being it its this that these those what who when where why how "
    "i you we they he she them us our your their can could should would will "
    "do does did have has had".split()
)


def _query_terms(query: str, intent_dict: dict) -> set[str]:
    terms: set[str] = set()
    for w in re.findall(r"[A-Za-z0-9][\w\-']+", query.lower()):
        if len(w) >= 3 and w not in _STOPWORD_SET:
            terms.add(w)
    for e in intent_dict.get("entities") or []:
        for w in re.findall(r"[A-Za-z0-9][\w\-']+", str(e).lower()):
            if len(w) >= 2:
                terms.add(w)
    return terms


def _heuristic_relevance(text: str, terms: set[str]) -> str:
    if not terms:
        return "medium"
    lower = text.lower()
    hits = sum(1 for t in terms if t in lower)
    ratio = hits / len(terms)
    if ratio >= 0.5 or hits >= 4:
        return "high"
    if ratio >= 0.25 or hits >= 2:
        return "medium"
    return "low"


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_JUNK_SENTENCE_HINTS = (
    "cookie", "subscribe", "sign in", "log in", "click here", "read more",
    "all rights reserved", "privacy policy", "terms of service",
    "accept all", "manage preferences",
)


def _split_sentences(text: str) -> list[str]:
    """Best-effort sentence splitter. Also splits on newlines so scraped text
    with line-break-separated paragraphs still decomposes.
    """
    out: list[str] = []
    for chunk in text.split("\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = _SENTENCE_SPLIT.split(chunk)
        for p in parts:
            p = p.strip()
            if p:
                out.append(p)
    return out


def _select_relevant_excerpt(
    text: str,
    terms: set[str],
    entity_terms: set[str],
    char_budget: int,
) -> str:
    """Pick the most query-relevant sentences up to ``char_budget``.

    Scoring per sentence:
      +2 per entity term hit (distinct) — entities are highest-signal
      +1 per other query term hit (distinct)
      +1 bonus if the sentence contains a digit (stats, dates, versions)
      -3 if the sentence looks like boilerplate/CTA

    Sentences with score <= 0 are skipped unless the doc has *no* positive
    sentences at all, in which case we fall back to the first chars of text
    so the main model still gets something.
    """
    if not text:
        return ""
    if not terms and not entity_terms:
        return text[:char_budget].strip()

    sentences = _split_sentences(text)
    scored: list[tuple[int, int, str]] = []  # (score, original_index, sentence)
    for i, s in enumerate(sentences):
        if len(s) < _SENTENCE_MIN_CHARS or len(s) > _SENTENCE_MAX_CHARS:
            continue
        lower = s.lower()
        if any(j in lower for j in _JUNK_SENTENCE_HINTS):
            continue
        score = 0
        for t in entity_terms:
            if t in lower:
                score += 2
        for t in terms:
            if t in entity_terms:
                continue
            if t in lower:
                score += 1
        if score <= 0:
            continue
        if re.search(r"\d", s):
            score += 1
        scored.append((score, i, s))

    if not scored:
        return text[:_HEAD_FALLBACK_CHARS].strip()

    # sort by score desc, then original doc order to keep high-signal early sentences
    scored.sort(key=lambda t: (-t[0], t[1]))

    picked: list[tuple[int, str]] = []
    used = 0
    for score, idx, sent in scored:
        cost = len(sent) + 1  # joining space
        if used + cost > char_budget and picked:
            break
        picked.append((idx, sent))
        used += cost
        if used >= char_budget:
            break

    # re-sort selected sentences by original position so the excerpt reads naturally
    picked.sort(key=lambda t: t[0])
    return " ".join(s for _, s in picked).strip()


def _guess_source_type(url: str) -> str:
    host = ""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc.lower()
    except Exception:
        return "unknown"
    if not host:
        return "unknown"
    if host.endswith(".gov") or ".gov." in host:
        return "government"
    if "wikipedia.org" in host:
        return "encyclopedic"
    if any(p in host for p in ("docs.", "developer.", "developers.", "/docs")):
        return "official_docs"
    if any(p in host for p in ("stackoverflow.com", "reddit.com", "news.ycombinator", "discourse.")):
        return "forum"
    if any(p in host for p in ("arxiv.org", "pubmed", "nature.com", "sciencedirect")):
        return "research_paper"
    if any(p in host for p in ("nytimes.com", "bbc.", "reuters.com", "theguardian.", "bloomberg.", "apnews.")):
        return "news_article"
    if any(p in host for p in (".shop", "amazon.", "ebay.", "store.")):
        return "product_page"
    return "blog_post"


def extract_from_pages(
    pages: list[dict],
    query: str,
    intent_dict: dict,
    org_id: int | None = None,
    fire_graph_writes: bool = True,
    function_name: str = "search_extraction",
) -> list[dict | None]:
    """Passthrough extractor: no LLM summarisation.

    Runs the heuristic quality gate and content-type classifier, then trims the
    scraped text to a per-page char budget and forwards it verbatim as the
    summary. Relevance is scored by query-term overlap. The main model gets
    the raw scraped excerpts and does the synthesis itself.
    """
    if not pages:
        return []

    try:
        from shared.quality import (
            _classify_content_type_heuristic,
            _heuristic_quality_gate,
        )
    except Exception as e:
        _log.warning("extraction helpers import failed: %s", e)
        return [None] * len(pages)

    started = time.time()
    results: list[dict | None] = [None] * len(pages)
    terms = _query_terms(query, intent_dict)
    entity_terms: set[str] = set()
    for e in intent_dict.get("entities") or []:
        for w in re.findall(r"[A-Za-z0-9][\w\-']+", str(e).lower()):
            if len(w) >= 2:
                entity_terms.add(w)
    accepted = 0
    dropped_gate = 0
    dropped_type = 0

    for i, page in enumerate(pages):
        text = page.get("text", "")
        if not text or not text.strip():
            continue

        passed, gate_reason, gate_metrics = _heuristic_quality_gate(text)
        if not passed:
            dropped_gate += 1
            _log.debug("extraction drop  gate=%s metrics=%s", gate_reason, gate_metrics)
            continue

        content_type, _cls_signals = _classify_content_type_heuristic(text)
        if content_type not in _ACCEPTABLE_CONTENT_TYPES and content_type not in _SOFT_CONTENT_TYPES:
            dropped_type += 1
            _log.debug("extraction drop  content_type=%s", content_type)
            continue

        result = page.get("result", {}) if isinstance(page.get("result"), dict) else {}
        url = result.get("url", "")
        summary = _select_relevant_excerpt(
            text, terms, entity_terms, _PASSTHROUGH_CHARS_PER_PAGE,
        )
        if not summary:
            continue

        results[i] = {
            "summary": summary,
            "relevance": _heuristic_relevance(summary, terms),
            "source_type": _guess_source_type(url),
            "content_type": content_type,
            "tokens": 0,
        }
        accepted += 1

    elapsed = round(time.time() - started, 2)
    _log.info(
        "extract_from_pages passthrough  pages=%d accepted=%d dropped_gate=%d "
        "dropped_type=%d chars_per_page=%d %.2fs",
        len(pages), accepted, dropped_gate, dropped_type,
        _PASSTHROUGH_CHARS_PER_PAGE, elapsed,
    )

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
