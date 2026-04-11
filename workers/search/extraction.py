from __future__ import annotations

import json
import logging
import re
import threading
import time

from workers.search.engine import SUMMARY_INPUT_CHAR_CAP, SUMMARY_MAX_TOKENS
from workers.search.intent import (
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
from workers.search.temporal import build_prompt_date_header

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

    # deferred: workers.enrichment pulls FalkorDB/NocoDB/crawler at import time
    try:
        from workers.enrichment.models import _fast_call
        from workers.enrichment.quality import (
            _classify_content_type,
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

    content_type, _classifier_raw, _classifier_tokens = _classify_content_type(page_text)
    if content_type is None:
        content_type = "UNCLEAR"
    if content_type not in _ACCEPTABLE_CONTENT_TYPES and content_type not in _SOFT_CONTENT_TYPES:
        _log.debug("extraction drop  content_type=%s", content_type)
        return None

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
        from workers.enrichment.relationships import _extract_relationships
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
    if not pages:
        return []

    try:
        from workers.crawler import fan_out
    except Exception:
        _log.warning("fan_out unavailable — falling back to sequential extraction")
        return [_extract_one_page(p.get("text", ""), query, intent_dict) for p in pages]

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
