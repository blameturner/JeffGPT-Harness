from __future__ import annotations

import json
import logging
import re
import threading
import time

import httpx

from config import CATEGORY_COLLECTIONS
from memory import remember
from workers.search.engine import FAST_TIMEOUT, _dedupe, searxng_search
from workers.search.extraction import extract_from_pages
from workers.search.intent import (
    SEARCH_POLICY_CONTEXTUAL,
    SEARCH_POLICY_FOCUSED,
    SEARCH_POLICY_FULL,
    SEARCH_POLICY_NONE,
    classify_message_intent,
)
from workers.search.models import _tool_model
from workers.search.queries import (
    build_failure_context,
    generate_search_queries,
    reformulate_query,
    rerank_candidates,
)
from workers.search.scraping import scrape_page

_log = logging.getLogger("web_search.orchestrator")


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
    from workers.styles import search_context_for

    queries_tried: list[str] = []
    queries = generate_search_queries(intent_dict, message=query)
    if not queries:
        _log.info("search skip  no queries generated")
        return "", [], "none"

    raw_results: list[dict] = []
    for q in queries:
        queries_tried.append(q)
        raw_results.extend(searxng_search(q, max_results=budget["max_candidates"]))
        if len(raw_results) >= budget["max_candidates"]:
            break
    raw_results = _dedupe(raw_results)[: budget["max_candidates"]]

    if raw_results:
        ranked = rerank_candidates(raw_results, intent_dict, max_candidates=budget["rerank_max"])
        kept = [c for c, score in ranked if score >= budget["rerank_drop_threshold"]]
    else:
        kept = []

    if not kept and raw_results:
        _log.info("rerank dropped all candidates — attempting reformulation")
        reformulated = reformulate_query(queries[0], intent_dict)
        if reformulated:
            queries_tried.append(reformulated)
            raw_results = _dedupe(searxng_search(reformulated, max_results=budget["max_candidates"]))
            if raw_results:
                ranked = rerank_candidates(raw_results, intent_dict, max_candidates=budget["rerank_max"])
                kept = [c for c, score in ranked if score >= budget["rerank_drop_threshold"]]

    if not raw_results:
        _log.info("searxng returned nothing — attempting reformulation")
        reformulated = reformulate_query(queries[0], intent_dict)
        if reformulated:
            queries_tried.append(reformulated)
            raw_results = _dedupe(searxng_search(reformulated, max_results=budget["max_candidates"]))
            if raw_results:
                ranked = rerank_candidates(raw_results, intent_dict, max_candidates=budget["rerank_max"])
                kept = [c for c, score in ranked if score >= budget["rerank_drop_threshold"]]

    if not kept:
        _log.warning("search failed  queries_tried=%s", queries_tried)
        context_block = build_failure_context(queries_tried, intent_dict)
        return context_block, [], "failed"

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

    assessments = extract_from_pages(
        scraped_pages,
        query,
        intent_dict,
        org_id=org_id,
        fire_graph_writes=True,
    )

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

    template_key = intent_dict.get("response_template") or "direct_answer"
    template_body = search_context_for(template_key)

    fact_blocks: list[str] = []
    for i, entry in enumerate(sorted_results, start=1):
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

    # Contextual enrichment has a hard latency cap; wrap the inner pipeline
    # in a future and return "deferred" on timeout. The background thread
    # continues — Python can't cancel blocking httpx cleanly.
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
            ex.shutdown(wait=False)

    return _run_search_inner(query, org_id, intent_dict, budget)


def _suggest_sources_from_search(
    summaries: list[tuple[str, str, str]],
    query: str,
    org_id: int,
) -> None:
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

    # Lazy import: enrichment_agent imports from web_search.
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
