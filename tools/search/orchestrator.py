from __future__ import annotations

import logging
import time

from infra.memory import remember
from tools.search.engine import _dedupe, searxng_search
from tools.search.extraction import extract_from_pages
from tools.search.intent import (
    SEARCH_POLICY_CONTEXTUAL,
    SEARCH_POLICY_FOCUSED,
    SEARCH_POLICY_FULL,
    SEARCH_POLICY_NONE,
    classify_message_intent,
)
from tools.search.queries import (
    build_failure_context,
    generate_search_queries,
    reformulate_query,
    rerank_candidates,
)
from tools.search.scraping import scrape_page

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
        "max_candidates": 30,
        "rerank_max": 20,
        "max_scrape": 12,
        "rerank_drop_threshold": 2,
        "hard_cap_s": 60.0,
    },
}


def _run_search_inner(
    query: str,
    org_id: int,
    intent_dict: dict,
    budget: dict,
    extraction_function_name: str = "search_extraction",
) -> tuple[str, list[dict], str]:
    from tools.search.config import search_context_for

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
        if not kept and ranked:
            # Do not hard-fail when reranker is overly strict/noisy; keep a small
            # best-effort head so downstream scrape+extraction can decide quality.
            fallback_keep = min(3, len(ranked))
            kept = [c for c, _score in ranked[:fallback_keep]]
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
                if not kept and ranked:
                    fallback_keep = min(3, len(ranked))
                    kept = [c for c, _score in ranked[:fallback_keep]]

    if not raw_results:
        _log.info("searxng returned nothing — attempting reformulation")
        reformulated = reformulate_query(queries[0], intent_dict)
        if reformulated:
            queries_tried.append(reformulated)
            raw_results = _dedupe(searxng_search(reformulated, max_results=budget["max_candidates"]))
            if raw_results:
                ranked = rerank_candidates(raw_results, intent_dict, max_candidates=budget["rerank_max"])
                kept = [c for c, score in ranked if score >= budget["rerank_drop_threshold"]]
                if not kept and ranked:
                    fallback_keep = min(3, len(ranked))
                    kept = [c for c, _score in ranked[:fallback_keep]]

    if not kept:
        _log.warning("search failed  queries_tried=%s", queries_tried)
        context_block = build_failure_context(queries_tried, intent_dict)
        return context_block, [], "failed"

    to_scrape = kept[: budget["max_scrape"]]
    scraped_pages: list[dict] = []
    if to_scrape:
        import concurrent.futures as _futures

        scrape_started = time.time()
        workers = min(len(to_scrape), 8)
        # Preserve input order so downstream rankers stay stable.
        indexed: list[tuple[int, str | None]] = [(i, None) for i in range(len(to_scrape))]
        with _futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="scrape"
        ) as pool:
            futs = {
                pool.submit(scrape_page, r["url"], r.get("snippet", "")): i
                for i, r in enumerate(to_scrape)
            }
            for fut in _futures.as_completed(futs):
                i = futs[fut]
                try:
                    text = fut.result()
                except Exception as e:
                    _log.warning("scrape error  url=%s error=%s", to_scrape[i]["url"][:80], e)
                    text = ""
                indexed[i] = (i, text)

        for i, text in indexed:
            if not text:
                continue
            scraped_pages.append({"result": to_scrape[i], "text": text})

        _log.info(
            "scrape batch  urls=%d workers=%d ok=%d %.2fs",
            len(to_scrape), workers, len(scraped_pages),
            round(time.time() - scrape_started, 2),
        )

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
        function_name=extraction_function_name,
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

    return context_block, sources, confidence


def run_web_search(
    query: str,
    org_id: int,
    intent_dict: dict | None = None,
    history: list[dict] | None = None,
    extraction_function_name: str = "search_extraction",
) -> tuple[str, list[dict], str]:
    _log.debug("search start  query=%s org=%d", query[:100], org_id)

    if intent_dict is None:
        intent_dict = classify_message_intent(query, history=history)
    if not isinstance(intent_dict, dict):
        _log.warning("search intent resolution returned non-dict: %r", type(intent_dict).__name__)
        return "", [], "failed"
    resolved_intent = intent_dict

    policy = resolved_intent.get("search_policy") or SEARCH_POLICY_NONE
    if policy == SEARCH_POLICY_NONE:
        return "", [], "none"

    budget = _POLICY_BUDGETS.get(policy)
    if not budget:
        _log.warning("unknown search policy %r, defaulting to focused", policy)
        budget = _POLICY_BUDGETS[SEARCH_POLICY_FOCUSED]

    # contextual: hard latency cap via future; the bg thread keeps running (can't cancel blocking httpx)
    if policy == SEARCH_POLICY_CONTEXTUAL:
        import concurrent.futures as _futures
        ex = _futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="contextual-search")
        try:
            fut = ex.submit(_run_search_inner, query, org_id, resolved_intent, budget, extraction_function_name)
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

    return _run_search_inner(query, org_id, resolved_intent, budget, extraction_function_name)

