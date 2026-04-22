from __future__ import annotations

import asyncio
import logging
import re
import time

from infra.config import get_feature, get_function_config
from tools.contract import ToolName, ToolResult
from tools.dispatcher import register_executor
from tools.search.engine import PER_PAGE_CHAR_CAP, searxng_search
_log = logging.getLogger("tools.web_search")

# Last-resort defaults — every use site reads the live value from config via
# _cfg_int() so operators can retune without a redeploy.
BASIC_MAX_URLS = 8
STANDARD_MAX_URLS = 8
SUPPLEMENTARY_EXTRACT_CHARS = 1200
PRIORITY_EXTRACT_CHARS = min(6000, PER_PAGE_CHAR_CAP)
MAX_SUMMARY_CHARS = 1500
SEARXNG_PER_QUERY = 10
BATCH_TARGET = 5
PRIORITY_URLS = 2
RERANK_WHEN_CANDIDATES_OVER = 5
RERANK_MAX = 8
BASIC_QUERY_CAP = 5
STANDARD_QUERY_CAP = 3
RELEVANCE_KEYWORD_THRESHOLD = 0.16
RELEVANCE_MIN_ABS_HITS = 2
RELEVANCE_MIN_KEEP = 4
# Per-URL hard cap: httpx (10s) + playwright fallback (25s) worst-case is ~35s;
# we cut the tail at 28s so one stuck URL can never stall its whole tier.
SCRAPE_DEADLINE_S = 28
# basic mode only: if SearXNG's snippet is already this long, skip the fetch
# and summarise the snippet directly — trades a little depth for a lot of speed.
BASIC_SNIPPET_DIRECT_MIN_CHARS = 400


def _cfg_int(key: str, default: int) -> int:
    try:
        return int(get_feature("web_search", key, default) or default)
    except (TypeError, ValueError):
        return default


async def _search_one(query: str) -> list[dict]:
    per_query = _cfg_int("searxng_per_query", SEARXNG_PER_QUERY)
    try:
        results = await asyncio.to_thread(
            searxng_search, query, per_query,
        )
    except Exception as e:
        _log.warning("searxng failed q=%r: %s", query[:80], e)
        return []
    return results or []


async def _search_all(queries: list[str], url_cap: int) -> list[dict]:
    all_results = await asyncio.gather(*[_search_one(q) for q in queries])

    seen: set[str] = set()
    deduped: list[dict] = []
    for result_set in all_results:
        for r in result_set:
            url = (r.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            deduped.append(r)
    return deduped[:url_cap]


def _filter_results_by_relevance(
    results: list[dict],
    queries: list[str],
) -> list[dict]:
    if not results or RELEVANCE_KEYWORD_THRESHOLD <= 0:
        return results

    from tools.search.queries import _extract_keywords
    keyword_sets: list[set[str]] = []
    for q in queries:
        kws = {
            kw.lower() for kw in _extract_keywords(q)
            if len(kw) > 2 and not kw.isdigit()
        }
        if kws:
            keyword_sets.append(kws)

    if not keyword_sets:
        return results

    threshold = float(get_feature("web_search", "relevance_keyword_threshold", RELEVANCE_KEYWORD_THRESHOLD) or RELEVANCE_KEYWORD_THRESHOLD)
    min_abs_hits = int(get_feature("web_search", "relevance_min_abs_hits", RELEVANCE_MIN_ABS_HITS) or RELEVANCE_MIN_ABS_HITS)
    min_keep_cfg = int(get_feature("web_search", "relevance_min_keep", RELEVANCE_MIN_KEEP) or RELEVANCE_MIN_KEEP)

    kept: list[dict] = []
    scored: list[tuple[dict, float, int]] = []
    dropped = 0
    for r in results:
        haystack = f"{r.get('title', '')} {r.get('snippet', '')}".lower()
        best_ratio = 0.0
        best_hits = 0
        for kws in keyword_sets:
            hits = sum(1 for kw in kws if kw in haystack)
            ratio = hits / max(1, len(kws))
            if ratio > best_ratio or (ratio == best_ratio and hits > best_hits):
                best_ratio = ratio
                best_hits = hits

        scored.append((r, best_ratio, best_hits))

        if best_ratio >= threshold or best_hits >= min_abs_hits:
            kept.append(r)
        else:
            dropped += 1
            _log.debug(
                "relevance_filter drop  url=%s ratio=%.2f hits=%d",
                (r.get("url") or "")[:80], best_ratio, best_hits,
            )

    min_keep = max(1, min(min_keep_cfg, len(results)))
    if len(kept) < min_keep and scored:
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        fallback = [row for row, _ratio, _hits in scored[:min_keep]]
        kept_map = {(r.get("url") or ""): r for r in kept}
        for r in fallback:
            u = r.get("url") or ""
            if u not in kept_map:
                kept.append(r)
                kept_map[u] = r

    if dropped:
        _log.info(
            "relevance_filter  kept=%d dropped=%d query_sets=%d threshold=%.2f min_hits=%d min_keep=%d",
            len(kept), dropped, len(keyword_sets), threshold, min_abs_hits, min_keep,
        )
    return kept


async def _scrape_one(item: dict, extract_cap: int) -> dict:
    from tools.search.scraping import scrape_page

    url = item["url"]
    snippet = item.get("snippet") or ""
    title = item.get("title") or url

    meta: dict = {}
    try:
        text = await asyncio.wait_for(
            asyncio.to_thread(scrape_page, url, snippet, None, meta),
            timeout=SCRAPE_DEADLINE_S,
        )
    except asyncio.TimeoutError:
        _log.warning("scrape deadline exceeded (%ds)  url=%s", SCRAPE_DEADLINE_S, url[:80])
        text = snippet
        meta["path"] = "deadline"
    except Exception as e:
        _log.warning("scrape_page raised for %s: %s", url, e)
        text = snippet
        meta["path"] = "error"

    cap = min(max(1, extract_cap), PER_PAGE_CHAR_CAP)
    return {
        "url": url,
        "title": title,
        "text": (text or "")[:cap],
        "path": meta.get("path", "unknown"),
    }


def _parse_relevance(text: str) -> tuple[str, str]:
    lines = text.strip().rsplit("\n", 1)
    relevance = "medium"
    summary = text.strip()
    if len(lines) == 2 and "RELEVANCE:" in lines[1].upper():
        summary = lines[0].strip()
        tag = lines[1].upper().replace("RELEVANCE:", "").strip()
        if "HIGH" in tag:
            relevance = "high"
        elif "LOW" in tag:
            relevance = "low"
    return summary, relevance


async def _summarise_one(
    url: str,
    text: str,
    user_query: str,
    function_name: str = "web_search_summarise",
    priority: bool = True,
) -> dict:
    max_summary = _cfg_int("max_summary_chars", MAX_SUMMARY_CHARS)
    if len(text) < 100:
        return {"summary": text[:max_summary], "relevance": "low", "source_type": "snippet"}

    cfg = get_function_config(function_name)
    max_input = cfg.get("max_input_chars", 12000)

    prompt = (
        f"Summarise this page for: {user_query}\n"
        f"- Under 200 words.\n"
        f"- Facts, numbers, dates, names only.\n"
        f"- Last line only: RELEVANCE: high|medium|low\n\n"
        f"URL: {url}\n{text[:max_input]}"
    )

    try:
        from shared.models import model_call
        raw, _tokens = await asyncio.to_thread(
            model_call, function_name, prompt, priority,
        )
        if not raw:
            return {"summary": text[:max_summary], "relevance": "low", "source_type": "unknown"}

        summary, relevance = _parse_relevance(raw)
        _log.info("summarise ok  url=%s chars=%d relevance=%s", url[:80], len(summary), relevance)
        return {"summary": summary[:max_summary], "relevance": relevance, "source_type": "article"}
    except Exception as e:
        _log.warning("summarise failed  url=%s: %s %r", url[:80], type(e).__name__, e)
        return {"summary": text[:max_summary], "relevance": "low", "source_type": "unknown"}


def _build_batches(
    pages: list[dict],
    max_input: int,
    batch_target: int,
) -> list[list[dict]]:
    prompt_overhead = 400
    per_page_overhead = 80
    budget = max_input - prompt_overhead

    batches: list[list[dict]] = []
    current: list[dict] = []
    current_chars = 0

    for page in pages:
        page_chars = len(page["text"]) + per_page_overhead
        fits = (current_chars + page_chars) <= budget
        under_target = len(current) < batch_target

        if current and (not fits or not under_target):
            batches.append(current)
            current = []
            current_chars = 0

        current.append(page)
        current_chars += page_chars

    if current:
        batches.append(current)

    return batches


async def _summarise_batch(
    pages: list[dict],
    user_query: str,
) -> list[dict]:
    if len(pages) == 1:
        return [await _summarise_one(pages[0]["url"], pages[0]["text"], user_query)]

    cfg = get_function_config("web_search_summarise_batch")
    max_input = cfg.get("max_input_chars", 14000)

    sections: list[str] = []
    for i, p in enumerate(pages, 1):
        sections.append(f"--- PAGE {i} ---\nURL: {p['url']}\n{p['text']}")
    body = "\n\n".join(sections)

    prompt = (
        f"Summarise each page for: {user_query}\n"
        f"- Each section starts 'PAGE N:'\n"
        f"- 3 bullets, under 50 words per page.\n"
        f"- Facts, numbers, dates, names only.\n"
        f"- End each section with RELEVANCE: high|medium|low\n\n"
        f"{body}"
    )

    try:
        from shared.models import model_call
        raw, _tokens = await asyncio.to_thread(
            model_call, "web_search_summarise_batch", prompt, True,
        )
        if not raw:
            _log.warning("batch summarise empty response — falling back to individual")
            return await _fallback_individual(pages, user_query)

        results = _parse_batch_response(raw, len(pages))
        if len(results) == len(pages):
            for p, r in zip(pages, results):
                _log.info(
                    "batch_summarise ok  url=%s chars=%d relevance=%s",
                    p["url"][:80], len(r["summary"]), r["relevance"],
                )
            return results

        _log.warning(
            "batch parse got %d results for %d pages — falling back",
            len(results), len(pages),
        )
        return await _fallback_individual(pages, user_query)

    except Exception as e:
        _log.warning("batch summarise failed: %s %r — falling back", type(e).__name__, e)
        return await _fallback_individual(pages, user_query)


def _parse_batch_response(raw: str, expected: int) -> list[dict]:
    parts = re.split(r"(?:^|\n)\s*PAGE\s+\d+\s*:\s*", raw, flags=re.IGNORECASE)
    sections = [p.strip() for p in parts if p.strip()]

    max_summary = _cfg_int("max_summary_chars", MAX_SUMMARY_CHARS)
    results: list[dict] = []
    for section in sections[:expected]:
        summary, relevance = _parse_relevance(section)
        results.append({
            "summary": summary[:max_summary],
            "relevance": relevance,
            "source_type": "article",
        })
    return results


async def _fallback_individual(pages: list[dict], user_query: str) -> list[dict]:
    _sem = asyncio.Semaphore(2)

    async def _bounded(p):
        async with _sem:
            return await _summarise_one(p["url"], p["text"], user_query)

    return list(await asyncio.gather(*[_bounded(p) for p in pages]))



@register_executor(ToolName.WEB_SEARCH)
async def execute(params: dict, emit) -> ToolResult:

    raw_queries = params.get("queries") or []
    if isinstance(raw_queries, str):
        raw_queries = [raw_queries]
    mode = (params.get("mode") or "basic").lower()
    if mode not in ("basic", "standard"):
        mode = "basic"

    query_cap = _cfg_int(
        "standard_query_cap" if mode == "standard" else "basic_query_cap",
        STANDARD_QUERY_CAP if mode == "standard" else BASIC_QUERY_CAP,
    )
    default_url_cap = STANDARD_MAX_URLS if mode == "standard" else BASIC_MAX_URLS
    url_cap = _cfg_int(
        "standard_max_urls" if mode == "standard" else "basic_max_urls",
        default_url_cap,
    )

    priority_urls = max(0, _cfg_int("priority_urls", PRIORITY_URLS))
    priority_cap = _cfg_int("priority_extract_chars", PRIORITY_EXTRACT_CHARS)
    supp_cap = _cfg_int("supplementary_extract_chars", SUPPLEMENTARY_EXTRACT_CHARS)
    rerank_threshold = _cfg_int("rerank_when_candidates_over", RERANK_WHEN_CANDIDATES_OVER)
    rerank_max = _cfg_int("rerank_max", RERANK_MAX)
    batch_target = max(1, _cfg_int("batch_target", BATCH_TARGET))
    max_summary = _cfg_int("max_summary_chars", MAX_SUMMARY_CHARS)
    # Pool cap gives the reranker something to choose from — we want the raw
    # dedupe pool to be comfortably larger than url_cap when rerank runs.
    pool_cap = max(url_cap, rerank_max * 2)

    queries = [str(q).strip() for q in raw_queries if str(q).strip()][:query_cap]
    if not queries:
        return ToolResult(
            tool=ToolName.WEB_SEARCH, action_index=0, ok=False,
            data="No search queries provided",
        )

    from tools._org import resolve_org_id
    org_id = resolve_org_id(params.get("_org_id") or params.get("org_id"), fallback=0)
    if org_id <= 0:
        return ToolResult(
            tool=ToolName.WEB_SEARCH, action_index=0, ok=False,
            data="Missing org_id for web search",
        )

    emit({"type": "searching", "queries": queries, "mode": mode})
    t0 = time.time()

    # keep queue consumers backing off for the whole search window
    from workers.tool_queue import touch_chat_activity

    results = await _search_all(queries, url_cap=pool_cap)
    touch_chat_activity()
    _log.info(
        "searxng  mode=%s queries=%d urls_deduped=%d pool=%d url_cap=%d",
        mode, len(queries), len(results), pool_cap, url_cap,
    )

    if not results:
        emit({
            "type": "search_complete", "source_count": 0, "ok": False,
            "confidence": "failed", "sources": [],
        })
        return ToolResult(
            tool=ToolName.WEB_SEARCH, action_index=0, ok=False,
            data="No search results from any query.",
            elapsed_s=round(time.time() - t0, 2),
        )


    results = _filter_results_by_relevance(results, queries)

    if not results:
        emit({
            "type": "search_complete", "source_count": 0, "ok": False,
            "confidence": "failed", "sources": [],
        })
        return ToolResult(
            tool=ToolName.WEB_SEARCH, action_index=0, ok=False,
            data="Search results were all off-topic — no relevant pages found.",
            elapsed_s=round(time.time() - t0, 2),
        )

    # Conditional rerank: only when the post-filter pool is large enough that
    # the extra LLM call pays for itself. Below the threshold, trust searxng's
    # ordering (already the result of a relevance-ranked search).
    reranked = False
    if len(results) > rerank_threshold:
        try:
            from tools.search.queries import rerank_candidates, _extract_keywords
            entities = [
                kw for kw in _extract_keywords(queries[0])
                if len(kw) > 2 and not kw.isdigit()
            ][:8]
            intent_dict = {"intent": "search_explicit", "entities": entities}
            ranked = await asyncio.to_thread(
                rerank_candidates, results, intent_dict, rerank_max,
            )
            if ranked:
                results = [r for r, _score in ranked]
                reranked = True
        except Exception as e:
            _log.warning("rerank skipped: %s %r", type(e).__name__, e)

    results = results[:url_cap]

    # Tier split: top priority_urls get the deep scrape + individual summary,
    # the rest get a smaller extract and go through the batch summariser.
    priority_candidates = results[:priority_urls]
    supplementary_candidates = results[priority_urls:]

    # Focus hint for every summariser call — cap so long fan-outs don't bloat prompts.
    query_str = " | ".join(queries)[:160]

    # basic-mode shortcut: if SearXNG already gave us a substantial snippet for a
    # supplementary URL, skip the fetch entirely and feed the snippet to the batch
    # summariser. Cuts N httpx/Playwright round-trips out of the supp tier.
    supp_direct: list[dict] = []
    supp_to_fetch: list[dict] = supplementary_candidates
    if mode == "basic":
        supp_direct = []
        supp_to_fetch = []
        for r in supplementary_candidates:
            snip = (r.get("snippet") or "").strip()
            if len(snip) >= BASIC_SNIPPET_DIRECT_MIN_CHARS:
                supp_direct.append({
                    "url": r["url"],
                    "title": r.get("title") or r["url"],
                    "text": snip[:supp_cap],
                    "path": "snippet_direct",
                })
            else:
                supp_to_fetch.append(r)
        if supp_direct:
            _log.info(
                "basic snippet-direct  skipped_fetch=%d fetching=%d",
                len(supp_direct), len(supp_to_fetch),
            )

    # Each tier runs scrape → summarise as a unit, and the two tiers run in
    # parallel. This means priority summarisation can start the moment the 2
    # priority pages have scraped, without waiting for the supp tier's fetches.
    async def _priority_tier() -> tuple[list[dict], list[dict], list[dict]]:
        if not priority_candidates:
            return [], [], []
        scraped_t = list(await asyncio.gather(
            *[_scrape_one(r, priority_cap) for r in priority_candidates]
        ))
        with_text = [s for s in scraped_t if s["text"] and len(s["text"]) >= 100]
        if not with_text:
            return scraped_t, [], []
        # Only 2 priority pages by default — run them fully in parallel.
        summaries = list(await asyncio.gather(
            *[_summarise_one(p["url"], p["text"], query_str) for p in with_text]
        ))
        return scraped_t, with_text, summaries

    async def _supp_tier() -> tuple[list[dict], list[dict], list[dict]]:
        fetched = list(await asyncio.gather(
            *[_scrape_one(r, supp_cap) for r in supp_to_fetch]
        )) if supp_to_fetch else []
        scraped_t = supp_direct + fetched
        with_text = [s for s in scraped_t if s["text"] and len(s["text"]) >= 100]
        if not with_text:
            return scraped_t, [], []
        batch_cfg = get_function_config("web_search_summarise_batch")
        batch_max_input = batch_cfg.get("max_input_chars", 14000)
        batches = _build_batches(with_text, batch_max_input, batch_target)
        _log.info(
            "summarise batches=%d pages=%d target=%d",
            len(batches), len(with_text), batch_target,
        )
        # Sem=2 so batches can overlap; the priority tier runs on its own
        # anyway, so there's nothing to starve.
        _sem = asyncio.Semaphore(2)

        async def _bounded_batch(batch):
            async with _sem:
                return await _summarise_batch(batch, query_str)

        batch_results = await asyncio.gather(*[_bounded_batch(b) for b in batches])
        out: list[dict] = []
        for br in batch_results:
            out.extend(br)
        return scraped_t, with_text, out

    (
        (priority_scraped, priority_with_text, priority_summaries),
        (supplementary_scraped, supp_with_text, supp_summaries),
    ) = await asyncio.gather(_priority_tier(), _supp_tier())
    touch_chat_activity()

    scraped = priority_scraped + supplementary_scraped
    summarised_set = {id(s) for s in priority_with_text} | {id(s) for s in supp_with_text}
    snippet_only = [s for s in scraped if s["text"] and id(s) not in summarised_set]

    snippet_summaries = [
        {"summary": s["text"][:max_summary], "relevance": "low", "source_type": "snippet"}
        for s in snippet_only
    ]

    combined_items = priority_with_text + supp_with_text + snippet_only
    summary_results = priority_summaries + supp_summaries + snippet_summaries

    context_parts: list[str] = []
    sources_meta: list[dict] = []
    for s, sr in zip(combined_items, summary_results):
        context_parts.append(
            f"SOURCE: {s['url']}\n(fetched via {s['path']})\n{sr['summary']}"
        )
        sources_meta.append({
            "url": s["url"],
            "title": s["title"],
            "snippet": sr["summary"][:200],
            "source_type": sr.get("source_type", s["path"]),
            "relevance": sr["relevance"],
            "content_type": "article",
        })

    combined = "\n\n".join(context_parts).strip()
    elapsed = round(time.time() - t0, 2)

    high_count = sum(1 for sr in summary_results if sr["relevance"] == "high")
    med_count = sum(1 for sr in summary_results if sr["relevance"] == "medium")

    counts = {
        "queries": len(queries),
        "urls": len(results),
        "scraped_ok": len([s for s in scraped if s["text"] and s["path"] not in ("snippet", "fallback", "pdf_failed", "playwright_auto_failed", "snippet_direct", "deadline", "error")]),
        "snippet_only": len([s for s in scraped if s["path"] in ("snippet", "fallback", "pdf_failed", "playwright_auto_failed", "snippet_direct", "deadline", "error")]),
        "summarised": len(priority_with_text) + len(supp_with_text),
        "chars": len(combined),
    }
    path_counts: dict[str, int] = {}
    for s in scraped:
        path_counts[s["path"]] = path_counts.get(s["path"], 0) + 1
    _log.info(
        "search_metrics queries=%d urls=%d reranked=%s priority=%d supplementary=%d scraped_ok=%d snippet_only=%d summarised=%d chars=%d paths=%s elapsed=%.2fs",
        counts["queries"], counts["urls"], reranked,
        len(priority_with_text), len(supp_with_text),
        counts["scraped_ok"], counts["snippet_only"],
        counts["summarised"], counts["chars"],
        path_counts, elapsed,
    )

    if high_count >= 1:
        confidence = "high"
    elif med_count >= 1:
        confidence = "medium"
    elif sources_meta:
        confidence = "low"
    else:
        confidence = "failed"

    emit({
        "type": "search_complete",
        "source_count": len(sources_meta),
        "ok": bool(sources_meta),
        "confidence": confidence,
        "sources": sources_meta,
    })

    if combined:
        try:
            from infra.memory import remember
            await asyncio.to_thread(
                remember,
                combined,
                {"source": "web_search", "queries": ", ".join(queries)[:500]},
                int(org_id),
                "web_search",
            )
        except Exception:
            _log.warning("chroma remember failed for web_search result", exc_info=True)

    if not combined:
        return ToolResult(
            tool=ToolName.WEB_SEARCH, action_index=0, ok=False,
            data="Searched but all pages were empty.",
            elapsed_s=elapsed,
        )

    return ToolResult(
        tool=ToolName.WEB_SEARCH, action_index=0, ok=True,
        data=combined, elapsed_s=elapsed,
    )
