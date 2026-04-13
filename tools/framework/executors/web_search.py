"""
Web search executor — inline search with RWKV summarisation.

Pipeline (all inline, no queue):
  1. Up to 5 heuristic queries fired at SearXNG in parallel.
  2. De-dupe by URL, scrape via `workers.search.scraping.scrape_page`.
  3. Summarise pages via RWKV in batches (`web_search_summarise_batch`
     config) with per-source relevance scoring (high/medium/low).
     Pages that don't fit a batch fall back to single-page summarisation.
  4. Store combined context in ChromaDB for RAG.

No tool model (t3_tool) is involved — RWKV handles all summarisation.
Scraping logic lives in `workers.search.scraping`, shared with enrichment.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

from config import get_function_config
from tools.framework.contract import ToolName, ToolResult
from tools.framework.dispatcher import register_executor
from workers.search.engine import PER_PAGE_CHAR_CAP, searxng_search
_log = logging.getLogger("tools.web_search")

# --- Tunables ---
MAX_URLS_TO_PROCESS = 5
MAX_EXTRACT_CHARS = min(2500, PER_PAGE_CHAR_CAP)
MAX_SUMMARY_CHARS = 1500
SEARXNG_PER_QUERY = 10
BATCH_TARGET = 5  # aim for this many pages per batch call
# Minimum fraction of query keywords that must appear in a result's
# title+snippet for it to pass the relevance filter (0.0 = no filter).
RELEVANCE_KEYWORD_THRESHOLD = 0.25


# ---------------- SearXNG ----------------

async def _search_one(query: str) -> list[dict]:
    """Run one SearXNG query via the existing engine helper, off-thread."""
    try:
        results = await asyncio.to_thread(
            searxng_search, query, SEARXNG_PER_QUERY,
        )
    except Exception as e:
        _log.warning("searxng failed q=%r: %s", query[:80], e)
        return []
    return results or []


async def _search_all(queries: list[str]) -> list[dict]:
    """Parallel SearXNG + URL de-dupe. Blocklist filtering is handled downstream by scrape_page."""
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
    return deduped[:MAX_URLS_TO_PROCESS]


def _filter_results_by_relevance(
    results: list[dict],
    queries: list[str],
) -> list[dict]:
    """Drop SearXNG results whose title+snippet share no meaningful keywords
    with the search queries.  This prevents clearly off-topic pages from
    being scraped and summarised (expensive).

    Uses a lightweight keyword-overlap heuristic — no model calls.
    """
    if not results or RELEVANCE_KEYWORD_THRESHOLD <= 0:
        return results

    # Build the set of meaningful keywords from all queries.
    from workers.search.queries import _extract_keywords
    query_keywords: set[str] = set()
    for q in queries:
        for kw in _extract_keywords(q):
            query_keywords.add(kw.lower())

    if not query_keywords:
        return results

    kept: list[dict] = []
    dropped = 0
    for r in results:
        haystack = f"{r.get('title', '')} {r.get('snippet', '')}".lower()
        # Count how many query keywords appear in the result.
        hits = sum(1 for kw in query_keywords if kw in haystack)
        ratio = hits / len(query_keywords)

        if ratio >= RELEVANCE_KEYWORD_THRESHOLD:
            kept.append(r)
        else:
            dropped += 1
            _log.debug(
                "relevance_filter drop  url=%s ratio=%.2f hits=%d/%d",
                (r.get("url") or "")[:80], ratio, hits, len(query_keywords),
            )

    if dropped:
        _log.info(
            "relevance_filter  kept=%d dropped=%d keywords=%d threshold=%.2f",
            len(kept), dropped, len(query_keywords), RELEVANCE_KEYWORD_THRESHOLD,
        )

    # If ALL results are irrelevant, return empty — don't waste time
    # scraping and summarising pages that have nothing to do with the query.
    return kept


# ---------------- Scrape (delegates to shared pipeline) ----------------

async def _scrape_one(item: dict) -> dict:
    """
    Delegate to workers.search.scraping.scrape_page via asyncio.to_thread.

    Returns {url, title, text, path} where `path` is the fetch route tag
    reported by scrape_page ("scraper", "playwright_auto", "pdf", "snippet"
    etc. — see scrape_page source).
    """
    from workers.search.scraping import scrape_page

    url = item["url"]
    snippet = item.get("snippet") or ""
    title = item.get("title") or url

    meta: dict = {}
    try:
        text = await asyncio.to_thread(
            scrape_page, url, snippet, None, meta,
        )
    except Exception as e:
        _log.warning("scrape_page raised for %s: %s", url, e)
        text = snippet
        meta["path"] = "error"

    return {
        "url": url,
        "title": title,
        "text": (text or "")[:MAX_EXTRACT_CHARS],
        "path": meta.get("path", "unknown"),
    }


# ---------------- Summarisation ----------------

def _parse_relevance(text: str) -> tuple[str, str]:
    """Extract (summary_body, relevance) from RWKV output that ends with
    ``RELEVANCE: high|medium|low``."""
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
) -> dict:
    """
    Summarise one scraped page via RWKV (exp_rwkv_r) using the config-driven
    model_call dispatch.  Returns {summary, relevance, source_type}.

    The RWKV prompt asks for a structured output so we get relevance scoring
    without needing the tool model.
    """
    if len(text) < 100:
        return {"summary": text[:MAX_SUMMARY_CHARS], "relevance": "low", "source_type": "snippet"}

    cfg = get_function_config(function_name)
    max_input = cfg.get("max_input_chars", 12000)

    prompt = (
        f"Summarise the following web page content. Focus ONLY on information "
        f"relevant to: {user_query}\n\n"
        f"Rules:\n"
        f"- Keep under 300 words.\n"
        f"- Include specific facts, numbers, dates, names.\n"
        f"- Skip navigation, boilerplate, cookie notices, unrelated content.\n"
        f"- On the LAST line output ONLY: RELEVANCE: high|medium|low\n"
        f"  (high = directly answers the query, medium = related context, "
        f"low = tangential or mostly irrelevant)\n\n"
        f"URL: {url}\n\n"
        f"Content:\n{text[:max_input]}"
    )

    try:
        from workers.enrichment.models import model_call
        raw, _tokens = await asyncio.to_thread(
            model_call, function_name, prompt, True,  # priority=True
        )
        if not raw:
            return {"summary": text[:MAX_SUMMARY_CHARS], "relevance": "low", "source_type": "unknown"}

        summary, relevance = _parse_relevance(raw)
        _log.info("summarise ok  url=%s chars=%d relevance=%s", url[:80], len(summary), relevance)
        return {"summary": summary[:MAX_SUMMARY_CHARS], "relevance": relevance, "source_type": "article"}
    except Exception as e:
        _log.warning("summarise failed  url=%s: %s %r", url[:80], type(e).__name__, e)
        return {"summary": text[:MAX_SUMMARY_CHARS], "relevance": "low", "source_type": "unknown"}


def _build_batches(
    pages: list[dict],
    max_input: int,
    batch_target: int,
) -> list[list[dict]]:
    """Group pages into batches that fit within max_input chars.

    Each page dict has {url, text}. Batches aim for ``batch_target`` pages
    but will split early when adding a page would exceed ``max_input``.
    Oversized single pages get their own batch (they'll be truncated at
    prompt-build time).
    """
    prompt_overhead = 400  # room for instructions, delimiters
    per_page_overhead = 80  # URL header, separator per page
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
    """Summarise multiple pages in a single RWKV call.

    Falls back to individual ``_summarise_one`` calls if the batch response
    can't be parsed (e.g. model only outputs partial markers).
    """
    if len(pages) == 1:
        return [await _summarise_one(pages[0]["url"], pages[0]["text"], user_query)]

    cfg = get_function_config("web_search_summarise_batch")
    max_input = cfg.get("max_input_chars", 14000)

    # Build multi-page prompt — each page already capped at MAX_EXTRACT_CHARS,
    # so give each page its full text rather than dividing the budget.
    sections: list[str] = []
    for i, p in enumerate(pages, 1):
        sections.append(f"--- PAGE {i} ---\nURL: {p['url']}\n{p['text']}")
    body = "\n\n".join(sections)

    prompt = (
        f"Summarise each web page below. Focus ONLY on information "
        f"relevant to: {user_query}\n\n"
        f"Rules:\n"
        f"- For EACH page, output a section starting with PAGE N:\n"
        f"- Use 3-5 bullet points per page, under 80 words total.\n"
        f"- Include specific facts, numbers, dates, names.\n"
        f"- Skip navigation, boilerplate, cookie notices, unrelated content.\n"
        f"- End each section with RELEVANCE: high|medium|low\n"
        f"  (high = directly answers the query, medium = related context, "
        f"low = tangential or mostly irrelevant)\n\n"
        f"{body}"
    )

    try:
        from workers.enrichment.models import model_call
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
    """Parse a multi-page batch response into per-page dicts."""
    # Split on PAGE N: markers
    parts = re.split(r"(?:^|\n)\s*PAGE\s+\d+\s*:\s*", raw, flags=re.IGNORECASE)
    # First element is usually empty or preamble
    sections = [p.strip() for p in parts if p.strip()]

    results: list[dict] = []
    for section in sections[:expected]:
        summary, relevance = _parse_relevance(section)
        results.append({
            "summary": summary[:MAX_SUMMARY_CHARS],
            "relevance": relevance,
            "source_type": "article",
        })
    return results


async def _fallback_individual(pages: list[dict], user_query: str) -> list[dict]:
    """Fallback: summarise pages individually when batch parsing fails."""
    _sem = asyncio.Semaphore(2)

    async def _bounded(p):
        async with _sem:
            return await _summarise_one(p["url"], p["text"], user_query)

    return list(await asyncio.gather(*[_bounded(p) for p in pages]))


# ---------------- Executor ----------------

@register_executor(ToolName.WEB_SEARCH)
async def execute(params: dict, emit) -> ToolResult:
    """
    Full web_search pipeline. See module docstring.

    params:
      queries: list[str]  — 1-3 planner-generated queries
      _org_id: int        — injected by chat_agent for ChromaDB scoping
    """
    raw_queries = params.get("queries") or []
    if isinstance(raw_queries, str):
        raw_queries = [raw_queries]
    queries = [str(q).strip() for q in raw_queries if str(q).strip()][:5]
    if not queries:
        return ToolResult(
            tool=ToolName.WEB_SEARCH, action_index=0, ok=False,
            data="No search queries provided",
        )

    org_id = params.get("_org_id") or 0
    if not org_id:
        return ToolResult(
            tool=ToolName.WEB_SEARCH, action_index=0, ok=False,
            data="web_search missing org context",
        )

    emit({"type": "searching", "queries": queries})
    t0 = time.time()

    # 1. Parallel SearXNG
    results = await _search_all(queries)
    _log.info("searxng  queries=%d urls_deduped=%d", len(queries), len(results))

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

    # 1b. Keyword-overlap filter — drop clearly off-topic results before
    #     spending time scraping and summarising them.
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

    # 2. Parallel scrape via the shared scrape_page pipeline
    scraped = await asyncio.gather(*[_scrape_one(r) for r in results])

    # 3. Summarise pages via RWKV in batches — reduces model calls.
    with_text = [s for s in scraped if s["text"] and len(s["text"]) >= 100]
    query_str = " | ".join(queries)

    summary_results: list[dict] = []
    if with_text:
        batch_cfg = get_function_config("web_search_summarise_batch")
        batch_max_input = batch_cfg.get("max_input_chars", 14000)
        batches = _build_batches(with_text, batch_max_input, BATCH_TARGET)
        _log.info(
            "summarise batches=%d pages=%d target=%d",
            len(batches), len(with_text), BATCH_TARGET,
        )

        _sem = asyncio.Semaphore(2)

        async def _bounded_batch(batch):
            async with _sem:
                return await _summarise_batch(batch, query_str)

        batch_results = await asyncio.gather(*[
            _bounded_batch(b) for b in batches
        ])
        for br in batch_results:
            summary_results.extend(br)

    # Include snippet-only items with low relevance
    snippet_only = [s for s in scraped if s["text"] and s not in with_text]
    for s in snippet_only:
        summary_results.append({
            "summary": s["text"][:MAX_SUMMARY_CHARS],
            "relevance": "low",
            "source_type": "snippet",
        })
    combined_items = with_text + snippet_only

    # 4. Build combined context + per-source metadata for UI + ChromaDB.
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
        "scraped_ok": len([s for s in scraped if s["text"] and s["path"] not in ("snippet", "fallback", "pdf_failed", "playwright_auto_failed")]),
        "snippet_only": len([s for s in scraped if s["path"] in ("snippet", "fallback", "pdf_failed", "playwright_auto_failed")]),
        "summarised": len(with_text),
        "chars": len(combined),
    }
    path_counts: dict[str, int] = {}
    for s in scraped:
        path_counts[s["path"]] = path_counts.get(s["path"], 0) + 1
    _log.info(
        "search_metrics queries=%d urls=%d scraped_ok=%d snippet_only=%d summarised=%d chars=%d paths=%s elapsed=%.2fs",
        counts["queries"], counts["urls"], counts["scraped_ok"],
        counts["snippet_only"], counts["summarised"], counts["chars"],
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

    # 5. Store combined context into ChromaDB for future RAG lookups.
    if combined:
        try:
            from memory import remember
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
