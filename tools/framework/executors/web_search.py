"""
Web search executor — parallel SearXNG queries + reuse of the existing
`workers.search.scraping.scrape_page` pipeline + parallel tool-model summarisation.

Design:
  1. The planner decides the queries — this executor never re-generates them.
  2. Fire all SearXNG queries concurrently.
  3. De-dupe by URL, filter the existing url blocklist.
  4. Scrape each URL via `workers.search.scraping.scrape_page` (via
     asyncio.to_thread — it's sync). That helper already has:
       - httpx + BeautifulSoup + injection-residue stripping + PDF handling
       - Main-content JS extraction via the in-process Playwright worker
       - Stealth JS injection + cookie/consent banner nuking + anti-bot detection
       - Auto-promotion of failing domains to use_playwright=True in NocoDB
       - Per-page char cap and snippet fallback
  5. Filter out snippet-only fallbacks from the summarisation step (there's
     nothing to summarise beyond the snippet itself — we already have it).
  6. Summarise each non-trivial extract in parallel via the tool model,
     using `acquire_model("tool")` so pick + slot is atomic and load-aware.
  7. Store the combined context in ChromaDB via the existing `memory.remember`
     so the RAG collection populates over time.

This executor deliberately does NOT re-implement scraping. Every byte of
scraping logic lives in `workers.search.scraping` and is shared with the
enrichment path.
"""

from __future__ import annotations

import asyncio
import logging
import time

from config import get_function_config
from tools.framework.contract import ToolName, ToolResult
from tools.framework.dispatcher import register_executor
from workers.search.engine import PER_PAGE_CHAR_CAP, searxng_search

_log = logging.getLogger("tools.web_search")

# --- Tunables ---
MAX_URLS_TO_PROCESS = 10
MAX_EXTRACT_CHARS = min(4000, PER_PAGE_CHAR_CAP)
MAX_SUMMARY_CHARS = 1500
SEARXNG_PER_QUERY = 10


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

        # Parse relevance from last line
        lines = raw.strip().rsplit("\n", 1)
        relevance = "medium"
        summary = raw.strip()
        if len(lines) == 2 and "RELEVANCE:" in lines[1].upper():
            summary = lines[0].strip()
            tag = lines[1].upper().replace("RELEVANCE:", "").strip()
            if "HIGH" in tag:
                relevance = "high"
            elif "LOW" in tag:
                relevance = "low"

        _log.info("summarise ok  url=%s chars=%d relevance=%s", url[:80], len(summary), relevance)
        return {"summary": summary[:MAX_SUMMARY_CHARS], "relevance": relevance, "source_type": "article"}
    except Exception as e:
        _log.warning("summarise failed  url=%s: %s %r", url[:80], type(e).__name__, e)
        return {"summary": text[:MAX_SUMMARY_CHARS], "relevance": "low", "source_type": "unknown"}


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

    # 2. Parallel scrape via the shared scrape_page pipeline
    scraped = await asyncio.gather(*[_scrape_one(r) for r in results])

    # 3. Summarise every non-trivial page via RWKV — includes relevance scoring.
    with_text = [s for s in scraped if s["text"] and len(s["text"]) >= 100]
    _sem = asyncio.Semaphore(2)

    async def _bounded_summarise(url, text, query):
        async with _sem:
            return await _summarise_one(url, text, query, "web_search_summarise")

    summary_results: list[dict] = []
    if with_text:
        summary_results = await asyncio.gather(*[
            _bounded_summarise(s["url"], s["text"], " | ".join(queries))
            for s in with_text
        ])

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
