"""
Deep search executor — queued variant of web search.

Used by agents for broad research. Runs more queries, scrapes more URLs,
and all jobs go through the tool job queue for background processing.

Results land in ChromaDB.  The agent reads them on its next cycle.
"""

from __future__ import annotations

import asyncio
import logging
import time

from tools.framework.contract import ToolName, ToolResult
from tools.framework.dispatcher import register_executor
from tools.framework.executors.web_search import _search_all

_log = logging.getLogger("tools.deep_search")


@register_executor(ToolName.DEEP_SEARCH)
async def execute(params: dict, emit) -> ToolResult:
    """
    QUEUED deep search.  No one is waiting in real-time.

    1. SearXNG queries in parallel (more queries than web_search)
    2. Enqueue ALL result URLs into the tool job queue
    3. Return immediately with count of queued jobs
    4. Workers process in background (scrape → summarise pipeline)
    5. Results land in ChromaDB for retrieval
    """
    raw_queries = params.get("queries") or []
    if isinstance(raw_queries, str):
        raw_queries = [raw_queries]
    queries = [str(q).strip() for q in raw_queries if str(q).strip()][:10]
    if not queries:
        return ToolResult(
            tool=ToolName.DEEP_SEARCH, action_index=0, ok=False,
            data="No search queries provided",
        )

    org_id = params.get("_org_id") or 0
    conversation_id = params.get("_conversation_id")
    if not org_id:
        return ToolResult(
            tool=ToolName.DEEP_SEARCH, action_index=0, ok=False,
            data="deep_search missing org context",
        )

    emit({"type": "searching", "queries": queries, "mode": "deep"})
    t0 = time.time()

    # 1. Parallel SearXNG
    results = await _search_all(queries)
    _log.info("deep_search searxng  queries=%d urls=%d", len(queries), len(results))

    if not results:
        emit({"type": "search_complete", "source_count": 0, "ok": False,
              "confidence": "failed", "sources": []})
        return ToolResult(
            tool=ToolName.DEEP_SEARCH, action_index=0, ok=False,
            data="No search results from any query.",
            elapsed_s=round(time.time() - t0, 2),
        )

    # 2. Enqueue all URLs into the tool job queue
    from workers.tool_queue import get_tool_queue
    tq = get_tool_queue()
    if not tq:
        return ToolResult(
            tool=ToolName.DEEP_SEARCH, action_index=0, ok=False,
            data="Tool job queue not available.",
            elapsed_s=round(time.time() - t0, 2),
        )

    job_ids: list[str] = []
    for r in results:
        ids = tq.submit_pipeline(
            url=r["url"],
            org_id=int(org_id),
            collection="web_search",
            source="deep_search",
            priority=2,
            metadata={
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "queries": queries,
                "conversation_id": conversation_id,
            },
        )
        job_ids.extend(ids)

    elapsed = round(time.time() - t0, 2)
    _log.info("deep_search queued  urls=%d jobs=%d elapsed=%.2fs",
              len(results), len(job_ids), elapsed)

    emit({
        "type": "search_complete",
        "source_count": len(results),
        "ok": True,
        "confidence": "queued",
        "sources": [{"url": r["url"], "title": r.get("title", "")} for r in results[:5]],
    })

    return ToolResult(
        tool=ToolName.DEEP_SEARCH, action_index=0, ok=True,
        data=f"Queued {len(results)} URLs for deep research. "
             f"Results will appear in the knowledge base shortly.",
        elapsed_s=elapsed,
    )
