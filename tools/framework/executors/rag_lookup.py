"""
RAG lookup executor — wraps existing rag.retrieve (which handles query
chunking + ChromaDB lookup + BGE reranking) across a set of likely collections.

If the planner didn't specify a collection, we search the conversation-scoped
collection and the web_search collection in parallel, then merge + dedupe.
Results are formatted with the source metadata the existing retrieve() helper
provides.
"""

from __future__ import annotations

import asyncio
import logging
import time

from tools.framework.contract import ToolName, ToolResult
from tools.framework.dispatcher import register_executor

_log = logging.getLogger("tools.rag_lookup")

MAX_CHUNKS = 5
N_RESULTS_PER_COLLECTION = 8
TOP_K_PER_COLLECTION = 3

DEFAULT_COLLECTIONS = ("chat_knowledge", "web_search")


@register_executor(ToolName.RAG_LOOKUP)
async def execute(params: dict, emit) -> ToolResult:
    """
    Query one or more ChromaDB collections with the given semantic query.

    params:
      query: str          — semantic search string
      _org_id: int        — injected
      _collection: str    — injected conversation-default collection
      collections: list   — optional override; search these collections instead
    """
    query = str(params.get("query") or "").strip()
    if not query:
        return ToolResult(
            tool=ToolName.RAG_LOOKUP, action_index=0, ok=False,
            data="No query provided for RAG lookup",
        )

    org_id = params.get("_org_id") or 0
    default_collection = params.get("_collection") or "chat_knowledge"
    if not org_id:
        return ToolResult(
            tool=ToolName.RAG_LOOKUP, action_index=0, ok=False,
            data="rag_lookup missing org context",
        )

    collections = params.get("collections")
    if not collections:
        collections = list({default_collection, *DEFAULT_COLLECTIONS})
    collections = [str(c) for c in collections if c][:4]

    t0 = time.time()

    try:
        from rag import retrieve
    except Exception as e:
        _log.error("rag import failed: %s", e, exc_info=True)
        return ToolResult(
            tool=ToolName.RAG_LOOKUP, action_index=0, ok=False,
            data=f"rag module unavailable: {e}",
        )

    async def _one(collection: str) -> tuple[str, str]:
        try:
            block = await asyncio.to_thread(
                retrieve, query, int(org_id), collection,
                N_RESULTS_PER_COLLECTION, TOP_K_PER_COLLECTION,
            )
            return collection, block or ""
        except Exception as e:
            _log.warning("rag retrieve failed collection=%s: %s", collection, e)
            return collection, ""

    pairs = await asyncio.gather(*[_one(c) for c in collections])

    non_empty = [(col, block) for col, block in pairs if block and block.strip()]
    elapsed = round(time.time() - t0, 2)

    if not non_empty:
        _log.info("rag_lookup no hits  collections=%s query=%s", collections, query[:80])
        return ToolResult(
            tool=ToolName.RAG_LOOKUP, action_index=0, ok=True,
            data="No prior context found in the knowledge base.",
            elapsed_s=elapsed,
        )

    parts: list[str] = []
    for col, block in non_empty:
        parts.append(f"=== Collection: {col} ===")
        parts.append(block)

    combined = "\n\n".join(parts).strip()
    _log.info(
        "rag_lookup ok  collections_hit=%d/%d chars=%d elapsed=%.2fs",
        len(non_empty), len(collections), len(combined), elapsed,
    )
    return ToolResult(
        tool=ToolName.RAG_LOOKUP, action_index=0, ok=True,
        data=combined, elapsed_s=elapsed,
    )
