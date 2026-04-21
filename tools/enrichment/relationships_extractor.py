"""Falkor relationship-extraction job.

One job fans out: for a list of Chroma chunk IDs, fetch each chunk's text,
extract entity relationships via the existing `shared.relationships`
extractor, and MERGE them into the org's Falkor graph. Scraper enqueues this
after a successful scrape so the scrape handler stays fast (fetch + embed only).
"""
from __future__ import annotations

import logging

from infra.memory import get_collection
from shared.relationships import _extract_relationships

_log = logging.getLogger("enrichment.relationships")


def extract_relationships_job(payload: dict) -> dict:
    """Tool-queue handler.

    payload: {"chunk_ids": [str, ...], "org_id": int, "scrape_target_id"?: int, "url"?: str}
    """
    chunk_ids = list(payload.get("chunk_ids") or [])
    from tools._org import resolve_org_id
    org_id = resolve_org_id(payload.get("org_id"))
    if not chunk_ids or org_id <= 0:
        return {"status": "error", "reason": "missing_chunk_ids_or_org"}

    collection = get_collection(org_id, "discovery")
    try:
        result = collection.get(ids=chunk_ids, include=["documents"])
    except Exception as e:
        _log.warning("chroma fetch failed  org=%d  error=%s", org_id, e, exc_info=True)
        return {"status": "error", "reason": "chroma_fetch_failed"}

    docs = [d for d in (result.get("documents") or []) if d]
    if not docs:
        return {"status": "no_docs", "chunk_ids": len(chunk_ids)}

    text = "\n\n".join(docs)[:20000]
    try:
        written, tokens = _extract_relationships(text, org_id)
    except Exception as e:
        _log.warning("relationships extraction failed  org=%d  error=%s", org_id, e, exc_info=True)
        return {"status": "error", "reason": str(e)[:200]}

    _log.info(
        "relationships ok  org=%d chunks=%d written=%d tokens=%d url=%s",
        org_id, len(docs), written, tokens, str(payload.get("url") or "")[:100],
    )
    return {
        "status": "ok",
        "org_id": org_id,
        "chunks_read": len(docs),
        "relationships_written": written,
        "tokens": tokens,
    }
