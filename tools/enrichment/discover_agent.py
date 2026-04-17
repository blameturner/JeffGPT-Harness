"""Autonomous discovery agent.

Runs every ``discover_agent.run_interval_minutes`` (default 20) when the
system is idle.  Workflow per invocation:

  1. Sample N random text chunks from the ChromaDB discovery collection to
     get a diverse cross-section of what the knowledge base already contains.
  2. Ask the tool LLM to propose web-search queries inspired by those topics —
     queries aimed at finding NEW authoritative root pages, not repeating known
     content.
  3. Execute the queries through SearXNG.
  4. Normalise every result URL down to its root (scheme + host), deduplicate,
     and skip hosts already registered in discovery or scrape_targets.
  5. Upsert each novel root into the discovery table so the pathfinder scheduler
     picks it up on the next crawl cycle.

Configuration lives under ``features.discover_agent`` in config.json.
"""
from __future__ import annotations

import json
import logging
import random
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

from infra.config import get_feature
from infra.memory import get_collection
from infra.nocodb_client import NocodbClient
from tools.search.engine import searxng_search

_log = logging.getLogger("discover_agent")

# ── config helpers ────────────────────────────────────────────────────────────

def _cfg(key: str, default):
    return get_feature("discover_agent", key, default)


# ── ChromaDB random sampler ───────────────────────────────────────────────────

def _sample_random_chunks(org_id: int, n: int = 12) -> list[str]:
    """Return up to *n* randomly-offset documents from the discovery collection.

    Uses ``collection.get(limit, offset)`` rather than a semantic query so we
    get genuine breadth instead of topically-clustered neighbours.
    """
    collection_name = str(_cfg("sample_collection", "discovery"))
    try:
        col = get_collection(org_id, collection_name)
        count = col.count()
        if count == 0:
            # Fallback: try agent_outputs
            col = get_collection(org_id, "agent_outputs")
            count = col.count()
        if count == 0:
            _log.info("discover_agent: chroma collections are empty  org_id=%d", org_id)
            return []
        offset = random.randint(0, max(0, count - n))
        result = col.get(limit=n, offset=offset, include=["documents"])
        docs = result.get("documents") or []
        # Trim each chunk so the LLM prompt stays compact
        max_chars = int(_cfg("sample_max_chars_per_chunk", 400))
        return [str(d)[:max_chars] for d in docs if d]
    except Exception:
        _log.warning("discover_agent: chroma sample failed  org_id=%d", org_id, exc_info=True)
        return []


# ── LLM query generation ──────────────────────────────────────────────────────

_QUERY_GEN_PROMPT = """\
You are a web-discovery assistant. Below are {n_chunks} text excerpts sampled \
from a knowledge base.

Your task: propose {n_queries} diverse, specific web-search queries that would \
surface NEW authoritative root websites (company sites, research institutes, \
government portals, reference databases) relevant to the topics in the excerpts. \
Avoid repeating topics that seem already covered. Prefer queries likely to return \
a home page or top-level domain rather than a deep article.

Return ONLY a JSON array of query strings — no explanation, no extra keys.
Example: ["site query one", "site query two"]

EXCERPTS:
{excerpts}
"""


def _generate_queries(chunks: list[str], n_queries: int = 6) -> list[str]:
    """Call the tool LLM to turn sampled chunks into search queries."""
    from shared.models import model_call

    function_name = str(_cfg("query_model", "discover_agent_queries"))
    excerpts = "\n---\n".join(chunks)
    prompt = _QUERY_GEN_PROMPT.format(
        n_chunks=len(chunks),
        n_queries=n_queries,
        excerpts=excerpts[:6000],
    )
    try:
        raw, _tokens = model_call(function_name, prompt)
        raw = (raw or "").strip()
        # Try to extract a JSON array even if the model wrapped it in markdown
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            queries = json.loads(match.group())
            if isinstance(queries, list):
                return [str(q).strip() for q in queries if q][:n_queries]
    except Exception:
        _log.warning("discover_agent: query generation failed", exc_info=True)
    return []


# ── URL root extraction ───────────────────────────────────────────────────────

def _root_url(url: str) -> str:
    """Return 'scheme://host' for a URL, or '' if unparseable / not HTTP(S)."""
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https") or not p.netloc:
            return ""
        host = p.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return f"{p.scheme}://{host}"
    except Exception:
        return ""


def _host_only(url: str) -> str:
    """Return normalised hostname only (no scheme) for dedup comparisons.

    Treats ``http://`` and ``https://`` variants of the same domain as identical
    so we don't re-discover a host we already have under a different scheme.
    """
    try:
        p = urlparse(url)
        host = p.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _existing_hosts(client: NocodbClient, org_id: int) -> set[str]:
    """Collect normalised hostnames already present in discovery + scrape_targets.

    Uses hostname-only comparison (no scheme) so http/https variants don't
    get re-discovered.  Fetches up to 2 000 rows per table which covers large
    knowledge bases while keeping the query fast.
    """
    hosts: set[str] = set()
    for table in ("discovery", "scrape_targets"):
        try:
            rows = client._get_paginated(table, params={
                "where": f"(org_id,eq,{org_id})",
                "fields": "url",
                "limit": 2000,
            })
            for row in rows:
                h = _host_only(row.get("url") or "")
                if h:
                    hosts.add(h)
        except Exception:
            _log.debug("discover_agent: existing host scan failed table=%s", table, exc_info=True)
    return hosts


# ── Cooldown gate ─────────────────────────────────────────────────────────────

def _seconds_since_last_completion(client: NocodbClient) -> float:
    """Return seconds since the last non-skipped discover_agent_run completed.

    Checks ``result_json`` (the actual DB column name) and ``result`` as a
    fallback so the skipped_cooldown filter works correctly regardless of which
    field NocoDB returns.
    """
    try:
        rows = client._get("tool_jobs", params={
            "where": "(type,eq,discover_agent_run)~and(status,eq,completed)",
            "sort": "-completed_at",
            "limit": 10,
        }).get("list", [])
    except Exception:
        return float("inf")
    for row in rows:
        result_str = row.get("result_json") or row.get("result") or ""
        if "skipped_cooldown" in result_str:
            continue
        ts_str = row.get("completed_at") or ""
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - ts).total_seconds()
        except Exception:
            continue
    return float("inf")


# ── Main job handler ──────────────────────────────────────────────────────────

def discover_agent_job(payload: dict | None = None) -> dict:
    """Tool-queue handler.  One invocation per scheduler tick.

    ``payload`` may carry ``{"org_id": N}`` from the jumpstart dispatcher;
    when absent we derive org_id from the most recent discovery row.
    """
    payload = payload or {}

    if not _cfg("enabled", True):
        return {"status": "disabled"}

    client = NocodbClient()

    # ── Cooldown gate ──────────────────────────────────────────────────────────
    min_interval = float(_cfg("min_run_interval_seconds", 1080))  # 18 min default
    elapsed = _seconds_since_last_completion(client)
    if elapsed < min_interval:
        _log.info("discover_agent skip: last run %.1fs ago gate=%.0fs", elapsed, min_interval)
        return {"status": "skipped_cooldown", "elapsed_s": round(elapsed, 1)}

    # ── Determine org_id ───────────────────────────────────────────────────────
    org_id = int(payload.get("org_id") or 0)
    if not org_id:
        # Derive from most-recent discovery row
        try:
            rows = client._get("discovery", params={
                "limit": 1, "sort": "-CreatedAt", "fields": "org_id",
            }).get("list", [])
            org_id = int((rows[0].get("org_id") if rows else None) or 0)
        except Exception:
            pass
    if not org_id:
        _log.info("discover_agent: no org context available, skipping run")
        return {"status": "no_org_context"}

    n_samples = int(_cfg("sample_chunks", 12))
    n_queries = int(_cfg("queries_per_run", 6))
    results_per_query = int(_cfg("results_per_query", 8))
    max_roots_per_run = int(_cfg("max_roots_per_run", 20))

    _log.info("discover_agent start  org_id=%d samples=%d queries=%d", org_id, n_samples, n_queries)

    # ── 1. Sample from Chroma ─────────────────────────────────────────────────
    chunks = _sample_random_chunks(org_id, n=n_samples)
    if not chunks:
        _log.info("discover_agent: no chunks available — nothing to discover")
        return {"status": "no_chunks", "org_id": org_id}

    # ── 2. Generate search queries ────────────────────────────────────────────
    queries = _generate_queries(chunks, n_queries=n_queries)
    if not queries:
        _log.warning("discover_agent: LLM returned no queries  org_id=%d", org_id)
        return {"status": "no_queries", "org_id": org_id, "chunks_sampled": len(chunks)}

    _log.info("discover_agent queries generated  count=%d  queries=%s", len(queries), queries)

    # ── 3. Search ─────────────────────────────────────────────────────────────
    seen_roots: set[str] = set()
    all_search_hits: list[dict] = []
    for q in queries:
        try:
            hits = searxng_search(q, max_results=results_per_query)
            all_search_hits.extend(hits)
        except Exception:
            _log.warning("discover_agent: search failed  query=%s", q[:80], exc_info=True)

    if not all_search_hits:
        _log.info("discover_agent: no search results returned  org_id=%d", org_id)
        return {"status": "no_results", "org_id": org_id, "queries": queries}

    # ── 4. Extract and deduplicate roots ──────────────────────────────────────
    for hit in all_search_hits:
        root = _root_url(hit.get("url") or "")
        if root:
            seen_roots.add(root)

    # Skip hosts already known (scheme-agnostic comparison via _host_only) so
    # we don't thrash pathfinder re-crawling sites it already tracks.
    existing = _existing_hosts(client, org_id)
    novel_roots = [r for r in seen_roots if _host_only(r) not in existing]
    _log.info(
        "discover_agent roots  found=%d existing=%d novel=%d",
        len(seen_roots), len(existing), len(novel_roots),
    )

    if not novel_roots:
        return {
            "status": "ok",
            "org_id": org_id,
            "queries": queries,
            "roots_found": len(seen_roots),
            "roots_added": 0,
            "note": "all roots already known",
        }

    # ── 5. Upsert into discovery queue ────────────────────────────────────────
    from tools.enrichment.pathfinder import upsert_discovery_root

    # Shuffle so repeated runs don't always take the same top-N
    random.shuffle(novel_roots)
    added = 0
    for root in novel_roots[:max_roots_per_run]:
        try:
            row_id = upsert_discovery_root(client, root, org_id, score=80.0)
            if row_id:
                added += 1
                _log.debug("discover_agent: queued  root=%s  id=%s", root, row_id)
        except Exception:
            _log.warning("discover_agent: upsert failed  root=%s", root, exc_info=True)

    _log.info(
        "discover_agent done  org_id=%d chunks=%d queries=%d roots_novel=%d roots_added=%d",
        org_id, len(chunks), len(queries), len(novel_roots), added,
    )
    return {
        "status": "ok",
        "org_id": org_id,
        "chunks_sampled": len(chunks),
        "queries": queries,
        "roots_found": len(seen_roots),
        "roots_novel": len(novel_roots),
        "roots_added": added,
    }




