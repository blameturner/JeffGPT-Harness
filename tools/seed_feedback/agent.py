"""Seed feedback agent.

Runs three signals and writes the output back into the existing
`suggested_scrape_targets` / `domain_stats` tables so the discover agent and
its approval flow can act on them without any changes of their own.

Signals:
  1. **Graph sparsity** — FalkorDB concepts with outbound degree < threshold
     become seed queries ("authoritative sources on X").
  2. **Domain quality** — per-domain aggregates computed from `messages` +
     `scrape_targets` and written to `domain_stats` for later consumption.
  3. **Weak-RAG turns** — recent `messages` rows where search produced no
     usable result get turned into seed queries.

Signals 1 and 3 share a common "query → searxng → classify → insert" path;
signal 2 just computes aggregates. Single-org per tick.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from infra.config import (
    NOCODB_TABLE_MESSAGES,
    NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS,
    get_feature,
    is_feature_enabled,
)
from infra.graph import get_sparse_concepts
from infra.nocodb_client import NocodbClient
from tools._org import resolve_org_id
from tools.enrichment.discover_agent import _classify_hit, _existing_urls
from tools.search.engine import searxng_search

_log = logging.getLogger("seed_feedback")

DOMAIN_STATS_TABLE = "domain_stats"

_RELEVANCE_RANK = {"rejected": 0, "low": 1, "medium": 2, "high": 3}


def _cfg(key: str, default):
    return get_feature("seed_feedback", key, default)


def _parse_iso(value) -> datetime | None:
    if value in (None, ""):
        return None
    s = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        try:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _recent_pending_queries(client: NocodbClient, org_id: int, days: int) -> set[str]:
    """Queries already pending/approved recently, so we don't re-submit them."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, days))
    try:
        rows = client._get_paginated(NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS, params={
            "where": f"(org_id,eq,{org_id})~and(status,in,pending,approved)",
            "fields": "query,CreatedAt",
            "limit": int(_cfg("recent_dedup_scan_limit", 500)),
            "sort": "-CreatedAt",
        })
    except Exception:
        _log.debug("recent pending query scan failed", exc_info=True)
        return set()
    out: set[str] = set()
    for r in rows:
        created = _parse_iso(r.get("CreatedAt"))
        if created and created < cutoff:
            continue
        q = (r.get("query") or "").strip().lower()
        if q:
            out.add(q)
    return out


def _insert_query_seed(
    client: NocodbClient,
    org_id: int,
    url: str,
    title: str,
    query: str,
    reason_prefix: str,
    verdict: dict,
) -> int | None:
    payload = {
        "org_id": org_id,
        "url": url,
        "title": title[:255],
        "query": query[:500],
        "relevance": verdict["relevance"],
        "score": verdict["score"],
        "reason": f"{reason_prefix} | {verdict.get('reason', '')}"[:500],
        "status": "pending",
    }
    try:
        row = client._post(NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS, payload)
        return row.get("Id")
    except Exception as e:
        _log.warning("seed insert failed  url=%s err=%s", url[:80], e)
        return None


def _run_query_seed(
    client: NocodbClient,
    org_id: int,
    query: str,
    reason_prefix: str,
    existing_urls: set[str],
    min_rank: int,
    results_per_query: int,
    max_hits_per_query: int,
) -> int:
    try:
        hits = searxng_search(query, max_results=results_per_query)
    except Exception:
        _log.warning("searxng failed  query=%s", query[:80], exc_info=True)
        return 0

    added = 0
    for hit in hits:
        if added >= max_hits_per_query:
            break
        url = (hit.get("url") or "").strip()
        if not url or url in existing_urls:
            continue
        verdict = _classify_hit(query, hit) or {
            "relevance": "medium", "score": 50, "reason": "classifier_unavailable",
        }
        if _RELEVANCE_RANK.get(verdict["relevance"], 1) < min_rank:
            continue
        sid = _insert_query_seed(client, org_id, url, hit.get("title") or "", query,
                                 reason_prefix, verdict)
        if sid:
            existing_urls.add(url)
            added += 1
    return added


# --- Signal 1: graph sparsity ------------------------------------------------

def _graph_sparsity_seeds(
    client: NocodbClient,
    org_id: int,
    existing_urls: set[str],
    recent_queries: set[str],
    min_rank: int,
    results_per_query: int,
    max_hits_per_query: int,
) -> dict:
    limit = int(_cfg("sparse_concepts_limit", 10))
    max_degree = int(_cfg("sparse_max_degree", 3))
    concepts = get_sparse_concepts(org_id, limit=limit, max_degree=max_degree)
    if not concepts:
        return {"concepts": 0, "queries": 0, "added": 0}

    added = 0
    issued = 0
    for concept in concepts:
        query = f"authoritative sources on {concept}"
        if query.lower() in recent_queries:
            continue
        issued += 1
        added += _run_query_seed(
            client, org_id, query, f"graph_sparse:{concept}",
            existing_urls, min_rank, results_per_query, max_hits_per_query,
        )
        recent_queries.add(query.lower())
    return {"concepts": len(concepts), "queries": issued, "added": added}


# --- Signal 2: domain quality -----------------------------------------------

def _extract_urls_from_context(text: str) -> list[str]:
    if not text:
        return []
    # Cheap URL grab — the context text in messages contains "SOURCE: <url>" lines.
    urls: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("http://") or s.startswith("https://"):
            urls.append(s.split()[0])
            continue
        if s.upper().startswith("SOURCE:"):
            rest = s.split(":", 1)[1].strip()
            if rest.startswith("http"):
                urls.append(rest.split()[0])
    return urls


def _host(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _domain_quality_stats(client: NocodbClient, org_id: int, window_days: int) -> dict:
    if DOMAIN_STATS_TABLE not in client.tables:
        _log.info("domain_stats table absent — skipping signal 2")
        return {"domains": 0, "skipped": True}

    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, window_days))

    try:
        msg_rows = client._get_paginated(NOCODB_TABLE_MESSAGES, params={
            "where": f"(org_id,eq,{org_id})~and(search_used,eq,1)",
            "fields": "search_context_text,CreatedAt",
            "limit": int(_cfg("domain_stats_message_scan_limit", 1000)),
            "sort": "-CreatedAt",
        })
    except Exception:
        _log.warning("messages scan failed  org_id=%d", org_id, exc_info=True)
        return {"domains": 0}

    cites_by_domain: dict[str, int] = defaultdict(int)
    for r in msg_rows:
        ts = _parse_iso(r.get("CreatedAt"))
        if ts and ts < cutoff:
            continue
        for url in _extract_urls_from_context(r.get("search_context_text") or ""):
            host = _host(url)
            if host:
                cites_by_domain[host] += 1

    try:
        target_rows = client._get_paginated("scrape_targets", params={
            "where": f"(org_id,eq,{org_id})",
            "fields": "domain,status",
            "limit": int(_cfg("domain_stats_target_scan_limit", 5000)),
        })
    except Exception:
        target_rows = []

    domain_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "ok": 0})
    for r in target_rows:
        host = (r.get("domain") or "").strip().lower()
        if not host:
            continue
        domain_counts[host]["total"] += 1
        if (r.get("status") or "").strip().lower() == "ok":
            domain_counts[host]["ok"] += 1

    all_domains = set(cites_by_domain) | set(domain_counts)
    window_end = datetime.now(timezone.utc).isoformat()
    window_start = cutoff.isoformat()
    written = 0
    for host in all_domains:
        cite_count = cites_by_domain.get(host, 0)
        counts = domain_counts.get(host, {"total": 0, "ok": 0})
        total = counts["total"]
        ok = counts["ok"]
        pass_rate = round(ok / total, 3) if total else 0.0
        cite_rate = round(cite_count / max(1, total), 3) if total else float(cite_count)
        weight = round(0.5 * pass_rate + 0.5 * min(1.0, cite_rate), 3)
        payload = {
            "org_id": org_id,
            "domain": host,
            "window_start": window_start,
            "window_end": window_end,
            "cite_count": cite_count,
            "cite_rate": cite_rate,
            "pass_rate": pass_rate,
            "page_count": total,
            "weight": weight,
        }
        try:
            client._post(DOMAIN_STATS_TABLE, payload)
            written += 1
        except Exception:
            _log.debug("domain_stats write failed  host=%s", host, exc_info=True)

    return {"domains": len(all_domains), "written": written}


# --- Signal 3: weak-RAG turns ------------------------------------------------

def _weak_rag_seeds(
    client: NocodbClient,
    org_id: int,
    existing_urls: set[str],
    recent_queries: set[str],
    min_rank: int,
    results_per_query: int,
    max_hits_per_query: int,
) -> dict:
    days = int(_cfg("weak_rag_window_days", 7))
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, days))
    try:
        rows = client._get_paginated(NOCODB_TABLE_MESSAGES, params={
            "where": f"(org_id,eq,{org_id})~and(role,eq,user)",
            "fields": "Id,content,search_status,search_confidence,search_used,CreatedAt",
            "limit": int(_cfg("weak_rag_scan_limit", 300)),
            "sort": "-CreatedAt",
        })
    except Exception:
        _log.warning("weak_rag scan failed  org_id=%d", org_id, exc_info=True)
        return {"considered": 0, "queries": 0, "added": 0}

    weak: list[dict] = []
    for r in rows:
        ts = _parse_iso(r.get("CreatedAt"))
        if ts and ts < cutoff:
            continue
        status = (r.get("search_status") or "").strip().lower()
        confidence = (r.get("search_confidence") or "").strip().lower()
        search_used = bool(r.get("search_used"))
        if status in ("no_results", "error"):
            weak.append(r)
        elif search_used and confidence == "none":
            weak.append(r)

    max_queries = int(_cfg("weak_rag_max_queries_per_run", 10))
    issued = 0
    added = 0
    for r in weak[:max_queries]:
        raw = (r.get("content") or "").strip()
        if not raw:
            continue
        query = raw[:200]
        if query.lower() in recent_queries:
            continue
        issued += 1
        msg_id = r.get("Id") or 0
        added += _run_query_seed(
            client, org_id, query, f"weak_rag:msg_{msg_id}",
            existing_urls, min_rank, results_per_query, max_hits_per_query,
        )
        recent_queries.add(query.lower())

    return {"considered": len(weak), "queries": issued, "added": added}


# --- Handler -----------------------------------------------------------------

def seed_feedback_job(payload: dict | None = None) -> dict:
    """Tool-queue handler. One invocation per scheduler tick."""
    payload = payload or {}
    if not is_feature_enabled("seed_feedback"):
        return {"status": "disabled"}

    org_id = resolve_org_id(payload.get("org_id"))
    min_relevance = str(_cfg("min_relevance", "medium")).lower()
    min_rank = _RELEVANCE_RANK.get(min_relevance, 2)
    results_per_query = int(_cfg("results_per_query", 6))
    max_hits_per_query = int(_cfg("max_hits_per_query", 2))
    recent_dedup_days = int(_cfg("recent_dedup_days", 7))
    domain_window_days = int(_cfg("domain_stats_window_days", 14))

    client = NocodbClient()
    existing_urls = _existing_urls(client, org_id)
    recent_queries = _recent_pending_queries(client, org_id, recent_dedup_days)

    graph_stats = _graph_sparsity_seeds(
        client, org_id, existing_urls, recent_queries,
        min_rank, results_per_query, max_hits_per_query,
    )
    domain_stats = _domain_quality_stats(client, org_id, domain_window_days)
    weak_rag_stats = _weak_rag_seeds(
        client, org_id, existing_urls, recent_queries,
        min_rank, results_per_query, max_hits_per_query,
    )

    total_added = int(graph_stats.get("added", 0)) + int(weak_rag_stats.get("added", 0))
    _log.info(
        "seed_feedback done  org_id=%d graph=%s domain=%s weak_rag=%s total_suggestions=%d",
        org_id, graph_stats, domain_stats, weak_rag_stats, total_added,
    )
    return {
        "status": "ok",
        "org_id": org_id,
        "graph_sparsity": graph_stats,
        "domain_quality": domain_stats,
        "weak_rag": weak_rag_stats,
        "suggestions_added": total_added,
    }
