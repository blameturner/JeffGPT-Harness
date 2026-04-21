"""Corpus maintenance agent — one periodic pass over `scrape_targets`, two outcomes.

Pass A — **Stale refresh**: re-enqueue a `scrape_page` job for targets scraped
more than N days ago that are still being cited in recent messages. Content-hash
dedup in the scraper means unchanged pages are a near-noop.

Pass B — **Near-dup detection**: bucket rows by domain, compute pairwise
Jaccard similarity on k-shingles of each row's summary, pick the winner in each
cluster (highest chunk_count, then most recent), and mark the losers with
`dup_of = <winner_id>`. Downstream paths (`rag_lookup`, `summariser`) filter
`dup_of IS NULL`.

Similarity uses naive O(n²) within-domain Jaccard — fine at current scale
(twice-daily over <5000 rows). Upgrade path: MinHash + LSH if n grows.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from infra.config import (
    NOCODB_TABLE_MESSAGES,
    is_feature_enabled,
)
from infra.nocodb_client import NocodbClient
from tools._org import resolve_org_id

_log = logging.getLogger("corpus_maintenance")

SCRAPE_TARGETS_TABLE = "scrape_targets"


def _cfg(key: str, default):
    from infra.config import get_feature
    return get_feature("corpus_maintenance", key, default)


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


# --- Pass A: stale refresh ---------------------------------------------------

def _recent_cited_urls(client: NocodbClient, org_id: int, window_days: int,
                       scan_limit: int) -> set[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, window_days))
    try:
        rows = client._get_paginated(NOCODB_TABLE_MESSAGES, params={
            "where": f"(org_id,eq,{org_id})~and(search_used,eq,1)",
            "fields": "search_context_text,CreatedAt",
            "limit": int(scan_limit),
            "sort": "-CreatedAt",
        })
    except Exception:
        _log.warning("messages scan for cited URLs failed", exc_info=True)
        return set()
    out: set[str] = set()
    for r in rows:
        ts = _parse_iso(r.get("CreatedAt"))
        if ts and ts < cutoff:
            continue
        text = r.get("search_context_text") or ""
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("http://") or s.startswith("https://"):
                out.add(s.split()[0])
            elif s.upper().startswith("SOURCE:"):
                rest = s.split(":", 1)[1].strip()
                if rest.startswith("http"):
                    out.add(rest.split()[0])
    return out


def _stale_refresh(
    client: NocodbClient,
    org_id: int,
    rows: list[dict],
) -> dict:
    stale_days = int(_cfg("stale_days", 30))
    cite_window_days = int(_cfg("stale_cite_window_days", 14))
    max_per_run = int(_cfg("max_refresh_per_run", 25))
    cite_scan_limit = int(_cfg("stale_cite_scan_limit", 500))

    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, stale_days))
    cited = _recent_cited_urls(client, org_id, cite_window_days, cite_scan_limit)

    try:
        from workers.tool_queue import get_tool_queue
        tq = get_tool_queue()
    except Exception:
        tq = None
    if not tq:
        _log.info("stale refresh: tool queue unavailable, skipping")
        return {"candidates": 0, "enqueued": 0}

    enqueued = 0
    candidates = 0
    for r in rows:
        if enqueued >= max_per_run:
            break
        if (r.get("status") or "").strip().lower() != "ok":
            continue
        last = _parse_iso(r.get("last_scraped_at"))
        if not last or last >= cutoff:
            continue
        url = (r.get("url") or "").strip()
        if not url or url not in cited:
            continue
        target_id = int(r.get("Id") or 0)
        if not target_id:
            continue
        candidates += 1
        try:
            tq.submit(
                "scrape_page",
                {"target_id": target_id, "org_id": org_id},
                source="corpus_maintenance_refresh",
                priority=5,
                org_id=org_id,
            )
            enqueued += 1
        except Exception:
            _log.warning("stale refresh enqueue failed  target_id=%d", target_id, exc_info=True)

    _log.info(
        "stale refresh  org_id=%d cutoff=%s cited=%d candidates=%d enqueued=%d",
        org_id, cutoff.isoformat(), len(cited), candidates, enqueued,
    )
    return {"candidates": candidates, "enqueued": enqueued}


# --- Pass B: near-dup detection ---------------------------------------------

def _shingles(text: str, k: int) -> set[str]:
    tokens = re.findall(r"\w+", (text or "").lower())
    if len(tokens) < k:
        return set()
    return {" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


def _winner_key(r: dict) -> tuple:
    """Higher is better. Prefer more chunks (more content), then most recent."""
    chunks = int(r.get("chunk_count") or 0)
    last = _parse_iso(r.get("last_scraped_at")) or datetime.min.replace(tzinfo=timezone.utc)
    return (chunks, last)


def _purge_chroma_chunks(org_id: int, scrape_target_id: int) -> None:
    """Remove a scrape_target's chunks from the `discovery` and
    `discovery_summaries` collections. Called when marking a row as dup so
    downstream RAG/summariser paths don't surface redundant content without
    needing a `dup_of IS NULL` filter on Chroma."""
    try:
        from infra.memory import get_collection
    except Exception:
        _log.debug("infra.memory import failed, skipping chroma purge", exc_info=True)
        return
    for name in ("discovery", "discovery_summaries"):
        try:
            col = get_collection(org_id, name)
            col.delete(where={"scrape_target_id": scrape_target_id})
        except Exception:
            _log.debug("chroma purge failed  collection=%s id=%d", name, scrape_target_id, exc_info=True)


def _dup_detect(client: NocodbClient, org_id: int, rows: list[dict]) -> dict:
    threshold = float(_cfg("dup_threshold", 0.85))
    shingle_k = int(_cfg("dup_shingle_k", 5))
    max_per_domain = int(_cfg("dup_max_per_domain", 200))
    purge_chroma = bool(_cfg("dup_purge_chroma", True))

    # Group by domain; only compare within domain.
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        host = (r.get("domain") or "").strip().lower()
        if not host:
            continue
        if (r.get("status") or "").strip().lower() != "ok":
            continue
        summary = (r.get("summary") or "").strip()
        if len(summary) < 100:
            continue
        if r.get("dup_of") not in (None, "", 0):
            continue
        by_domain[host].append(r)

    marked = 0
    compared = 0
    clusters_with_dups = 0

    for host, group in by_domain.items():
        if len(group) < 2:
            continue
        group = group[:max_per_domain]
        shingles: list[set[str]] = [_shingles(r.get("summary") or "", shingle_k) for r in group]

        # Union-find over indices
        parent = list(range(len(group)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                compared += 1
                if _jaccard(shingles[i], shingles[j]) >= threshold:
                    union(i, j)

        # For each cluster, pick winner and mark losers.
        clusters: dict[int, list[int]] = defaultdict(list)
        for i in range(len(group)):
            clusters[find(i)].append(i)
        for members in clusters.values():
            if len(members) < 2:
                continue
            clusters_with_dups += 1
            members.sort(key=lambda idx: _winner_key(group[idx]), reverse=True)
            winner = group[members[0]]
            winner_id = int(winner.get("Id") or 0)
            if not winner_id:
                continue
            for idx in members[1:]:
                loser = group[idx]
                loser_id = int(loser.get("Id") or 0)
                if not loser_id or loser_id == winner_id:
                    continue
                try:
                    client._patch(SCRAPE_TARGETS_TABLE, loser_id, {
                        "Id": loser_id,
                        "dup_of": winner_id,
                    })
                    marked += 1
                except Exception:
                    _log.debug("dup_of patch failed  id=%d", loser_id, exc_info=True)
                    continue
                if purge_chroma:
                    _purge_chroma_chunks(org_id, loser_id)

    _log.info(
        "near_dup  org_id=%d domains=%d compared=%d clusters_with_dups=%d marked=%d threshold=%.2f",
        org_id, len(by_domain), compared, clusters_with_dups, marked, threshold,
    )
    return {
        "domains": len(by_domain),
        "compared": compared,
        "clusters_with_dups": clusters_with_dups,
        "marked": marked,
    }


# --- Handler -----------------------------------------------------------------

def _load_rows(client: NocodbClient, org_id: int) -> list[dict]:
    scan_limit = int(_cfg("scan_limit", 3000))
    try:
        return client._get_paginated(SCRAPE_TARGETS_TABLE, params={
            "where": f"(org_id,eq,{org_id})",
            "limit": scan_limit,
            "sort": "-last_scraped_at",
        })
    except Exception:
        _log.warning("scrape_targets scan failed  org_id=%d", org_id, exc_info=True)
        return []


def corpus_maintenance_job(payload: dict | None = None) -> dict:
    """Tool-queue handler. One invocation per scheduler tick."""
    payload = payload or {}
    if not is_feature_enabled("corpus_maintenance"):
        return {"status": "disabled"}

    org_id = resolve_org_id(payload.get("org_id"))
    client = NocodbClient()
    rows = _load_rows(client, org_id)
    if not rows:
        return {"status": "no_rows", "org_id": org_id}

    refresh_stats = {"candidates": 0, "enqueued": 0, "skipped": True}
    if bool(_cfg("stale_refresh_enabled", True)):
        refresh_stats = _stale_refresh(client, org_id, rows)

    dup_stats = {"domains": 0, "compared": 0, "clusters_with_dups": 0, "marked": 0, "skipped": True}
    if bool(_cfg("dup_detect_enabled", True)):
        dup_stats = _dup_detect(client, org_id, rows)

    _log.info(
        "corpus_maintenance done  org_id=%d rows=%d refresh=%s dup=%s",
        org_id, len(rows), refresh_stats, dup_stats,
    )
    return {
        "status": "ok",
        "org_id": org_id,
        "rows_scanned": len(rows),
        "stale_refresh": refresh_stats,
        "near_dup": dup_stats,
    }
