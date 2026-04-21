"""Daily digest agent.

Once per day, pull the last 24h of completed `scrape_targets` for a single
org, cluster by domain, summarise each cluster with `model_call("daily_digest",
...)`, and emit three artefacts:

  1. Markdown file at `{DIGEST_DIR}/{org_id}/YYYY-MM-DD.md` — human-readable.
  2. A row in the `daily_digests` NocoDB table (best-effort; skipped if the
     table doesn't exist yet) indexing the file for UI enumeration.
  3. A Chroma embedding in the `daily_digests` collection so the digest is
     recallable via the existing RAG path.

Single-org per tick (same pattern as the enrichment dispatchers); no fan-out.
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from infra.config import get_feature, get_function_config, is_feature_enabled
from infra.memory import remember
from infra.nocodb_client import NocodbClient
from shared.models import model_call
from tools._org import resolve_org_id

_log = logging.getLogger("digest")

DEFAULT_DIGEST_DIR = os.getenv("DIGEST_DIR", "/app/data/digests")
DEFAULT_FUNCTION = "daily_digest"
DIGESTS_TABLE = "daily_digests"


def _cfg(key: str, default):
    return get_feature("daily_digest", key, default)


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


def _recent_scrape_targets(client: NocodbClient, org_id: int, window_hours: int) -> list[dict]:
    """Pull the N most recent completed scrape_targets for this org and filter
    in Python to those scraped within `window_hours`. NocoDB `where` on
    datetime columns isn't reliably supported, so do the cutoff locally."""
    scan_limit = max(50, int(_cfg("scan_limit", 500)))
    try:
        rows = client._get_paginated("scrape_targets", params={
            "where": f"(org_id,eq,{org_id})~and(status,eq,ok)",
            "sort": "-last_scraped_at",
            "limit": scan_limit,
        })
    except Exception:
        _log.warning("scrape_targets scan failed  org_id=%d", org_id, exc_info=True)
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    out: list[dict] = []
    for r in rows:
        ts = _parse_iso(r.get("last_scraped_at"))
        if not ts or ts < cutoff:
            continue
        if not (r.get("summary") or "").strip():
            continue
        out.append(r)
    return out


def _cluster_by_domain(rows: list[dict]) -> dict[str, list[dict]]:
    clusters: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        host = (r.get("domain") or "").strip().lower()
        if not host:
            url = (r.get("url") or "").strip()
            try:
                from urllib.parse import urlparse
                host = urlparse(url).netloc.lower() or "unknown"
            except Exception:
                host = "unknown"
        clusters[host].append(r)
    return dict(clusters)


_CLUSTER_PROMPT = """\
Below are {n_pages} page summaries scraped in the last {hours} hours, all from \
the domain `{domain}`.

Write a concise "what we learned" section as markdown bullet points. Focus on \
the concrete facts, claims, data, and developments in these pages. Do not \
fabricate details — if the summaries are vague, be vague. Do not repeat the \
domain name in every bullet. Keep it under 8 bullets.

SUMMARIES:
{summaries}
"""


def _summarise_cluster(domain: str, rows: list[dict], window_hours: int, function_name: str) -> str:
    try:
        cfg = get_function_config(function_name)
    except KeyError:
        _log.warning("digest model %r not configured", function_name)
        return ""
    max_input = int(cfg.get("max_input_chars", 12000))

    chunks: list[str] = []
    for r in rows:
        summary = (r.get("summary") or "").strip()
        url = (r.get("url") or "").strip()
        if summary:
            chunks.append(f"- {url}\n  {summary}")
    joined = "\n".join(chunks)[:max_input]

    prompt = _CLUSTER_PROMPT.format(
        n_pages=len(rows),
        hours=window_hours,
        domain=domain,
        summaries=joined,
    )
    try:
        text, _tokens = model_call(function_name, prompt)
        return (text or "").strip()
    except Exception:
        _log.warning("digest model call failed  domain=%s rows=%d", domain, len(rows), exc_info=True)
        return ""


def _render_markdown(org_id: int, date_str: str, clusters: dict[str, list[dict]], cluster_texts: dict[str, str]) -> str:
    total_pages = sum(len(v) for v in clusters.values())
    parts: list[str] = [
        f"# Daily digest — {date_str}",
        "",
        f"_org_id={org_id} · clusters={len(clusters)} · pages={total_pages}_",
        "",
    ]
    # Sort clusters by page count desc for a natural reading order.
    for domain in sorted(clusters.keys(), key=lambda d: (-len(clusters[d]), d)):
        rows = clusters[domain]
        parts.append(f"## {domain}  _({len(rows)} page{'s' if len(rows) != 1 else ''})_")
        parts.append("")
        body = (cluster_texts.get(domain) or "").strip()
        if body:
            parts.append(body)
        else:
            parts.append("_(no summary produced)_")
        parts.append("")
        parts.append("**Sources:**")
        parts.append("")
        for r in rows:
            url = (r.get("url") or "").strip()
            if url:
                parts.append(f"- {url}")
        parts.append("")
    return "\n".join(parts)


def _write_markdown(org_id: int, date_str: str, markdown: str) -> Path:
    base = Path(_cfg("dir", DEFAULT_DIGEST_DIR)).expanduser()
    org_dir = base / str(org_id)
    org_dir.mkdir(parents=True, exist_ok=True)
    out = org_dir / f"{date_str}.md"
    out.write_text(markdown, encoding="utf-8")
    return out


def _persist_row(client: NocodbClient, org_id: int, date_str: str, path: Path,
                 cluster_count: int, source_count: int) -> int | None:
    if DIGESTS_TABLE not in client.tables:
        _log.info("daily_digests table absent — skipping NocoDB index write")
        return None
    try:
        row = client._post(DIGESTS_TABLE, {
            "org_id": org_id,
            "digest_date": date_str,
            "markdown_path": str(path),
            "cluster_count": cluster_count,
            "source_count": source_count,
        })
        return row.get("Id")
    except Exception:
        _log.warning("daily_digests write failed  org_id=%d date=%s", org_id, date_str, exc_info=True)
        return None


def _embed_digest(org_id: int, date_str: str, markdown: str) -> int:
    try:
        ids = remember(
            markdown,
            {"kind": "daily_digest", "digest_date": date_str, "source": "digest"},
            org_id,
            collection_name="daily_digests",
        )
        return len(ids or [])
    except Exception:
        _log.warning("digest embed failed  org_id=%d date=%s", org_id, date_str, exc_info=True)
        return 0


def daily_digest_job(payload: dict | None = None) -> dict:
    """Tool-queue handler. One invocation per scheduler tick."""
    payload = payload or {}
    if not is_feature_enabled("daily_digest"):
        return {"status": "disabled"}

    org_id = resolve_org_id(payload.get("org_id"))
    window_hours = int(_cfg("window_hours", 24))
    function_name = str(_cfg("model", DEFAULT_FUNCTION))

    client = NocodbClient()
    rows = _recent_scrape_targets(client, org_id, window_hours)
    if not rows:
        _log.info("daily_digest no content  org_id=%d window=%dh", org_id, window_hours)
        return {"status": "no_content", "org_id": org_id, "window_hours": window_hours}

    clusters = _cluster_by_domain(rows)
    cluster_texts: dict[str, str] = {}
    for domain, cluster_rows in clusters.items():
        cluster_texts[domain] = _summarise_cluster(domain, cluster_rows, window_hours, function_name)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    markdown = _render_markdown(org_id, date_str, clusters, cluster_texts)
    path = _write_markdown(org_id, date_str, markdown)
    source_count = sum(len(v) for v in clusters.values())
    nocodb_id = _persist_row(client, org_id, date_str, path, len(clusters), source_count)
    embedded = _embed_digest(org_id, date_str, markdown)

    _log.info(
        "daily_digest done  org_id=%d date=%s clusters=%d sources=%d path=%s embedded=%d",
        org_id, date_str, len(clusters), source_count, path, embedded,
    )
    return {
        "status": "ok",
        "org_id": org_id,
        "date": date_str,
        "clusters": len(clusters),
        "sources": source_count,
        "path": str(path),
        "nocodb_id": nocodb_id,
        "embedded_chunks": embedded,
    }
