"""Pathfinder: turn an approved suggestion into scrape_targets rows.

One job == one approved `suggested_scrape_targets` row. Fetch the URL, extract
same-host `<a href>` children, and insert each child URL (depth 1) into
`scrape_targets`. The scraper picks them up oldest-first on its own.

Discovery/user-approval flow lives in `discover_agent.py` and the enrichment
router. This module is deliberately narrow.
"""
from __future__ import annotations

import logging
import re
from urllib.parse import parse_qsl, urldefrag, urlencode, urlparse, urlunparse

from infra.config import NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS, get_feature
from infra.nocodb_client import NocodbClient
from tools.scraper.pathfinder import PathfinderScraper

_log = logging.getLogger("pathfinder")

DEFAULT_MAX_LINKS = 100

_BINARY_EXT = re.compile(
    r"\.(jpg|jpeg|png|gif|webp|svg|ico|bmp|tif|tiff|"
    r"mp3|mp4|avi|mov|webm|wav|flac|ogg|"
    r"zip|tar|gz|7z|rar|dmg|exe|msi|iso|"
    r"pdf|doc|docx|xls|xlsx|ppt|pptx|csv|rtf|epub)(\?.*)?$",
    re.IGNORECASE,
)

_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "msclkid", "mc_cid", "mc_eid", "ref", "referrer",
    "_ga", "_gl", "yclid", "igshid",
}

_JUNK_PATH = re.compile(
    r"/(login|logout|signin|signout|signup|register|cart|checkout|"
    r"admin|wp-admin|wp-login|account|profile|settings|preferences|"
    r"feed|rss|atom|print|share|email|subscribe)(/|$|\?)",
    re.IGNORECASE,
)


def _normalize(url: str) -> str:
    if not url:
        return ""
    url, _ = urldefrag(url.strip())
    try:
        parts = urlparse(url)
    except Exception:
        return ""
    if parts.scheme not in ("http", "https"):
        return ""
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = re.sub(r"/+", "/", parts.path) or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    params = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=False)
              if k.lower() not in _TRACKING_PARAMS]
    params.sort()
    return urlunparse((parts.scheme, host, path, "", urlencode(params, doseq=True), ""))


def _host(url: str) -> str:
    try:
        h = urlparse(url).netloc.lower()
        return h[4:] if h.startswith("www.") else h
    except Exception:
        return ""


def _is_binary(url: str) -> bool:
    return bool(_BINARY_EXT.search(url))


def _is_junk(url: str) -> bool:
    try:
        parts = urlparse(url)
    except Exception:
        return True
    return bool(_JUNK_PATH.search(parts.path))


def _derive_name(url: str) -> str:
    parts = urlparse(url)
    path = (parts.path or "/").strip("/").replace("/", " · ") or parts.netloc
    return f"{parts.netloc}: {path}"[:255]


def _scrape_target_exists(client: NocodbClient, url: str, org_id: int) -> bool:
    try:
        rows = client._get("scrape_targets", params={
            "where": f"(url,eq,{url})~and(org_id,eq,{org_id})",
            "limit": 1,
        }).get("list", [])
        return bool(rows)
    except Exception:
        return False


def _insert_scrape_target(
    client: NocodbClient,
    url: str,
    source_url: str,
    org_id: int,
    suggested_id: int | None,
) -> bool:
    if _scrape_target_exists(client, url, org_id):
        return False
    payload = {
        "org_id": org_id,
        "url": url,
        "name": _derive_name(url),
        "category": "auto",
        "active": 1,
        "frequency_hours": 24,
        "depth": 1,
        "discovered_from": source_url or "",
        "auto_crawled": 1,
        "consecutive_failures": 0,
        "consecutive_unchanged": 0,
        "chunk_count": 0,
    }
    if suggested_id:
        payload["suggested_id"] = suggested_id
    try:
        client._post("scrape_targets", payload)
        return True
    except Exception as e:
        _log.warning("scrape_targets insert failed  url=%s  error=%s", url[:80], e)
        return False


def pathfinder_extract_job(payload: dict | None = None) -> dict:
    """Tool-queue handler. Extract child URLs from an approved suggestion.

    payload: {"suggested_id": int, "org_id": int, "bypass_idle"?: bool}
    """
    payload = payload or {}
    suggested_id = int(payload.get("suggested_id") or 0)
    if suggested_id <= 0:
        return {"status": "error", "reason": "missing_suggested_id"}

    client = NocodbClient()
    try:
        rows = client._get(NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS, params={
            "where": f"(Id,eq,{suggested_id})",
            "limit": 1,
        }).get("list", [])
    except Exception:
        _log.warning("suggested_scrape_targets lookup failed  id=%d", suggested_id, exc_info=True)
        return {"status": "error", "suggested_id": suggested_id}
    if not rows:
        return {"status": "not_found", "suggested_id": suggested_id}

    row = rows[0]
    row_status = str(row.get("status") or "").strip().lower()
    if row_status in ("extracted", "rejected"):
        return {"status": "skipped", "reason": f"status_{row_status}", "suggested_id": suggested_id}

    seed_url = _normalize(str(row.get("url") or "")) or str(row.get("url") or "")
    from tools._org import resolve_org_id
    org_id = resolve_org_id(row.get("org_id"))
    if not seed_url or org_id <= 0:
        _mark_suggested(client, suggested_id, "rejected", "missing_url_or_org")
        return {"status": "error", "reason": "missing_url_or_org", "suggested_id": suggested_id}

    same_host_only = bool(get_feature("pathfinder", "same_host_only", True))
    max_links = int(get_feature("pathfinder", "max_links_per_seed", DEFAULT_MAX_LINKS))

    scraper = PathfinderScraper(timeout=30)
    try:
        result = scraper.scrape(seed_url)
    except Exception as e:
        _log.warning("pathfinder seed fetch failed  url=%s  error=%s", seed_url[:100], e)
        _mark_suggested(client, suggested_id, "failed", f"seed_fetch: {e}"[:500])
        return {"status": "error", "suggested_id": suggested_id, "reason": "seed_fetch"}

    if result.get("status") != "ok":
        err = result.get("error") or "seed_failed"
        _mark_suggested(client, suggested_id, "failed", err[:500])
        return {"status": "failed", "suggested_id": suggested_id, "reason": err}

    seed_host = _host(seed_url)
    links = result.get("links") or []
    added = 0
    filtered = 0
    source_url = _normalize(result.get("final_url") or seed_url) or seed_url

    # Insert the seed itself as a scrape_target too, so the seed page gets scraped.
    if _insert_scrape_target(client, source_url, "", org_id, suggested_id):
        added += 1

    for link in links[:max(1, max_links)]:
        n = _normalize(link)
        if not n or _is_binary(n) or _is_junk(n):
            filtered += 1
            continue
        if same_host_only and _host(n) != seed_host:
            filtered += 1
            continue
        if _insert_scrape_target(client, n, source_url, org_id, suggested_id):
            added += 1

    _mark_suggested(client, suggested_id, "extracted", "")
    _log.info(
        "pathfinder extract  suggested_id=%d url=%s links=%d added=%d filtered=%d",
        suggested_id, seed_url[:100], len(links), added, filtered,
    )
    return {
        "status": "ok",
        "suggested_id": suggested_id,
        "url": seed_url,
        "links_seen": len(links),
        "added": added,
        "filtered": filtered,
    }


def _mark_suggested(
    client: NocodbClient,
    suggested_id: int,
    status: str,
    error: str,
) -> None:
    try:
        client._patch(NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS, suggested_id, {
            "Id": suggested_id,
            "status": status,
            "error_message": error[:500],
        })
    except Exception:
        _log.debug(
            "suggested_scrape_targets patch failed  id=%d status=%s",
            suggested_id, status, exc_info=True,
        )


def preview_next_approved(org_id: int | None = None) -> dict | None:
    """Dashboard preview: next approved suggestion that pathfinder will pick up."""
    client = NocodbClient()
    try:
        where = "(status,eq,approved)"
        if org_id and int(org_id) > 0:
            where = f"{where}~and(org_id,eq,{int(org_id)})"
        rows = client._get(NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS, params={
            "where": where,
            "sort": "CreatedAt",
            "limit": 1,
        }).get("list", [])
    except Exception:
        return None
    return rows[0] if rows else None
