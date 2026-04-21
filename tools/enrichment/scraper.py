"""Single-page scraper.

One Huey job == one scrape_targets row. Fetch, chunk, write to Chroma
`discovery`, and enqueue two follow-up jobs: relationship extraction (Falkor)
and page summarisation. Oldest-first selection lives in `fetch_due_target`;
`scrape_page_job` is the handler.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone

from infra.config import is_feature_enabled
from infra.memory import remember
from infra.nocodb_client import NocodbClient
from tools.scraper.pathfinder import PathfinderScraper

_log = logging.getLogger("scraper")

DEFAULT_FREQUENCY_HOURS = 24
DEFAULT_MAX_FAILURES_BEFORE_DEACTIVATE = 8
BACKOFF_BASE_HOURS = 1
BACKOFF_MAX_HOURS = 168  # 7 days

_NOCODB_DT_FORMAT = "%Y-%m-%dT%H:%M:%S"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime(_NOCODB_DT_FORMAT)


def _next_crawl_at(frequency_hours: int, failures: int) -> str:
    base = max(int(frequency_hours or DEFAULT_FREQUENCY_HOURS), 1)
    if failures <= 0:
        delta = timedelta(hours=base)
    else:
        backoff = min(BACKOFF_BASE_HOURS * (2 ** (failures - 1)), BACKOFF_MAX_HOURS)
        delta = timedelta(hours=max(base, backoff))
    return (datetime.now(timezone.utc) + delta).strftime(_NOCODB_DT_FORMAT)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _parse_iso(value) -> datetime | None:
    if value is None or value == "":
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


def _due_key(row: dict) -> tuple[datetime, int]:
    return (
        _parse_iso(row.get("next_crawl_at"))
        or _parse_iso(row.get("CreatedAt"))
        or datetime.min.replace(tzinfo=timezone.utc),
        int(row.get("Id") or 0),
    )


SCRAPING_LOCK_TTL_SECONDS = 1800  # 30 minutes — recover rows stuck in status=scraping from a crashed handler


def _is_due(r: dict, now: datetime) -> bool:
    """True if this row is selectable: not locked by a recent in-flight scrape,
    and either never-scraped or past its next_crawl_at."""
    if str(r.get("status") or "").strip().lower() == "scraping":
        # Respect the lock only while it's fresh. If UpdatedAt is older than the
        # TTL, treat the row as eligible so crashed handlers don't wedge it.
        updated = _parse_iso(r.get("UpdatedAt") or r.get("updated_at"))
        if updated and (now - updated).total_seconds() < SCRAPING_LOCK_TTL_SECONDS:
            return False
    if r.get("last_scraped_at") in (None, ""):
        return True
    nca = _parse_iso(r.get("next_crawl_at"))
    return nca is None or nca <= now


def fetch_due_target(client: NocodbClient, org_id: int | None = None) -> dict | None:
    """Return the single oldest-due active scrape_targets row, or None.

    "Oldest" means earliest `next_crawl_at` (nulls treated as now), falling
    back to CreatedAt, then Id.
    """
    try:
        where = "(active,eq,1)"
        if org_id and int(org_id) > 0:
            where = f"{where}~and(org_id,eq,{int(org_id)})"
        rows = client._get_paginated("scrape_targets", params={
            "where": where,
            "limit": 500,
            "sort": "next_crawl_at,CreatedAt",
        })
    except Exception:
        _log.warning("fetch_due_target query failed", exc_info=True)
        return None

    now = datetime.now(timezone.utc)
    due = [r for r in rows if _is_due(r, now) and not r.get("dup_of")]
    if not due:
        return None
    due.sort(key=_due_key)
    return due[0]


def _patch_target(client: NocodbClient, target_id: int, payload: dict) -> None:
    try:
        client._patch("scrape_targets", target_id, {"Id": target_id, **payload})
    except Exception:
        _log.warning("scrape_target patch failed  id=%d", target_id, exc_info=True)


def scrape_page_job(payload: dict | int | None = None) -> dict:
    """Tool-queue handler. Scrape ONE page and write it to memory.

    payload: {"target_id": int, "org_id": int, "bypass_idle"?: bool}
    (int payloads are treated as target_id for back-compat.)
    """
    if not is_feature_enabled("scraper"):
        return {"status": "disabled"}

    if isinstance(payload, int):
        target_id = payload
    elif isinstance(payload, dict) and payload.get("target_id"):
        target_id = int(payload["target_id"])
    else:
        # No target_id: pick the oldest-due row.
        client = NocodbClient()
        from tools._org import resolve_org_id
        req_org = resolve_org_id((payload or {}).get("org_id"), fallback=0)
        row = fetch_due_target(client, org_id=(req_org or None))
        if not row:
            return {"status": "idle"}
        target_id = int(row.get("Id") or 0)
        return _scrape_one(target_id, client=client, row=row)

    return _scrape_one(target_id)


def _scrape_one(target_id: int, client: NocodbClient | None = None, row: dict | None = None) -> dict:
    db: NocodbClient = client if client is not None else NocodbClient()

    if row is None:
        try:
            rows = db._get("scrape_targets", params={
                "where": f"(Id,eq,{target_id})",
                "limit": 1,
            }).get("list", [])
        except Exception:
            _log.warning("scrape_target lookup failed  id=%d", target_id, exc_info=True)
            return {"status": "error", "target_id": target_id}
        if not rows:
            return {"status": "not_found", "target_id": target_id}
        row = rows[0]

    url = row.get("url") or ""
    from tools._org import resolve_org_id
    org_id = resolve_org_id(row.get("org_id"))
    frequency_hours = int(row.get("frequency_hours") or DEFAULT_FREQUENCY_HOURS)
    consecutive_failures = int(row.get("consecutive_failures") or 0)
    consecutive_unchanged = int(row.get("consecutive_unchanged") or 0)
    prior_hash = row.get("content_hash") or ""
    if not url or org_id <= 0:
        _patch_target(db, target_id, {"status": "error", "last_scrape_error": "missing_url_or_org"})
        return {"status": "error", "reason": "missing_url_or_org", "target_id": target_id}

    _patch_target(db, target_id, {"status": "scraping", "last_scrape_error": ""})

    scraper = PathfinderScraper(timeout=30)
    try:
        result = scraper.scrape(url)
    except Exception as e:
        new_failures = consecutive_failures + 1
        _patch_target(db, target_id, {
            "status": "error",
            "last_scrape_error": str(e)[:500],
            "consecutive_failures": new_failures,
            "next_crawl_at": _next_crawl_at(frequency_hours, new_failures),
        })
        _log.warning("scrape exception  id=%d url=%s error=%s", target_id, url[:100], e)
        return {"status": "error", "target_id": target_id, "url": url}

    if result.get("status") != "ok":
        err = result.get("error") or "scrape_failed"
        new_failures = consecutive_failures + 1
        patch: dict = {
            "status": "error",
            "last_scrape_error": err[:500],
            "consecutive_failures": new_failures,
            "next_crawl_at": _next_crawl_at(frequency_hours, new_failures),
        }
        if new_failures >= DEFAULT_MAX_FAILURES_BEFORE_DEACTIVATE:
            patch["active"] = 0
            _log.warning("scrape deactivated after %d failures  id=%d url=%s",
                         new_failures, target_id, url[:100])
        _patch_target(db, target_id, patch)
        return {"status": "failed", "target_id": target_id, "url": url, "reason": err}

    text = (result.get("text") or "").strip()
    if not text:
        new_failures = consecutive_failures + 1
        _patch_target(db, target_id, {
            "status": "error",
            "last_scrape_error": "empty_text",
            "consecutive_failures": new_failures,
            "next_crawl_at": _next_crawl_at(frequency_hours, new_failures),
        })
        return {"status": "failed", "target_id": target_id, "url": url, "reason": "empty_text"}

    new_hash = _content_hash(text)
    unchanged = (prior_hash == new_hash)
    chunk_ids: list[str] = []

    if unchanged:
        consecutive_unchanged += 1
    else:
        consecutive_unchanged = 0
        metadata = {
            "url": result.get("final_url") or url,
            "canonical": result.get("canonical") or url,
            "source": "scrape_target",
            "domain": result.get("domain") or "",
            "scrape_target_id": target_id,
        }
        try:
            chunk_ids = remember(text, metadata, org_id, collection_name="discovery") or []
        except Exception:
            _log.warning("scrape embed failed  id=%d", target_id, exc_info=True)

    total_chunks = int(row.get("chunk_count") or 0) + len(chunk_ids)
    _patch_target(db, target_id, {
        "status": "ok",
        "last_scraped_at": _now_iso(),
        "next_crawl_at": _next_crawl_at(frequency_hours, 0),
        "consecutive_failures": 0,
        "consecutive_unchanged": consecutive_unchanged,
        "content_hash": new_hash,
        "chunk_count": total_chunks,
        "last_scrape_error": "",
    })

    # Only enqueue downstream work when content actually changed.
    if not unchanged:
        _enqueue_followups(target_id, url, text, chunk_ids, org_id, result.get("final_url") or url)

    _log.info("scrape ok  id=%d url=%s chars=%d chunks=%d unchanged=%s",
              target_id, url[:100], len(text), len(chunk_ids), unchanged)
    return {
        "status": "ok",
        "target_id": target_id,
        "url": url,
        "chunks": len(chunk_ids),
        "unchanged": unchanged,
    }


def _enqueue_followups(
    target_id: int,
    url: str,
    text: str,
    chunk_ids: list[str],
    org_id: int,
    final_url: str,
) -> None:
    try:
        from workers.tool_queue import get_tool_queue
        tq = get_tool_queue()
    except Exception:
        tq = None
    if not tq:
        return

    if chunk_ids:
        try:
            tq.submit(
                "extract_relationships",
                {"chunk_ids": chunk_ids, "org_id": org_id, "scrape_target_id": target_id, "url": final_url},
                source="scrape_page",
                priority=5,
                org_id=org_id,
            )
        except Exception:
            _log.warning("relationships enqueue failed  target_id=%d", target_id, exc_info=True)

    try:
        tq.submit(
            "summarise_page",
            {
                "url": final_url,
                "text": text[:30000],
                "org_id": org_id,
                "source": "scrape_page",
                "scrape_target_id": target_id,
            },
            source="scrape_page",
            priority=4,
            org_id=org_id,
        )
    except Exception:
        _log.warning("summarise enqueue failed  target_id=%d", target_id, exc_info=True)
