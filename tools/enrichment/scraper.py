from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone

from infra.config import get_feature, is_feature_enabled
from infra.memory import remember
from infra.nocodb_client import NocodbClient
from tools.scraper.pathfinder import PathfinderScraper

_log = logging.getLogger("scraper")

DEFAULT_BATCH_SIZE = 10
DEFAULT_FREQUENCY_HOURS = 24
DEFAULT_MAX_FAILURES_BEFORE_DEACTIVATE = 8
BACKOFF_BASE_HOURS = 1
BACKOFF_MAX_HOURS = 168  # 7 days


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _next_crawl_at(frequency_hours: int, failures: int) -> str:
    base = max(int(frequency_hours or DEFAULT_FREQUENCY_HOURS), 1)
    if failures <= 0:
        delta = timedelta(hours=base)
    else:
        # exponential backoff: 1h, 2h, 4h, 8h ... capped
        backoff = min(BACKOFF_BASE_HOURS * (2 ** (failures - 1)), BACKOFF_MAX_HOURS)
        delta = timedelta(hours=max(base, backoff))
    return (datetime.now(timezone.utc) + delta).isoformat()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _parse_iso(value) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        ts = value / 1000.0 if value > 1e12 else float(value)
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
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


def fetch_due_targets(client: NocodbClient, limit: int = 50) -> list[dict]:
    """Return scrape_targets rows that need scraping. NocoDB v1's where parser rejects
    gte/lte filters on datetime columns, so we fetch active rows and filter in Python."""
    rows: list[dict] = []
    try:
        rows = client._get("scrape_targets", params={
            "where": "(active,eq,1)",
            "limit": 1000,
            "sort": "next_crawl_at,CreatedAt",
        }).get("list", [])
    except Exception:
        _log.warning("fetch_due_targets query failed", exc_info=True)
        return []

    now = datetime.now(timezone.utc)
    never: list[dict] = []
    due: list[dict] = []
    for r in rows:
        if r.get("last_scraped_at") in (None, ""):
            never.append(r)
            continue
        nca = _parse_iso(r.get("next_crawl_at"))
        if nca is None or nca <= now:
            due.append(r)
    return (never + due)[:limit]


def _patch_target(client: NocodbClient, target_id: int, payload: dict) -> None:
    try:
        client._patch("scrape_targets", target_id, payload)
    except Exception:
        _log.warning("scrape_target patch failed  id=%d", target_id, exc_info=True)


def _chain_next_job(reason: str = "chain") -> None:
    """Submit a follow-up scrape_target job (no payload — handler picks next due target)."""
    try:
        from workers.tool_queue import get_tool_queue
        tq = get_tool_queue()
        if tq:
            tq.submit("scrape_target", {}, source=f"scraper_{reason}", priority=5)
    except Exception:
        _log.warning("scrape_target chain submit failed  reason=%s", reason, exc_info=True)


def scrape_target_job(payload: dict | int | None = None) -> dict:
    """Tool-queue handler: pick (or use given) due scrape_target, scrape + summarise + embed,
    then re-queue another scrape_target job so the worker keeps draining the backlog.
    Stops chaining when there are no due targets — the 5-min dispatcher will jumpstart it again."""
    if not is_feature_enabled("scraper"):
        return {"status": "disabled"}

    # accept payload as dict (preferred), int (legacy direct id), or None (pick next)
    if isinstance(payload, int):
        target_id = payload
    elif isinstance(payload, dict):
        target_id = payload.get("target_id")
    else:
        target_id = None

    client = NocodbClient()

    if not target_id:
        due = fetch_due_targets(client, limit=1)
        if not due:
            _log.debug("scraper: no due targets, idling — waiting for dispatcher")
            return {"status": "idle"}
        target_id = int(due[0].get("Id") or 0)
        if not target_id:
            return {"status": "error", "reason": "bad_target_row"}

    try:
        rows = client._get("scrape_targets", params={
            "where": f"(Id,eq,{target_id})",
            "limit": 1,
        }).get("list", [])
    except Exception:
        _log.warning("scrape_target_job lookup failed  id=%d", target_id, exc_info=True)
        _chain_next_job("after_lookup_error")
        return {"status": "error", "target_id": target_id}
    if not rows:
        _chain_next_job("after_not_found")
        return {"status": "not_found", "target_id": target_id}

    row = rows[0]
    url = row.get("url") or ""
    org_id = row.get("org_id") or 0
    frequency_hours = int(row.get("frequency_hours") or DEFAULT_FREQUENCY_HOURS)
    consecutive_failures = int(row.get("consecutive_failures") or 0)
    consecutive_unchanged = int(row.get("consecutive_unchanged") or 0)
    prior_hash = row.get("content_hash") or ""
    if not url:
        _patch_target(client, target_id, {"status": "error", "last_scrape_error": "no_url"})
        return {"status": "error", "reason": "no_url", "target_id": target_id}

    scraper = PathfinderScraper(timeout=30)
    try:
        result = scraper.scrape(url)
    except Exception as e:
        new_failures = consecutive_failures + 1
        _patch_target(client, target_id, {
            "status": "error",
            "last_scrape_error": str(e)[:500],
            "consecutive_failures": new_failures,
            "next_crawl_at": _next_crawl_at(frequency_hours, new_failures),
        })
        _log.warning("scrape_target exception  id=%d url=%s error=%s", target_id, url[:100], e)
        _chain_next_job("after_exception")
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
            _log.warning("scrape_target deactivated after %d failures  id=%d url=%s",
                         new_failures, target_id, url[:100])
        _patch_target(client, target_id, patch)
        _log.info("scrape_target failed  id=%d url=%s reason=%s failures=%d",
                  target_id, url[:100], err, new_failures)
        _chain_next_job("after_failure")
        return {"status": "failed", "target_id": target_id, "url": url, "reason": err}

    text = (result.get("text") or "").strip()
    if not text:
        new_failures = consecutive_failures + 1
        _patch_target(client, target_id, {
            "status": "error",
            "last_scrape_error": "empty_text",
            "consecutive_failures": new_failures,
            "next_crawl_at": _next_crawl_at(frequency_hours, new_failures),
        })
        _chain_next_job("after_empty")
        return {"status": "failed", "target_id": target_id, "url": url, "reason": "empty_text"}

    new_hash = _content_hash(text)
    unchanged = (prior_hash == new_hash)
    chunks_added = 0

    if unchanged:
        consecutive_unchanged += 1
        _log.info("scrape_target unchanged  id=%d url=%s consecutive_unchanged=%d",
                  target_id, url[:100], consecutive_unchanged)
    else:
        consecutive_unchanged = 0
        metadata = {
            "url": result.get("final_url") or url,
            "canonical": result.get("canonical") or url,
            "source": "scrape_target",
            "domain": result.get("domain") or "",
            "scrape_target_id": target_id,
            "depth": row.get("depth") or 0,
        }
        try:
            chunk_ids = remember(text, metadata, org_id, collection_name="discovery")
            chunks_added = len(chunk_ids or [])
        except Exception:
            _log.warning("scrape_target embed failed  id=%d", target_id, exc_info=True)
            chunks_added = 0

    total_chunks = int(row.get("chunk_count") or 0) + chunks_added
    _patch_target(client, target_id, {
        "status": "ok",
        "last_scraped_at": _now_iso(),
        "next_crawl_at": _next_crawl_at(frequency_hours, 0),
        "consecutive_failures": 0,
        "consecutive_unchanged": consecutive_unchanged,
        "content_hash": new_hash,
        "chunk_count": total_chunks,
        "last_scrape_error": "",
    })

    # if content changed, queue an out-of-band summarise job (priority 5 = background)
    if not unchanged:
        try:
            from workers.tool_queue import get_tool_queue
            tq = get_tool_queue()
            if tq:
                tq.submit(
                    "summarise_page",
                    {
                        "url": result.get("final_url") or url,
                        "text": text[:30000],
                        "org_id": org_id,
                        "source": "scrape_target",
                        "scrape_target_id": target_id,
                    },
                    source="scrape_target",
                    priority=5,
                )
        except Exception:
            _log.warning("summarise queue failed  target_id=%d", target_id, exc_info=True)

    _log.info("scrape_target ok  id=%d url=%s chars=%d chunks=%d unchanged=%s",
              target_id, url[:100], len(text), chunks_added, unchanged)
    _chain_next_job("after_ok")
    return {
        "status": "ok",
        "target_id": target_id,
        "url": url,
        "chunks": chunks_added,
        "unchanged": unchanged,
    }


# --- legacy helpers retained for the existing /scraper/run endpoint -----------

def scrape_next() -> dict | None:
    client = NocodbClient()
    rows = fetch_due_targets(client, limit=1)
    return rows[0] if rows else None


def run_scraper(batch_size: int | None = None) -> dict:
    """Manual trigger: scrape up to N due targets in-process. The scheduler usually drives this."""
    if not is_feature_enabled("scraper"):
        return {"status": "disabled", "successful": 0}
    if batch_size is None:
        batch_size = get_feature("scraper", "batch_size", DEFAULT_BATCH_SIZE)

    client = NocodbClient()
    targets = fetch_due_targets(client, limit=batch_size * 2)
    successful = 0
    failed = 0
    chunks = 0
    for t in targets[:batch_size]:
        out = scrape_target_job(int(t.get("Id") or 0))
        if out.get("status") == "ok":
            successful += 1
            chunks += int(out.get("chunks") or 0)
        else:
            failed += 1
    return {
        "successful": successful,
        "failed": failed,
        "chunks": chunks,
        "considered": len(targets),
        "target": batch_size,
    }
