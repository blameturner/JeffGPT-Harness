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


_NOCODB_DT_FORMAT = "%Y-%m-%dT%H:%M:%S"


def _now_iso() -> str:
    # NocoDB v1 DateTime columns reject microseconds + tz suffix (isoformat()); stick to
    # ISO-short so PATCHes succeed and last_scraped_at/next_crawl_at actually persist.
    return datetime.now(timezone.utc).strftime(_NOCODB_DT_FORMAT)


def _next_crawl_at(frequency_hours: int, failures: int) -> str:
    base = max(int(frequency_hours or DEFAULT_FREQUENCY_HOURS), 1)
    if failures <= 0:
        delta = timedelta(hours=base)
    else:
        # exponential backoff: 1h, 2h, 4h, 8h ... capped
        backoff = min(BACKOFF_BASE_HOURS * (2 ** (failures - 1)), BACKOFF_MAX_HOURS)
        delta = timedelta(hours=max(base, backoff))
    return (datetime.now(timezone.utc) + delta).strftime(_NOCODB_DT_FORMAT)


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
    """Return scrape_targets rows that need scraping. Sort by CreatedAt (oldest first)
    so never-scraped rows surface in order. Any datetime-column sort here can silently
    fail on NocoDB v1 when the column has mixed null/datetime values — CreatedAt is
    always populated, so it's the reliable ordering."""
    rows: list[dict] = []
    try:
        # No `fields=` — NocoDB v1 404s the whole request if any listed column
        # is missing from the live schema, and this query must not fail or the
        # scraper never picks anything up.
        rows = client._get("scrape_targets", params={
            "where": "(active,eq,1)",
            "limit": 1000,
            "sort": "CreatedAt",
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
    _log.debug("fetch_due_targets  total=%d never=%d due=%d", len(rows), len(never), len(due))
    return (never + due)[:limit]


def _patch_target(client: NocodbClient, target_id: int, payload: dict) -> None:
    try:
        client._patch("scrape_targets", target_id, payload)
    except Exception:
        _log.warning("scrape_target patch failed  id=%d", target_id, exc_info=True)


def _scrape_one_target(target_id: int) -> dict:
    """Process a single scrape_target row: scrape, embed on change, queue summarise,
    update bookkeeping. Does NOT chain — caller batches."""
    client = NocodbClient()

    try:
        rows = client._get("scrape_targets", params={
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
    org_id = int(row.get("org_id") or 0)
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

    # on content change, queue one summarise_page job per scrape (one per change, not one per attempt)
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
                    priority=4,
                    org_id=org_id,
                )
        except Exception:
            _log.warning("summarise queue failed  target_id=%d", target_id, exc_info=True)

    _log.info("scrape_target ok  id=%d url=%s chars=%d chunks=%d unchanged=%s",
              target_id, url[:100], len(text), chunks_added, unchanged)
    return {
        "status": "ok",
        "target_id": target_id,
        "url": url,
        "chunks": chunks_added,
        "unchanged": unchanged,
    }


def _seconds_since_last_scrape_target_completion(client: NocodbClient) -> float:
    """Newest REAL completed scrape_target batch's age in seconds. Skips past
    rows whose result carries "skipped_cooldown" — those are themselves the
    cooldown gate firing and counting them would let a flood of queued jobs
    reset the timer every iteration, blocking real runs indefinitely.

    Returns inf if no real completion is found in the recent window.
    """
    try:
        rows = client._get("tool_jobs", params={
            "where": "(type,eq,scrape_target)~and(status,eq,completed)",
            "sort": "-completed_at",
            "limit": 20,
        }).get("list", [])
    except Exception:
        return float("inf")
    for row in rows:
        result_str = row.get("result") or ""
        if "skipped_cooldown" in result_str:
            continue
        ts_str = row.get("completed_at") or ""
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        return (datetime.now(timezone.utc) - ts).total_seconds()
    return float("inf")


def scrape_target_job(payload: dict | int | None = None) -> dict:
    """Tool-queue handler. Runs as a BATCH: processes up to `scraper.batch_size`
    due targets then exits. No self-chain — the 5-min dispatcher schedules the
    next batch. Legacy int payload is treated as a specific target_id.
    """
    if not is_feature_enabled("scraper"):
        return {"status": "disabled"}

    if isinstance(payload, int):
        return _scrape_one_target(payload)
    if isinstance(payload, dict) and payload.get("target_id"):
        return _scrape_one_target(int(payload["target_id"]))

    # Cooldown guard for the scheduled/batch path: if another scrape_target batch
    # completed in the last `scraper.min_run_interval_seconds` (default 60s), skip
    # this run. Protects against stray submitters flooding the queue (the chain
    # bug that spun pathfinder every 1.5s could happen here too).
    min_interval = float(get_feature("scraper", "min_run_interval_seconds", 60))
    if min_interval > 0:
        client_cooldown = NocodbClient()
        elapsed = _seconds_since_last_scrape_target_completion(client_cooldown)
        if elapsed < min_interval:
            _log.info(
                "scraper batch skip: last completion %.1fs ago, gate=%.0fs",
                elapsed, min_interval,
            )
            return {"status": "skipped_cooldown", "elapsed_s": round(elapsed, 1)}

    # Scheduled / jumpstart path: batch
    batch_size = int(get_feature("scraper", "batch_size", DEFAULT_BATCH_SIZE))
    client = NocodbClient()
    targets = fetch_due_targets(client, limit=batch_size)
    if not targets:
        _log.info("scraper batch: no due targets — idle")
        return {"status": "idle", "processed": 0}

    processed: list[dict] = []
    for t in targets:
        tid = int(t.get("Id") or 0)
        if not tid:
            continue
        out = _scrape_one_target(tid)
        processed.append(out)

    successful = sum(1 for p in processed if p.get("status") == "ok")
    chunks = sum(int(p.get("chunks") or 0) for p in processed)
    _log.info(
        "scraper batch done  processed=%d successful=%d chunks=%d",
        len(processed), successful, chunks,
    )
    return {
        "status": "ok",
        "processed": len(processed),
        "successful": successful,
        "chunks": chunks,
    }


# --- legacy helpers retained for the existing /scraper/run endpoint -----------

def scrape_next() -> dict | None:
    client = NocodbClient()
    rows = fetch_due_targets(client, limit=1)
    return rows[0] if rows else None


def run_scraper(batch_size: int | None = None) -> dict:
    """Manual trigger: scrape up to N due targets in-process. Same as the batch
    handler above but callable from the legacy endpoint."""
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
        out = _scrape_one_target(int(t.get("Id") or 0))
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
