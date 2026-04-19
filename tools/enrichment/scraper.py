from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

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


def _is_truthy_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) != 0
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _created_at_key(row: dict) -> tuple[datetime, int]:
    return (
        _parse_iso(row.get("CreatedAt") or row.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc),
        int(row.get("Id") or 0),
    )


def _due_at_key(row: dict) -> tuple[datetime, datetime, int]:
    return (
        _parse_iso(row.get("next_crawl_at")) or datetime.min.replace(tzinfo=timezone.utc),
        _parse_iso(row.get("CreatedAt") or row.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc),
        int(row.get("Id") or 0),
    )


def _target_host(row: dict) -> str:
    try:
        host = urlparse(str(row.get("url") or "")).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def _scrape_lock_ttl_seconds() -> int:
    raw = get_feature("scraper", "in_progress_lock_ttl_seconds", 1800)
    try:
        val = int(raw)
        return val if val > 0 else 1800
    except Exception:
        return 1800


def _max_per_domain_per_batch() -> int:
    raw = get_feature("scraper", "max_per_domain_per_batch", 2)
    try:
        val = int(raw)
        return val if val > 0 else 2
    except Exception:
        return 2


def _bucket_reason(bucket: str) -> str:
    return {
        "manual_never": "manual target never scraped",
        "manual_due": "manual target due for recrawl",
        "auto_due": "auto-discovered target due for recrawl",
        "auto_never": "auto-discovered target never scraped",
    }.get(bucket, "eligible scrape target")


def fetch_due_targets(client: NocodbClient, limit: int = 50, org_id: int | None = None) -> list[dict]:
    rows: list[dict] = []
    try:
        where = "(active,eq,1)"
        if org_id and int(org_id) > 0:
            where = f"{where}~and(org_id,eq,{int(org_id)})"
        rows = client._get_paginated("scrape_targets", params={
            "where": where,
            "limit": 1000,
            "sort": "CreatedAt",
        })
    except Exception:
        _log.warning("fetch_due_targets query failed", exc_info=True)
        return []

    now = datetime.now(timezone.utc)
    manual_never: list[dict] = []
    manual_due: list[dict] = []
    auto_due: list[dict] = []
    auto_never: list[dict] = []
    lock_ttl = _scrape_lock_ttl_seconds()
    for r in rows:
        if org_id and int(org_id) > 0 and int(r.get("org_id") or 0) != int(org_id):
            continue
        # Avoid repeatedly selecting rows that are currently being processed.
        # If the lock is stale, allow re-selection so wedged rows recover.
        if str(r.get("status") or "").strip().lower() == "scraping":
            updated = _parse_iso(r.get("UpdatedAt") or r.get("updated_at"))
            if updated and (now - updated).total_seconds() < lock_ttl:
                continue
        is_auto = _is_truthy_flag(r.get("auto_crawled"))
        if r.get("last_scraped_at") in (None, ""):
            bucket = auto_never if is_auto else manual_never
            bucket_name = "auto_never" if is_auto else "manual_never"
            bucket.append({
                **r,
                "_selection_bucket": bucket_name,
                "_selection_reason": _bucket_reason(bucket_name),
                "_eligible_at": r.get("CreatedAt") or r.get("created_at") or None,
            })
            continue
        nca = _parse_iso(r.get("next_crawl_at"))
        if nca is None or nca <= now:
            bucket = auto_due if is_auto else manual_due
            bucket_name = "auto_due" if is_auto else "manual_due"
            bucket.append({
                **r,
                "_selection_bucket": bucket_name,
                "_selection_reason": _bucket_reason(bucket_name),
                "_eligible_at": (nca.isoformat() if nca else None),
            })

    manual_never.sort(key=_created_at_key)
    manual_due.sort(key=_due_at_key)
    auto_due.sort(key=_due_at_key)
    auto_never.sort(key=_created_at_key)

    ordered = manual_never + manual_due + auto_due + auto_never
    per_domain_cap = _max_per_domain_per_batch()
    domain_counts: dict[str, int] = {}
    selected: list[dict] = []
    for row in ordered:
        if len(selected) >= limit:
            break
        host = _target_host(row)
        if host:
            if domain_counts.get(host, 0) >= per_domain_cap:
                continue
            domain_counts[host] = domain_counts.get(host, 0) + 1
        selected.append(row)
    _log.debug(
        "fetch_due_targets total=%d manual_never=%d manual_due=%d auto_due=%d auto_never=%d",
        len(rows), len(manual_never), len(manual_due), len(auto_due), len(auto_never),
    )
    return selected


def _patch_target(client: NocodbClient, target_id: int, payload: dict) -> None:
    try:
        client._patch("scrape_targets", target_id, {"Id": target_id, **payload})
    except Exception:
        _log.warning("scrape_target patch failed  id=%d", target_id, exc_info=True)


def _scrape_one_target(target_id: int, client: NocodbClient | None = None, selection: dict | None = None) -> dict:
    db: NocodbClient = client if client is not None else NocodbClient()

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
    org_id = int(row.get("org_id") or 0)
    frequency_hours = int(row.get("frequency_hours") or DEFAULT_FREQUENCY_HOURS)
    consecutive_failures = int(row.get("consecutive_failures") or 0)
    consecutive_unchanged = int(row.get("consecutive_unchanged") or 0)
    prior_hash = row.get("content_hash") or ""
    if not url:
        _patch_target(db, target_id, {"status": "error", "last_scrape_error": "no_url"})
        return {"status": "error", "reason": "no_url", "target_id": target_id}

    # Lightweight lease so parallel workers don't repeatedly select the same row.
    selection_bucket = str((selection or {}).get("selection_bucket") or (selection or {}).get("_selection_bucket") or "")
    selection_reason = str((selection or {}).get("selection_reason") or (selection or {}).get("_selection_reason") or "")
    _patch_target(db, target_id, {
        "status": "scraping",
        "last_scrape_error": "",
        "last_selected_at": _now_iso(),
        "last_selection_bucket": selection_bucket,
        "last_selection_reason": selection_reason,
        "selection_count": int(row.get("selection_count") or 0) + 1,
    })

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
        _patch_target(db, target_id, patch)
        _log.info("scrape_target failed  id=%d url=%s reason=%s failures=%d",
                  target_id, url[:100], err, new_failures)
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

    try:
        from workers.tool_queue import get_tool_queue
        tq = get_tool_queue()
        if tq:
            # Only re-classify and re-summarise when content has actually changed.
            # On unchanged pages the existing relevance label is still valid — no
            # need to burn an LLM call to confirm what we already know.
            if not unchanged:
                tq.submit(
                    "classify_relevance",
                    {
                        "target_id": target_id,
                        "url": result.get("final_url") or url,
                        "text": text[:20000],
                        "org_id": org_id,
                    },
                    source="scrape_target",
                    priority=4,
                    org_id=org_id,
                )
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
        _log.warning("scrape_target follow-up queue failed  target_id=%d", target_id, exc_info=True)

    _log.info("scrape_target ok  id=%d url=%s chars=%d chunks=%d unchanged=%s",
              target_id, url[:100], len(text), chunks_added, unchanged)
    return {
        "status": "ok",
        "target_id": target_id,
        "url": url,
        "chunks": chunks_added,
        "unchanged": unchanged,
        "selection_bucket": selection_bucket or None,
        "selection_reason": selection_reason or None,
    }


def _seconds_since_last_scrape_target_completion(client: NocodbClient) -> float:
    try:
        rows = client._get("tool_jobs", params={
            "where": "(type,eq,scrape_target)~and(status,eq,completed)",
            "sort": "-completed_at",
            "limit": 20,
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
        except Exception:
            continue
        return (datetime.now(timezone.utc) - ts).total_seconds()
    return float("inf")


def scrape_target_job(payload: dict | int | None = None) -> dict:
    if not is_feature_enabled("scraper"):
        return {"status": "disabled"}

    if isinstance(payload, int):
        return _scrape_one_target(payload, selection={"selection_bucket": "direct", "selection_reason": "explicit target run"})
    if isinstance(payload, dict) and payload.get("target_id"):
        return _scrape_one_target(
            int(payload["target_id"]),
            selection={
                "selection_bucket": payload.get("selection_bucket") or "direct",
                "selection_reason": payload.get("selection_reason") or "explicit target run",
            },
        )

    requested_org_id = 0
    if isinstance(payload, dict):
        try:
            requested_org_id = int(payload.get("org_id") or 0)
        except Exception:
            requested_org_id = 0

    # Reuse a single client for all batch-level DB calls so we pay _load_tables()
    # once per invocation instead of once per target.
    client = NocodbClient()

    min_interval = float(get_feature("scraper", "min_run_interval_seconds", 60))
    if min_interval > 0:
        elapsed = _seconds_since_last_scrape_target_completion(client)
        if elapsed < min_interval:
            _log.info(
                "scraper batch skip: last completion %.1fs ago, gate=%.0fs",
                elapsed, min_interval,
            )
            return {"status": "skipped_cooldown", "elapsed_s": round(elapsed, 1)}

    batch_size = int(get_feature("scraper", "batch_size", DEFAULT_BATCH_SIZE))
    targets = fetch_due_targets(client, limit=batch_size, org_id=requested_org_id or None)
    if not targets:
        _log.info("scraper batch: no due targets — idle")
        return {"status": "idle", "processed": 0, "org_id": requested_org_id or None}

    bucket_counts: dict[str, int] = {}
    for t in targets:
        bucket = str(t.get("_selection_bucket") or "unknown")
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    _log.info(
        "scraper batch selected  count=%d buckets=%s first_ids=%s",
        len(targets), bucket_counts, [int(t.get("Id") or 0) for t in targets[:5]],
    )

    processed: list[dict] = []
    for t in targets:
        tid = int(t.get("Id") or 0)
        if not tid:
            continue
        out = _scrape_one_target(
            tid,
            client=client,
            selection={
                "selection_bucket": t.get("_selection_bucket") or "",
                "selection_reason": t.get("_selection_reason") or "",
            },
        )
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
        "org_id": requested_org_id or None,
    }

# legacy

def scrape_next(org_id: int | None = None) -> dict | None:
    client = NocodbClient()
    rows = fetch_due_targets(client, limit=1, org_id=org_id)
    return rows[0] if rows else None


def run_scraper(batch_size: int | None = None) -> dict:
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
        out = _scrape_one_target(int(t.get("Id") or 0), client=client)
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
