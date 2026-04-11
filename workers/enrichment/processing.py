from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from config import CATEGORY_COLLECTIONS
from memory import remember
from workers.crawler import check_robots, compute_next_crawl_at, expand_frontier, fan_out
from workers.enrichment.db import EnrichmentDB
from workers.enrichment.models import _tool_call
from workers.enrichment.quality import _content_hash, _validate_content
from workers.enrichment.relationships import _extract_relationships
from workers.enrichment.sources import _discover_sources
from workers.enrichment.summarise import _summarise
from workers.search.scraping import scrape_page

_log = logging.getLogger("enrichment_agent.processing")


def _process_source(
    source: dict,
    org_id: int,
    cycle_id: str,
    db: EnrichmentDB,
    budget_remaining: int,
    sparse_concepts: list[str] | None = None,
) -> int:
    # returns tokens consumed; sparse_concepts passed through to expand_frontier for graph-aware link ranking
    url = source.get("url") or ""
    target_id = source.get("Id")
    _log.debug("processing source %s (id=%s, org=%d)", url[:80], target_id, org_id)
    category = (source.get("category") or "documentation").lower()
    collection = CATEGORY_COLLECTIONS.get(category, "scraped_documentation")
    started = time.time()
    now_utc = datetime.now(timezone.utc)
    base_hours = float(source.get("frequency_hours") or source.get("frequency") or 24)
    prev_unchanged = int(source.get("consecutive_unchanged") or 0)

    if not url:
        _log.warning("source %s rejected: empty url", target_id)
        db.log_event(cycle_id, "source_rejected", org_id, target_id, url, "empty url")
        return 0

    if not check_robots(url):
        _log.info("source %s rejected: robots.txt disallow for %s", target_id, url)
        db.log_event(cycle_id, "source_rejected", org_id, target_id, url, "robots.txt disallow")
        return 0

    fetch_meta: dict[str, Any] = {}
    text = scrape_page(url, source=source, meta_out=fetch_meta)
    fetch_path = fetch_meta.get("path", "unknown")
    if not text:
        _log.warning(
            "source %s scrape failed: no text extracted from %s (path=%s)",
            target_id, url, fetch_path,
        )
        db.log_event(
            cycle_id, "source_error", org_id, target_id, url,
            message=f"scrape failed (path={fetch_path})",
            flags=[f"fetch_path:{fetch_path}"],
        )
        # retry on normal cadence — don't penalise source with exponential backoff for transient scrape failures
        retry_at = (now_utc + timedelta(hours=base_hours)).isoformat()
        db.update_scrape_target(target_id, status="error", next_crawl_at=retry_at)
        return 0

    _log.debug(
        "source %s scraped %d chars from %s (path=%s)",
        target_id, len(text), url, fetch_path,
    )
    new_hash = _content_hash(text)
    is_parent = source.get("parent_target") is None
    if source.get("content_hash") == new_hash:
        next_unchanged = prev_unchanged + 1
        next_at = compute_next_crawl_at(now_utc, base_hours, next_unchanged)

        children_created = 0
        if is_parent:
            children_created, _extra_tokens = expand_frontier(
                parent=source,
                db=db,
                org_id=org_id,
                category=category,
                budget_remaining=budget_remaining,
                tool_call=_tool_call,
                sparse_concepts=sparse_concepts,
            )
            if children_created:
                _log.info(
                    "source %s unchanged but expanded frontier: %d new children from %s",
                    target_id, children_created, url[:80],
                )

        db.update_scrape_target(
            target_id,
            last_scraped_at=now_utc.isoformat(),
            status="ok",
            consecutive_unchanged=next_unchanged,
            next_crawl_at=next_at.isoformat(),
        )
        _log.debug(
            "source %s unchanged (hash match) consecutive=%d next=%s",
            target_id, next_unchanged, next_at.isoformat(),
        )
        db.log_event(
            cycle_id, "source_unchanged", org_id, target_id, url,
            message=f"consecutive_unchanged={next_unchanged} next={next_at.isoformat()} children={children_created}",
            duration_seconds=time.time() - started,
        )
        return 0

    total_tokens = 0

    vr = _validate_content(text)
    total_tokens += vr["tokens"]
    head = re.sub(r"\s+", " ", text[:200]).strip()
    tail = re.sub(r"\s+", " ", text[-200:]).strip() if len(text) > 400 else ""
    validator_flags = [f"fetch_path:{fetch_path}"] + vr["flags"]
    validator_detail = (
        f"code={vr['reason_code']} "
        f"class={vr['classification']} "
        f"path={fetch_path} "
        f"len={len(text)} "
        f"metrics={vr['metrics']} "
        f"detail={vr['message']} "
        f"head={head!r} "
        f"tail={tail!r}"
    )[:1500]
    if not vr["ok"]:
        _log.info(
            "source %s rejected by validator: code=%s class=%s path=%s (%s)",
            target_id, vr["reason_code"], vr["classification"], fetch_path, url,
        )
        db.log_event(
            cycle_id, "source_rejected", org_id, target_id, url,
            message=validator_detail,
            tokens_used=vr["tokens"],
            flags=validator_flags,
        )
        # reuse exponential backoff from stable-content so rejected pages are re-probed less often but still retried
        rejection_next_at = compute_next_crawl_at(now_utc, base_hours, prev_unchanged + 1)
        db.update_scrape_target(
            target_id,
            status="rejected",
            last_scraped_at=now_utc.isoformat(),
            next_crawl_at=rejection_next_at.isoformat(),
        )
        return total_tokens

    _log.info(
        "source %s accepted by validator: code=%s class=%s path=%s",
        target_id, vr["reason_code"], vr["classification"], fetch_path,
    )

    if total_tokens >= budget_remaining:
        db.log_event(cycle_id, "budget_exhausted", org_id, target_id, url)
        return total_tokens

    # fan-out: summarise, extract_relationships, discover_sources write to different destinations; safe to run concurrently
    fanout_results = fan_out(
        [
            lambda: _summarise(text),
            lambda: _extract_relationships(text, org_id),
            lambda: _discover_sources(text, url, org_id, cycle_id, db),
        ],
        label="process_source",
        max_workers=3,
    )
    summary_result, rels_result, disc_result = fanout_results

    if summary_result is None:
        summary, s_tokens = "", 0
    else:
        summary, s_tokens = summary_result
    if rels_result is None:
        rels, r_tokens = 0, 0
    else:
        rels, r_tokens = rels_result
    d_tokens = disc_result if isinstance(disc_result, int) else 0
    total_tokens += s_tokens + r_tokens + d_tokens

    if not summary:
        _log.warning("source %s summariser returned empty for %s", target_id, url)
        db.log_event(cycle_id, "source_error", org_id, target_id, url, "summariser failed")
        return total_tokens

    chunks = 0
    try:
        ids = remember(
            text=summary,
            metadata={
                "url": url,
                "name": source.get("name") or url,
                "category": category,
                "fetched_at": time.time(),
                "cycle_id": cycle_id,
            },
            org_id=org_id,
            collection_name=collection,
        )
        chunks = len(ids or [])
    except Exception as e:
        db.log_event(cycle_id, "source_error", org_id, target_id, url, f"chroma: {e}")
        return total_tokens

    # external-domain discoveries handled above by _discover_sources in the fan-out batch
    children_created = 0
    if total_tokens < budget_remaining:
        children_created, _child_tokens = expand_frontier(
            parent=source,
            db=db,
            org_id=org_id,
            category=category,
            budget_remaining=budget_remaining - total_tokens,
            tool_call=_tool_call,
            sparse_concepts=sparse_concepts,
        )
        if children_created:
            _log.info(
                "source %s frontier expanded: %d new children from %s",
                target_id, children_created, url[:80],
            )

    next_at = compute_next_crawl_at(now_utc, base_hours, 0)
    db.update_scrape_target(
        target_id,
        last_scraped_at=now_utc.isoformat(),
        content_hash=new_hash,
        chunk_count=chunks,
        status="ok",
        consecutive_unchanged=0,
        next_crawl_at=next_at.isoformat(),
    )
    elapsed = round(time.time() - started, 2)
    _log.info("source %s done  url=%s chunks=%d rels=%d tokens=%d %.1fs",
              target_id, url, chunks, rels, total_tokens, elapsed)
    db.log_event(
        cycle_id, "source_scraped", org_id, target_id, url,
        message=(
            f"rels={rels} class={vr['classification']} "
            f"code={vr['reason_code']} path={fetch_path} "
            f"len={len(text)}"
        ),
        chunks_stored=chunks,
        tokens_used=total_tokens,
        duration_seconds=elapsed,
        flags=validator_flags,
    )
    return total_tokens
