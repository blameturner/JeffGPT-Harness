from __future__ import annotations

import logging

from infra.config import get_feature, is_feature_enabled
from infra.memory import remember
from infra.nocodb_client import NocodbClient
from tools.scraper.pathfinder import PathfinderScraper

_log = logging.getLogger("scraper")

DEFAULT_BATCH_SIZE = 10


def scrape_next() -> dict | None:
    client = NocodbClient()
    try:
        data = client._get("discovery", params={
            "where": "(status,eq,discovered)",
            "sort": "-score,depth",
            "limit": 1,
        })
        rows = data.get("list", [])
        return rows[0] if rows else None
    except Exception:
        _log.warning("scrape_next failed")
        return None


def mark_complete(url_id: int, chunks: int = 0) -> None:
    client = NocodbClient()
    try:
        client._patch("discovery", url_id, {"status": "scraped"})
    except Exception:
        _log.warning("mark_complete failed  id=%d", url_id)


def mark_failed(url_id: int, error: str) -> None:
    client = NocodbClient()
    try:
        client._patch("discovery", url_id, {"status": "failed", "error_message": error[:500]})
    except Exception:
        _log.warning("mark_failed failed  id=%d", url_id)


def run_scraper(batch_size: int | None = None) -> dict:
    if not is_feature_enabled("scraper"):
        _log.info("scraper feature disabled, skipping")
        return {"processed": 0, "chunks": 0, "failed": 0, "successful": 0}

    if batch_size is None:
        batch_size = get_feature("scraper", "batch_size", DEFAULT_BATCH_SIZE)

    max_attempts_multiplier = get_feature("scraper", "max_attempts_multiplier", 5)
    max_attempts = max(batch_size * max_attempts_multiplier, batch_size + 5)

    client = NocodbClient()
    scraper = PathfinderScraper(timeout=30)
    attempts = 0
    successful = 0
    chunks_total = 0
    failed = 0
    empty_text = 0

    _log.info("scraper run start  target_successful=%d max_attempts=%d", batch_size, max_attempts)

    while successful < batch_size and attempts < max_attempts:
        row = scrape_next()
        if not row:
            _log.info("scraper queue exhausted  successful=%d/%d after %d attempts", successful, batch_size, attempts)
            break

        attempts += 1
        url_id: int = row.get("Id", 0)
        url: str = row.get("url", "")
        org_id: int = row.get("org_id", 0)
        if not url or not url_id:
            continue

        try:
            client._patch("discovery", url_id, {"status": "scraping"})
        except Exception:
            _log.warning("scraper claim patch failed  id=%d", url_id)

        try:
            result = scraper.scrape(url)
            if result.get("status") != "ok":
                err = result.get("error") or "scrape_failed"
                _log.info("scraper failed  id=%d url=%s reason=%s", url_id, url[:100], err)
                mark_failed(url_id, err)
                failed += 1
                continue

            text = (result.get("text") or "").strip()
            if not text:
                _log.info("scraper empty text  id=%d url=%s", url_id, url[:100])
                mark_failed(url_id, "empty_text")
                failed += 1
                empty_text += 1
                continue

            metadata: dict = {
                "url": result.get("final_url") or url,
                "canonical": result.get("canonical") or url,
                "source": "pathfinder",
                "domain": result.get("domain") or "",
                "discovery_id": url_id,
                "depth": row.get("depth") or 0,
            }
            chunk_ids = remember(text, metadata, org_id, collection_name="discovery")

            if chunk_ids:
                mark_complete(url_id, len(chunk_ids))
                chunks_total += len(chunk_ids)
                successful += 1
                _log.info("scraper ok  id=%d url=%s chars=%d chunks=%d  (%d/%d successful)",
                          url_id, url[:100], len(text), len(chunk_ids), successful, batch_size)
            else:
                mark_failed(url_id, "no_chunks")
                failed += 1
        except Exception as e:
            _log.warning("scraper exception  id=%d url=%s error=%s", url_id, url[:100], e, exc_info=True)
            mark_failed(url_id, str(e)[:200])
            failed += 1

    if successful < batch_size and attempts >= max_attempts:
        _log.warning(
            "scraper run hit attempts ceiling  successful=%d/%d attempts=%d failed=%d",
            successful, batch_size, attempts, failed,
        )

    _log.info(
        "scraper run done  successful=%d/%d attempts=%d chunks=%d failed=%d empty_text=%d",
        successful, batch_size, attempts, chunks_total, failed, empty_text,
    )
    return {
        "successful": successful,
        "target": batch_size,
        "attempts": attempts,
        "chunks": chunks_total,
        "failed": failed,
        "empty_text": empty_text,
        "processed": attempts,  # backwards compat
    }
