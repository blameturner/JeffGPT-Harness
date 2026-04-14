import logging
from urllib.parse import urlparse

from infra.config import is_feature_enabled
from infra.memory import remember
from infra.nocodb_client import NocodbClient
from tools.search.scraping import scrape_page

_log = logging.getLogger("scraper")

def scrape_next() -> dict | None:
    client = NocodbClient()
    try:
        data = client._get("discovery", params={
            "where": "(status,eq,discovered)",
            "limit": 1,
            "sort": "depth"
        })
        rows = data.get("list", [])
        return rows[0] if rows else None
    except Exception:
        _log.warning("scrape_next failed")
        return None


def mark_complete(url_id: int, chunk_count: int) -> None:
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


def run_scraper(batch_size: int = 10) -> dict:
    if not is_feature_enabled("scraper"):
        _log.info("scraper feature disabled, skipping")
        return {"processed": 0, "chunks": 0, "failed": 0}

    client = NocodbClient()
    processed = 0
    chunks_total = 0
    failed = 0

    for _ in range(batch_size):
        row = scrape_next()
        if not row:
            break

        url_id: int = row.get("Id", 0)
        url: str = row.get("url", "")
        org_id: int = row.get("org_id", 0)
        if not url or not url_id:
            continue

        try:
            client._patch("discovery", url_id, {"status": "scraping"})
        except Exception:
            pass

        try:
            text = scrape_page(url)
            if not text:
                mark_failed(url_id, "empty_response")
                failed += 1
                continue

            metadata: dict = {
                "url": url,
                "source": "pathfinder",
                "domain": urlparse(url).netloc.lower(),
            }
            chunk_ids: list = remember(text, metadata, org_id, collection_name="discovery")

            if chunk_ids:
                mark_complete(url_id, len(chunk_ids))
                chunks_total += len(chunk_ids)
            else:
                mark_failed(url_id, "no_chunks")
                failed += 1
        except Exception as e:
            mark_failed(url_id, str(e)[:200])
            failed += 1

        processed += 1

    return {"processed": processed, "chunks": chunks_total, "failed": failed}