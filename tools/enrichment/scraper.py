import logging
from urllib.parse import urlparse

from infra.memory import remember
from infra.nocodb_client import NocodbClient
from tools.search.scraping import scrape_page

_log = logging.getLogger("scraper")

DEFAULT_RATE_LIMIT_MINUTES = 10


def _scrape_and_index(url: str, org_id: int = 0) -> dict:
    result = {"url": url, "chunks": 0, "status": "failed", "error": None}

    try:
        text = scrape_page(url)
        if not text:
            result["error"] = "empty_response"
            return result

        metadata = {
            "url": url,
            "source": "pathfinder",
            "domain": urlparse(url).netloc.lower(),
        }
        chunk_ids = remember(text, metadata, org_id, collection_name="discovery")

        if chunk_ids:
            result["chunks"] = len(chunk_ids)
            result["status"] = "scrape_success"
    except Exception as e:
        result["error"] = str(e)[:200]
        _log.warning("scrape failed  url=%s  error=%s", url[:80], e)

    return result


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


def run_scraper(batch_size: int = 10, org_id: int = 0) -> dict:
    client = NocodbClient()
    processed = 0
    chunks_total = 0
    failed = 0

    for _ in range(batch_size):
        row = scrape_next()
        if not row:
            break

        url_id = row.get("Id")
        url = row.get("url")
        if not url or not url_id:
            continue

        client._patch("discovery", url_id, {"status": "scraping"})

        result = _scrape_and_index(url, org_id)
        if result["status"] == "scrape_success":
            mark_complete(url_id, result["chunks"])
            chunks_total += result["chunks"]
        else:
            mark_failed(url_id, result.get("error", "unknown"))
            failed += 1
        processed += 1

    return {"processed": processed, "chunks": chunks_total, "failed": failed}