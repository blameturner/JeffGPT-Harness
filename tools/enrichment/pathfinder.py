import hashlib
import logging
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from infra.nocodb_client import NocodbClient

_log = logging.getLogger("pathfinder")

DEFAULT_MAX_DEPTH = 3


def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:16]


def _get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _extract_links(url: str, html: str) -> list[str]:
    try:
        soup = BeautifulSoup(html, "lxml")
        links = []
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if href.startswith(("mailto:", "tel:", "javascript:")):
                continue
            normalized = urljoin(url, href)
            if normalized.startswith("http"):
                links.append(normalized)
        return list(set(links))
    except Exception:
        return []


def _url_exists(client: NocodbClient, url_hash: str, org_id: int) -> bool:
    try:
        data = client._get("discovery", params={
            "where": f"(url_hash,eq,{url_hash})~and(org_id,eq,{org_id})",
            "limit": 1
        })
        return bool(data.get("list"))
    except Exception:
        return False


def _add_url(client: NocodbClient, url: str, source_url: str, depth: int, org_id: int) -> bool:
    url_hash = _url_hash(url)
    if _url_exists(client, url_hash, org_id):
        return False
    try:
        client._post("discovery", {
            "org_id": org_id,
            "url": url,
            "url_hash": url_hash,
            "source_url": source_url,
            "depth": depth,
            "domain": _get_domain(url),
            "status": "discovered"
        })
        return True
    except Exception:
        _log.warning("failed to add url  url=%s", url[:80])
        return False


def discover(url: str, org_id: int, max_depth: int = DEFAULT_MAX_DEPTH) -> dict:
    client = NocodbClient()
    to_process = [url]
    processed = 0
    added = 0

    while processed < len(to_process):
        current = to_process[processed]
        current_depth = current.count("/") - url.count("/")
        processed += 1

        if current_depth >= max_depth:
            continue

        try:
            resp = httpx.get(current, timeout=30, follow_redirects=True)
            resp.raise_for_status()
            html = resp.text
        except Exception:
            _log.warning("fetch failed  url=%s", current[:80])
            continue

        for link in _extract_links(current, html):
            if link in to_process:
                continue
            if _add_url(client, link, current, current_depth + 1, org_id):
                to_process.append(link)
                added += 1

    return {"processed": processed, "added": added}


def fetch_next() -> dict | None:
    client = NocodbClient()
    try:
        data = client._get("discovery", params={
            "where": "(status,eq,discovered)~or(status,eq,newly_added)",
            "limit": 1,
            "sort": "depth"
        })
        rows = data.get("list", [])
        return rows[0] if rows else None
    except Exception:
        _log.warning("fetch_next failed")
        return None


def mark_processed(url_id: int) -> None:
    client = NocodbClient()
    try:
        client._patch("discovery", url_id, {"status": "processed"})
    except Exception:
        _log.warning("mark_processed failed  id=%d", url_id)


def mark_failed(url_id: int, error: str) -> None:
    client = NocodbClient()
    try:
        client._patch("discovery", url_id, {"status": "failed", "error_message": error})
    except Exception:
        _log.warning("mark_failed failed  id=%d", url_id)