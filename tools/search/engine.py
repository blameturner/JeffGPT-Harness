from __future__ import annotations

import logging
from typing import Iterable

import httpx

from infra.config import MAX_SUMMARY_INPUT_CHARS, SEARXNG_URL

_log = logging.getLogger("web_search.engine")


MAX_SOURCES = 8
PER_PAGE_CHAR_CAP = 20_000
SUMMARY_MAX_TOKENS = 500
SUMMARY_INPUT_CHAR_CAP = MAX_SUMMARY_INPUT_CHARS
SEARXNG_TIMEOUT = 10
SCRAPE_TIMEOUT = 15
FAST_TIMEOUT = 60


def searxng_search(query: str, max_results: int = MAX_SOURCES) -> list[dict]:
    search_url = f"{SEARXNG_URL}/search"
    _log.debug("searxng request  url=%s query=%s", search_url, query[:120])
    try:
        resp = httpx.get(
            search_url,
            params={"q": query, "format": "json"},
            timeout=SEARXNG_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        _log.error("searxng %d for '%s': %s", e.response.status_code, query, e.response.text[:300])
        return []
    except httpx.TimeoutException:
        _log.error("searxng timeout after %ds for '%s'", SEARXNG_TIMEOUT, query)
        return []
    except Exception:
        _log.error("searxng failed for '%s'", query, exc_info=True)
        return []

    results = data.get("results") or []
    out: list[dict] = []
    for r in results[: max_results * 2]:
        url = r.get("url")
        if not url:
            continue
        out.append({
            "title": (r.get("title") or "").strip()[:200],
            "url": url,
            "snippet": (r.get("content") or "").strip(),
        })
    _log.debug("searxng returned %d raw results, kept %d", len(results), len(out))
    return out


def _dedupe(results: Iterable[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for r in results:
        url = r.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(r)
    return out
