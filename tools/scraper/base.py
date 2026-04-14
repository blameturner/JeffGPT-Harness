from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from tools.search.scraping import scrape_page as _search_scrape_text

_log = logging.getLogger("scraper.base")

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; JeffGPT-Pathfinder/1.0; +https://jeffgpt.local/bot)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class BaseScraper(ABC):
    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def fetch_html(self, url: str) -> tuple[str, str]:
        try:
            resp = httpx.get(
                url, headers=_HEADERS, timeout=self.timeout, follow_redirects=True
            )
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if "html" not in ct and "xml" not in ct:
                return "", str(resp.url)
            return resp.text, str(resp.url)
        except Exception as e:
            _log.debug("fetch_html failed  url=%s  error=%s", url[:80], e)
            return "", url

    def fetch_text(self, url: str) -> str:
        try:
            return _search_scrape_text(url) or ""
        except Exception as e:
            _log.debug("fetch_text failed  url=%s  error=%s", url[:80], e)
            return ""

    def extract_links(self, html: str, base_url: str) -> list[str]:
        if not html:
            return []
        try:
            soup = BeautifulSoup(html, "lxml")
            out: set[str] = set()
            for a in soup.find_all("a", href=True):
                href = (a.get("href") or "").strip()
                if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
                    continue
                joined = urljoin(base_url, href)
                if joined.startswith(("http://", "https://")):
                    out.add(joined)
            return list(out)
        except Exception:
            return []

    def extract_text(self, html: str) -> str:
        if not html:
            return ""
        try:
            soup = BeautifulSoup(html, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)
        except Exception:
            return ""

    def canonical_url(self, html: str, fallback: str) -> str:
        try:
            soup = BeautifulSoup(html, "lxml")
            for tag in soup.find_all("link"):
                rel = tag.get("rel") or []
                if isinstance(rel, str):
                    rel = [rel]
                if "canonical" in rel and tag.get("href"):
                    return urljoin(fallback, str(tag["href"]))
        except Exception:
            pass
        return fallback

    def get_domain(self, url: str) -> str:
        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return ""

    @abstractmethod
    def scrape(self, url: str) -> dict:
        ...