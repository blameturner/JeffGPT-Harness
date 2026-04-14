import logging
from abc import ABC, abstractmethod
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from tools.search.scraping import scrape_page as default_scrape

_log = logging.getLogger("scraper.base")


class BaseScraper(ABC):
    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def fetch(self, url: str) -> str:
        try:
            return default_scrape(url) or self._fallback_fetch(url)
        except Exception as e:
            _log.warning("scrape failed  url=%s  error=%s", url[:80], e)
            return ""

    def _fallback_fetch(self, url: str) -> str:
        try:
            resp = httpx.get(url, timeout=self.timeout, follow_redirects=True)
            resp.raise_for_status()
            return resp.text
        except Exception:
            return ""

    def extract_links(self, html: str, base_url: str) -> list[str]:
        links = []
        try:
            soup = BeautifulSoup(html, "lxml")
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                if href.startswith(("mailto:", "tel:", "javascript:")):
                    continue
                if not href.startswith("http"):
                    from urllib.parse import urljoin
                    href = urljoin(base_url, href)
                if href.startswith("http"):
                    links.append(href)
        except Exception:
            pass
        return list(set(links))

    def extract_text(self, html: str) -> str:
        try:
            soup = BeautifulSoup(html, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)
        except Exception:
            return ""

    def get_domain(self, url: str) -> str:
        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return ""

    @abstractmethod
    def scrape(self, url: str) -> dict:
        pass