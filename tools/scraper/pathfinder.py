from __future__ import annotations

import logging
import re

import httpx
from bs4 import BeautifulSoup

from tools.scraper.base import BaseScraper
from tools.search.scraping import playwright_fetch_html
from tools.search.urls import BROWSER_HEADERS

_log = logging.getLogger("scraper.pathfinder")

_HTML_SNIFF = re.compile(rb"<\s*(?:html|head|body|!doctype|a\b|link\b)", re.IGNORECASE)
_ANTIBOT_MARKERS = (
    "checking your browser", "cloudflare", "ddos protection",
    "enable javascript", "verify you are human", "captcha",
    "access denied", "just a moment", "attention required",
    "one more step",
)


def _looks_like_antibot_html(html: str) -> bool:
    if not html:
        return False
    if len(html) > 8000:
        return False
    lower = html.lower()
    return any(m in lower for m in _ANTIBOT_MARKERS)


class PathfinderScraper(BaseScraper):
    """Scraper tuned for link discovery: realistic UA, relaxed CT check, Playwright fallback,
    and aggressive link extraction (anchors, areas, link rel=alternate/next/prev, og:url)."""

    def __init__(self, timeout: int = 30, headers: dict | None = None):
        super().__init__(timeout=timeout)
        self.headers = headers or BROWSER_HEADERS

    def fetch_html(self, url: str) -> tuple[str, str]:
        html, final_url, ok = self._fetch_httpx(url)
        if ok and html and not _looks_like_antibot_html(html):
            return html, final_url

        reason = "no html" if not html else ("antibot" if _looks_like_antibot_html(html) else "weak")
        _log.info("pathfinder httpx %s, falling back to playwright  url=%s", reason, url[:120])
        pw_html, pw_final = playwright_fetch_html(url)
        if pw_html:
            return pw_html, pw_final or final_url or url
        _log.warning("pathfinder fetch failed (httpx + playwright)  url=%s", url[:120])
        return "", final_url or url

    def _fetch_httpx(self, url: str) -> tuple[str, str, bool]:
        try:
            resp = httpx.get(
                url, headers=self.headers, timeout=self.timeout,
                follow_redirects=True,
            )
        except httpx.ConnectError as e:
            err = str(e).lower()
            if "ssl" in err or "certificate" in err:
                try:
                    resp = httpx.get(
                        url, headers=self.headers, timeout=self.timeout,
                        follow_redirects=True, verify=False,
                    )
                except Exception as e2:
                    _log.debug("pathfinder ssl retry failed  url=%s  error=%s", url[:80], e2)
                    return "", url, False
            else:
                _log.debug("pathfinder connect error  url=%s  error=%s", url[:80], e)
                return "", url, False
        except Exception as e:
            _log.debug("pathfinder fetch_html failed  url=%s  error=%s", url[:80], e)
            return "", url, False

        final_url = str(resp.url)
        if resp.status_code >= 400:
            _log.info("pathfinder http %d  url=%s", resp.status_code, url[:80])
            return "", final_url, False

        body = resp.content or b""
        if len(body) < 100:
            return "", final_url, False

        ct = (resp.headers.get("content-type") or "").lower()
        if ct and not any(t in ct for t in ("html", "xml", "text/plain")):
            sniff = body[:1024]
            if not _HTML_SNIFF.search(sniff):
                _log.info("pathfinder skip non-html  ct=%s  url=%s", ct.split(";")[0], url[:80])
                return "", final_url, False

        try:
            text = resp.text
        except Exception:
            return "", final_url, False
        return text, final_url, True

    def extract_links(self, html: str, base_url: str) -> list[str]:
        if not html:
            return []
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            return []

        from urllib.parse import urljoin

        out: set[str] = set()

        for a in soup.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
                continue
            joined = urljoin(base_url, href)
            if joined.startswith(("http://", "https://")):
                out.add(joined)

        for area in soup.find_all("area", href=True):
            href = (area.get("href") or "").strip()
            if href and not href.startswith(("mailto:", "tel:", "javascript:", "#")):
                joined = urljoin(base_url, href)
                if joined.startswith(("http://", "https://")):
                    out.add(joined)

        for link in soup.find_all("link", href=True):
            rel = link.get("rel") or []
            if isinstance(rel, str):
                rel = [rel]
            rel_lc = {r.lower() for r in rel if isinstance(r, str)}
            if rel_lc & {"alternate", "next", "prev", "canonical"}:
                joined = urljoin(base_url, link.get("href", "").strip())
                if joined.startswith(("http://", "https://")):
                    out.add(joined)

        for meta in soup.find_all("meta", property=True):
            prop = (meta.get("property") or "").lower()
            if prop in ("og:url", "og:see_also"):
                content = (meta.get("content") or "").strip()
                if content:
                    joined = urljoin(base_url, content)
                    if joined.startswith(("http://", "https://")):
                        out.add(joined)

        for iframe in soup.find_all("iframe", src=True):
            src = (iframe.get("src") or "").strip()
            if src and not src.startswith(("data:", "javascript:", "#")):
                joined = urljoin(base_url, src)
                if joined.startswith(("http://", "https://")):
                    out.add(joined)

        return list(out)

    def scrape(self, url: str) -> dict:
        result = {
            "url": url,
            "final_url": url,
            "text": "",
            "links": [],
            "domain": "",
            "canonical": url,
            "status": "failed",
            "error": "",
        }

        html, final_url = self.fetch_html(url)
        if not html:
            result["error"] = "empty_response"
            return result

        try:
            result["final_url"] = final_url
            result["canonical"] = self.canonical_url(html, final_url)
            result["text"] = self.extract_text(html)
            result["links"] = self.extract_links(html, final_url)
            result["domain"] = self.get_domain(final_url)
            result["status"] = "ok"
        except Exception as e:
            result["error"] = str(e)[:200]
            _log.warning("pathfinder scrape failed  url=%s  error=%s", url[:80], e)

        return result
