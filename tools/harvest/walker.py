"""Link extraction strategies for the harvest walker.

Pure functions on already-fetched HTML. Returns absolute URLs, deduped.
"""
from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin, urlparse

_log = logging.getLogger("harvest.walker")

_BINARY_EXTS = (
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp", ".ico",
    ".mp3", ".mp4", ".mov", ".avi", ".webm", ".wav", ".ogg",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".css", ".js", ".map",
)
_JUNK_PATTERNS = (
    "/login", "/signin", "/signup", "/register", "/logout",
    "/cdn-cgi/", "javascript:", "mailto:", "tel:",
    "/wp-json/", "/feed/", "/feed.xml",
)


def _normalise(url: str) -> str:
    if not url:
        return ""
    url = url.strip()
    url, _ = urldefrag(url)
    return url


def _is_binary(url: str) -> bool:
    low = url.lower().split("?", 1)[0]
    return any(low.endswith(ext) for ext in _BINARY_EXTS)


def _is_junk(url: str) -> bool:
    low = url.lower()
    return any(p in low for p in _JUNK_PATTERNS)


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


class _LinkParser(HTMLParser):
    def __init__(self, base_url: str, link_class: str):
        super().__init__()
        self.base_url = base_url
        self.link_class = link_class
        self.links: list[str] = []
        self._in_article = 0
        self._article_only = (link_class == "article")

    def handle_starttag(self, tag, attrs):
        if tag in ("article", "main"):
            self._in_article += 1
        if tag != "a":
            return
        if self._article_only and self._in_article == 0:
            return
        href = ""
        for k, v in attrs:
            if k == "href" and v:
                href = v
                break
        if not href:
            return
        absolute = _normalise(urljoin(self.base_url, href))
        if absolute:
            self.links.append(absolute)

    def handle_endtag(self, tag):
        if tag in ("article", "main") and self._in_article > 0:
            self._in_article -= 1


def extract_links(html: str, base_url: str, *,
                  same_host_only: bool = True,
                  link_class: str = "all",
                  url_pattern: str | None = None,
                  max_links: int = 200) -> list[str]:
    """Extract candidate URLs from a fetched HTML page."""
    if not html:
        return []
    parser = _LinkParser(base_url=base_url, link_class=link_class)
    try:
        parser.feed(html)
    except Exception as e:
        _log.debug("link parse failed for %s: %s", base_url[:100], e)
        return []

    base_host = _host(base_url)
    pattern_re = re.compile(url_pattern) if url_pattern else None

    seen: set[str] = set()
    out: list[str] = []
    for u in parser.links:
        if not u or u in seen:
            continue
        seen.add(u)
        if _is_binary(u) or _is_junk(u):
            continue
        if same_host_only and _host(u) != base_host:
            continue
        if pattern_re and not pattern_re.search(u):
            continue
        out.append(u)
        if len(out) >= max_links:
            break
    return out


def extract_sitemap_urls(xml_text: str, *, max_urls: int = 1000) -> list[str]:
    """Pull `<loc>` entries from a sitemap XML body. Handles sitemap-index
    nesting one level deep (returns child sitemap URLs alongside docs)."""
    if not xml_text:
        return []
    # Cheap regex-based extraction; avoids pulling in lxml unless needed.
    locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml_text, flags=re.IGNORECASE)
    seen: set[str] = set()
    out: list[str] = []
    for u in locs:
        u = _normalise(u)
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(u)
        if len(out) >= max_urls:
            break
    return out


def extract_rss_entries(rss_text: str, *, max_entries: int = 200) -> list[dict]:
    """Pull RSS/Atom entries. Returns list of {url, title, published, summary}.

    Uses feedparser if available, otherwise a regex fallback for `<link>`.
    """
    if not rss_text:
        return []
    try:
        import feedparser  # type: ignore
    except ImportError:
        # Minimal fallback — extract <link> URLs only
        links = re.findall(r"<link[^>]*>([^<]+)</link>", rss_text)
        return [
            {"url": _normalise(u), "title": "", "published": "", "summary": ""}
            for u in links[:max_entries]
            if u.strip().startswith("http")
        ]

    parsed = feedparser.parse(rss_text)
    out: list[dict] = []
    for e in (parsed.entries or [])[:max_entries]:
        url = getattr(e, "link", "") or ""
        if not url:
            continue
        out.append({
            "url": _normalise(url),
            "title": getattr(e, "title", "") or "",
            "published": getattr(e, "published", "") or getattr(e, "updated", "") or "",
            "summary": getattr(e, "summary", "") or "",
        })
    return out
