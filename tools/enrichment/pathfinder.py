from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
import urllib.robotparser as robotparser
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import parse_qsl, urldefrag, urlencode, urljoin, urlparse, urlunparse

import httpx

from infra.nocodb_client import NocodbClient
from tools.scraper.search import SearchScraper

_log = logging.getLogger("pathfinder")

DEFAULT_MAX_DEPTH = 3
DEFAULT_MAX_PAGES = 200
DEFAULT_CONCURRENCY = 4
DEFAULT_PER_HOST_DELAY_S = 1.0
ROBOTS_CACHE_TTL_S = 3600

_BINARY_EXT = re.compile(
    r"\.(jpg|jpeg|png|gif|webp|svg|ico|bmp|tif|tiff|"
    r"mp3|mp4|avi|mov|webm|wav|flac|ogg|"
    r"zip|tar|gz|7z|rar|dmg|exe|msi|iso|"
    r"pdf|doc|docx|xls|xlsx|ppt|pptx|csv|rtf|epub)(\?.*)?$",
    re.IGNORECASE,
)

_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "msclkid", "mc_cid", "mc_eid", "ref", "referrer",
    "_ga", "_gl", "yclid", "igshid",
}

_JUNK_PATH_PATTERNS = re.compile(
    r"/(login|logout|signin|signout|signup|register|cart|checkout|"
    r"admin|wp-admin|wp-login|account|profile|settings|preferences|"
    r"feed|rss|atom|print|share|email|subscribe)(/|$|\?)",
    re.IGNORECASE,
)

_JUNK_QUERY_PATTERNS = re.compile(
    r"(?:^|&)(sort|order|view|display|sessionid|phpsessid|jsessionid)=",
    re.IGNORECASE,
)


def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:16]


def _normalize(url: str) -> str:
    if not url:
        return ""
    url, _ = urldefrag(url.strip())
    try:
        parts = urlparse(url)
    except Exception:
        return ""
    if parts.scheme not in ("http", "https"):
        return ""

    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = re.sub(r"/+", "/", parts.path) or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    params = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=False)
              if k.lower() not in _TRACKING_PARAMS]
    params.sort()
    query = urlencode(params, doseq=True)

    return urlunparse((parts.scheme, host, path, "", query, ""))


def _is_binary(url: str) -> bool:
    return bool(_BINARY_EXT.search(url))


def _is_junk(url: str) -> bool:
    try:
        parts = urlparse(url)
    except Exception:
        return True
    if _JUNK_PATH_PATTERNS.search(parts.path):
        return True
    if _JUNK_QUERY_PATTERNS.search(parts.query):
        return True
    return False


def _host(url: str) -> str:
    try:
        h = urlparse(url).netloc.lower()
        return h[4:] if h.startswith("www.") else h
    except Exception:
        return ""


def _score(url: str, seed_host: str, seed_path: str, depth: int) -> float:
    score = 100.0 - depth * 15.0
    try:
        parts = urlparse(url)
    except Exception:
        return 0.0

    if _host(url) == seed_host:
        score += 25.0
    else:
        score -= 40.0

    if seed_path and seed_path != "/" and parts.path.startswith(seed_path):
        score += 15.0

    if parts.query:
        score -= min(10.0, len(parts.query) / 20.0)

    segments = [s for s in parts.path.split("/") if s]
    score -= min(15.0, len(segments) * 2.0)

    if parts.path in ("/", "") or parts.path.endswith(("/about", "/about-us", "/team",
                                                         "/products", "/services",
                                                         "/research", "/blog", "/news")):
        score += 5.0

    return round(max(0.0, score), 2)


class _RobotsCache:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[robotparser.RobotFileParser, float]] = {}
        self._lock = threading.Lock()

    def allowed(self, url: str, user_agent: str) -> bool:
        host_key = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        now = time.time()
        with self._lock:
            entry = self._cache.get(host_key)
            if entry and now - entry[1] < ROBOTS_CACHE_TTL_S:
                rp = entry[0]
            else:
                rp = robotparser.RobotFileParser()
                rp.set_url(f"{host_key}/robots.txt")
                try:
                    rp.read()
                except Exception:
                    rp = robotparser.RobotFileParser()
                self._cache[host_key] = (rp, now)
        try:
            return rp.can_fetch(user_agent, url)
        except Exception:
            return True

    def sitemaps(self, url: str) -> list[str]:
        host_key = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        with self._lock:
            entry = self._cache.get(host_key)
        if entry:
            try:
                sm = entry[0].site_maps() or []
                return list(sm)
            except Exception:
                return []
        return []


class _HostRateLimiter:
    def __init__(self, delay: float = DEFAULT_PER_HOST_DELAY_S):
        self._delay = delay
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> None:
        with self._lock:
            last = self._last.get(host, 0.0)
            now = time.time()
            wait_s = self._delay - (now - last)
            self._last[host] = now + max(0.0, wait_s)
        if wait_s > 0:
            time.sleep(wait_s)


def _fetch_sitemap_urls(sitemap_url: str, limit: int = 500) -> list[str]:
    try:
        resp = httpx.get(sitemap_url, timeout=15, follow_redirects=True)
        resp.raise_for_status()
        body = resp.text
    except Exception:
        return []
    urls = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", body)
    return urls[:limit]


def _url_exists(client: NocodbClient, url_hash: str, org_id: int) -> bool:
    try:
        data = client._get("discovery", params={
            "where": f"(url_hash,eq,{url_hash})~and(org_id,eq,{org_id})",
            "limit": 1,
        })
        return bool(data.get("list"))
    except Exception:
        return False


def _add_url(
    client: NocodbClient,
    url: str,
    source_url: str,
    depth: int,
    org_id: int,
    score: float,
) -> bool:
    uh = _url_hash(url)
    if _url_exists(client, uh, org_id):
        return False
    try:
        client._post("discovery", {
            "org_id": org_id,
            "url": url,
            "url_hash": uh,
            "source_url": source_url,
            "depth": depth,
            "domain": _host(url),
            "score": score,
            "status": "discovered",
        })
        return True
    except Exception as e:
        _log.warning("add url failed  url=%s  error=%s", url[:80], e)
        return False


def discover(
    seed_url: str,
    org_id: int,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_pages: int = DEFAULT_MAX_PAGES,
    same_host_only: bool = True,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict:
    seed = _normalize(seed_url)
    if not seed:
        return {"processed": 0, "added": 0, "error": "invalid_seed"}

    seed_host = _host(seed)
    seed_path = urlparse(seed).path or "/"

    scraper = SearchScraper(timeout=30)
    robots = _RobotsCache()
    limiter = _HostRateLimiter()
    client = NocodbClient()
    ua = "JeffGPT-Pathfinder/1.0"

    visited: set[str] = set()
    visited_lock = threading.Lock()
    frontier: deque[tuple[str, int, str]] = deque()  # (url, depth, source)
    frontier.append((seed, 0, ""))

    # seed with sitemap hints (robots.txt Sitemap: lines)
    if robots.allowed(seed, ua):
        for sm in robots.sitemaps(seed):
            for u in _fetch_sitemap_urls(sm, limit=200):
                n = _normalize(u)
                if n and (not same_host_only or _host(n) == seed_host):
                    frontier.append((n, 1, sm))

    added = 0
    processed = 0
    errors = 0
    skipped_robots = 0
    skipped_filter = 0

    def _process(url: str, depth: int, source: str) -> tuple[int, int, int]:
        """Returns (added, skipped_filter, skipped_robots)."""
        if depth > max_depth:
            return 0, 0, 0
        if _is_binary(url) or _is_junk(url):
            return 0, 1, 0
        if not robots.allowed(url, ua):
            return 0, 0, 1

        limiter.wait(_host(url))

        result = scraper.scrape(url)
        if result.get("status") != "ok":
            return 0, 0, 0

        final = _normalize(result.get("final_url") or url)
        canon = _normalize(result.get("canonical") or final)
        record_url = canon or final or url

        sc = _score(record_url, seed_host, seed_path, depth)
        local_added = 1 if _add_url(client, record_url, source or "", depth, org_id, sc) else 0

        local_filtered = 0
        for link in result.get("links", []):
            n = _normalize(link)
            if not n:
                local_filtered += 1
                continue
            if same_host_only and _host(n) != seed_host:
                local_filtered += 1
                continue
            if _is_binary(n) or _is_junk(n):
                local_filtered += 1
                continue
            with visited_lock:
                if n in visited:
                    continue
                visited.add(n)
            frontier.append((n, depth + 1, record_url))

        return local_added, local_filtered, 0

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        in_flight: set = set()
        while (frontier or in_flight) and processed < max_pages:
            while frontier and len(in_flight) < concurrency and processed + len(in_flight) < max_pages:
                url, depth, source = frontier.popleft()
                with visited_lock:
                    if url in visited:
                        continue
                    visited.add(url)
                in_flight.add(pool.submit(_process, url, depth, source))

            if not in_flight:
                break

            done, in_flight = _wait_any(in_flight)
            for fut in done:
                processed += 1
                try:
                    a, sf, sr = fut.result()
                    added += a
                    skipped_filter += sf
                    skipped_robots += sr
                except Exception as e:
                    errors += 1
                    _log.debug("worker error: %s", e)

    _log.info(
        "pathfinder done  seed=%s  processed=%d  added=%d  filtered=%d  robots_blocked=%d  errors=%d",
        seed[:80], processed, added, skipped_filter, skipped_robots, errors,
    )
    return {
        "processed": processed,
        "added": added,
        "skipped_filter": skipped_filter,
        "skipped_robots": skipped_robots,
        "errors": errors,
        "frontier_remaining": len(frontier),
    }


def _wait_any(futures: set) -> tuple[list, set]:
    done = []
    pending = set(futures)
    for fut in as_completed(futures, timeout=None):
        done.append(fut)
        pending.discard(fut)
        break
    return done, pending


def fetch_next() -> dict | None:
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
        client._patch("discovery", url_id, {"status": "failed", "error_message": error[:500]})
    except Exception:
        _log.warning("mark_failed failed  id=%d", url_id)
