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

from infra.config import get_feature
from infra.nocodb_client import NocodbClient
from tools.scraper.pathfinder import PathfinderScraper

_log = logging.getLogger("pathfinder")

DEFAULT_MAX_DEPTH = 0
DEFAULT_MAX_PAGES = 500
DEFAULT_CONCURRENCY = 8
DEFAULT_BATCH_SIZE = 5
DEFAULT_PER_HOST_DELAY_S = 0.5
DEFAULT_ROBOTS_CACHE_TTL_S = 3600
DEFAULT_SITEMAP_LIMIT = 500
SITEMAP_CANDIDATES = ("/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml", "/sitemap.xml.gz")


def _cfg(key: str, default):
    return get_feature("pathfinder", key, default)


ROBOTS_CACHE_TTL_S = _cfg("robots_cache_ttl_s", DEFAULT_ROBOTS_CACHE_TTL_S)

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


def _discovery_url_exists(client: NocodbClient, url_hash: str, org_id: int) -> int | None:
    try:
        data = client._get("discovery", params={
            "where": f"(url_hash,eq,{url_hash})~and(org_id,eq,{org_id})",
            "limit": 1,
        })
        rows = data.get("list", [])
        return rows[0].get("Id") if rows else None
    except Exception:
        return None


def _scrape_target_exists(client: NocodbClient, url: str, org_id: int) -> bool:
    try:
        data = client._get("scrape_targets", params={
            "where": f"(url,eq,{url})~and(org_id,eq,{org_id})",
            "limit": 1,
        })
        return bool(data.get("list"))
    except Exception:
        return False


def upsert_discovery_root(
    client: NocodbClient,
    url: str,
    org_id: int,
    score: float = 100.0,
    reset_status: bool = False,
) -> int | None:
    """Ensure the seed URL is recorded in the discovery table; return its row Id.

    The URL is always normalized before hashing so that a raw URL from the API
    and the normalized URL produced inside ``discover()`` resolve to the same row
    and don't create duplicates.

    If the row already exists and ``reset_status`` is True (set by the API
    endpoint on explicit re-submission), the status is reset to 'discovered' so
    the scheduler can pick it up on the next cycle even if the in-memory queue
    job was lost due to a restart.
    """
    url = _normalize(url) or url  # always hash the canonical form
    if not url:
        return None
    uh = _url_hash(url)
    existing_id = _discovery_url_exists(client, uh, org_id)
    if existing_id:
        if reset_status:
            try:
                client._patch("discovery", existing_id, {
                    "status": "discovered",
                    "error_message": "",
                })
                _log.info("upsert_discovery_root reset status=discovered  id=%d  url=%s", existing_id, url[:80])
            except Exception:
                _log.warning("upsert_discovery_root status reset failed  id=%d", existing_id, exc_info=True)
        return existing_id
    try:
        row = client._post("discovery", {
            "org_id": org_id,
            "url": url,
            "url_hash": uh,
            "source_url": "",
            "depth": 0,
            "domain": _host(url),
            "score": score,
            "status": "discovered",
        })
        return row.get("Id")
    except Exception as e:
        _log.warning("discovery seed insert failed  url=%s  error=%s", url[:80], e)
        return None


def _derive_target_name(url: str) -> str:
    parts = urlparse(url)
    path = (parts.path or "/").strip("/").replace("/", " · ") or parts.netloc
    return f"{parts.netloc}: {path}"[:255]


def _add_to_scrape_targets(
    client: NocodbClient,
    url: str,
    source_url: str,
    depth: int,
    org_id: int,
    score: float,
    parent_target_id: int | None = None,
    frequency_hours: int = 24,
) -> bool:
    """Insert a discovered URL into scrape_targets so the enrichment scraper picks it up."""
    if _scrape_target_exists(client, url, org_id):
        return False
    payload = {
        "org_id": org_id,
        "url": url,
        "name": _derive_target_name(url),
        "category": "auto",
        "active": 1,
        "frequency_hours": frequency_hours,
        "depth": depth,
        "discovered_from": source_url or "",
        "auto_crawled": 1,
        "consecutive_failures": 0,
        "consecutive_unchanged": 0,
        "chunk_count": 0,
    }
    if parent_target_id:
        payload["parent_target"] = parent_target_id
    try:
        client._post("scrape_targets", payload)
        return True
    except Exception as e:
        _log.warning("scrape_targets insert failed  url=%s  error=%s", url[:80], e)
        return False


def discover(
    seed_url: str,
    org_id: int,
    max_depth: int | None = None,
    max_pages: int | None = None,
    same_host_only: bool = True,
    concurrency: int | None = None,
    register_as_root: bool = True,
) -> dict:
    if not _cfg("enabled", True):
        return {"processed": 0, "added": 0, "error": "pathfinder_disabled"}

    max_depth = max_depth if max_depth is not None else _cfg("max_depth", DEFAULT_MAX_DEPTH)
    max_pages = max_pages if max_pages is not None else _cfg("max_pages", DEFAULT_MAX_PAGES)
    concurrency = concurrency if concurrency is not None else _cfg("concurrency", DEFAULT_CONCURRENCY)
    sitemap_limit = _cfg("sitemap_limit", DEFAULT_SITEMAP_LIMIT)

    seed = _normalize(seed_url)
    if not seed:
        return {"processed": 0, "added": 0, "error": "invalid_seed"}

    seed_host = _host(seed)
    seed_path = urlparse(seed).path or "/"

    scraper = PathfinderScraper(timeout=30)
    robots = _RobotsCache()
    limiter = _HostRateLimiter(delay=_cfg("per_host_delay_s", DEFAULT_PER_HOST_DELAY_S))
    client = NocodbClient()
    ua = "Mozilla/5.0 (compatible; JeffGPT-Pathfinder/1.0)"

    # ensure the seed exists as a discovery row (root) so future re-crawls have an anchor.
    # Skip when the caller is crawling a scrape_target URL as a fallback seed — those
    # shouldn't be promoted to discovery roots.
    discovery_root_id = upsert_discovery_root(client, seed, org_id) if register_as_root else None

    # `enqueued` tracks every URL that's ever been put in the frontier (so we don't
    # enqueue duplicates). The outer submit loop trusts that everything in the
    # frontier is unique and just submits.
    enqueued: set[str] = set()
    enqueued_lock = threading.Lock()
    frontier: deque[tuple[str, int, str]] = deque()
    enqueued.add(seed)
    frontier.append((seed, 0, ""))

    # seed with sitemap hints (robots.txt Sitemap: lines, then common fallback paths)
    sitemap_urls_seeded = 0
    if robots.allowed(seed, ua):
        sitemaps = list(robots.sitemaps(seed))
        if not sitemaps:
            parts = urlparse(seed)
            base = f"{parts.scheme}://{parts.netloc}"
            sitemaps = [f"{base}{path}" for path in SITEMAP_CANDIDATES]
        for sm in sitemaps:
            for u in _fetch_sitemap_urls(sm, limit=sitemap_limit):
                n = _normalize(u)
                if not n:
                    continue
                if same_host_only and _host(n) != seed_host:
                    continue
                if n in enqueued:
                    continue
                enqueued.add(n)
                frontier.append((n, 1, sm))
                sitemap_urls_seeded += 1
    _log.info("pathfinder seed=%s  sitemap_urls_seeded=%d", seed[:80], sitemap_urls_seeded)

    added = 0
    processed = 0
    errors = 0
    skipped_robots = 0
    skipped_filter = 0
    scrape_failed = 0
    seed_ok = False
    seed_link_count = 0
    seed_text_container: list[str] = [""]  # captured by _process when it scrapes the seed

    def _process(url: str, depth: int, source: str) -> tuple[int, int, int, int, bool, int]:
        """Returns (added, skipped_filter, skipped_robots, scrape_failed, is_seed_ok, link_count)."""
        if depth > max_depth:
            return 0, 0, 0, 0, False, 0
        if _is_binary(url) or _is_junk(url):
            return 0, 1, 0, 0, False, 0
        if not robots.allowed(url, ua):
            return 0, 0, 1, 0, False, 0

        limiter.wait(_host(url))

        result = scraper.scrape(url)
        if result.get("status") != "ok":
            err = result.get("error") or "unknown"
            _log.info("pathfinder scrape failed  depth=%d  url=%s  reason=%s", depth, url[:100], err)
            return 0, 0, 0, 1, False, 0

        final = _normalize(result.get("final_url") or url)
        canon = _normalize(result.get("canonical") or final)
        record_url = canon or final or url

        # the seed itself stays in `discovery` (already upserted as root); every other
        # URL pathfinder reaches gets recorded as a scrape_targets row so the
        # enrichment scraper can pick it up later.
        local_added = 0
        if depth > 0:
            sc = _score(record_url, seed_host, seed_path, depth)
            if _add_to_scrape_targets(
                client,
                record_url,
                source or "",
                depth,
                org_id,
                sc,
                parent_target_id=None,
            ):
                local_added = 1

        # queue a summarise tool_job for the page we just scraped, so the model run
        # happens out-of-band on the tool queue (priority 5 = background)
        page_text = (result.get("text") or "").strip()
        if depth == 0 and page_text:
            # stash the seed's text so the caller can pass it to the classifier
            seed_text_container[0] = page_text
        if page_text:
            try:
                from workers.tool_queue import get_tool_queue
                tq = get_tool_queue()
                if tq:
                    payload = {
                        "url": record_url,
                        "text": page_text[:30000],
                        "org_id": org_id,
                        "source": "pathfinder",
                    }
                    if depth == 0 and discovery_root_id:
                        payload["discovery_id"] = discovery_root_id
                    tq.submit("summarise_page", payload, source="pathfinder", priority=4, org_id=org_id)
            except Exception:
                _log.warning("summarise queue failed  url=%s", record_url[:80], exc_info=True)

        local_filtered = 0
        links = result.get("links") or []
        link_count = len(links)
        # Hard cap so a single MDN-style page can't create thousands of scrape_targets
        # rows + summarise_page jobs in one invocation. Remaining links from this page
        # will be rediscovered on next pathfinder run (capped rotation).
        max_links = int(_cfg("max_links_per_seed", 100))
        kept = 0
        for link in links:
            if kept >= max_links:
                local_filtered += (len(links) - max_links)  # tally the skipped tail once
                break
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
            with enqueued_lock:
                if n in enqueued:
                    continue
                enqueued.add(n)
            link_score = _score(n, seed_host, seed_path, depth + 1)
            if _add_to_scrape_targets(client, n, record_url, depth + 1, org_id, link_score):
                local_added += 1
                kept += 1
            if depth + 1 <= max_depth:
                frontier.append((n, depth + 1, record_url))

        is_seed = (depth == 0)
        _log.info(
            "pathfinder scraped  depth=%d  url=%s  links=%d  added=%d  filtered=%d",
            depth, record_url[:100], link_count, local_added, local_filtered,
        )
        return local_added, local_filtered, 0, 0, is_seed, link_count

    attempts_multiplier = _cfg("max_attempts_multiplier", 5)
    max_attempts = max(max_pages * attempts_multiplier, max_pages + 20)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        in_flight: set = set()
        while (frontier or in_flight) and added < max_pages and processed < max_attempts:
            while (frontier and len(in_flight) < concurrency
                   and added + len(in_flight) < max_pages
                   and processed + len(in_flight) < max_attempts):
                url, depth, source = frontier.popleft()
                # already deduped at frontier-append time via `enqueued`; just submit
                in_flight.add(pool.submit(_process, url, depth, source))

            if not in_flight:
                break

            done, in_flight = _wait_any(in_flight)
            for fut in done:
                processed += 1
                try:
                    a, sf, sr, sc_fail, is_seed_ok, link_count = fut.result()
                    added += a
                    skipped_filter += sf
                    skipped_robots += sr
                    scrape_failed += sc_fail
                    if is_seed_ok:
                        seed_ok = True
                        seed_link_count = link_count
                except Exception as e:
                    errors += 1
                    _log.debug("worker error: %s", e)

    if added < max_pages and processed >= max_attempts:
        _log.warning(
            "pathfinder hit attempts ceiling  added=%d/%d processed=%d scrape_failed=%d",
            added, max_pages, processed, scrape_failed,
        )

    if not seed_ok:
        _log.warning("pathfinder SEED SCRAPE FAILED  seed=%s — no links discovered from seed", seed[:120])
    elif seed_link_count == 0:
        _log.warning("pathfinder seed scraped but ZERO links extracted  seed=%s", seed[:120])

    # Update the discovery root row with just the status + error_message.
    # We deliberately DON'T write processed_at: NocoDB's Timestamp column has been
    # silently 400-ing PATCHes in this deployment, which left rows stuck with
    # processed_at=null and caused the picker to re-crawl the same 10 URLs forever.
    # The picker now uses the auto-managed `UpdatedAt` column (which refreshes on
    # every successful patch) to detect staleness, so we only need this PATCH to
    # succeed — not to carry a datetime value.
    if discovery_root_id:
        try:
            client._patch("discovery", discovery_root_id, {
                "status": "scraped" if seed_ok else "failed",
                "error_message": "" if seed_ok else "seed scrape failed",
            })
        except Exception:
            _log.warning("discovery root patch failed  id=%s", discovery_root_id, exc_info=True)

    _log.info(
        "pathfinder done  seed=%s  added=%d/%d  attempts=%d  filtered=%d  scrape_failed=%d  robots_blocked=%d  errors=%d  seed_ok=%s  seed_links=%d",
        seed[:80], added, max_pages, processed, skipped_filter, scrape_failed, skipped_robots, errors, seed_ok, seed_link_count,
    )
    return {
        "added": added,
        "target": max_pages,
        "attempts": processed,
        "processed": processed,  # backwards compat
        "skipped_filter": skipped_filter,
        "skipped_robots": skipped_robots,
        "scrape_failed": scrape_failed,
        "errors": errors,
        "frontier_remaining": len(frontier),
        "seed_ok": seed_ok,
        "seed_link_count": seed_link_count,
        "sitemap_urls_seeded": sitemap_urls_seeded,
        "discovery_root_id": discovery_root_id,
        # full seed page text so the caller (pathfinder_crawl_job) can forward it to
        # the classifier agent without re-scraping
        "seed_text": seed_text_container[0],
    }


def _pick_next_discovery_root(client: NocodbClient) -> dict | None:
    """Pick the next discovery root to crawl.

    Priority order:
    1. Never-crawled roots (status='discovered'), oldest first.
    2. Stale roots (status='scraped'/'failed') whose UpdatedAt is older than
       ``pathfinder.stale_after_minutes`` (default 60), rotated by UpdatedAt asc.

    Returns None when nothing is due — caller falls through to scrape_targets.
    """
    # 1. Never-crawled first.
    try:
        never = client._get("discovery", params={
            "where": "(depth,eq,0)~and(status,eq,discovered)",
            "limit": 1,
            "sort": "CreatedAt",
        }).get("list", [])
        if never:
            return never[0]
    except Exception:
        _log.warning("pathfinder picker never-crawled query failed", exc_info=True)

    # 2. Stale re-crawl: find the oldest previously-crawled root whose UpdatedAt
    #    pre-dates the stale window. We can't do date-arithmetic inside NocoDB's
    #    filter syntax, so we pull the oldest-UpdatedAt row and check client-side.
    #    Failed roots use a 3× longer window to avoid hammering broken URLs.
    stale_minutes = float(_cfg("stale_after_minutes", 60))
    if stale_minutes <= 0:
        return None
    try:
        from datetime import datetime, timezone
        rows = client._get("discovery", params={
            "where": "(depth,eq,0)~and(status,neq,discovered)",
            "limit": 1,
            "sort": "UpdatedAt,CreatedAt",
        }).get("list", [])
        if not rows:
            return None
        row = rows[0]
        row_status = row.get("status") or ""
        effective_stale = stale_minutes * 3 if row_status == "failed" else stale_minutes
        updated_str = row.get("UpdatedAt") or row.get("updated_at") or ""
        if not updated_str:
            # No timestamp — treat as stale only for non-failed rows to be safe
            return row if row_status != "failed" else None
        ts = datetime.fromisoformat(str(updated_str).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_minutes = (datetime.now(timezone.utc) - ts).total_seconds() / 60
        if age_minutes >= effective_stale:
            _log.info(
                "pathfinder stale re-crawl  id=%s url=%s status=%s age_min=%.1f threshold=%.0f",
                row.get("Id"), str(row.get("url") or "")[:80], row_status, age_minutes, effective_stale,
            )
            return row
    except Exception:
        _log.warning("pathfinder picker stale query failed", exc_info=True)
    return None


def _pick_next_scrape_target_seed(client: NocodbClient) -> dict | None:
    """Fallback picker: grab any active scrape_target to crawl as a new seed. Used
    when every discovery root is fresh. Sorted by UpdatedAt asc so we rotate through
    rows naturally — pathfinder touching a row bumps its UpdatedAt and it falls to
    the back of the queue until every other active target has also been tried."""
    if not _cfg("fallback_to_scrape_targets", True):
        return None
    try:
        rows = client._get("scrape_targets", params={
            "where": "(active,eq,1)",
            "limit": 1,
            "sort": "UpdatedAt,CreatedAt",
            # limit the payload so a big m2m expansion can't choke this call
            "fields": "Id,org_id,url,depth",
        }).get("list", [])
        return rows[0] if rows else None
    except Exception:
        _log.debug("pathfinder fallback scrape_targets query failed", exc_info=True)
        return None


# NOTE: `_chain_next_pathfinder` and `_count_queued_pathfinder` previously lived
# here and self-submitted the next pathfinder_crawl at the end of every handler
# invocation. With no per-type throttle and chat idle for long periods, that
# produced one job every ~1.5 s (the runaway the user reported). Both have been
# removed rather than left as dead code, so they can't be accidentally re-wired.
# The sole driver now is `jumpstart_pathfinder` in dispatcher.py (10-minute
# IntervalTrigger) backed by the handler-level cooldown gate above.


def _queue_classifier(target_id: int | None, url: str, text: str, org_id: int) -> None:
    """Queue the relevance classifier for this scrape_targets row. No-op if the
    row id isn't known (seed pages aren't in scrape_targets) or if text is empty."""
    if not target_id or not url or not text:
        return
    try:
        from workers.tool_queue import get_tool_queue
        tq = get_tool_queue()
        if tq:
            tq.submit(
                "classify_relevance",
                {"target_id": target_id, "url": url, "text": text[:20000], "org_id": org_id},
                source="pathfinder",
                priority=4,
                org_id=org_id,
            )
    except Exception:
        _log.warning("classifier queue failed  target_id=%s", target_id, exc_info=True)


def _process_one_seed(
    client: NocodbClient,
    seed_url: str,
    org_id: int,
    seed_source: str,
    discovery_id: int | None = None,
    scrape_target_id: int | None = None,
) -> dict:
    """Run discover() for a single URL, classify it if we have a scrape_target row,
    and update the originating row's bookkeeping so the picker rotates correctly."""
    if not seed_url:
        return {"status": "error", "reason": "no_url", "seed_source": seed_source}

    try:
        result = discover(seed_url, org_id, register_as_root=(seed_source == "discovery"))
    except Exception as e:
        _log.warning("pathfinder discover failed  source=%s url=%s error=%s",
                     seed_source, seed_url[:80], e, exc_info=True)
        result = {"status": "error", "reason": str(e)[:200]}

    # Queue the relevance classifier whenever we have a scrape_target row for this URL,
    # regardless of whether the seed came from discovery or a direct scrape_target pick.
    # Also nudge UpdatedAt on the scrape_target so the fallback picker rotates correctly.
    if scrape_target_id:
        seed_text = (result.get("seed_text") or "").strip() if isinstance(result, dict) else ""
        _queue_classifier(scrape_target_id, seed_url, seed_text, org_id)
        try:
            client._patch("scrape_targets", scrape_target_id, {"active": 1})
        except Exception:
            _log.debug("pathfinder scrape_target UpdatedAt nudge failed", exc_info=True)

    # Decide the job-level status. discover() uses two shapes for failure:
    #   - exception path: {"status": "error", "reason": ...}
    #   - early-return path: {"processed":0, "added":0, "error": "..."}
    # Either counts as an error for us. Strip `seed_text` before spreading (up to
    # 30k chars) so the tool_jobs.result_json stays small.
    is_error = bool(result.get("error")) or result.get("status") == "error"
    slim = {k: v for k, v in result.items() if k != "seed_text"} if isinstance(result, dict) else {}
    return {
        "status": "error" if is_error else "ok",
        "seed_source": seed_source,
        "discovery_id": discovery_id,
        "scrape_target_id": scrape_target_id,
        **slim,
    }


def _seconds_since_last_pathfinder_completion(client: NocodbClient) -> float:
    """How long since the newest REAL completed pathfinder_crawl finished. We
    skip past rows whose result carries "skipped_cooldown" because those are
    themselves products of this gate — counting them would let a flood of
    queued jobs reset the timer to "now" every time one of them skips, which
    would block real runs indefinitely as long as the flood lasts.

    Returns inf if no real completion is found in the recent window.
    """
    from datetime import datetime, timezone
    try:
        rows = client._get("tool_jobs", params={
            "where": "(type,eq,pathfinder_crawl)~and(status,eq,completed)",
            "sort": "-completed_at",
            "limit": 20,
        }).get("list", [])
    except Exception:
        return float("inf")
    for row in rows:
        result_str = row.get("result") or ""
        if "skipped_cooldown" in result_str:
            continue
        ts_str = row.get("completed_at") or ""
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        return (datetime.now(timezone.utc) - ts).total_seconds()
    return float("inf")


def pathfinder_crawl_job(payload: dict | int | None = None) -> dict:
    """Tool-queue handler. Processes ONE URL per invocation then EXITS.
    No self-chain — the `pathfinder.recrawl_interval_minutes` cron (default 10 min)
    is the sole driver, so pathfinder runs at most once per cron cycle regardless
    of chat activity or backoff state. This avoids the runaway chain that fired
    a new job every few seconds."""
    if isinstance(payload, int):
        direct_discovery_id: int | None = payload
    elif isinstance(payload, dict):
        direct_discovery_id = payload.get("discovery_id")
    else:
        direct_discovery_id = None

    client = NocodbClient()
    chain_org_id = 1  # default tenant if we can't derive from a picked row

    # Cooldown guard: bail if another pathfinder_crawl completed recently. Applies
    # only to scheduled/background runs (direct_discovery_id paths are user-initiated
    # and should run immediately). Prevents the 1.5s runaway log even if an unknown
    # submitter is flooding the queue.
    if direct_discovery_id is None:
        min_interval = float(_cfg("min_run_interval_seconds", 60))
        if min_interval > 0:
            elapsed = _seconds_since_last_pathfinder_completion(client)
            if elapsed < min_interval:
                _log.info(
                    "pathfinder skip: last completion %.1fs ago, gate=%.0fs",
                    elapsed, min_interval,
                )
                return {"status": "skipped_cooldown", "elapsed_s": round(elapsed, 1)}

    # UI-triggered path: specific discovery row.
    if direct_discovery_id:
        try:
            rows = client._get("discovery", params={
                "where": f"(Id,eq,{direct_discovery_id})",
                "limit": 1,
            }).get("list", [])
        except Exception:
            _log.warning("pathfinder direct lookup failed  id=%d", direct_discovery_id, exc_info=True)
            # no chain — cron (every pathfinder.recrawl_interval_minutes) drives the next run
            return {"status": "error", "discovery_id": direct_discovery_id}
        if not rows:
            # no chain — cron (every pathfinder.recrawl_interval_minutes) drives the next run
            return {"status": "not_found", "discovery_id": direct_discovery_id}
        row = rows[0]
        chain_org_id = int(row.get("org_id") or 0) or 1
        res = _process_one_seed(
            client,
            seed_url=row.get("url") or "",
            org_id=chain_org_id,
            seed_source="discovery",
            discovery_id=int(row.get("Id") or 0) or None,
        )
        # no chain — cron (every pathfinder.recrawl_interval_minutes) drives the next run
        return res

    # Scheduled/chained path: pick ONE seed. Stale discovery first, then oldest scrape_target.
    seed = _pick_next_discovery_root(client)
    if seed:
        chain_org_id = int(seed.get("org_id") or 0) or 1
        res = _process_one_seed(
            client,
            seed_url=seed.get("url") or "",
            org_id=chain_org_id,
            seed_source="discovery",
            discovery_id=int(seed.get("Id") or 0) or None,
        )
        # no chain — cron (every pathfinder.recrawl_interval_minutes) drives the next run
        return res

    fallback = _pick_next_scrape_target_seed(client)
    if fallback:
        chain_org_id = int(fallback.get("org_id") or 0) or 1
        fallback_url = _normalize(fallback.get("url") or "") or (fallback.get("url") or "")
        fallback_target_id = int(fallback.get("Id") or 0) or None

        # Register into discovery so it is tracked and won't be re-picked
        # as a blind fallback indefinitely. upsert_discovery_root is a no-op
        # if the row already exists.
        discovery_id = upsert_discovery_root(client, fallback_url, chain_org_id)
        _log.info(
            "pathfinder fallback scrape_target promoted to discovery  "
            "target_id=%s discovery_id=%s url=%s",
            fallback_target_id, discovery_id, fallback_url[:80],
        )

        res = _process_one_seed(
            client,
            seed_url=fallback_url,
            org_id=chain_org_id,
            seed_source="discovery",
            discovery_id=discovery_id,
            scrape_target_id=fallback_target_id,
        )
        # no chain — cron (every pathfinder.recrawl_interval_minutes) drives the next run
        return res

    # Nothing to do. Don't chain — let the 10-min cron restart us when work appears.
    _log.info("pathfinder idle: no stale discovery + no scrape_target fallback")
    return {"status": "idle"}


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
