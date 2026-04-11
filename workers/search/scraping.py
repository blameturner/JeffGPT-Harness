from __future__ import annotations

import logging
import queue
import re
import threading
import time
from concurrent.futures import Future

import httpx
from bs4 import BeautifulSoup

from workers.search.engine import PER_PAGE_CHAR_CAP, SCRAPE_TIMEOUT
from workers.search.urls import (
    BROWSER_HEADERS,
    BROWSER_UA,
    _is_blocklisted,
    _is_safe_url,
    _sanitise_url,
    _strip_injection_patterns,
)

_log = logging.getLogger("web_search.scraping")


def _nocodb_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


_STEALTH_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
window.chrome = {runtime: {}};
"""


# sync_playwright() is pinned to the thread that called .start(). All playwright
# calls run on ONE dedicated worker thread owned by this module; callers submit
# (url, Future) pairs. Any other arrangement violates the thread-affinity
# contract and crashes enrichment worker threads.
_PW_QUEUE_SENTINEL = object()
_pw_queue: queue.Queue | None = None
_pw_worker: threading.Thread | None = None
_pw_worker_lock = threading.Lock()
PLAYWRIGHT_FETCH_TIMEOUT = 60
MAX_RESPONSE_BYTES = 5_000_000


def _playwright_worker_main() -> None:
    pw_instance = None
    browser = None

    def _launch_browser():
        nonlocal pw_instance, browser
        from playwright.sync_api import sync_playwright
        pw_instance = sync_playwright().start()
        browser = pw_instance.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-extensions",
                "--disable-gpu",
                "--disable-background-networking",
                "--disable-default-apps",
                "--disable-sync",
                "--disable-translate",
            ],
        )
        _log.info("playwright chromium launched (worker thread)")

    def _teardown_browser():
        nonlocal pw_instance, browser
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass
        if pw_instance is not None:
            try:
                pw_instance.stop()
            except Exception:
                pass
        browser = None
        pw_instance = None

    def _browser_alive() -> bool:
        if browser is None:
            return False
        try:
            _ = browser.contexts
            return True
        except Exception:
            return False

    while True:
        item = _pw_queue.get()
        if item is _PW_QUEUE_SENTINEL:
            _teardown_browser()
            return
        url, fut = item
        if fut.cancelled():
            continue
        started = time.time()
        try:
            if not _browser_alive():
                _teardown_browser()
                _launch_browser()
            assert browser is not None
            context = browser.new_context(
                user_agent=BROWSER_UA,
                viewport={"width": 1280, "height": 800},
                java_script_enabled=True,
                service_workers="block",
                permissions=[],
                accept_downloads=False,
                locale="en-US",
                timezone_id="America/New_York",
            )
            try:
                page = context.new_page()
                page.add_init_script(_STEALTH_SCRIPT)

                _allowed_types = {"document", "stylesheet", "font", "image", "script", "xhr", "fetch"}
                page.route("**/*", lambda route: (
                    route.continue_() if route.request.resource_type in _allowed_types
                    else route.abort()
                ))

                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                try:
                    page.wait_for_load_state("networkidle", timeout=5_000)
                except Exception:
                    pass

                final_url = page.url
                if not _is_safe_url(final_url):
                    _log.warning("playwright_fetch blocked redirect to %s", final_url[:120])
                    fut.set_result("")
                    continue

                text = page.inner_text("body")
                text = _strip_injection_patterns(text)[:PER_PAGE_CHAR_CAP]
                elapsed = round(time.time() - started, 2)
                _log.info("playwright_fetch ok  url=%s chars=%d %.2fs", url[:120], len(text), elapsed)
                fut.set_result(text)
            finally:
                try:
                    context.close()
                except Exception:
                    pass
        except Exception as e:
            elapsed = round(time.time() - started, 2)
            _log.warning("playwright_fetch failed  url=%s error=%s %.2fs", url[:120], e, elapsed)
            if not _browser_alive():
                _log.warning("playwright browser appears dead, resetting")
                _teardown_browser()
            fut.set_result("")


def _ensure_playwright_worker() -> None:
    global _pw_queue, _pw_worker
    if _pw_worker is not None and _pw_worker.is_alive():
        return
    with _pw_worker_lock:
        if _pw_worker is not None and _pw_worker.is_alive():
            return
        _pw_queue = queue.Queue()
        _pw_worker = threading.Thread(
            target=_playwright_worker_main,
            name="playwright-worker",
            daemon=True,
        )
        _pw_worker.start()
        _log.info("playwright worker thread started")


def playwright_fetch(url: str) -> str:
    if not _is_safe_url(url):
        _log.warning("playwright_fetch blocked unsafe url %s", url[:120])
        return ""
    url = _sanitise_url(url)
    _ensure_playwright_worker()
    assert _pw_queue is not None
    fut: Future = Future()
    _pw_queue.put((url, fut))
    try:
        return fut.result(timeout=PLAYWRIGHT_FETCH_TIMEOUT)
    except Exception as e:
        _log.warning("playwright_fetch wait failed  url=%s error=%s", url[:120], e)
        return ""


def _scrape_with_httpx(url: str) -> str:
    started = time.time()
    try:
        resp = httpx.get(
            url,
            timeout=SCRAPE_TIMEOUT,
            follow_redirects=True,
            headers=BROWSER_HEADERS,
        )
        resp.raise_for_status()
        if len(resp.content) > MAX_RESPONSE_BYTES:
            _log.warning("scrape skip  response too large (%d bytes) for %s", len(resp.content), url)
            return ""
        final_url = str(resp.url)
        if final_url != url and not _is_safe_url(final_url):
            _log.warning("scrape blocked redirect to unsafe url %s", final_url[:120])
            return ""
    except httpx.HTTPStatusError as e:
        _log.warning("scrape %d for %s (%s)", e.response.status_code, url, e.response.reason_phrase)
        return ""
    except httpx.TimeoutException:
        _log.warning("scrape timeout after %ds for %s", SCRAPE_TIMEOUT, url)
        return ""
    except Exception as e:
        _log.warning("scrape failed for %s: %s", url, e)
        return ""

    elapsed = round(time.time() - started, 2)
    content_type = (resp.headers.get("content-type") or "").lower()
    if content_type and "text/html" not in content_type and "text/plain" not in content_type:
        _log.debug("scrape skip  non-html content-type=%s for %s", content_type.split(";")[0], url)
        return ""
    _log.debug("scrape ok    %s  status=%d size=%d %.2fs", url, resp.status_code, len(resp.text), elapsed)

    try:
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        _log.warning("html parse failed for %s: %s", url, e)
        return ""

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _strip_injection_patterns(text)
    text = text[:PER_PAGE_CHAR_CAP]
    if not text:
        _log.warning("scrape empty after extraction for %s (raw html was %d chars)", url, len(resp.text))
    else:
        _log.debug("scrape extracted %d chars from %s", len(text), url)
    return text


def scrape_page(
    url: str,
    snippet: str = "",
    source: dict | None = None,
    meta_out: dict | None = None,
) -> str:
    def _set_meta(path: str, chars: int) -> None:
        if meta_out is not None:
            meta_out["path"] = path
            meta_out["chars"] = chars

    fallback = _strip_injection_patterns(snippet)[:PER_PAGE_CHAR_CAP] if snippet else ""

    if not _is_safe_url(url):
        _log.warning("scrape skip  unsafe url %s", url[:120])
        _set_meta("unsafe", len(fallback))
        return fallback

    if _is_blocklisted(url):
        _log.debug("scrape skip  blocklisted %s", url)
        _set_meta("blocked", len(fallback))
        return fallback

    if source and _nocodb_truthy(source.get("use_playwright")):
        text = playwright_fetch(url)
        _log.info("path=playwright_direct  url=%s ok=%s", url[:120], bool(text))
        out = text or fallback
        _set_meta("playwright_direct", len(out))
        return out

    text = _scrape_with_httpx(url)

    if text and text != fallback:
        _log.info("path=scraper  url=%s chars=%d", url[:120], len(text))
        _set_meta("scraper", len(text))
        return text

    if source is not None:
        pw_text = playwright_fetch(url)
        if pw_text:
            target_id = source.get("Id")
            if target_id:
                try:
                    from workers.enrichment.db import EnrichmentDB
                    EnrichmentDB().update_scrape_target(target_id, use_playwright=True)
                    _log.info("auto-promoted to playwright_direct id=%s", target_id)
                except Exception as e:
                    _log.warning("playwright promotion failed id=%s: %s", target_id, e)
            _log.info("path=playwright_auto  url=%s chars=%d", url[:120], len(pw_text))
            _set_meta("playwright_auto", len(pw_text))
            return pw_text
        _log.info("path=playwright_auto  url=%s failed", url[:120])
        _set_meta("playwright_auto_failed", len(fallback))
        return fallback

    _set_meta("fallback", len(fallback))
    return fallback
