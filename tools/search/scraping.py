from __future__ import annotations

import logging
import queue
import re
import threading
import time
from concurrent.futures import Future

import httpx
from bs4 import BeautifulSoup

from tools.search.engine import PER_PAGE_CHAR_CAP, SCRAPE_TIMEOUT
from tools.search.urls import (
    BROWSER_HEADERS,
    BROWSER_UA,
    _is_blocklisted,
    _is_safe_url,
    _sanitise_url,
    _strip_injection_patterns,
)

_log = logging.getLogger("web_search.scraping")


def _looks_like_real_text(text: str) -> bool:
    if not text or len(text) < 80:
        return False
    printable = sum(1 for c in text if c.isprintable() or c.isspace())
    if printable / len(text) < 0.85:
        return False
    words = text.split()
    if len(words) < 15:
        return False
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len > 25 or avg_word_len < 2:
        return False
    return True


def _looks_like_pdf_url(url: str) -> bool:
    lower = url.lower().split("?", 1)[0].split("#", 1)[0]
    return lower.endswith(".pdf")


def _extract_pdf_text(url: str) -> str:
    try:
        import io
        import pdfplumber
        resp = httpx.get(url, headers=BROWSER_HEADERS, timeout=SCRAPE_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        if len(resp.content) > MAX_RESPONSE_BYTES:
            _log.warning("pdf too large  url=%s bytes=%d", url, len(resp.content))
            return ""
        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            pages = []
            max_pages = 20
            max_chars = PER_PAGE_CHAR_CAP
            chars = 0
            for i, p in enumerate(pdf.pages):
                if i >= max_pages or chars >= max_chars:
                    break
                text = p.extract_text() or ""
                remaining = max_chars - chars
                if len(text) > remaining:
                    text = text[:remaining]
                pages.append(text)
                chars += len(text)
        text = "\n\n".join(pages).strip()
        _log.info("pdf extracted  url=%s pages=%d chars=%d", url, len(pages), len(text))
        return text[:PER_PAGE_CHAR_CAP] if text else ""
    except ImportError:
        _log.warning("pdfplumber not installed; skipping pdf  url=%s", url)
        return ""
    except Exception:
        _log.warning("pdf extraction failed  url=%s", url, exc_info=True)
        return ""


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

_COOKIE_BANNER_NUKE = """
() => {
    const sel = [
        '[id*="cookie"]', '[class*="cookie"]', '[id*="consent"]', '[class*="consent"]',
        '[id*="gdpr"]', '[class*="gdpr"]', '[aria-label*="cookie" i]',
        '#onetrust-banner-sdk', '.onetrust-banner', '.ot-sdk-container',
        '[id*="CybotCookiebot"]', '[class*="CybotCookiebot"]',
        '[class*="cc-banner"]', '[class*="cookieconsent"]',
    ];
    sel.forEach(s => {
        try { document.querySelectorAll(s).forEach(el => el.remove()); } catch (e) {}
    });
}
"""

_MAIN_CONTENT_EXTRACT = """
() => {
    const candidates = [
        'main', 'article', '[role=main]', '#content', '#main',
        '.content', '.main-content', '.post-content', '.markdown-body',
        '.article-body', '.entry-content',
    ];
    for (const sel of candidates) {
        const el = document.querySelector(sel);
        if (el && el.innerText && el.innerText.length > 300) return el.innerText;
    }
    return document.body ? document.body.innerText : '';
}
"""


def _looks_like_antibot(text: str) -> bool:
    if not text or len(text) >= 2000:
        return False
    lower = text.lower()
    markers = (
        "checking your browser", "cloudflare", "ddos protection",
        "enable javascript", "please enable cookies", "verify you are human",
        "captcha", "access denied", "just a moment", "attention required",
        "one more step", "needs to review the security",
    )
    return any(m in lower for m in markers)


# thread-affinity: sync_playwright() is pinned to its starting thread; each worker owns its own
# browser instance, pulling from a single shared queue so flaky URLs don't block the pool.
_PW_QUEUE_SENTINEL = object()
_pw_queue: queue.Queue | None = None
_pw_workers: list[threading.Thread] = []
_pw_worker_lock = threading.Lock()
PLAYWRIGHT_FETCH_TIMEOUT = 25
PLAYWRIGHT_WORKERS = 2
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
        if len(item) == 3:
            url, fut, mode = item
        else:
            url, fut = item
            mode = "text"
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

                page.goto(url, wait_until="domcontentloaded", timeout=15_000)
                try:
                    page.wait_for_load_state("networkidle", timeout=5_000)
                except Exception:
                    pass
                try:
                    page.wait_for_function(
                        "() => document.body && document.body.innerText.length > 200",
                        timeout=5_000,
                    )
                except Exception:
                    pass

                final_url = page.url
                if not _is_safe_url(final_url):
                    _log.warning("playwright_fetch blocked redirect to %s", final_url[:120])
                    fut.set_result("")
                    continue

                try:
                    page.evaluate(_COOKIE_BANNER_NUKE)
                except Exception:
                    pass

                if mode == "html":
                    try:
                        html = page.content() or ""
                    except Exception:
                        html = ""
                    text_probe = ""
                    try:
                        text_probe = page.evaluate("() => document.body ? document.body.innerText : ''") or ""
                    except Exception:
                        pass
                    if text_probe and _looks_like_antibot(text_probe):
                        elapsed = round(time.time() - started, 2)
                        _log.warning("playwright_fetch_html anti-bot wall  url=%s final=%s %.2fs", url[:120], final_url[:120], elapsed)
                        fut.set_result({"html": "", "final_url": final_url})
                        continue
                    elapsed = round(time.time() - started, 2)
                    _log.info("playwright_fetch_html ok  url=%s html_chars=%d %.2fs", url[:120], len(html), elapsed)
                    fut.set_result({"html": html, "final_url": final_url})
                    continue

                try:
                    text = page.evaluate(_MAIN_CONTENT_EXTRACT) or ""
                except Exception:
                    text = page.inner_text("body")

                if _looks_like_antibot(text):
                    elapsed = round(time.time() - started, 2)
                    _log.warning("playwright_fetch anti-bot wall  url=%s chars=%d %.2fs", url[:120], len(text), elapsed)
                    fut.set_result("")
                    continue

                text = re.sub(r"\n{3,}", "\n\n", text)
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
            if mode == "html":
                fut.set_result({"html": "", "final_url": url})
            else:
                fut.set_result("")


def _ensure_playwright_worker() -> None:
    global _pw_queue
    with _pw_worker_lock:
        if _pw_queue is None:
            _pw_queue = queue.Queue()
        alive = [w for w in _pw_workers if w.is_alive()]
        _pw_workers[:] = alive
        while len(_pw_workers) < PLAYWRIGHT_WORKERS:
            t = threading.Thread(
                target=_playwright_worker_main,
                name=f"playwright-worker-{len(_pw_workers)}",
                daemon=True,
            )
            t.start()
            _pw_workers.append(t)
            _log.info(
                "playwright worker thread started  pool=%d/%d",
                len(_pw_workers), PLAYWRIGHT_WORKERS,
            )


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


def playwright_fetch_html(url: str) -> tuple[str, str]:
    """Render the page in Chromium and return (html, final_url). Empty html on failure."""
    if not _is_safe_url(url):
        _log.warning("playwright_fetch_html blocked unsafe url %s", url[:120])
        return "", url
    url = _sanitise_url(url)
    _ensure_playwright_worker()
    assert _pw_queue is not None
    fut: Future = Future()
    _pw_queue.put((url, fut, "html"))
    try:
        out = fut.result(timeout=PLAYWRIGHT_FETCH_TIMEOUT)
    except Exception as e:
        _log.warning("playwright_fetch_html wait failed  url=%s error=%s", url[:120], e)
        return "", url
    if isinstance(out, dict):
        return out.get("html", "") or "", out.get("final_url", url) or url
    return "", url


HTTPX_FETCH_TIMEOUT = 10


def _scrape_with_httpx(url: str) -> str:
    started = time.time()
    resp = None
    try:
        resp = httpx.get(
            url,
            timeout=HTTPX_FETCH_TIMEOUT,
            follow_redirects=True,
            headers=BROWSER_HEADERS,
        )
        resp.raise_for_status()
    except httpx.ConnectError as e:
        err_str = str(e).lower()
        if "ssl" in err_str or "certificate" in err_str or "verify" in err_str:
            _log.warning("scrape ssl error, retrying without verification  url=%s", url[:80])
            try:
                resp = httpx.get(url, timeout=HTTPX_FETCH_TIMEOUT, follow_redirects=True, headers=BROWSER_HEADERS, verify=False)
                resp.raise_for_status()
            except Exception as e2:
                _log.warning("scrape ssl retry also failed  url=%s: %s", url[:80], e2)
                return ""
        else:
            _log.warning("scrape connect error  url=%s: %s", url[:80], e)
            return ""
    except httpx.HTTPStatusError as e:
        _log.warning("scrape %d  url=%s (%s)", e.response.status_code, url[:80], e.response.reason_phrase)
        return ""
    except httpx.TimeoutException:
        _log.warning("scrape timeout after %ds  url=%s", HTTPX_FETCH_TIMEOUT, url[:80])
        return ""
    except Exception as e:
        _log.warning("scrape failed  url=%s: %s (%s)", url[:80], type(e).__name__, e)
        return ""

    if resp is None:
        return ""

    if len(resp.content) > MAX_RESPONSE_BYTES:
        _log.warning("scrape skip  response too large (%d bytes)  url=%s", len(resp.content), url[:80])
        return ""
    if len(resp.content) < 100:
        _log.warning("scrape empty  status=%d bytes=%d  url=%s", resp.status_code, len(resp.content), url[:80])
        return ""

    final_url = str(resp.url)
    if final_url != url and not _is_safe_url(final_url):
        _log.warning("scrape blocked redirect to unsafe url %s", final_url[:120])
        return ""

    elapsed = round(time.time() - started, 2)
    content_type = (resp.headers.get("content-type") or "").lower()
    if content_type and not any(t in content_type for t in ("text/html", "text/plain", "application/xhtml")):
        _log.info("scrape skip  non-html content-type=%s  url=%s", content_type.split(";")[0], url[:80])
        return ""
    _log.info("scrape ok    %s  status=%d size=%d %.2fs", url[:80], resp.status_code, len(resp.text), elapsed)

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
    if text and not _looks_like_real_text(text):
        _log.info("scrape httpx output failed quality check, falling through to playwright  url=%s len=%d", url, len(text))
        return ""
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

    if _looks_like_pdf_url(url):
        pdf_text = _extract_pdf_text(url)
        if pdf_text:
            _log.info("path=pdf  url=%s chars=%d", url[:120], len(pdf_text))
            _set_meta("pdf", len(pdf_text))
            return pdf_text
        _log.info("path=pdf  url=%s failed", url[:120])
        _set_meta("pdf_failed", len(fallback))
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

    pw_text = playwright_fetch(url)
    if pw_text:
        _log.info("path=playwright_auto  url=%s chars=%d", url[:120], len(pw_text))
        _set_meta("playwright_auto", len(pw_text))
        return pw_text

    _log.info("path=playwright_auto_failed  url=%s", url[:120])
    _set_meta("playwright_auto_failed", len(fallback))
    return fallback
