from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Iterable

from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from config import CATEGORY_COLLECTIONS, MAX_SUMMARY_INPUT_CHARS, MODELS, SEARXNG_URL
from memory import remember

_log = logging.getLogger("web_search")

SCRAPE_BLOCKLIST = {
    "reddit.com",
    "news.com.au",
    "medium.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "facebook.com",
    "instagram.com",
    "nytimes.com",
    "wsj.com",
    "ft.com",
}


def _is_blocklisted(url: str) -> bool:
    try:
        host = (urlparse(url).netloc or "").lower()
    except Exception:
        return False
    if host.startswith("www."):
        host = host[4:]
    return any(host == d or host.endswith("." + d) for d in SCRAPE_BLOCKLIST)

MAX_SOURCES = 8
OVERFETCH_FACTOR = 3
PER_PAGE_CHAR_CAP = 20_000
SUMMARY_MAX_TOKENS = 500
SUMMARY_INPUT_CHAR_CAP = MAX_SUMMARY_INPUT_CHARS
SUMMARY_BATCH_SIZE = 3
SEARXNG_TIMEOUT = 10
SCRAPE_TIMEOUT = 15
FAST_TIMEOUT = 60

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
BROWSER_HEADERS = {
    "User-Agent": BROWSER_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

_INJECTION_PATTERNS = [
    re.compile(r"<\s*/?\s*(?:system|assistant|user|s|\|im_start\||\|im_end\|)[^>]*>", re.I),
    re.compile(r"ignore (?:all )?(?:previous|prior|above|earlier) (?:instructions|messages|context|prompts?)", re.I),
    re.compile(r"disregard (?:all )?(?:prior|previous|above|earlier) (?:instructions|messages|context|prompts?)", re.I),
    re.compile(r"(?:you are|act as|pretend (?:you are|to be)|roleplay as|behave as) (?:a |an )?[a-z ]{3,30}", re.I),
    re.compile(r"new (?:instructions?|rules?|persona|role):", re.I),
    re.compile(r"(?:system|admin|root|developer) (?:prompt|message|override|mode):", re.I),
    re.compile(r"(?:forget|override|bypass|skip) (?:all |your )?(?:previous |prior )?(?:instructions|rules|guidelines|safety)", re.I),
    re.compile(r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>", re.I),
    re.compile(r"human:|assistant:|###\s*(?:system|instruction|human|assistant)", re.I),
    re.compile(r"(?:do not|don'?t) (?:follow|obey|listen to) (?:your |the )?(?:previous|original|system)", re.I),
    re.compile(r"(?:reveal|show|print|output|repeat) (?:your |the )?(?:system ?prompt|instructions|rules)", re.I),
]


def _strip_injection_patterns(text: str) -> str:
    for pat in _INJECTION_PATTERNS:
        text = pat.sub("[redacted]", text)
    return text


def _is_safe_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if host in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]"):
        return False
    if host.startswith("10.") or host.startswith("192.168.") or host.startswith("169.254."):
        return False
    if host.startswith("172."):
        # RFC 1918: 172.16.0.0 – 172.31.255.255
        try:
            second_octet = int(host.split(".")[1])
            if 16 <= second_octet <= 31:
                return False
        except (IndexError, ValueError):
            return False
    if host.endswith(".local") or host.endswith(".internal"):
        return False
    return True


def _fast_model() -> tuple[str | None, str | None]:
    entry = MODELS.get("fast")
    if isinstance(entry, dict):
        return entry.get("url"), entry.get("model_id") or "fast"
    for v in MODELS.values():
        if isinstance(v, dict) and v.get("url"):
            _log.debug("no 'fast' role in catalog, falling back to %s", v.get("role"))
            return v.get("url"), v.get("model_id") or v.get("role")
    return None, None


def _tool_model() -> tuple[str | None, str | None]:
    entry = MODELS.get("tool")
    if isinstance(entry, dict):
        return entry.get("url"), entry.get("model_id") or "tool"
    _log.debug("no 'tool' role in catalog, falling back to fast model")
    return _fast_model()


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


def generate_search_queries(message: str) -> list[str]:
    tool_url, tool_model = _tool_model()
    if not tool_url:
        _log.debug("no tool model, using raw message as search query")
        return [message.strip()[:200]]

    prompt = (
        "You write web search queries. Given a user question, output 1-3 "
        "short, high-recall queries as a JSON list of strings. No prose.\n\n"
        f"Question: {message}"
    )
    try:
        resp = httpx.post(
            f"{tool_url}/v1/chat/completions",
            json={
                "model": tool_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 120,
            },
            timeout=FAST_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        cleaned = _parse_query_list(raw)
        if cleaned:
            queries = cleaned[:3]
            _log.debug("generated queries: %s", queries)
            return queries
        _log.warning("query generation returned no parseable list; raw=%s", raw[:200])
    except Exception as e:
        _log.error("query generation failed", exc_info=True)

    return [message.strip()[:200]]


def _parse_query_list(raw: str) -> list[str]:
    if not raw:
        return []
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = text.rstrip("`").strip()

    match = re.search(r"\[.*?\]", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                out = [str(q).strip().strip('"\'') for q in data]
                out = [q for q in out if q]
                if out:
                    return out
        except Exception:
            pass

    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line)
        line = line.strip().strip('"\'')
        if len(line) > 200 or line.endswith(":"):
            continue
        if line:
            out.append(line)
    return out


def _nocodb_truthy(value) -> bool:
    """NocoDB checkboxes can return bool, int, or string representations."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


_pw_lock = threading.Lock()
_pw_instance = None
_pw_browser = None


def _sanitise_url(url: str) -> str:
    """Strip fragments and dangerous characters from a URL before navigating."""
    parsed = urlparse(url)
    clean = parsed._replace(fragment="")
    return clean.geturl()


_STEALTH_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
window.chrome = {runtime: {}};
"""


def _get_browser():
    """Lazy singleton — launches Chromium once, reuses across all fetches."""
    global _pw_instance, _pw_browser
    if _pw_browser is not None:
        return _pw_browser
    with _pw_lock:
        if _pw_browser is not None:
            return _pw_browser
        from playwright.sync_api import sync_playwright
        _pw_instance = sync_playwright().start()
        _pw_browser = _pw_instance.chromium.launch(
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
        _log.info("playwright chromium launched")
        return _pw_browser


def _reset_browser() -> None:
    """Kill the cached browser so the next call to _get_browser() relaunches."""
    global _pw_instance, _pw_browser
    with _pw_lock:
        if _pw_browser is not None:
            try:
                _pw_browser.close()
            except Exception:
                pass
        if _pw_instance is not None:
            try:
                _pw_instance.stop()
            except Exception:
                pass
        _pw_browser = None
        _pw_instance = None
        _log.info("playwright browser reset")


def playwright_fetch(url: str) -> str:
    """Fetch page text using in-process Playwright/Chromium."""
    if not _is_safe_url(url):
        _log.warning("playwright_fetch blocked unsafe url %s", url[:120])
        return ""
    url = _sanitise_url(url)
    started = time.time()
    try:
        browser = _get_browser()
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
            # Brief settle for JS-rendered content without waiting on
            # straggler analytics/websocket requests that block networkidle.
            try:
                page.wait_for_load_state("networkidle", timeout=5_000)
            except Exception:
                pass  # DOM is loaded, proceed with what we have

            final_url = page.url
            if not _is_safe_url(final_url):
                _log.warning("playwright_fetch blocked redirect to %s", final_url[:120])
                return ""

            text = page.inner_text("body")
            text = _strip_injection_patterns(text)[:PER_PAGE_CHAR_CAP]
            elapsed = round(time.time() - started, 2)
            _log.info("playwright_fetch ok  url=%s chars=%d %.2fs", url[:120], len(text), elapsed)
            return text
        finally:
            context.close()
    except Exception as e:
        elapsed = round(time.time() - started, 2)
        _log.warning("playwright_fetch failed  url=%s error=%s %.2fs", url[:120], e, elapsed)
        # If the browser itself is broken (not just a page-level error),
        # reset so the next call gets a fresh instance.
        if "browser" not in str(type(e).__name__).lower():
            try:
                _get_browser().contexts  # probe: is it still alive?
            except Exception:
                _log.warning("playwright browser appears dead, resetting")
                _reset_browser()
        else:
            _reset_browser()
        return ""


MAX_RESPONSE_BYTES = 5_000_000  # 5 MB — skip anything larger


def _scrape_with_httpx(url: str) -> str:
    """Plain httpx + BeautifulSoup scraper. Returns extracted text or empty string."""
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


def scrape_page(url: str, snippet: str = "", source: dict | None = None) -> str:
    """Fetch and extract page text. Falls back to `snippet` on any failure."""
    fallback = _strip_injection_patterns(snippet)[:PER_PAGE_CHAR_CAP] if snippet else ""

    if not _is_safe_url(url):
        _log.warning("scrape skip  unsafe url %s", url[:120])
        return fallback

    if _is_blocklisted(url):
        _log.debug("scrape skip  blocklisted %s", url)
        return fallback

    # Playwright direct — skip scraper entirely
    if source and _nocodb_truthy(source.get("use_playwright")):
        text = playwright_fetch(url)
        _log.info("path=playwright_direct  url=%s ok=%s", url[:120], bool(text))
        return text or fallback

    # Plain scraper
    text = _scrape_with_httpx(url)

    if text and text != fallback:
        _log.info("path=scraper  url=%s chars=%d", url[:120], len(text))
        return text

    # Scraper returned nothing useful — try Playwright when called with a source (enrichment)
    if source is not None:
        pw_text = playwright_fetch(url)
        if pw_text:
            target_id = source.get("Id")
            if target_id:
                try:
                    from workers.enrichment_agent import EnrichmentDB
                    EnrichmentDB().update_scrape_target(target_id, use_playwright=True)
                    _log.info("auto-promoted to playwright_direct id=%s", target_id)
                except Exception as e:
                    _log.warning("playwright promotion failed id=%s: %s", target_id, e)
            _log.info("path=playwright_auto  url=%s chars=%d", url[:120], len(pw_text))
            return pw_text
        _log.info("path=playwright_auto  url=%s failed", url[:120])
        return fallback

    return fallback



def summarise_page(text: str, query: str) -> dict | None:
    """Summarise page content and assess its relevance to the query.

    Returns {"summary": str, "relevance": "high"|"medium"|"low", "source_type": str}
    or None if irrelevant / empty.
    """
    if not text.strip():
        return None
    tool_url, tool_model = _tool_model()
    if not tool_url:
        _log.warning("no tool model available, returning raw truncation for '%s'", query[:80])
        return {"summary": text[:1200], "relevance": "unknown", "source_type": "unknown"}

    prompt = (
        "You are summarising a web page for someone who asked: "
        f"'{query[:500]}'\n\n"
        "Return a JSON object with these fields:\n"
        "- summary: factual summary of the page (up to 400 words). Preserve "
        "key names, numbers, dates, and direct quotes relevant to the question.\n"
        "- relevance: how relevant this page is to the question — "
        "\"high\" (directly answers it), \"medium\" (related/useful context), "
        "or \"low\" (tangentially related at best).\n"
        "- source_type: what kind of source this is — e.g. \"official_docs\", "
        "\"news_article\", \"blog_post\", \"research_paper\", \"forum\", "
        "\"product_page\", \"government\", \"unknown\".\n\n"
        "If the page is completely irrelevant to the question, return: "
        '{\"irrelevant\": true}\n\n'
        f"PAGE:\n{text[:SUMMARY_INPUT_CHAR_CAP]}"
    )
    started = time.time()
    try:
        resp = httpx.post(
            f"{tool_url}/v1/chat/completions",
            json={
                "model": tool_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": SUMMARY_MAX_TOKENS,
            },
            timeout=FAST_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        elapsed = round(time.time() - started, 2)

        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip()).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            if raw.upper().startswith("IRRELEVANT") or raw.strip() == '{"irrelevant": true}':
                _log.debug("summarise    irrelevant for '%s' %.2fs", query[:80], elapsed)
                return None
            _log.debug("summarise    unparseable json, using raw text for '%s'", query[:80])
            return {"summary": raw[:2000], "relevance": "unknown", "source_type": "unknown"}

        if data.get("irrelevant"):
            _log.debug("summarise    irrelevant for '%s' %.2fs", query[:80], elapsed)
            return None

        summary = str(data.get("summary") or "").strip()
        if not summary:
            return None

        result = {
            "summary": summary,
            "relevance": str(data.get("relevance") or "unknown").lower(),
            "source_type": str(data.get("source_type") or "unknown").lower(),
        }
        _log.debug("summarise    ok for '%s' relevance=%s type=%s %d chars %.2fs",
                    query[:80], result["relevance"], result["source_type"], len(summary), elapsed)
        return result
    except Exception:
        _log.error("summarisation failed for '%s'", query, exc_info=True)
        return {"summary": text[:1200], "relevance": "unknown", "source_type": "unknown"}



def summarise_pages_batch(
    pages: list[dict], query: str
) -> list[dict | None]:
    """Summarise multiple pages in a single tool-model call.

    *pages* is a list of {"index": int, "text": str, ...}.
    Returns a list aligned with *pages* — each element is the assessment dict
    or None (irrelevant / parse failure).  Falls back to per-page calls on
    any top-level failure.
    """
    if not pages:
        return []

    tool_url, tool_model = _tool_model()
    if not tool_url:
        return [
            {"summary": p["text"][:1200], "relevance": "unknown", "source_type": "unknown"}
            for p in pages
        ]

    # Divide the char budget evenly across pages
    per_page_cap = max(800, SUMMARY_INPUT_CHAR_CAP // len(pages))

    page_blocks = []
    for i, p in enumerate(pages):
        page_blocks.append(f"--- PAGE {i + 1} ---\n{p['text'][:per_page_cap]}")
    all_pages_text = "\n\n".join(page_blocks)

    prompt = (
        "You are summarising multiple web pages for someone who asked: "
        f"'{query[:500]}'\n\n"
        f"There are {len(pages)} pages below. For EACH page, return a JSON object with:\n"
        "- page: the page number (1-indexed)\n"
        "- summary: factual summary (up to 200 words). Preserve key names, numbers, dates.\n"
        "- relevance: \"high\", \"medium\", or \"low\"\n"
        "- source_type: e.g. \"official_docs\", \"news_article\", \"blog_post\", "
        "\"research_paper\", \"forum\", \"product_page\", \"government\", \"unknown\"\n\n"
        "If a page is completely irrelevant, return: {\"page\": N, \"irrelevant\": true}\n\n"
        "Return a JSON array of objects, one per page. Example:\n"
        '[{"page": 1, "summary": "...", "relevance": "high", "source_type": "news_article"}, '
        '{"page": 2, "irrelevant": true}]\n\n'
        f"{all_pages_text}"
    )

    started = time.time()
    try:
        resp = httpx.post(
            f"{tool_url}/v1/chat/completions",
            json={
                "model": tool_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": SUMMARY_MAX_TOKENS * len(pages),
            },
            timeout=FAST_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        elapsed = round(time.time() - started, 2)
        _log.debug("batch summarise  %d pages %.2fs", len(pages), elapsed)

        # Strip markdown fences and any prose before/after the JSON
        cleaned = raw.strip()
        fence_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", cleaned)
        if fence_match:
            cleaned = fence_match.group(1)
        else:
            cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned).strip()
        # If model prefixed with prose, find the array
        if not cleaned.startswith("["):
            arr_match = re.search(r"\[[\s\S]*\]", cleaned)
            if arr_match:
                cleaned = arr_match.group(0)
        items = json.loads(cleaned)
        if not isinstance(items, list):
            items = [items]

        results: list[dict | None] = [None] * len(pages)
        for item in items:
            idx = item.get("page")
            if not isinstance(idx, int) or idx < 1 or idx > len(pages):
                continue
            if item.get("irrelevant"):
                results[idx - 1] = None
                continue
            summary = str(item.get("summary") or "").strip()
            if not summary:
                continue
            results[idx - 1] = {
                "summary": summary,
                "relevance": str(item.get("relevance") or "unknown").lower(),
                "source_type": str(item.get("source_type") or "unknown").lower(),
            }
        return results

    except Exception:
        _log.warning("batch summarisation failed, falling back to per-page", exc_info=True)
        return [summarise_page(p["text"], query) for p in pages]


_EXPLICIT_SEARCH_PATTERNS = [
    re.compile(r"\b(?:search|look up|google|find|lookup)\b.*\b(?:for|about|this|that|it|me)\b", re.I),
    re.compile(r"\b(?:search|look up|google|find|lookup)\s+(?:for\s+)?(?:me\s+)?[\"'].+[\"']", re.I),
    re.compile(r"\bwhat(?:'s| is| are)\b.*\b(?:today|tonight|this week|this weekend|right now|currently|latest|recent)\b", re.I),
    re.compile(r"\b(?:current|latest|recent|live|today'?s?|tonight'?s?)\b.*\b(?:price|score|weather|news|results?|standings?|fixtures?|schedule|status)\b", re.I),
    re.compile(r"\b(?:who won|who is winning|what happened|what's happening)\b", re.I),
    re.compile(r"\bcan you (?:search|look|find|check)\b", re.I),
    re.compile(r"\b(?:give me|get me|pull up|show me)\b.*\b(?:info|information|details|data|docs|documentation)\b", re.I),
    re.compile(r"\b(?:best|top|recommended)\b.*\b(?:resources?|sources?|articles?|guides?|tutorials?|docs)\b", re.I),
]


def _explicit_search_intent(message: str) -> tuple[bool, str]:
    for pat in _EXPLICIT_SEARCH_PATTERNS:
        if pat.search(message):
            return True, "explicit search request"
    return False, ""


def needs_web_search(message: str) -> tuple[bool, str, str]:
    """Classify whether a message needs web search.

    Returns (needs_search, reason, confidence).
    confidence is "high" (auto-invoke), "medium" (consent prompt), or "" (no search).
    Defaults to (False, "", "") on any failure.
    """
    msg = (message or "").strip()
    if not msg:
        return False, "", ""

    explicit, reason = _explicit_search_intent(msg)
    if explicit:
        _log.info("search intent detected (pattern): %s", reason)
        return True, reason, "high"

    tool_url, tool_model = _tool_model()
    if not tool_url:
        return False, "", ""

    prompt = (
        "Decide whether answering this user message would benefit from a web search. "
        "Lean towards YES — it's better to search and find nothing than to miss "
        "useful information. Return ONLY a JSON object:\n"
        '{"needs_search": true|false, "confidence": "high"|"medium"|"low", '
        '"reason": "<15 words or fewer>"}\n\n'
        "needs_search=true (confidence high) when:\n"
        "- Current events, scores, prices, weather, news, schedules\n"
        "- References to 'today', 'this week', 'latest', 'recent', 'current'\n"
        "- Specific products, frameworks, tools, or technologies\n"
        "- Company or person's recent activity\n"
        "- Fact-checking or verification requests\n"
        "- Requests for documentation, resources, or guides\n\n"
        "needs_search=true (confidence medium) when:\n"
        "- The topic might have recent developments\n"
        "- Unclear whether the user needs current vs general info\n"
        "- A web search could add value but isn't essential\n\n"
        "needs_search=false when:\n"
        "- Clearly asking for help writing code, prose, or analysis\n"
        "- Casual conversation or follow-up within an ongoing thread\n"
        "- Abstract or philosophical questions\n\n"
        f"Message: {msg[:500]}"
    )
    try:
        resp = httpx.post(
            f"{tool_url}/v1/chat/completions",
            json={
                "model": tool_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 80,
            },
            timeout=20,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        match = re.search(r"\{.*?\}", raw, re.S)
        if not match:
            return False, "", ""
        data = json.loads(match.group(0))
        needs = bool(data.get("needs_search"))
        reason = str(data.get("reason") or "").strip()[:120]
        confidence = str(data.get("confidence") or "medium").lower()
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"
        if needs:
            _log.info("search intent detected (classifier): confidence=%s reason=%s", confidence, reason)
        return needs, reason, confidence if needs else ""
    except Exception as e:
        _log.warning("needs_web_search classifier failed: %s", e)
        return False, "", ""


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


def run_web_search(query: str, org_id: int) -> tuple[str, list[dict], str]:
    _log.debug("search start  query=%s org=%d", query[:100], org_id)
    queries = generate_search_queries(query)

    max_candidates = MAX_SOURCES * OVERFETCH_FACTOR
    raw_results: list[dict] = []
    for q in queries:
        raw_results.extend(searxng_search(q, max_results=MAX_SOURCES * 2))
        if len(raw_results) >= max_candidates:
            break
    raw_results = _dedupe(raw_results)[:max_candidates]

    results: list[dict] = []
    scrape_failures = 0
    for start in range(0, len(raw_results), SUMMARY_BATCH_SIZE):
        if len(results) >= MAX_SOURCES:
            break

        batch = raw_results[start:start + SUMMARY_BATCH_SIZE]
        scraped_pages: list[dict] = []
        for r in batch:
            text = scrape_page(r["url"], snippet=r.get("snippet", ""))
            if not text:
                scrape_failures += 1
                continue
            scraped_pages.append({"result": r, "text": text})

        if not scraped_pages:
            continue

        assessments = summarise_pages_batch(
            [{"index": i, "text": page["text"]} for i, page in enumerate(scraped_pages)],
            query,
        )
        for page, assessed in zip(scraped_pages, assessments):
            if len(results) >= MAX_SOURCES:
                break
            if not assessed:
                continue

            result = page["result"]
            entry = {
                "title": result["title"] or result["url"],
                "url": result["url"],
                "summary": assessed["summary"],
                "relevance": assessed.get("relevance", "unknown"),
                "source_type": assessed.get("source_type", "unknown"),
            }
            results.append(entry)

            try:
                remember(
                    text=f"{entry['title']}\n\n{entry['summary']}",
                    metadata={
                        "url": result["url"],
                        "title": result["title"],
                        "query": query,
                        "relevance": entry["relevance"],
                        "source_type": entry["source_type"],
                        "fetched_at": time.time(),
                    },
                    org_id=org_id,
                    collection_name="web_search",
                )
            except Exception:
                _log.error("chroma write failed for %s", result["url"], exc_info=True)

    if results:
        high = [r for r in results if r["relevance"] == "high"]
        medium = [r for r in results if r["relevance"] == "medium"]
        low = [r for r in results if r["relevance"] not in ("high", "medium")]

        if high:
            confidence = "high"
            confidence_note = f"{len(high)} highly relevant source(s) found."
        elif medium:
            confidence = "medium"
            confidence_note = f"No directly relevant sources, but {len(medium)} related source(s) found."
        else:
            confidence = "low"
            confidence_note = "Only tangentially related sources found."

        sorted_results = high + medium + low

        source_types = sorted({e["source_type"] for e in results if e["source_type"] != "unknown"})
        type_note = f" Source types: {', '.join(source_types)}." if source_types else ""

        context_parts = [
            f"WEB SEARCH RESULTS — confidence: {confidence}. {confidence_note}{type_note}\n"
            "In your answer:\n"
            "- Cite sources inline by number [1], [2], etc.\n"
            "- Prioritise high-relevance sources from authoritative source types "
            "(official_docs, research_paper, government) over blog posts or forums.\n"
            "- Briefly explain to the user WHY you trust certain sources more — "
            "e.g. 'According to the official documentation [1]...' or "
            "'A blog post [3] suggests X, though this is less authoritative.'\n"
            "- If overall confidence is medium or low, tell the user explicitly "
            "and explain what's missing or uncertain.\n"
            "- For research or exploratory questions where the results leave gaps, "
            "suggest specific follow-up topics or source types worth investigating. "
            "Do NOT suggest follow-ups for simple factual lookups.\n"
        ]
        for i, entry in enumerate(sorted_results, start=1):
            rel_tag = f"[relevance: {entry['relevance']}, type: {entry['source_type']}]"
            context_parts.append(
                f"[{i}] {entry['title']} ({entry['url']}) {rel_tag}\n{entry['summary']}\n"
            )
        context_block = "\n".join(context_parts)
        sources = [
            {"index": i + 1, "title": e["title"], "url": e["url"],
             "relevance": e["relevance"], "source_type": e["source_type"],
             "snippet": e["summary"][:200]}
            for i, e in enumerate(sorted_results)
        ]
    elif raw_results:
        _log.warning("scraping failed on all %d candidates, returning raw SearxNG results", len(raw_results))
        confidence = "low"
        context_parts = [
            "WEB SEARCH RESULTS — confidence: low. "
            "Search found results but full page content could not be retrieved. "
            "Use the titles, URLs, and snippets below to inform your answer. "
            "Cite the URLs so the user can visit them directly. "
            "Caveat that you could not verify the full page content.\n"
        ]
        for i, r in enumerate(raw_results[:MAX_SOURCES], start=1):
            title = _strip_injection_patterns(r.get("title") or r["url"])
            snippet = _strip_injection_patterns(r.get("snippet") or "")
            context_parts.append(f"[{i}] {title} ({r['url']})\n{snippet}\n")
        context_block = "\n".join(context_parts)
        sources = [
            {"index": i + 1, "title": _strip_injection_patterns(r.get("title") or r["url"]),
             "url": r["url"], "relevance": "unknown", "source_type": "unknown",
             "snippet": _strip_injection_patterns((r.get("snippet") or "")[:200])}
            for i, r in enumerate(raw_results[:MAX_SOURCES])
        ]
    else:
        _log.warning("search returned no results for query=%s", query[:100])
        return "", [], "none"

    _log.info("search done   queries=%d candidates=%d results=%d scrape_failures=%d",
              len(queries), len(raw_results), len(results), scrape_failures)

    if results:
        summaries_for_suggest = [(e["title"], e["url"], e["summary"]) for e in results]
        threading.Thread(
            target=_suggest_sources_from_search,
            args=(summaries_for_suggest, query, org_id),
            daemon=True,
        ).start()

    return context_block, sources, confidence


def _suggest_sources_from_search(
    summaries: list[tuple[str, str, str]],
    query: str,
    org_id: int,
) -> None:
    """Evaluate web search results and suggest worthy ones as scrape targets."""
    if not summaries:
        return

    tool_url, tool_model = _tool_model()
    if not tool_url:
        return

    sources_text = "\n".join(
        f"{i+1}. {title} ({url})\n   {summary[:300]}"
        for i, (title, url, summary) in enumerate(summaries)
    )
    prompt = (
        "You are evaluating web search results to find sources worth ONGOING "
        "monitoring in a knowledge base. Most search results are NOT good "
        "monitoring targets — be very selective.\n\n"
        "A source MUST have ALL of:\n"
        "- INSTITUTIONAL AUTHORITY: maintained by a recognised organisation\n"
        "- REGULAR UPDATES: publishes new content on a recurring basis\n"
        "- ORIGINAL CONTENT: primary source, not aggregator or reposter\n"
        "- EDITORIAL STANDARDS: institutional accountability, not anonymous\n\n"
        "ALWAYS REJECT: social media, forums, Medium/Substack (unless institutional), "
        "paywalled sites, personal blogs, content farms, aggregators, YouTube, "
        "one-off news articles (suggest the publication's section page instead).\n\n"
        f"Search context: user asked about '{query}'\n\n"
        f"RESULTS:\n{sources_text}\n\n"
        "For each result worth monitoring long-term, return:\n"
        "- index: the result number\n"
        "- name: the organisation or publication\n"
        "- category: one of documentation, news, competitive, regulatory, "
        "  research, security, model_releases\n"
        "- authority: WHO maintains this and WHY they are credible\n"
        "- score: 1-10 (8+ = clearly authoritative, 7 = probably good)\n"
        "- suggested_url: the best URL to monitor (may differ from the search "
        "  result — prefer a section/feed page over a single article)\n\n"
        "Return a JSON array. If NONE are worth monitoring, return []. "
        "0 suggestions is the expected outcome for most searches."
    )

    try:
        resp = httpx.post(
            f"{tool_url}/v1/chat/completions",
            json={
                "model": tool_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 400,
            },
            timeout=FAST_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        _log.debug("search suggestion evaluation failed", exc_info=True)
        return

    try:
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip()).strip()
        items = json.loads(cleaned)
    except Exception:
        _log.debug("search suggestion unparseable: %s", raw[:200])
        return
    if not isinstance(items, list) or not items:
        return

    # Lazy import to avoid circular dependency (enrichment_agent imports from web_search)
    try:
        from workers.enrichment_agent import EnrichmentDB
        db = EnrichmentDB()
    except Exception:
        _log.debug("could not init EnrichmentDB for search suggestions")
        return

    cycle_id = f"websearch_{int(time.time())}"
    recorded = 0
    for item in items[:3]:
        try:
            score = int(item.get("score") or 0)
            if score < 7:
                continue
            category = str(item.get("category") or "").lower()
            if category not in CATEGORY_COLLECTIONS:
                continue
            idx = int(item.get("index", 0)) - 1
            if idx < 0 or idx >= len(summaries):
                continue

            url = str(item.get("suggested_url") or summaries[idx][1]).strip()
            if not url.startswith("http"):
                continue

            authority = str(item.get("authority") or "")
            name = str(item.get("name") or summaries[idx][0])
            confidence = "high" if score >= 8 else "medium"

            db.record_suggestion(
                org_id=org_id,
                url=url,
                name=name,
                category=category,
                reason=f"Web search: {authority}"[:500],
                confidence=confidence,
                confidence_score=score,
                suggested_by_url=summaries[idx][1],
                suggested_by_cycle=cycle_id,
            )
            recorded += 1
        except Exception:
            _log.debug("search suggestion record failed", exc_info=True)

    if recorded:
        _log.info("search suggestions recorded=%d from query='%s'", recorded, query[:80])
