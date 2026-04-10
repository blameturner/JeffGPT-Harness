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

from config import CATEGORY_COLLECTIONS, MODELS, SEARXNG_URL
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
    re.compile(r"ignore (?:all )?previous (?:instructions|messages)", re.I),
    re.compile(r"disregard (?:all )?(?:prior|previous) (?:instructions|messages)", re.I),
    re.compile(r"you are now [a-z ]+", re.I),
    re.compile(r"new instructions?:", re.I),
]


def _strip_injection_patterns(text: str) -> str:
    for pat in _INJECTION_PATTERNS:
        text = pat.sub("[redacted]", text)
    return text


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
        _pw_browser = _pw_instance.chromium.launch(headless=True)
        _log.info("playwright chromium launched")
        return _pw_browser


def playwright_fetch(url: str) -> str:
    """Fetch page text using in-process Playwright/Chromium."""
    started = time.time()
    try:
        browser = _get_browser()
        context = browser.new_context(
            user_agent=BROWSER_UA,
            viewport={"width": 1280, "height": 800},
        )
        try:
            page = context.new_page()
            page.goto(url, wait_until="networkidle", timeout=30_000)
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
        return ""


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
        f"PAGE:\n{text}"
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



def needs_web_search(message: str) -> tuple[bool, str]:
    """Fast-model classifier — does this question need live web info?

    Returns (needs_search, reason). Defaults to (False, "") on any failure so
    classifier hiccups never block a reply.
    """
    msg = (message or "").strip()
    if not msg:
        return False, ""

    tool_url, tool_model = _tool_model()
    if not tool_url:
        return False, ""

    prompt = (
        "Decide whether answering this user question requires LIVE web "
        "information — recent news, current scores/prices, 'today' / 'this "
        "week' events, things that change frequently, or anything after an "
        "LLM knowledge cutoff. Timeless or conceptual questions do NOT need "
        "search. Reply with ONLY a JSON object of the form "
        '{"needs_search": true|false, "reason": "<15 words or fewer>"}.\n\n'
        f"Question: {msg}"
    )
    try:
        resp = httpx.post(
            f"{tool_url}/v1/chat/completions",
            json={
                "model": tool_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 60,
            },
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        match = re.search(r"\{.*?\}", raw, re.S)
        if not match:
            return False, ""
        data = json.loads(match.group(0))
        needs = bool(data.get("needs_search"))
        reason = str(data.get("reason") or "").strip()[:120]
        return needs, reason
    except Exception as e:
        _log.warning("needs_web_search classifier failed: %s", e)
        return False, ""


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


def run_web_search(query: str, org_id: int) -> tuple[str, list[str]]:
    _log.debug("search start  query=%s org=%d", query[:100], org_id)
    queries = generate_search_queries(query)

    max_candidates = MAX_SOURCES * OVERFETCH_FACTOR
    raw_results: list[dict] = []
    for q in queries:
        raw_results.extend(searxng_search(q, max_results=MAX_SOURCES * 2))
        if len(raw_results) >= max_candidates:
            break
    raw_results = _dedupe(raw_results)[:max_candidates]

    summaries: list[tuple[str, str, str]] = []
    scrape_failures = 0
    for r in raw_results:
        if len(summaries) >= MAX_SOURCES:
            break
        text = scrape_page(r["url"], snippet=r.get("snippet", ""))
        if not text:
            scrape_failures += 1
            continue
        summary = summarise_page(text, query)
        if not summary:
            continue
        summaries.append((r["title"] or r["url"], r["url"], summary))

        try:
            remember(
                text=f"{r['title']}\n\n{summary}",
                metadata={
                    "url": r["url"],
                    "title": r["title"],
                    "query": query,
                    "fetched_at": time.time(),
                },
                org_id=org_id,
                collection_name="web_search",
            )
        except Exception as e:
            _log.error("chroma write failed for %s", r["url"], exc_info=True)

    if summaries:
        context_parts = [
            "The following web search results were retrieved for the user's question. "
            "Cite them inline where relevant.\n"
        ]
        for i, (title, url, summary) in enumerate(summaries, start=1):
            context_parts.append(f"[{i}] {title} ({url})\n{summary}\n")
        context_block = "\n".join(context_parts)
        sources = [f"{title}: {url}" for title, url, _ in summaries]
    elif raw_results:
        _log.warning("scraping failed on all %d candidates, returning raw SearxNG results", len(raw_results))
        context_parts = [
            "Web search found the following results but full page content could not be retrieved. "
            "Use the titles, URLs, and snippets below to inform your answer. "
            "Cite the URLs so the user can visit them directly. "
            "Note that you could not verify the full page content.\n"
        ]
        for i, r in enumerate(raw_results[:MAX_SOURCES], start=1):
            title = r.get("title") or r["url"]
            snippet = r.get("snippet") or ""
            context_parts.append(f"[{i}] {title} ({r['url']})\n{snippet}\n")
        context_block = "\n".join(context_parts)
        sources = [f"{r.get('title') or r['url']}: {r['url']}" for r in raw_results[:MAX_SOURCES]]
    else:
        _log.warning("search returned no results for query=%s", query[:100])
        return "", []

    _log.info("search done   queries=%d candidates=%d summaries=%d scrape_failures=%d",
              len(queries), len(raw_results), len(summaries), scrape_failures)

    if summaries:
        threading.Thread(
            target=_suggest_sources_from_search,
            args=(summaries, query, org_id),
            daemon=True,
        ).start()

    return context_block, sources


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
