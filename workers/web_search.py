from __future__ import annotations

import json
import logging
import re
import time
from typing import Iterable

from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from config import MODELS, SEARXNG_URL
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

MAX_SOURCES = 5
OVERFETCH_FACTOR = 4
PER_PAGE_CHAR_CAP = 20_000
SUMMARY_MAX_TOKENS = 300
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
    """Best-effort extraction of a list of query strings from fast-model output.

    Handles: fenced code blocks, JSON arrays embedded in prose, numbered lists,
    and plain newline-separated lines.
    """
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


def scrape_page(url: str, snippet: str = "") -> str:
    """Fetch and extract page text. Falls back to `snippet` on any failure."""
    fallback = _strip_injection_patterns(snippet)[:PER_PAGE_CHAR_CAP] if snippet else ""

    if _is_blocklisted(url):
        _log.debug("scrape skip  blocklisted %s", url)
        return fallback

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
        return fallback
    except httpx.TimeoutException:
        _log.warning("scrape timeout after %ds for %s", SCRAPE_TIMEOUT, url)
        return fallback
    except Exception as e:
        _log.warning("scrape failed for %s: %s", url, e)
        return fallback

    elapsed = round(time.time() - started, 2)
    _log.debug("scrape ok    %s  status=%d size=%d %.2fs", url, resp.status_code, len(resp.text), elapsed)

    try:
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        _log.warning("html parse failed for %s: %s", url, e)
        return fallback

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _strip_injection_patterns(text)
    text = text[:PER_PAGE_CHAR_CAP]
    extracted = len(text)
    if not text:
        _log.warning("scrape empty after extraction for %s (raw html was %d chars)", url, len(resp.text))
    else:
        _log.debug("scrape extracted %d chars from %s", extracted, url)
    return text or fallback



def summarise_page(text: str, query: str) -> str:
    if not text.strip():
        return ""
    tool_url, tool_model = _tool_model()
    if not tool_url:
        _log.warning("no tool model available, returning raw truncation for '%s'", query[:80])
        return text[:1200]

    prompt = (
        "Summarise the following page content for a user who asked: "
        f"'{query}'. Be factual, ≤ 250 words, preserve any key names, "
        "numbers, dates, and direct quotes relevant to the question. "
        "If the page is irrelevant, reply exactly: IRRELEVANT.\n\n"
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
        summary = resp.json()["choices"][0]["message"]["content"].strip()
        elapsed = round(time.time() - started, 2)
        if summary.upper().startswith("IRRELEVANT"):
            _log.debug("summarise    irrelevant for '%s' %.2fs", query[:80], elapsed)
            return ""
        _log.debug("summarise    ok for '%s' %d chars %.2fs", query[:80], len(summary), elapsed)
        return summary
    except Exception:
        _log.error("summarisation failed for '%s'", query, exc_info=True)
        return text[:1200]



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
    for r in raw_results:
        if len(summaries) >= MAX_SOURCES:
            break
        text = scrape_page(r["url"], snippet=r.get("snippet", ""))
        if not text:
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

    if not summaries:
        return "", []

    context_parts = ["The following web search results were retrieved for the user's question. Cite them inline where relevant.\n"]
    for i, (title, url, summary) in enumerate(summaries, start=1):
        context_parts.append(f"[{i}] {title} ({url})\n{summary}\n")
    context_block = "\n".join(context_parts)

    sources = [f"{title}: {url}" for title, url, _ in summaries]
    _log.info("search done   queries=%d candidates=%d summaries=%d", len(queries), len(raw_results), len(summaries))
    return context_block, sources
