from __future__ import annotations

import json
import re
import time
from typing import Iterable

from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from config import MODELS, SEARXNG_URL
from memory import remember

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
OVERFETCH_FACTOR = 4                # how many candidates per desired summary
PER_PAGE_CHAR_CAP = 20_000          # hard cap before summarisation
SUMMARY_MAX_TOKENS = 300            # fast-model cap per page
SEARXNG_TIMEOUT = 10
SCRAPE_TIMEOUT = 15
FAST_TIMEOUT = 60

# Realistic desktop Chrome UA — many news sites 403 anything that looks like a bot.
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
    if not isinstance(entry, dict):
        return None, None
    return entry.get("url"), entry.get("model_id") or "fast"


def searxng_search(query: str, max_results: int = MAX_SOURCES) -> list[dict]:
    try:
        resp = httpx.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json"},
            timeout=SEARXNG_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[web_search] searxng failed for '{query}': {e}")
        return []

    results = data.get("results") or []
    out: list[dict] = []
    for r in results[: max_results * 2]:  # overfetch; scrape may fail
        url = r.get("url")
        if not url:
            continue
        out.append({
            "title": (r.get("title") or "").strip()[:200],
            "url": url,
            "snippet": (r.get("content") or "").strip(),
        })
    return out


def generate_search_queries(message: str) -> list[str]:

    fast_url, fast_model = _fast_model()
    if not fast_url:
        return [message.strip()[:200]]

    prompt = (
        "You write web search queries. Given a user question, output 1-3 "
        "short, high-recall queries as a JSON list of strings. No prose.\n\n"
        f"Question: {message}"
    )
    try:
        resp = httpx.post(
            f"{fast_url}/v1/chat/completions",
            json={
                "model": fast_model,
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
            return cleaned[:3]
        print(f"[web_search] query generation returned no parseable list; raw={raw[:200]!r}")
    except Exception as e:
        print(f"[web_search] query generation failed: {e}")

    return [message.strip()[:200]]


def _parse_query_list(raw: str) -> list[str]:
    """Best-effort extraction of a list of query strings from fast-model output.

    Handles: fenced code blocks, JSON arrays embedded in prose, numbered lists,
    and plain newline-separated lines.
    """
    if not raw:
        return []
    text = raw.strip()
    # Strip ``` fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = text.rstrip("`").strip()

    # Try: JSON array anywhere in the text
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

    # Try: numbered / bulleted / plain lines
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip "1.", "1)", "- ", "* ", "• "
        line = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line)
        line = line.strip().strip('"\'')
        # Skip obvious prose preamble
        if len(line) > 200 or line.endswith(":"):
            continue
        if line:
            out.append(line)
    return out


def scrape_page(url: str, snippet: str = "") -> str:
    """Fetch and extract page text. Falls back to `snippet` on any failure
    (403, timeout, parse error, blocklisted domain)."""
    fallback = _strip_injection_patterns(snippet)[:PER_PAGE_CHAR_CAP] if snippet else ""

    if _is_blocklisted(url):
        if fallback:
            print(f"[web_search] blocklisted {url}; using snippet")
        return fallback

    try:
        resp = httpx.get(
            url,
            timeout=SCRAPE_TIMEOUT,
            follow_redirects=True,
            headers=BROWSER_HEADERS,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[web_search] scrape failed for {url}: {e}")
        return fallback

    try:
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        print(f"[web_search] parse failed for {url}: {e}")
        return fallback

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _strip_injection_patterns(text)
    text = text[:PER_PAGE_CHAR_CAP]
    return text or fallback



def summarise_page(text: str, query: str) -> str:
    if not text.strip():
        return ""
    fast_url, fast_model = _fast_model()
    if not fast_url:
        return text[:1200]

    prompt = (
        "Summarise the following page content for a user who asked: "
        f"'{query}'. Be factual, ≤ 250 words, preserve any key names, "
        "numbers, dates, and direct quotes relevant to the question. "
        "If the page is irrelevant, reply exactly: IRRELEVANT.\n\n"
        f"PAGE:\n{text}"
    )
    try:
        resp = httpx.post(
            f"{fast_url}/v1/chat/completions",
            json={
                "model": fast_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": SUMMARY_MAX_TOKENS,
            },
            timeout=FAST_TIMEOUT,
        )
        resp.raise_for_status()
        summary = resp.json()["choices"][0]["message"]["content"].strip()
        if summary.upper().startswith("IRRELEVANT"):
            return ""
        return summary
    except Exception as e:
        print(f"[web_search] summarisation failed for {query}: {e}")
        return text[:1200]



def needs_web_search(message: str) -> tuple[bool, str]:
    """Fast-model classifier — does this question need live web info?

    Returns (needs_search, reason). Defaults to (False, "") on any failure so
    classifier hiccups never block a reply.
    """
    msg = (message or "").strip()
    if not msg:
        return False, ""

    fast_url, fast_model = _fast_model()
    if not fast_url:
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
            f"{fast_url}/v1/chat/completions",
            json={
                "model": fast_model,
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
        print(f"[web_search] needs_web_search classifier failed: {e}")
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
    queries = generate_search_queries(query)

    max_candidates = MAX_SOURCES * OVERFETCH_FACTOR
    raw_results: list[dict] = []
    for q in queries:
        raw_results.extend(searxng_search(q, max_results=MAX_SOURCES * 2))
        if len(raw_results) >= max_candidates:
            break
    raw_results = _dedupe(raw_results)[:max_candidates]

    summaries: list[tuple[str, str, str]] = []  # (title, url, summary)
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
            print(f"[web_search] chroma write failed for {r['url']}: {e}")

    if not summaries:
        return "", []

    context_parts = ["The following web search results were retrieved for the user's question. Cite them inline where relevant.\n"]
    for i, (title, url, summary) in enumerate(summaries, start=1):
        context_parts.append(f"[{i}] {title} ({url})\n{summary}\n")
    context_block = "\n".join(context_parts)

    sources = [f"{title}: {url}" for title, url, _ in summaries]
    return context_block, sources
