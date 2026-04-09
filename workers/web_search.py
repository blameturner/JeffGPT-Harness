from __future__ import annotations

import json
import re
import time
from typing import Iterable

import httpx
from bs4 import BeautifulSoup

from config import MODELS, SEARXNG_URL
from memory import remember

MAX_SOURCES = 5
PER_PAGE_CHAR_CAP = 20_000          # hard cap before summarisation
SUMMARY_MAX_TOKENS = 300            # fast-model cap per page
SEARXNG_TIMEOUT = 10
SCRAPE_TIMEOUT = 15
FAST_TIMEOUT = 60

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
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?", "", raw).rstrip("`").strip()
        queries = json.loads(raw)
        if isinstance(queries, list):
            cleaned = [str(q).strip() for q in queries if str(q).strip()]
            if cleaned:
                return cleaned[:3]
    except Exception as e:
        print(f"[web_search] query generation failed: {e}")

    return [message.strip()[:200]]


def scrape_page(url: str) -> str:
    try:
        resp = httpx.get(
            url,
            timeout=SCRAPE_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; mst-harness/1.0)"},
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[web_search] scrape failed for {url}: {e}")
        return ""

    try:
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        print(f"[web_search] parse failed for {url}: {e}")
        return ""

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _strip_injection_patterns(text)
    return text[:PER_PAGE_CHAR_CAP]



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

    raw_results: list[dict] = []
    for q in queries:
        raw_results.extend(searxng_search(q))
        if len(raw_results) >= MAX_SOURCES * 2:
            break
    raw_results = _dedupe(raw_results)[: MAX_SOURCES * 2]

    summaries: list[tuple[str, str, str]] = []  # (title, url, summary)
    for r in raw_results:
        if len(summaries) >= MAX_SOURCES:
            break
        text = scrape_page(r["url"])
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
