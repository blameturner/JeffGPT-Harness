
from __future__ import annotations

import json
import logging
import random
import re
from urllib.parse import urlparse

from infra.config import (
    NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS,
    get_feature,
    get_function_config,
)
from infra.memory import get_collection
from infra.nocodb_client import NocodbClient
from tools.search.engine import searxng_search

_log = logging.getLogger("discover_agent")


def _cfg(key: str, default):
    return get_feature("discover_agent", key, default)


def _sample_random_chunks(org_id: int, n: int = 12) -> list[str]:
    collection_name = str(_cfg("sample_collection", "discovery"))
    try:
        col = get_collection(org_id, collection_name)
        count = col.count()
        if count == 0:
            col = get_collection(org_id, "agent_outputs")
            count = col.count()
        if count == 0:
            return []
        offset = random.randint(0, max(0, count - n))
        result = col.get(limit=n, offset=offset, include=["documents"])
        docs = result.get("documents") or []
        max_chars = int(_cfg("sample_max_chars_per_chunk", 400))
        return [str(d)[:max_chars] for d in docs if d]
    except Exception:
        _log.warning("chroma sample failed  org_id=%d", org_id, exc_info=True)
        return []


_QUERY_PROMPT = """\
You are a web-discovery assistant. Below are {n_chunks} text excerpts sampled \
from a knowledge base.

Propose {n_queries} diverse, specific web-search queries that would surface \
NEW authoritative pages relevant to the topics in the excerpts. Avoid \
repeating topics that seem already covered.

Return ONLY a JSON array of query strings. No explanation.
Example: ["query one", "query two"]

EXCERPTS:
{excerpts}
"""


def _generate_queries(chunks: list[str], n_queries: int) -> list[str]:
    from shared.models import model_call

    function_name = str(_cfg("query_model", "discover_agent_queries"))
    excerpts = "\n---\n".join(chunks)
    prompt = _QUERY_PROMPT.format(
        n_chunks=len(chunks),
        n_queries=n_queries,
        excerpts=excerpts[:6000],
    )
    try:
        raw, _ = model_call(function_name, prompt)
        raw = (raw or "").strip()
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            queries = json.loads(match.group())
            if isinstance(queries, list):
                return [str(q).strip() for q in queries if q][:n_queries]
    except Exception:
        _log.warning("query generation failed", exc_info=True)
    return []


_CLASSIFY_PROMPT = """You are reviewing a search result to decide if the page \
is worth scraping for an enrichment corpus.

QUERY: {query}
TITLE: {title}
URL: {url}
SNIPPET:
{snippet}

Respond with ONLY a JSON object, no prose:
{{
  "relevance": "high" | "medium" | "low" | "rejected",
  "score": <0-100 integer>,
  "reason": "<one short sentence>"
}}

"rejected" = spam, login wall, social media noise, unrelated topic.
"low" = thin/tangential.
"medium" = useful context.
"high" = substantive primary content.
"""


def _extract_json(raw: str) -> dict | None:
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s).rstrip("`").strip()
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _classify_hit(query: str, hit: dict) -> dict | None:
    from shared.models import model_call

    function_name = str(_cfg("classify_model", "scrape_classifier"))
    try:
        cfg = get_function_config(function_name)
    except KeyError:
        _log.warning("classifier model %r not configured — skipping classification", function_name)
        return None

    max_input = int(cfg.get("max_input_chars", 2000))
    snippet = (hit.get("content") or hit.get("snippet") or "").strip()[:max_input]
    prompt = _CLASSIFY_PROMPT.format(
        query=query[:200],
        title=(hit.get("title") or "")[:200],
        url=hit.get("url") or "",
        snippet=snippet,
    )
    try:
        raw, _ = model_call(function_name, prompt)
    except Exception as e:
        _log.debug("classifier call failed  url=%s  error=%s", (hit.get("url") or "")[:80], e)
        return None
    parsed = _extract_json(raw)
    if not parsed:
        return None
    label = str(parsed.get("relevance") or "").strip().lower()
    if label not in ("high", "medium", "low", "rejected"):
        label = "low"
    try:
        score = max(0, min(100, int(parsed.get("score") or 0)))
    except Exception:
        score = 0
    return {
        "relevance": label,
        "score": score,
        "reason": str(parsed.get("reason") or "")[:500],
    }


def _host_only(url: str) -> str:
    try:
        p = urlparse(url)
        host = p.netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def _existing_urls(client: NocodbClient, org_id: int) -> set[str]:
    """Return URLs already present in scrape_targets or suggested_scrape_targets."""
    urls: set[str] = set()
    scan_limit = int(_cfg("existing_scan_limit", 5000))
    for table in ("scrape_targets", NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS):
        try:
            rows = client._get_paginated(table, params={
                "where": f"(org_id,eq,{org_id})",
                "fields": "url",
                "limit": scan_limit,
            })
            for r in rows:
                u = (r.get("url") or "").strip()
                if u:
                    urls.add(u)
        except Exception:
            _log.debug("existing url scan failed  table=%s", table, exc_info=True)
    return urls


def _insert_suggestion(
    client: NocodbClient,
    org_id: int,
    url: str,
    title: str,
    query: str,
    classification: dict,
) -> int | None:
    payload = {
        "org_id": org_id,
        "url": url,
        "title": title[:255],
        "query": query[:500],
        "relevance": classification["relevance"],
        "score": classification["score"],
        "reason": classification["reason"],
        "status": "pending",
    }
    try:
        row = client._post(NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS, payload)
        return row.get("Id")
    except Exception as e:
        _log.warning("suggested_scrape_targets insert failed  url=%s  error=%s", url[:80], e)
        return None


def discover_agent_job(payload: dict | None = None) -> dict:
    """Tool-queue handler. One invocation per scheduler tick."""
    payload = payload or {}

    if not _cfg("enabled", True):
        return {"status": "disabled"}

    client = NocodbClient()

    org_id = int(payload.get("org_id") or 0)
    if not org_id:
        try:
            rows = client._get("scrape_targets", params={
                "limit": 1, "sort": "-CreatedAt", "fields": "org_id",
            }).get("list", [])
            org_id = int((rows[0].get("org_id") if rows else None) or 0)
        except Exception:
            pass
    if not org_id:
        return {"status": "no_org_context"}

    n_samples = int(_cfg("sample_chunks", 12))
    n_queries = int(_cfg("queries_per_run", 6))
    results_per_query = int(_cfg("results_per_query", 8))
    max_suggestions = int(_cfg("max_suggestions_per_run", 20))
    min_relevance = str(_cfg("min_relevance", "medium")).lower()
    _RELEVANCE_RANK = {"rejected": 0, "low": 1, "medium": 2, "high": 3}
    min_rank = _RELEVANCE_RANK.get(min_relevance, 2)

    _log.info("discover_agent start  org_id=%d samples=%d queries=%d", org_id, n_samples, n_queries)

    chunks = _sample_random_chunks(org_id, n=n_samples)
    if not chunks:
        return {"status": "no_chunks", "org_id": org_id}

    queries = _generate_queries(chunks, n_queries=n_queries)
    if not queries:
        return {"status": "no_queries", "org_id": org_id}

    existing = _existing_urls(client, org_id)

    seen_urls: set[str] = set()
    candidates: list[tuple[str, dict, str]] = []  # (query, hit, normalised_url)
    for q in queries:
        try:
            hits = searxng_search(q, max_results=results_per_query)
        except Exception:
            _log.warning("search failed  query=%s", q[:80], exc_info=True)
            continue
        for hit in hits:
            raw_url = (hit.get("url") or "").strip()
            if not raw_url:
                continue
            if raw_url in seen_urls or raw_url in existing:
                continue
            seen_urls.add(raw_url)
            candidates.append((q, hit, raw_url))

    _log.info("discover_agent candidates  queries=%d found=%d", len(queries), len(candidates))

    from workers.tool_queue import _background_idle_gate, seconds_since_chat

    added = 0
    classified = 0
    yielded_to_chat = False
    for q, hit, url in candidates:
        if added >= max_suggestions:
            break
        # Yield mid-loop if chat gets touched: classification LLM calls share the
        # model server with chat, so keep this loop cooperative.
        if seconds_since_chat() < _background_idle_gate():
            yielded_to_chat = True
            break
        verdict = _classify_hit(q, hit)
        if verdict is None:
            # Classifier unavailable — treat as medium so user still sees it.
            verdict = {"relevance": "medium", "score": 50, "reason": "classifier_unavailable"}
        classified += 1
        if _RELEVANCE_RANK.get(verdict["relevance"], 1) < min_rank:
            continue
        sid = _insert_suggestion(
            client, org_id, url, hit.get("title") or "", q, verdict,
        )
        if sid:
            added += 1

    _log.info(
        "discover_agent done  org_id=%d chunks=%d queries=%d candidates=%d classified=%d suggested=%d",
        org_id, len(chunks), len(queries), len(candidates), classified, added,
    )
    return {
        "status": "ok",
        "org_id": org_id,
        "chunks_sampled": len(chunks),
        "queries": queries,
        "candidates_found": len(candidates),
        "classified": classified,
        "suggested": added,
        "yielded_to_chat": yielded_to_chat,
    }
