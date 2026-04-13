"""Shared pipeline for deep search and research executors.

Common stages: search, scrape, summarise, store, deliver.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass

from nocodb_client import NocodbClient

_log = logging.getLogger("tools.search.pipeline")

MIN_USABLE_TEXT_CHARS = 200
MIN_SUMMARY_CHARS = 50


def parse_llm_json(raw: str) -> dict | list | None:
    """Strip markdown fences from LLM output and parse JSON."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        _log.warning("parse_llm_json failed: %s", cleaned[:200])
        return None


async def search_and_dedup(
    queries: list[str],
    max_results: int = 25,
    existing_urls: set[str] | None = None,
) -> list[dict]:
    """Parallel SearXNG search across all queries, dedup, blocklist filter."""
    from tools.framework.executors.search.web_search import _search_one
    from workers.search.urls import _is_blocklisted

    all_raw = await asyncio.gather(*[_search_one(q) for q in queries], return_exceptions=True)
    seen: set[str] = set(existing_urls or ())
    results: list[dict] = []
    for result_set in all_raw:
        if isinstance(result_set, Exception):
            continue
        for r in result_set:
            url = (r.get("url") or "").strip()
            if not url or url in seen or _is_blocklisted(url):
                continue
            seen.add(url)
            results.append(r)
    return results[:max_results]


async def scrape_sources(sources: list[dict]) -> list[dict]:
    """Parallel scrape, returns only sources with >= MIN_USABLE_TEXT_CHARS."""
    from tools.framework.executors.search.web_search import _scrape_one

    raw = await asyncio.gather(*[_scrape_one(s) for s in sources], return_exceptions=True)
    usable: list[dict] = []
    for i, r in enumerate(raw):
        if isinstance(r, Exception):
            _log.warning("scrape exception  url=%s  error=%s",
                         sources[i].get("url", "?")[:60], r)
            continue
        if len(r.get("text") or "") >= MIN_USABLE_TEXT_CHARS:
            usable.append(r)
    return usable


async def summarise_sources(
    sources: list[dict],
    question: str,
    function_name: str,
    max_sources: int | None = None,
) -> list[dict]:
    """Parallel-bounded summarisation. Returns [{url, title, summary}]."""
    from tools.framework.executors.search.web_search import _summarise_one

    to_process = sources[:max_sources] if max_sources else sources
    sem = asyncio.Semaphore(3)

    async def _bounded(source: dict) -> dict | None:
        async with sem:
            result = await _summarise_one(
                source["url"], source["text"], question, function_name, priority=False,
            )
            summary_text = result.get("summary", "")
            if summary_text and len(summary_text) >= MIN_SUMMARY_CHARS:
                return {"url": source["url"], "title": source.get("title", ""), "summary": summary_text}
            return None

    results = await asyncio.gather(*[_bounded(s) for s in to_process], return_exceptions=True)
    return [r for r in results if isinstance(r, dict)]


def store_summaries(
    summaries: list[dict],
    metadata_type: str,
    collection_name: str,
    question: str,
    conversation_id: int | str | None,
    org_id: int,
) -> None:
    from memory import remember
    from workers.chat.graph import extract_and_write_graph

    for s in summaries:
        try:
            remember(
                text=s["summary"],
                metadata={"url": s["url"], "type": metadata_type, "question": question[:500], "conversation_id": conversation_id},
                org_id=org_id, collection_name=collection_name,
            )
        except Exception:
            _log.warning("chroma store failed  url=%s", s["url"][:60], exc_info=True)

        try:
            extract_and_write_graph(f"{metadata_type} source: {s['url']}", s["summary"], conversation_id or 0, org_id)
        except Exception:
            _log.debug("graph extract failed  url=%s", s["url"][:60], exc_info=True)


def store_report(
    report: str,
    metadata_type: str,
    collection_name: str,
    question: str,
    conversation_id: int | str | None,
    org_id: int,
    source_count: int,
    iterations: int = 1,
) -> None:
    from memory import remember
    from workers.chat.graph import extract_and_write_graph

    try:
        remember(
            text=f"{metadata_type.upper()}: {question}\n\n{report}",
            metadata={"type": f"{metadata_type}_report", "question": question[:500], "sources": source_count, "iterations": iterations, "conversation_id": conversation_id},
            org_id=org_id, collection_name=collection_name,
        )
    except Exception:
        _log.error("report chroma store failed", exc_info=True)

    try:
        extract_and_write_graph(f"{metadata_type}: {question[:80]}", report[:8000], conversation_id or 0, org_id)
    except Exception:
        _log.debug("report graph extract failed", exc_info=True)


def build_evidence_block(summaries: list[dict]) -> str:
    parts: list[str] = []
    for i, s in enumerate(summaries, 1):
        parts.append(f"[{i}] {s['url']}\nTitle: {s['title']}\n{s['summary']}")
    return "\n\n".join(parts)


def deliver_to_conversation(
    conversation_id: int | str,
    org_id: int,
    content: str,
    model: str,
    search_status: str = "completed",
    search_confidence: str = "high",
    source_count: int = 0,
    tokens_output: int = 0,
) -> None:
    db = NocodbClient()
    db.add_message(
        conversation_id=int(conversation_id), org_id=org_id,
        role="assistant", content=content[:16000], model=model,
        tokens_input=0, tokens_output=tokens_output,
        search_used=True, search_status=search_status,
        search_confidence=search_confidence, search_source_count=source_count,
    )


@dataclass
class PendingPlanConfig:
    db_field: str
    plan_model: str


PLAN_CONFIGS = {
    "deep_search": PendingPlanConfig(db_field="pending_deep_search", plan_model="deep_search_plan"),
    "research": PendingPlanConfig(db_field="pending_research", plan_model="research_plan"),
}


def store_pending_plan(conversation_id: int | str, tool_label: str, plan_dict: dict) -> None:
    cfg = PLAN_CONFIGS[tool_label]
    NocodbClient().update_conversation(int(conversation_id), {cfg.db_field: json.dumps(plan_dict)})


def clear_pending_plan(conversation_id: int | str, tool_label: str) -> None:
    cfg = PLAN_CONFIGS[tool_label]
    NocodbClient().update_conversation(int(conversation_id), {cfg.db_field: ""})
