"""
Research agent — deep multi-source investigation that runs on idle compute.

Produces a formal cited document with hypothesis based on extensive research.
Can be triggered manually via POST /research or auto-queued when models are idle.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from config import get_model_url, no_think_params, MODELS
from memory import remember
from nocodb_client import NocodbClient
from tools.framework.executors.web_search import (
    _search_all,
    _scrape_one,
    _summarise_one,
    MAX_SUMMARY_CHARS,
)
from workers.search.intent import classify_message_intent
from workers.search.models import acquire_model, _free_slots, _sem_for
from workers.search.queries import generate_search_queries

_log = logging.getLogger("research")

MAX_RESEARCH_URLS = 15
MAX_RESEARCH_QUERIES = 8
RESEARCH_SUMMARY_CONCURRENCY = 2

RESEARCH_SYSTEM_PROMPT = (
    "You are a research analyst producing a formal report. "
    "Structure your response with: "
    "1. Executive Summary (2-3 sentences) "
    "2. Key Findings (numbered, with citations to source URLs) "
    "3. Analysis & Hypothesis "
    "4. Limitations & Gaps "
    "5. Sources (numbered list of URLs used) "
    "Be factual. Cite sources by number [1], [2] etc. "
    "State hypotheses clearly as hypotheses, not facts. "
    "If evidence is contradictory, note the disagreement."
)


def estimate_research_time() -> str:
    """Estimate when research might run based on current model load."""
    tool_roles = ["t3_tool", "t2_coder"]
    total_free = 0
    total_slots = 0
    for role in tool_roles:
        entry = MODELS.get(role)
        if isinstance(entry, dict) and entry.get("url"):
            sem = _sem_for(role)
            free = _free_slots(sem)
            from config import MODEL_PARALLEL_SLOTS
            total_free += free
            total_slots += MODEL_PARALLEL_SLOTS

    if total_free == total_slots:
        return "Models are idle — research can start immediately"
    elif total_free > 0:
        return "Some model capacity available — research can start within 1-2 minutes"
    else:
        return "All model slots busy — research will queue behind current work, estimated 5-15 minutes"


def run_research(
    question: str,
    org_id: int,
    model: str,
    job=None,
    conversation_id: int | None = None,
) -> dict | None:
    """Run a deep research investigation. Returns the research document or None on failure."""
    from workers.jobs import STORE

    def emit(event: dict):
        if job:
            STORE.append(job, event)

    db = NocodbClient()
    t0 = time.time()
    _log.info("research start  question=%s org=%d", question[:80], org_id)

    emit({"type": "research_status", "phase": "classifying", "message": "Analysing research question..."})

    # 1. Intent classification + query generation
    intent_dict = classify_message_intent(question)
    intent_dict["search_policy"] = "full"

    base_queries = generate_search_queries(intent_dict, message=question)
    # Expand with additional angles
    extra_queries = [
        f"{question} research review",
        f"{question} analysis comparison",
    ]
    all_queries = list(dict.fromkeys(base_queries + extra_queries))[:MAX_RESEARCH_QUERIES]
    _log.info("research queries  count=%d queries=%s", len(all_queries), all_queries)

    emit({"type": "research_status", "phase": "searching", "message": f"Searching {len(all_queries)} queries...", "queries": all_queries})

    # 2. Parallel SearXNG across all queries
    try:
        results = asyncio.run(_search_all(all_queries))
    except Exception:
        _log.error("research searxng failed", exc_info=True)
        emit({"type": "error", "message": "Search failed"})
        return None

    results = results[:MAX_RESEARCH_URLS]
    _log.info("research results  urls=%d", len(results))

    if not results:
        emit({"type": "error", "message": "No search results found for this research question"})
        return None

    emit({"type": "research_status", "phase": "scraping", "message": f"Scraping {len(results)} sources..."})

    # 3. Parallel scrape
    try:
        scraped = asyncio.run(asyncio.gather(*[_scrape_one(r) for r in results]))
    except Exception:
        _log.error("research scraping failed", exc_info=True)
        emit({"type": "error", "message": "Scraping failed"})
        return None

    with_text = [s for s in scraped if s["text"] and len(s["text"]) >= 200]
    _log.info("research scraped  total=%d with_text=%d", len(scraped), len(with_text))

    if not with_text:
        emit({"type": "error", "message": "All sources returned empty content"})
        return None

    emit({"type": "research_status", "phase": "summarising", "message": f"Summarising {len(with_text)} sources..."})

    # 4. Summarise each source through tool model
    _sem = asyncio.Semaphore(RESEARCH_SUMMARY_CONCURRENCY)

    async def _bounded(client, url, text, query):
        async with _sem:
            return await _summarise_one(client, url, text, query)

    try:
        async def _summarise_all():
            async with httpx.AsyncClient() as client:
                return await asyncio.gather(*[
                    _bounded(client, s["url"], s["text"], question)
                    for s in with_text
                ])
        summaries = asyncio.run(_summarise_all())
    except Exception:
        _log.error("research summarisation failed", exc_info=True)
        summaries = [s["text"][:MAX_SUMMARY_CHARS] for s in with_text]

    # 5. Build source list for citations
    sources_block = ""
    for i, (s, summary) in enumerate(zip(with_text, summaries), 1):
        sources_block += f"\n[{i}] {s['url']}\nTitle: {s['title']}\nSummary: {summary}\n"

    emit({"type": "research_status", "phase": "synthesising", "message": "Synthesising research document..."})

    # 6. Send to main model for synthesis
    model_url = get_model_url(model)
    if not model_url:
        emit({"type": "error", "message": f"Model '{model}' not available"})
        return None

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Research question: {question}\n\n"
                f"Below are {len(with_text)} sources gathered from web research. "
                f"Synthesise them into a formal research report.\n\n"
                f"SOURCES:\n{sources_block}"
            )},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    import requests as req
    chunks: list[str] = []
    try:
        with req.post(f"{model_url}/v1/chat/completions", json=payload, stream=True, timeout=(30, 3600)) as resp:
            resp.raise_for_status()
            resp.encoding = "utf-8"
            in_think = False
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data = raw_line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    import json
                    event = json.loads(data)
                except Exception:
                    continue
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = delta.get("content")
                if not text:
                    continue
                if "<think>" in text:
                    in_think = True
                    continue
                if in_think:
                    if "</think>" in text:
                        in_think = False
                    continue
                chunks.append(text)
                emit({"type": "chunk", "text": text})
    except Exception:
        _log.error("research model call failed", exc_info=True)
        emit({"type": "error", "message": "Model call failed during research synthesis"})
        return None

    output = "".join(chunks)
    elapsed = round(time.time() - t0, 1)
    _log.info("research done  question=%s sources=%d chars=%d elapsed=%.1fs", question[:60], len(with_text), len(output), elapsed)

    # 7. Store in ChromaDB
    if output:
        try:
            remember(
                text=f"RESEARCH: {question}\n\n{output}",
                metadata={"type": "research", "question": question[:500], "sources": len(with_text)},
                org_id=org_id,
                collection_name="research",
            )
        except Exception:
            _log.error("research chroma store failed", exc_info=True)

    source_list = [{"url": s["url"], "title": s["title"]} for s in with_text]

    emit({
        "type": "done",
        "mode": "research",
        "conversation_id": conversation_id,
        "model": model,
        "sources_count": len(with_text),
        "duration_seconds": elapsed,
        "output": output,
    })

    return {
        "output": output,
        "sources": source_list,
        "duration_seconds": elapsed,
        "queries_used": all_queries,
    }
