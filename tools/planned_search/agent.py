from __future__ import annotations

import asyncio
import json
import logging
import time

from infra.config import get_feature
from infra.nocodb_client import NocodbClient
from tools.contract import ToolName, ToolResult
from tools.dispatcher import register_executor
from tools.planned_search.planner import generate_planned_queries
from tools.scraper.pathfinder import PathfinderScraper
from tools.search.engine import searxng_search

_log = logging.getLogger("planned_search.agent")

# planned_search does its OWN searxng fan-out + relevance filter. It deliberately
# does NOT call tools.search.web_search (the LLM-orchestrated web_search tool) —
# planned_search is an explicit, user-approved query path and must not trigger
# the implicit web_search pipeline.
#
# `_PLANNED_MAX_URLS` sets how many candidate URLs survive the pre-scrape cap.
# It must be larger than `successful_scrapes_needed` because block pages and
# scrape failures eat into this budget — if they're equal, one block page
# means one fewer source reaches synthesis. We keep 3× the needed count so
# we have headroom after block-page/stub drops.
_PLANNED_SEARXNG_PER_QUERY = 10
_PLANNED_RELEVANCE_THRESHOLD = 0.25


def _planned_timeout_s(key: str, default_s: int) -> int:
    raw = get_feature("planned_search", key, default_s)
    try:
        val = int(raw)
        return val if val > 0 else default_s
    except Exception:
        return default_s


def _planned_max_urls() -> int:
    """Candidate-URL cap derived from the successful_scrapes_needed feature
    flag. See comment above for why we oversample by 3×."""
    needed = int(get_feature("planned_search", "successful_scrapes_needed", 10) or 10)
    return max(needed * 3, needed + 5)


async def _planned_search_all(queries: list[str]) -> list[dict]:
    async def _one(q: str) -> list[dict]:
        try:
            return await asyncio.to_thread(searxng_search, q, _PLANNED_SEARXNG_PER_QUERY) or []
        except Exception as e:
            _log.warning("planned_search searxng failed q=%r: %s", q[:80], e)
            return []

    results_per_query = await asyncio.gather(*[_one(q) for q in queries])
    seen: set[str] = set()
    deduped: list[dict] = []
    for rs in results_per_query:
        for r in rs:
            url = (r.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            deduped.append(r)
    return deduped[:_planned_max_urls()]


# Signatures of pages that returned HTTP 200 + HTML but carry no usable content:
# CDN/WAF challenges, captchas, security blocks, stubs, generic error pages.
# These otherwise sneak through the keyword relevance filter when the block page
# happens to contain the query terms (e.g. "Access to Rust was denied").
#
# Kept narrow: phrases here must appear in actual block interstitials and be
# unlikely in prose about those same topics. Words like "cloudflare",
# "captcha", "forbidden", "rate limited" appear in normal articles, so they
# are NOT listed — otherwise we'd drop legitimate content about those topics.
_BLOCK_PAGE_SIGNATURES = (
    "attention required! | cloudflare",
    "are you a human?",
    "checking your browser before accessing",
    "enable javascript and cookies to continue",
    "enable javascript to continue",
    "enable cookies to continue",
    "just a moment...",
    "please verify you are human",
    "please stand by, while we are checking",
    "this website is using a security service to protect itself",
    "you have been blocked",
    "your request has been blocked",
    "request unsuccessful. incapsula incident",
    "ray id:",
    "you don't have permission to access",
)
_MIN_USEFUL_SCRAPE_CHARS = 500


def _looks_like_block_page(text: str) -> bool:
    """True when the scraped 'text' is a WAF/captcha/block page or a useless stub.
    These pages commonly pass keyword relevance by accident and poison synthesis."""
    if not text:
        return True
    stripped = text.strip()
    if len(stripped) < _MIN_USEFUL_SCRAPE_CHARS:
        return True
    head = stripped[:1500].lower()
    return any(sig in head for sig in _BLOCK_PAGE_SIGNATURES)


def _planned_filter_by_relevance(results: list[dict], queries: list[str]) -> list[dict]:
    if not results or _PLANNED_RELEVANCE_THRESHOLD <= 0:
        return results
    from tools.search.queries import _extract_keywords
    keywords: set[str] = set()
    for q in queries:
        for kw in _extract_keywords(q):
            keywords.add(kw.lower())
    if not keywords:
        return results
    kept: list[dict] = []
    for r in results:
        haystack = f"{r.get('title', '')} {r.get('snippet', '')}".lower()
        hits = sum(1 for kw in keywords if kw in haystack)
        if hits / len(keywords) >= _PLANNED_RELEVANCE_THRESHOLD:
            kept.append(r)
    return kept


@register_executor(ToolName.PLANNED_SEARCH)
async def execute(params: dict, emit) -> ToolResult:
    user_question = params.get("question", "")
    conversation_id = params.get("_conversation_id") or params.get("conversation_id") or 0
    org_id = params.get("_org_id") or params.get("org_id") or 0

    if not user_question:
        return ToolResult(
            tool=ToolName.PLANNED_SEARCH,
            action_index=0,
            ok=False,
            data="No question provided",
        )

    emit({"type": "status", "phase": "planning", "message": "Generating search queries..."})

    t0 = time.time()
    queries = await generate_planned_queries(user_question)

    if not queries:
        return ToolResult(
            tool=ToolName.PLANNED_SEARCH,
            action_index=0,
            ok=False,
            data="Failed to generate search queries",
            elapsed_s=round(time.time() - t0, 2),
        )

    query_list = [{"query": q["query"], "reason": q.get("reason", "")} for q in queries]

    client = NocodbClient()
    query_text = json.dumps(query_list, ensure_ascii=False)

    try:
        msg = client.add_message(
            conversation_id=conversation_id,
            org_id=org_id,
            role="assistant",
            content=query_text,
            model="planned_search",
            pending_approval=1,
            search_status="awaiting_approval",
        )
        message_id = msg.get("Id") or msg.get("id")
        _log.info("planned_search message created  msg_id=%s queries=%d conv=%s",
                  message_id, len(query_list), conversation_id)
    except Exception:
        _log.warning("planned_search message create failed", exc_info=True)
        message_id = None

    render_content = f"[planned_search]\n" + "\n".join(
        f"- {q['query']}" + (f" ({q.get('reason')})" if q.get("reason") else "")
        for q in query_list
    )
    if message_id:
        render_content += f"\n\n[message_id:{message_id}]"

    return ToolResult(
        tool=ToolName.PLANNED_SEARCH,
        action_index=0,
        ok=True,
        data=render_content,
        elapsed_s=round(time.time() - t0, 2),
    )


def _relevance_label(ratio: float) -> str:
    if ratio >= 0.5:
        return "high"
    if ratio >= 0.3:
        return "medium"
    if ratio > 0:
        return "low"
    return "unknown"


def _score_against_keywords(text: str, query_keywords: set[str]) -> tuple[float, str]:
    if not query_keywords or not text:
        return 0.0, "unknown"
    haystack = text.lower()
    hits = sum(1 for kw in query_keywords if kw in haystack)
    ratio = hits / len(query_keywords)
    return round(ratio, 3), _relevance_label(ratio)


async def approve_searches(message_id: int, org_id: int) -> dict:
    """Endpoint-facing shell: validates queries, marks the message queued, submits a job.
    Returns fast so the UI can poll for status instead of blocking on scrape/synthesis.
    """
    client = NocodbClient()

    try:
        msgs = client._get("messages", params={"where": f"(Id,eq,{message_id})", "limit": 1})
        msg_row = (msgs.get("list") or [{}])[0]
        row_org_id = int(msg_row.get("org_id") or 0)
        if not row_org_id:
            return {"status": "error", "message": "Message has no org_id"}
        if int(org_id or 0) != row_org_id:
            return {"status": "error", "message": "org_id mismatch for message"}
        query_content = msg_row.get("content") or "{}"
    except Exception:
        return {"status": "error", "message": "Message not found"}

    try:
        queries_data = json.loads(query_content)
    except Exception:
        return {"status": "error", "message": "Invalid query format"}

    query_list = [q.get("query") for q in queries_data.get("queries", queries_data) if isinstance(q, dict) and q.get("query")] \
        if isinstance(queries_data, dict) else \
        [q["query"] for q in queries_data if isinstance(q, dict) and q.get("query")]
    if not query_list:
        return {"status": "error", "message": "No queries to execute"}

    try:
        client._patch("messages", message_id, {"search_status": "queued", "pending_approval": 0})
    except Exception:
        _log.warning("planned_search queued patch failed", exc_info=True)

    from workers.tool_queue import get_tool_queue
    tq = get_tool_queue()
    if tq:
        job_id = tq.submit(
            "planned_search_execute",
            {"message_id": message_id, "org_id": row_org_id},
            source="planned_search_api",
            priority=2,
            org_id=row_org_id,
        )
        _log.info("Queued planned_search job %s for message_id %s", job_id, message_id)
        return {"status": "queued", "message_id": message_id, "job_id": job_id}

    _log.warning("Tool queue not available, running planned_search synchronously for message_id %s", message_id)
    return await _run_planned_search_async(message_id, row_org_id)


def run_planned_search_job(message_id: int, org_id: int) -> dict:
    """Tool-queue handler entry point: runs the async worker on a worker thread via asyncio.run."""
    return asyncio.run(_run_planned_search_async(message_id, org_id))


def run_planned_search_scrape_job(payload: dict) -> dict:
    url = (payload.get("url") or "").strip()
    title = (payload.get("title") or "").strip()
    org_id = int(payload.get("org_id") or 0)
    query_keywords = set((payload.get("query_keywords") or []))

    if not url:
        return {"status": "error", "reason": "missing_url"}

    scraper = PathfinderScraper()
    scraped = scraper.scrape(url)
    if scraped.get("status") != "ok" or not scraped.get("text"):
        return {"status": "failed", "reason": scraped.get("error") or "scrape_failed", "url": url}

    text = scraped.get("text", "")
    if _looks_like_block_page(text):
        return {"status": "failed", "reason": "block_page", "url": url}

    snippet = text[:1000]
    ratio, label = _score_against_keywords(f"{title} {snippet}", query_keywords)
    return {
        "status": "ok",
        "url": scraped.get("canonical") or url,
        "title": title,
        "snippet": snippet,
        "relevance": label,
        "relevance_score": ratio,
        "org_id": org_id,
    }


async def _run_planned_search_async(message_id: int, org_id: int) -> dict:
    client = NocodbClient()

    try:
        msgs = client._get("messages", params={"where": f"(Id,eq,{message_id})", "limit": 1})
        msg_row = (msgs.get("list") or [{}])[0]
        query_content = msg_row.get("content") or "{}"
    except Exception:
        _log.warning("planned_search load message failed  id=%s", message_id, exc_info=True)
        return {"status": "error", "message": "Message not found"}

    try:
        queries_data = json.loads(query_content)
    except Exception:
        client._patch("messages", message_id, {"search_status": "failed"})
        return {"status": "error", "message": "Invalid query format"}

    if isinstance(queries_data, dict):
        raw_queries = queries_data.get("queries", [])
    else:
        raw_queries = queries_data
    query_list = [q["query"] for q in raw_queries if isinstance(q, dict) and q.get("query")]
    if not query_list:
        client._patch("messages", message_id, {"search_status": "failed"})
        return {"status": "error", "message": "No queries to execute"}

    try:
        client._patch("messages", message_id, {"search_status": "approved"})
    except Exception:
        pass

    from tools.search.queries import _extract_keywords
    query_keywords: set[str] = set()
    for q in query_list:
        for kw in _extract_keywords(q):
            query_keywords.add(kw.lower())

    results = await _planned_search_all(query_list)
    results = _planned_filter_by_relevance(results, query_list)

    needed = get_feature("planned_search", "successful_scrapes_needed", 10)

    scraped_results: list[dict] = []
    from workers.tool_queue import get_tool_queue
    tq = get_tool_queue()
    if tq:
        scrape_job_ids: list[str] = []
        keywords_payload = sorted(query_keywords)
        for r in results:
            url = (r.get("url") or "").strip()
            if not url:
                continue
            job_id = tq.submit(
                "planned_search_scrape",
                {
                    "url": url,
                    "title": r.get("title", ""),
                    "org_id": org_id,
                    "query_keywords": keywords_payload,
                },
                source="planned_search_execute",
                priority=2,
                org_id=org_id,
            )
            scrape_job_ids.append(job_id)

        scrape_timeout_s = _planned_timeout_s("scrape_timeout_s", 1800)
        deadline = time.time() + scrape_timeout_s
        pending = set(scrape_job_ids)
        while pending and len(scraped_results) < needed and time.time() < deadline:
            finished: list[str] = []
            for job_id in pending:
                job = tq.get_job(job_id)
                if not job or job.status in ("queued", "running"):
                    continue
                finished.append(job_id)
                if job.status == "completed":
                    res = job.result or {}
                    if res.get("status") == "ok":
                        scraped_results.append({
                            "url": res.get("url", ""),
                            "title": res.get("title", ""),
                            "snippet": res.get("snippet", ""),
                            "relevance": res.get("relevance", "unknown"),
                            "relevance_score": res.get("relevance_score", 0.0),
                        })
                        if len(scraped_results) >= needed:
                            break
            for job_id in finished:
                pending.discard(job_id)
            if pending and len(scraped_results) < needed:
                await asyncio.sleep(1)
        if pending and len(scraped_results) < needed:
            _log.warning(
                "planned_search scrape phase timeout after %ds  completed=%d/%d",
                scrape_timeout_s,
                len(scrape_job_ids) - len(pending),
                len(scrape_job_ids),
            )
    else:
        from tools.scraper.pathfinder import PathfinderScraper

        scraper = PathfinderScraper()

        def _scrape(url: str) -> dict:
            return scraper.scrape(url)

        for r in results:
            if len(scraped_results) >= needed:
                break
            url = (r.get("url") or "").strip()
            if not url:
                continue
            scraped = await asyncio.to_thread(_scrape, url)
            if scraped.get("status") == "ok" and scraped.get("text"):
                text = scraped.get("text", "")
                if _looks_like_block_page(text):
                    _log.info("planned_search drop  reason=block_page  url=%s chars=%d",
                              url[:100], len(text))
                    continue
                title = r.get("title", "")
                snippet = text[:1000]
                ratio, label = _score_against_keywords(f"{title} {snippet}", query_keywords)
                scraped_results.append({
                    "url": scraped.get("canonical") or url,
                    "title": title,
                    "snippet": snippet,
                    "relevance": label,
                    "relevance_score": ratio,
                })

    conversation_id = msg_row.get("conversation_id") or 0
    for src in scraped_results:
        try:
            client.add_message_search_sources(
                message_id=message_id,
                conversation_id=conversation_id,
                org_id=org_id,
                sources=[{
                    "title": src.get("title", "")[:255],
                    "url": src.get("url", ""),
                    "relevance": src.get("relevance", "unknown"),
                    "source_type": "web",
                    "snippet": src.get("snippet", ""),
                    "used_in_answer": True,
                }],
            )
        except Exception:
            _log.warning("planned_search add source failed", exc_info=True)

    user_question = _find_original_question(client, conversation_id, message_id)
    answer_text, answer_tokens = await _synthesize_answer(user_question, scraped_results)

    answer_msg_id = None
    if answer_text:
        try:
            answer_msg = client.add_message(
                conversation_id=conversation_id,
                org_id=org_id,
                role="assistant",
                content=answer_text,
                model="planned_search_answer",
                tokens_output=answer_tokens,
                search_used=True,
                search_status="completed",
                search_source_count=len(scraped_results),
            )
            answer_msg_id = answer_msg.get("Id") or answer_msg.get("id")
            for src in scraped_results:
                try:
                    client.add_message_search_sources(
                        message_id=answer_msg_id,
                        conversation_id=conversation_id,
                        org_id=org_id,
                        sources=[{
                            "title": src.get("title", "")[:255],
                            "url": src.get("url", ""),
                            "relevance": src.get("relevance", "unknown"),
                            "source_type": "web",
                            "snippet": src.get("snippet", ""),
                            "used_in_answer": True,
                        }],
                    )
                except Exception:
                    _log.warning("planned_search answer add source failed", exc_info=True)
        except Exception:
            _log.warning("planned_search answer message create failed", exc_info=True)

    final_status = "completed" if answer_text else "failed"
    try:
        client._patch("messages", message_id, {
            "pending_approval": 0,
            "search_used": 1,
            "search_source_count": len(scraped_results),
            "search_status": final_status,
        })
    except Exception:
        _log.warning("planned_search patch message failed", exc_info=True)

    if answer_text:
        convo_row = _load_conversation(client, conversation_id)
        rag_enabled = _truthy(convo_row.get("rag_enabled"))
        knowledge_enabled = _truthy(convo_row.get("knowledge_enabled"))
        rag_collection = (convo_row.get("rag_collection") or "agent_outputs").strip() or "agent_outputs"

        from workers.post_turn import ingest_output
        await asyncio.to_thread(
            ingest_output,
            output=answer_text,
            user_text=user_question,
            org_id=org_id,
            conversation_id=conversation_id,
            model="planned_search_answer",
            rag_collection=rag_collection if rag_enabled else "",
            knowledge_collection="chat_knowledge" if knowledge_enabled else "",
            source="planned_search",
            extra_metadata={
                "message_id": answer_msg_id,
                "proposal_message_id": message_id,
                "source_urls": [s.get("url") for s in scraped_results],
            },
        )

    return {
        "status": "ok",
        "message_id": message_id,
        "answer_message_id": answer_msg_id,
        "queries_executed": len(query_list),
        "results_found": len(results),
        "successful_scrapes": len(scraped_results),
        "answer_chars": len(answer_text),
    }


def _load_conversation(client: NocodbClient, conversation_id: int) -> dict:
    if not conversation_id:
        return {}
    try:
        rows = client._get("conversations", params={
            "where": f"(Id,eq,{conversation_id})",
            "limit": 1,
        }).get("list", [])
        return rows[0] if rows else {}
    except Exception:
        return {}


def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return False


def _find_original_question(client: NocodbClient, conversation_id: int, before_message_id: int) -> str:
    if not conversation_id:
        return ""
    try:
        rows = client._get("messages", params={
            "where": f"(conversation_id,eq,{conversation_id})~and(role,eq,user)~and(Id,lt,{before_message_id})",
            "sort": "-Id",
            "limit": 1,
        }).get("list", [])
        if rows:
            return (rows[0].get("content") or "").strip()
    except Exception:
        _log.warning("planned_search could not load original question", exc_info=True)
    return ""


async def _synthesize_answer(question: str, sources: list[dict]) -> tuple[str, int]:
    if not sources:
        return "", 0

    from infra.config import get_function_config
    from shared.models import model_call

    cfg = get_function_config("planned_search_answer")
    max_input = cfg.get("max_input_chars", 24000)

    context_parts: list[str] = []
    budget = max_input - 500
    for i, src in enumerate(sources, start=1):
        chunk = f"[{i}] {src.get('title') or src.get('url')}\nURL: {src.get('url')}\n{src.get('snippet', '')}"
        if budget - len(chunk) < 0:
            break
        context_parts.append(chunk)
        budget -= len(chunk)

    prompt = (
        "Answer the user's question using only the sources below. "
        "Cite sources inline as [1], [2] matching the numbered list. "
        "If sources disagree or are insufficient, say so.\n\n"
        f"Question: {question or '(original question unavailable)'}\n\n"
        "Sources:\n" + "\n\n".join(context_parts)
    )

    # model_call has no internal timeout; wrap in asyncio.wait_for so the whole
    # planned_search job can't hang on a stalled LLM. Same policy as research.
    try:
        timeout_s = _planned_timeout_s("synthesis_timeout_s", 1800)
        raw, tokens = await asyncio.wait_for(
            asyncio.to_thread(model_call, "planned_search_answer", prompt, True),
            timeout=timeout_s,
        )
        return (raw or "").strip(), tokens or 0
    except asyncio.TimeoutError:
        _log.warning("planned_search synthesis timeout after %ds", timeout_s)
        return "", 0
    except Exception:
        _log.warning("planned_search synthesis failed", exc_info=True)
        return "", 0


def reject_searches(message_id: int) -> dict:
    client = NocodbClient()

    try:
        client._patch("messages", message_id, {
            "pending_approval": 0,
            "search_status": "declined",
        })
        return {"status": "ok", "message_id": message_id}
    except Exception:
        _log.warning("planned_search reject failed", exc_info=True)
        return {"status": "error", "message": "Failed to reject"}


def get_pending_search(message_id: int) -> dict:
    client = NocodbClient()

    try:
        msgs = client._get("messages", params={"where": f"(Id,eq,{message_id})", "limit": 1})
        msg_row = (msgs.get("list") or [{}])[0]
        return {
            "status": "ok",
            "message": {
                "id": msg_row.get("Id"),
                "content": msg_row.get("content"),
                "pending_approval": msg_row.get("pending_approval"),
                "search_status": msg_row.get("search_status"),
            },
        }
    except Exception:
        return {"status": "error", "message": "Message not found"}


def get_search_results(message_id: int, org_id: int) -> dict:
    client = NocodbClient()

    try:
        msgs = client._get("messages", params={"where": f"(Id,eq,{message_id})", "limit": 1})
        msg_row = (msgs.get("list") or [{}])[0]
        row_org_id = int(msg_row.get("org_id") or 0)
        if not row_org_id:
            return {"status": "error", "message": "Message has no org_id"}
        if int(org_id or 0) != row_org_id:
            return {"status": "error", "message": "org_id mismatch for message"}
    except Exception:
        return {"status": "error", "message": "Message not found"}

    try:
        sources = client.list_message_search_sources(message_id=message_id)
    except Exception:
        return {"status": "error", "message": "Failed to get sources"}

    search_status = None
    source_count = None
    search_status = msg_row.get("search_status")
    source_count = msg_row.get("search_source_count")

    return {
        "status": "ok",
        "search_status": search_status,
        "source_count": source_count,
        "sources": sources,
    }