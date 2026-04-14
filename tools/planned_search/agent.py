from __future__ import annotations

import asyncio
import json
import logging
import time

from infra.config import get_feature_value
from infra.nocodb_client import NocodbClient
from tools.contract import ToolName, ToolResult
from tools.dispatcher import register_executor
from tools.planned_search.planner import generate_planned_queries
from tools.search.web_search import _filter_results_by_relevance, _search_all

_log = logging.getLogger("planned_search.agent")


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
        )
        message_id = msg.get("Id") or msg.get("id")
        _log.info("planned_search message created  msg_id=%s queries=%d", message_id, len(query_list))
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


async def approve_searches(message_id: int, org_id: int) -> dict:
    client = NocodbClient()

    try:
        msgs = client._get("messages", params={"where": f"(Id,eq,{message_id})", "limit": 1})
        msg_row = (msgs.get("list") or [{}])[0]
        query_content = msg_row.get("content") or "{}"
    except Exception:
        return {"status": "error", "message": "Message not found"}

    try:
        queries_data = json.loads(query_content)
    except Exception:
        return {"status": "error", "message": "Invalid query format"}

    query_list = [q["query"] for q in queries_data if q.get("query")]
    if not query_list:
        return {"status": "error", "message": "No queries to execute"}

    try:
        client._patch("messages", message_id, {"search_status": "running"})
    except Exception:
        pass

    results = await _search_all(query_list)
    results = _filter_results_by_relevance(results, query_list)

    needed = get_feature_value("planned_search_successful_scrapes_needed", 10)

    from tools.scraper.search import SearchScraper

    scraper = SearchScraper()
    scraped_results: list[dict] = []

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
            scraped_results.append({
                "url": scraped.get("canonical") or url,
                "title": r.get("title", ""),
                "snippet": scraped.get("text", "")[:1000],
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
                    "relevance": "high",
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
                            "relevance": "high",
                            "source_type": "web",
                            "snippet": src.get("snippet", ""),
                            "used_in_answer": True,
                        }],
                    )
                except Exception:
                    _log.warning("planned_search answer add source failed", exc_info=True)
        except Exception:
            _log.warning("planned_search answer message create failed", exc_info=True)

    try:
        client._patch("messages", message_id, {
            "pending_approval": 0,
            "search_used": 1,
            "search_source_count": len(scraped_results),
            "search_status": "completed",
        })
    except Exception:
        _log.warning("planned_search patch message failed", exc_info=True)

    return {
        "status": "ok",
        "message_id": message_id,
        "answer_message_id": answer_msg_id,
        "queries_executed": len(query_list),
        "results_found": len(results),
        "successful_scrapes": len(scraped_results),
        "answer_chars": len(answer_text),
    }


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

    try:
        raw, tokens = await asyncio.to_thread(model_call, "planned_search_answer", prompt, True)
        return (raw or "").strip(), tokens or 0
    except Exception:
        _log.warning("planned_search synthesis failed", exc_info=True)
        return "", 0


def reject_searches(message_id: int) -> dict:
    client = NocodbClient()

    try:
        client._patch("messages", message_id, {
            "pending_approval": 0,
            "search_status": "rejected",
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
        sources = client.list_message_search_sources(message_id=message_id)
        return {
            "status": "ok",
            "sources": sources,
        }
    except Exception:
        return {"status": "error", "message": "Failed to get sources"}