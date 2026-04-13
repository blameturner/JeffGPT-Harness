"""
Deep search executor — LLM-powered queued search with approval flow.

Unlike normal web search (heuristic queries, RWKV summarise, inline),
deep search uses:
  1. T3 model to generate smarter, more targeted search queries
  2. User reviews and approves the queries before execution
  3. T1 secondary to summarise each scraped page
  4. All work runs through the tool job queue (background)
  5. Results stored to ChromaDB + FalkorDB, delivered back to conversation
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time

from tools.framework.contract import ToolName, ToolResult
from tools.framework.dispatcher import register_executor
from tools.framework.executors.web_search import _search_all
from workers.enrichment.models import model_call

_log = logging.getLogger("tools.deep_search")

_QUERY_GENERATION_PROMPT = """You are a search query strategist. Given a user's question and optional conversation context, generate 6-10 precise search queries that will find high-quality, authoritative answers.

Rules:
- Each query should target a DIFFERENT angle or aspect of the question
- Use specific technical terms, not vague rephrasing
- Include at least one query with quoted phrases for exact matching
- Include at least one query targeting recent/current information (add year or "latest")
- Avoid generic queries that would match millions of irrelevant pages
- Think about what authoritative sources would title their articles

Return ONLY a JSON array of query strings. No prose, no markdown fences.
First character must be `[`, last must be `]`.

USER QUESTION:
{question}

{context_section}"""


def _generate_queries_with_llm(message: str, conversation_topics: list[str] | None = None) -> list[str]:
    """Use T3 to generate smart search queries."""
    context_section = ""
    if conversation_topics:
        context_section = f"CONVERSATION CONTEXT (topics discussed so far):\n{', '.join(conversation_topics[:10])}"

    prompt = _QUERY_GENERATION_PROMPT.format(
        question=message[:1500],
        context_section=context_section,
    )

    raw, tokens = model_call("deep_search_queries", prompt)
    _log.info("deep_search query generation  tokens=%d", tokens)

    if not raw:
        _log.warning("deep_search query generation returned empty")
        return []

    try:
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip()).strip()
        queries = json.loads(cleaned)
    except json.JSONDecodeError:
        _log.warning("deep_search query generation unparseable: %s", raw[:200])
        return []

    if not isinstance(queries, list):
        return []

    result = [str(q).strip() for q in queries if str(q).strip()][:10]
    _log.info("deep_search generated %d queries: %s", len(result), result)
    return result


@register_executor(ToolName.DEEP_SEARCH)
async def execute(params: dict, emit) -> ToolResult:
    """
    Deep search has two phases:

    Phase 1 (plan): T3 generates queries → SearXNG finds URLs → return plan
                    for user approval. Queries + URLs stored on conversation.

    Phase 2 (execute): After approval, queue all URLs into the tool queue
                       for T1 secondary summarisation.
    """
    phase = params.get("_phase", "plan")
    org_id = params.get("_org_id") or 0
    conversation_id = params.get("_conversation_id")

    if not org_id:
        return ToolResult(
            tool=ToolName.DEEP_SEARCH, action_index=0, ok=False,
            data="deep_search missing org context",
        )

    if phase == "execute":
        return await _execute_approved_plan(params, emit)

    # ---- Phase 1: Generate plan for approval ----
    user_message = params.get("_user_message") or ""
    conversation_topics = params.get("_conversation_topics") or []

    if not user_message:
        return ToolResult(
            tool=ToolName.DEEP_SEARCH, action_index=0, ok=False,
            data="No message provided for deep search.",
        )

    emit({"type": "tool_status", "phase": "planning", "message": "Generating search strategy..."})

    # T3 generates queries
    queries = await asyncio.get_event_loop().run_in_executor(
        None, _generate_queries_with_llm, user_message, conversation_topics,
    )

    if not queries:
        return ToolResult(
            tool=ToolName.DEEP_SEARCH, action_index=0, ok=False,
            data="Failed to generate search queries.",
        )

    emit({"type": "searching", "queries": queries, "mode": "deep"})
    t0 = time.time()

    # SearXNG in parallel to find candidate URLs
    results = await _search_all(queries)
    _log.info("deep_search searxng  queries=%d urls=%d", len(queries), len(results))

    elapsed = round(time.time() - t0, 2)

    if not results:
        emit({"type": "search_complete", "source_count": 0, "ok": False,
              "confidence": "failed", "sources": []})
        return ToolResult(
            tool=ToolName.DEEP_SEARCH, action_index=0, ok=False,
            data="No search results from any query.",
            elapsed_s=elapsed,
        )

    # Build the plan for approval
    plan = {
        "queries": queries,
        "urls": [{"url": r["url"], "title": r.get("title", ""), "snippet": r.get("snippet", "")} for r in results],
    }

    # Store on conversation for retrieval after approval
    if conversation_id:
        try:
            from nocodb_client import NocodbClient
            db = NocodbClient()
            db.update_conversation(int(conversation_id), {
                "pending_deep_search": json.dumps(plan),
            })
            _log.info("deep_search plan stored  conv=%s queries=%d urls=%d", conversation_id, len(queries), len(results))
        except Exception:
            _log.error("deep_search plan storage failed  conv=%s", conversation_id, exc_info=True)

    # Build plan summary for the chat model
    query_list = "\n".join(f"- {q}" for q in queries)
    source_list = "\n".join(f"- {r.get('title') or r['url'][:60]}" for r in results[:10])
    plan_summary = (
        f"I've prepared a deep search plan with {len(queries)} targeted queries "
        f"and found {len(results)} sources to analyse.\n\n"
        f"**Search queries:**\n{query_list}\n\n"
        f"**Sources to analyse ({len(results)}):**\n{source_list}\n\n"
        f"Please review and approve this plan, or suggest changes to the queries."
    )

    emit({
        "type": "search_complete",
        "source_count": len(results),
        "ok": True,
        "confidence": "awaiting_approval",
        "sources": [{"url": r["url"], "title": r.get("title", "")} for r in results[:10]],
    })

    return ToolResult(
        tool=ToolName.DEEP_SEARCH, action_index=0, ok=True,
        data=plan_summary,
        elapsed_s=elapsed,
    )


async def _execute_approved_plan(params: dict, emit) -> ToolResult:
    """Phase 2: Execute the approved plan by queuing all URLs."""
    org_id = int(params.get("_org_id") or 0)
    conversation_id = params.get("_conversation_id")
    plan = params.get("_plan") or {}

    queries = plan.get("queries", [])
    urls = plan.get("urls", [])

    if not urls:
        return ToolResult(
            tool=ToolName.DEEP_SEARCH, action_index=0, ok=False,
            data="No URLs in the approved plan.",
        )

    from workers.tool_queue import get_tool_queue
    tq = get_tool_queue()
    if not tq:
        return ToolResult(
            tool=ToolName.DEEP_SEARCH, action_index=0, ok=False,
            data="Tool job queue not available.",
        )

    t0 = time.time()
    job_ids: list[str] = []
    for r in urls:
        ids = tq.submit_pipeline(
            url=r["url"],
            org_id=org_id,
            collection="web_search",
            source="deep_search",
            priority=2,
            metadata={
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "queries": queries,
                "conversation_id": conversation_id,
            },
            summarise_function="deep_search_summarise",
        )
        job_ids.extend(ids)

    elapsed = round(time.time() - t0, 2)
    _log.info("deep_search executing  urls=%d jobs=%d elapsed=%.2fs", len(urls), len(job_ids), elapsed)

    # Clear the pending plan
    if conversation_id:
        try:
            from nocodb_client import NocodbClient
            db = NocodbClient()
            db.update_conversation(int(conversation_id), {
                "pending_deep_search": "",
            })
        except Exception:
            _log.warning("failed to clear pending_deep_search  conv=%s", conversation_id, exc_info=True)

    emit({
        "type": "jobs_queued",
        "tool": "deep_search",
        "message": f"Queued {len(urls)} sources for deep analysis.",
        "status": "running",
    })

    return ToolResult(
        tool=ToolName.DEEP_SEARCH, action_index=0, ok=True,
        data=f"Deep search approved and running. {len(urls)} sources queued for thorough analysis. "
             f"Results will be delivered to this conversation as they complete.",
        elapsed_s=elapsed,
    )