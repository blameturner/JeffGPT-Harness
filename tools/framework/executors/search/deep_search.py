"""Deep search executor — LLM-powered queued search with approval flow.

Two phases:
  1. T3 generates targeted queries → SearXNG finds URLs → plan returned
     for user approval.
  2. After approval, background job: scrape → summarise → synthesise → deliver.
"""

from __future__ import annotations

import asyncio
import logging
import time

from tools.framework.contract import ToolName, ToolResult
from tools.framework.dispatcher import register_executor
from tools.framework.executors.search.pipeline import (
    parse_llm_json,
    search_and_dedup,
    store_pending_plan,
    clear_pending_plan,
)
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


def _generate_queries(message: str, conversation_topics: list[str] | None = None) -> list[str]:
    """Use T3 to generate smart search queries."""
    context_section = ""
    if conversation_topics:
        context_section = f"CONVERSATION CONTEXT:\n{', '.join(conversation_topics[:10])}"

    prompt = _QUERY_GENERATION_PROMPT.format(
        question=message[:1500],
        context_section=context_section,
    )

    raw, tokens = model_call("deep_search_queries", prompt)
    _log.info("query generation  tokens=%d", tokens)

    if not raw:
        _log.warning("query generation returned empty")
        return []

    parsed = parse_llm_json(raw)
    if not isinstance(parsed, list):
        return []

    result = [str(q).strip() for q in parsed if str(q).strip()][:10]
    _log.info("generated %d queries: %s", len(result), result)
    return result


@register_executor(ToolName.DEEP_SEARCH)
async def execute(params: dict, emit) -> ToolResult:
    """Phase 1: generate plan. Phase 2: submit background job."""
    phase = params.get("_phase", "plan")
    org_id = params.get("_org_id") or 0
    conversation_id = params.get("_conversation_id")

    if not org_id:
        return ToolResult(tool=ToolName.DEEP_SEARCH, action_index=0, ok=False,
                          data="deep_search missing org context")

    if phase == "execute":
        return await _execute_approved(params, emit)

    # ---- Phase 1: Generate plan for approval ----
    user_message = params.get("_user_message") or ""
    conversation_topics = params.get("_conversation_topics") or []

    if not user_message:
        return ToolResult(tool=ToolName.DEEP_SEARCH, action_index=0, ok=False,
                          data="No message provided for deep search.")

    emit({"type": "tool_status", "phase": "planning", "message": "Generating search strategy..."})

    queries = await asyncio.get_running_loop().run_in_executor(
        None, _generate_queries, user_message, conversation_topics,
    )
    if not queries:
        return ToolResult(tool=ToolName.DEEP_SEARCH, action_index=0, ok=False,
                          data="Failed to generate search queries.")

    emit({"type": "searching", "queries": queries, "mode": "deep"})
    t0 = time.time()

    results = await search_and_dedup(queries, max_results=25)
    elapsed = round(time.time() - t0, 2)
    _log.info("searxng  queries=%d results=%d", len(queries), len(results))

    if not results:
        return ToolResult(tool=ToolName.DEEP_SEARCH, action_index=0, ok=False,
                          data="No search results from any query.", elapsed_s=elapsed)

    plan = {
        "question": user_message,
        "queries": queries,
        "urls": [{"url": r["url"], "title": r.get("title", ""), "snippet": r.get("snippet", "")} for r in results],
    }

    if conversation_id:
        try:
            store_pending_plan(conversation_id, "deep_search", plan)
            _log.info("plan stored  conv=%s queries=%d urls=%d", conversation_id, len(queries), len(results))
        except Exception:
            _log.error("plan storage failed  conv=%s", conversation_id, exc_info=True)

    query_list = "\n".join(f"- {q}" for q in queries)
    source_list = "\n".join(f"- {r.get('title') or r['url'][:60]}" for r in results[:10])
    plan_summary = (
        f"I've prepared a deep search plan with {len(queries)} targeted queries "
        f"and found {len(results)} sources to analyse.\n\n"
        f"**Search queries:**\n{query_list}\n\n"
        f"**Sources to analyse ({len(results)}):**\n{source_list}\n\n"
        f"Please review and approve this plan, or suggest changes to the queries."
    )

    return ToolResult(tool=ToolName.DEEP_SEARCH, action_index=0, ok=True,
                      data=plan_summary, elapsed_s=elapsed)


async def _execute_approved(params: dict, emit) -> ToolResult:
    """Phase 2: Submit background job to tool queue."""
    org_id = int(params.get("_org_id") or 0)
    conversation_id = params.get("_conversation_id")
    plan = params.get("_plan") or {}

    if not plan.get("urls"):
        return ToolResult(tool=ToolName.DEEP_SEARCH, action_index=0, ok=False,
                          data="No URLs in the approved plan.")

    from workers.tool_queue import get_tool_queue
    tq = get_tool_queue()
    if not tq:
        return ToolResult(tool=ToolName.DEEP_SEARCH, action_index=0, ok=False,
                          data="Tool job queue not available.")

    job_id = tq.submit(
        job_type="deep_search",
        payload={"plan": plan, "org_id": org_id, "conversation_id": conversation_id},
        source="deep_search", org_id=org_id, priority=2,
    )
    _log.info("job queued  conv=%s job=%s urls=%d", conversation_id, job_id, len(plan["urls"]))

    if conversation_id:
        try:
            clear_pending_plan(conversation_id, "deep_search")
        except Exception:
            _log.warning("failed to clear pending plan  conv=%s", conversation_id, exc_info=True)

    msg = f"Deep search queued with {len(plan['urls'])} sources. A synthesised response will be delivered when analysis completes."
    emit({"type": "jobs_queued", "tool": "deep_search", "message": msg, "status": "running"})

    return ToolResult(tool=ToolName.DEEP_SEARCH, action_index=0, ok=True, data=msg)