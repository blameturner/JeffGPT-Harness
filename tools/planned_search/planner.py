from __future__ import annotations

import logging
import re
import time

import httpx

from infra.config import get_function_config, no_think_params
from shared.model_pool import acquire_role

_log = logging.getLogger("planned_search.planner")

SYSTEM_PROMPT = """Output ONLY valid JSON. No markdown, no prose.
Generate diverse search queries to answer the user's question. Return an array of query objects with 'query' and 'reason' fields.
Queries should target different aspects of the question for comprehensive coverage.

User: What's the latest RBA rate decision?
{"queries":[{"query":"RBA cash rate decision 2025","reason":"find current official rate"},{"query":"Australian interest rate announcement April 2025","reason":"check recent announcements"},{"query":"RBA monetary policy statement inflation","reason":"understand policy rationale"},{"query":"Australia cash rate target April","reason":"verify current rate"},{"query":"RBA board meeting outcomes","reason":"find board decision details"}]}

User: {question}
{"queries":[{"query":"...","reason":"..."}]}"""


async def generate_planned_queries(
    user_question: str,
    conversation_summary: str = "",
) -> list[dict]:
    cfg = get_function_config("planned_search")

    user_prompt = f"User question: {user_question}"
    if conversation_summary:
        user_prompt = f"Conversation context: {conversation_summary}\n\n{user_prompt}"

    t0 = time.time()
    try:
        with acquire_role(cfg["role"], priority=True) as (tool_url, tool_model_id):
            if not tool_url:
                _log.warning("no planned_search model available")
                return []
            _log.info("planned_search planner call  model=%s", tool_model_id)
            async with httpx.AsyncClient(timeout=3600.0) as client:
                resp = await client.post(
                    f"{tool_url}/v1/chat/completions",
                    json={
                        "model": tool_model_id,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": cfg.get("temperature", 0.1),
                        "max_tokens": cfg.get("max_tokens", 500),
                        **no_think_params(),
                    },
                )
                resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        _log.info("planned_search planner response  chars=%d elapsed=%.2fs", len(raw), time.time() - t0)
    except Exception:
        _log.warning("planned_search planner call failed", exc_info=True)
        return []

    cleaned = raw
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = cleaned.rstrip("`").strip()
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if json_match:
        cleaned = json_match.group()

    import json
    try:
        data = json.loads(cleaned)
        queries = data.get("queries", [])
        return [{"query": q.get("query", ""), "reason": q.get("reason", "")} for q in queries if q.get("query")]
    except Exception:
        _log.warning("planned_search parse failed  raw=%s", raw[:200])
        return []