from __future__ import annotations

import logging
import re
import time

import httpx

from infra.config import get_function_config, no_think_params
from tools.contract import ToolPlan
from shared.model_pool import acquire_role

_log = logging.getLogger("tools.planner")


SYSTEM_PROMPT = """Output ONLY valid JSON. No markdown, no prose.
Tools: web_search(queries:[str]), rag_lookup(query:str)
web_search: quick inline search, 4-5 DIVERSE queries targeting different aspects. Results in seconds. Use for direct user questions needing current info.
rag_lookup: only when user references prior conversations.
Max 4 actions. "summary": one sentence shown to user.


User: What's the latest RBA rate decision?
{"actions":[{"tool":"web_search","params":{"queries":["RBA cash rate decision latest 2025","Australian interest rate announcement","RBA board statement inflation","RBA monetary policy update","Australia cash rate today"]},"reason":"current policy"}],"summary":"Checking the latest RBA rate decision..."}

User: What did we discuss about auth?
{"actions":[{"tool":"rag_lookup","params":{"query":"auth migration discussion"},"reason":"prior context"}],"summary":"Searching our previous discussions..."}

User: thanks
{"actions":[],"summary":""}"""


async def generate_plan(
    user_message: str,
    hints: set[str],
    conversation_summary: str = "",
) -> ToolPlan | None:
    # fail-open: never raises; caller proceeds toolless on None
    cfg = get_function_config("tool_planner")

    user_prompt_parts: list[str] = []
    if conversation_summary:
        user_prompt_parts.append(f"Conversation context: {conversation_summary}")
    if hints:
        user_prompt_parts.append(f"Hinted tools: {', '.join(sorted(hints))}")
    user_prompt_parts.append(f"User: {user_message}")

    t0 = time.time()
    try:
        with acquire_role(cfg["role"], priority=True) as (tool_url, tool_model_id):
            if not tool_url:
                _log.warning("no tool model available — skipping plan")
                return None
            _log.info("planner call  model=%s url=%s", tool_model_id, tool_url)
            async with httpx.AsyncClient(timeout=3600.0) as client:
                resp = await client.post(
                    f"{tool_url}/v1/chat/completions",
                    json={
                        "model": tool_model_id,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": "\n".join(user_prompt_parts)},
                        ],
                        "temperature": cfg.get("temperature", 0.1),
                        "max_tokens": cfg.get("max_tokens", 200),
                        **no_think_params(),
                    },
                )
                resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        _log.info("planner response  model=%s chars=%d elapsed=%.2fs", tool_model_id, len(raw), time.time() - t0)
    except Exception:
        _log.warning("planner call failed", exc_info=True)
        return None

    cleaned = raw
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = cleaned.rstrip("`").strip()
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if not json_match:
        _log.warning("planner: no JSON in response: %s", raw[:200])
        return None

    try:
        plan = ToolPlan.model_validate_json(json_match.group(0))
    except Exception:
        _log.warning("planner: JSON validation failed: %s", raw[:200])
        return None

    elapsed = round(time.time() - t0, 2)
    _log.info(
        "plan generated actions=%d tools=%s elapsed=%ss",
        len(plan.actions),
        [a.tool.value for a in plan.actions],
        elapsed,
    )

    if not plan.actions:
        return None
    return plan
