"""
Tool planner — calls the shared tool model to generate a ToolPlan.

Only invoked when the heuristic gate (tools.framework.gate) flags at least
one possible tool. Uses strict few-shot prompting + temperature=0.1 for
structured JSON output.

The planner is model-agnostic: URL + model id come from config.get_model_url("tool")
via workers.search.models._tool_model(). Swapping Qwen 3B for a more capable
planner later requires zero code changes — just flip the env var.
"""

from __future__ import annotations

import logging
import re
import time

import httpx

from config import no_think_params
from tools.framework.contract import ToolPlan
from workers.enrichment.models import _assert_not_reasoner
from workers.search.models import acquire_model

_log = logging.getLogger("tools.planner")


SYSTEM_PROMPT = """You are a tool planner. Given a user message, output a JSON ToolPlan.
Output ONLY valid JSON. No markdown, no backticks, no prose.

Tools:
- web_search: {"queries": ["q1","q2","q3"]}
  2-3 DIVERSE queries. Each query must target a different aspect.
  BAD (near-duplicates): ["RBA rate decision","RBA interest rate decision","RBA rate update"]
  GOOD (diverse angles): ["RBA cash rate decision April 2026","Australian mortgage rate forecast 2026","RBA governor statement inflation"]
  Never copy the user message verbatim.
- rag_lookup: {"query": "semantic terms"}
  Only when the user references prior conversations, "we discussed", "you mentioned", etc.
- code_exec: {"language":"python","code":"..."}
  ONLY when the user wants code to actually RUN and see output.
  "write me a script" / "show me code" / "how would I" = NO tool (main model writes the code).
  "run this" / "execute" / "calculate" / "what's the result of" = USE code_exec.

Rules:
- Maximum 4 actions total.
- "summary" is one sentence, conversational, shown to the user while tools run.
- If no tools are needed, return {"actions":[],"summary":""}.

Examples:

User: What's the latest RBA interest rate decision?
{"actions":[{"tool":"web_search","params":{"queries":["RBA cash rate decision latest","Reserve Bank Australia interest rate announcement","RBA board statement inflation"]},"reason":"current monetary policy + context"}],"summary":"Checking the latest RBA rate decision..."}

User: What happened in the NSW election?
{"actions":[{"tool":"web_search","params":{"queries":["NSW state election results","NSW election seat changes","NSW premier reaction to election"]},"reason":"results plus reactions"}],"summary":"Looking up the NSW election results..."}

User: Run a python script that prints fibonacci up to 100
{"actions":[{"tool":"code_exec","params":{"language":"python","code":"a,b=0,1\\nwhile a<=100:\\n    print(a)\\n    a,b=b,a+b"},"reason":"user asked to run code"}],"summary":"Running your Fibonacci script..."}

User: Write me a Python function that calculates fibonacci
{"actions":[],"summary":""}

User: What did we discuss about the Prodigi auth migration?
{"actions":[{"tool":"rag_lookup","params":{"query":"Prodigi auth migration"},"reason":"retrieve prior context"}],"summary":"Searching our previous discussions..."}

User: How do I use asyncio.gather with a semaphore in Python?
{"actions":[{"tool":"web_search","params":{"queries":["asyncio.gather semaphore pattern python","asyncio bounded concurrency example","python asyncio Semaphore with gather"]},"reason":"API usage lookup"}],"summary":"Looking up the asyncio.gather + semaphore pattern..."}

User: TypeError: 'NoneType' object is not subscriptable — in get_user()
{"actions":[{"tool":"rag_lookup","params":{"query":"get_user None subscriptable TypeError"},"reason":"check prior context on this function"},{"tool":"web_search","params":{"queries":["python TypeError NoneType object is not subscriptable common cause","python None return value unpacking"]},"reason":"diagnose error class"}],"summary":"Checking past discussion and common causes..."}

User: Compare the latest iPhone vs Samsung Galaxy pricing in Australia
{"actions":[{"tool":"web_search","params":{"queries":["iPhone latest price Australia 2026","Samsung Galaxy latest price Australia 2026","iPhone vs Samsung comparison review"]},"reason":"current pricing + reviews"}],"summary":"Checking current Australian pricing on both phones..."}

User: thanks
{"actions":[],"summary":""}

User: explain that again
{"actions":[],"summary":""}"""


async def generate_plan(
    user_message: str,
    hints: set[str],
    conversation_summary: str = "",
) -> ToolPlan | None:
    """
    Call the tool model to produce a ToolPlan.

    Returns None if the planner fails, times out, or decides no tools are needed.
    Fail-open — never raises. Caller proceeds without tools on None.
    """
    user_prompt_parts: list[str] = []
    if conversation_summary:
        user_prompt_parts.append(f"Conversation context: {conversation_summary}")
    if hints:
        user_prompt_parts.append(f"Hinted tools: {', '.join(sorted(hints))}")
    user_prompt_parts.append(f"User: {user_message}")

    t0 = time.time()
    try:
        with acquire_model("tool") as (tool_url, tool_model_id):
            if not tool_url:
                _log.warning("no tool model available — skipping plan")
                return None
            _assert_not_reasoner(tool_url)
            _log.info("planner call  model=%s url=%s", tool_model_id, tool_url)
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(
                    f"{tool_url}/v1/chat/completions",
                    json={
                        "model": tool_model_id,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": "\n".join(user_prompt_parts)},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 500,
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
