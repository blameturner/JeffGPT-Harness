"""
Tool dispatcher — routes ToolPlan actions to registered executors and runs
them concurrently with asyncio.gather.

Extensibility — to add a new tool:
  1. Add an enum value to tools.framework.contract.ToolName.
  2. Create tools/framework/executors/<name>.py with an async `execute(params, emit)`
     decorated with @register_executor(ToolName.<NAME>).
  3. Import the new module in the "trigger executor registration" block at the
     bottom of this file.

The dispatcher itself never changes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

from tools.framework.contract import (
    ToolAction,
    ToolContext,
    ToolName,
    ToolPlan,
    ToolResult,
)

_log = logging.getLogger("tools.dispatcher")

Emit = Callable[[dict], None]
Executor = Callable[[dict, Emit], Awaitable[ToolResult]]

EXECUTORS: dict[ToolName, Executor] = {}


def register_executor(tool: ToolName):
    """Decorator — registers an async executor for a ToolName."""

    def decorator(fn: Executor) -> Executor:
        EXECUTORS[tool] = fn
        return fn

    return decorator


async def execute_plan(
    plan: ToolPlan,
    emit: Emit | None = None,
) -> ToolContext:
    """
    Execute every action in the plan concurrently.

    Individual executor failures do not fail the whole plan — they return a
    non-OK ToolResult and the main model sees the failure in its context.
    """
    _emit: Emit = emit or (lambda _e: None)

    if not plan.actions:
        return ToolContext(plan_summary=plan.summary, results=[])

    async def run_one(index: int, action: ToolAction) -> ToolResult:
        executor = EXECUTORS.get(action.tool)
        if executor is None:
            _log.warning("no executor registered for %s", action.tool.value)
            return ToolResult(
                tool=action.tool, action_index=index, ok=False,
                data=f"no executor registered for {action.tool.value}",
            )

        _emit({
            "type": "tool_status",
            "phase": "start",
            "tool": action.tool.value,
            "index": index,
            "reason": action.reason,
        })

        t0 = time.time()
        try:
            result = await executor(action.params, _emit)
            # Override whatever the executor set — dispatcher owns these.
            result.action_index = index
            result.elapsed_s = round(time.time() - t0, 2)
        except Exception as e:
            _log.error("executor %s failed", action.tool.value, exc_info=True)
            result = ToolResult(
                tool=action.tool, action_index=index, ok=False,
                data=str(e), elapsed_s=round(time.time() - t0, 2),
            )

        _emit({
            "type": "tool_status",
            "phase": "end",
            "tool": action.tool.value,
            "index": index,
            "ok": result.ok,
            "elapsed_s": result.elapsed_s,
        })
        return result

    results = await asyncio.gather(
        *[run_one(i, a) for i, a in enumerate(plan.actions)]
    )
    return ToolContext(plan_summary=plan.summary, results=list(results))


# --- Trigger executor registration (import side-effect) ---
# Adding a new tool? Add one more import on the line below and you're done.
from tools.framework.executors import (  # noqa: E402, F401
    code_exec,
    rag_lookup,
    web_search,
)
