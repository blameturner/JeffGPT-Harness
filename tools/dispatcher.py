from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

from tools.contract import (
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
    def decorator(fn: Executor) -> Executor:
        EXECUTORS[tool] = fn
        return fn

    return decorator


async def execute_plan(
    plan: ToolPlan,
    emit: Emit | None = None,
) -> ToolContext:
    # per-action failures return a non-OK ToolResult; the whole plan never fails
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
            # dispatcher owns index/elapsed; overwrite whatever the executor set
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


# import for @register_executor side-effects — do not remove
from tools import rag_lookup  # noqa: E402, F401
from tools.search import web_search  # noqa: E402, F401
from tools.url_viewer import url_scraper as _url_scraper  # noqa: E402, F401
