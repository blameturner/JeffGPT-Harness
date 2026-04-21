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

# Chat-initiated tool work must keep the tool_queue's chat-idle gate alive for
# its full duration, or background jobs slip in mid-tool. Touch at the start of
# every action, and run a low-frequency heartbeat during the await so long
# searches/scrapes stay covered even if the tool itself forgets to touch.
# 10s keeps us well inside the default 30s gate; raise/lower in lockstep if you
# change `features.tool_queue.background_chat_idle_seconds`.
_CHAT_HEARTBEAT_S = 10.0


async def _chat_activity_heartbeat():
    from workers.tool_queue import touch_chat_activity
    try:
        while True:
            touch_chat_activity()
            await asyncio.sleep(_CHAT_HEARTBEAT_S)
    except asyncio.CancelledError:
        pass


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

        # Prime the chat-idle gate before awaiting the tool and keep it fresh
        # throughout — prevents background tool_queue workers from claiming
        # jobs while this tool is in flight.
        from workers.tool_queue import touch_chat_activity
        touch_chat_activity()
        heartbeat = asyncio.create_task(_chat_activity_heartbeat())

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
        finally:
            heartbeat.cancel()
            # One final touch so the gate covers the window between tool end
            # and whatever the caller does next (usually a model_call).
            touch_chat_activity()

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
