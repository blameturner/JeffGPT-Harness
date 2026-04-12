"""
Code execution executor — HTTP client for the mst-ag-sandbox service.

The sandbox runs as a separate container and exposes POST /exec. Same
architectural pattern as the browser service: stateless HTTP call, resource
limits enforced by Docker, no host filesystem access.

Params:
  language: "python" or "bash" (default "python")
  code: str — the code to execute
  timeout: int — per-execution timeout in seconds (default 30, max 120)
  stdin: str — optional stdin to pipe in
"""

from __future__ import annotations

import logging
import time

import httpx

from config import SANDBOX_URL
from tools.framework.contract import ToolName, ToolResult
from tools.framework.dispatcher import register_executor

_log = logging.getLogger("tools.code_exec")

_OUTPUT_CAP_FOR_MODEL = 6000  # chars injected into main-model context


@register_executor(ToolName.CODE_EXEC)
async def execute(params: dict, emit) -> ToolResult:
    language = str(params.get("language") or "python").lower()
    code = str(params.get("code") or "").strip()
    timeout = int(params.get("timeout") or 30)
    stdin = str(params.get("stdin") or "")

    if language not in ("python", "bash"):
        return ToolResult(
            tool=ToolName.CODE_EXEC, action_index=0, ok=False,
            data=f"Unsupported language: {language!r} (expected python or bash)",
        )

    if not code:
        return ToolResult(
            tool=ToolName.CODE_EXEC, action_index=0, ok=False,
            data="No code provided",
        )

    if not SANDBOX_URL:
        return ToolResult(
            tool=ToolName.CODE_EXEC, action_index=0, ok=False,
            data="SVC_SANDBOX_URL not configured",
        )

    t0 = time.time()
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{SANDBOX_URL}/exec",
                json={
                    "language": language,
                    "code": code,
                    "timeout": timeout,
                    "stdin": stdin,
                },
                timeout=float(timeout + 10),
            )
            resp.raise_for_status()
            result = resp.json()
    except Exception as e:
        _log.warning("sandbox call failed: %s", e)
        return ToolResult(
            tool=ToolName.CODE_EXEC, action_index=0, ok=False,
            data=f"Sandbox service error: {e}",
            elapsed_s=round(time.time() - t0, 2),
        )

    stdout = str(result.get("stdout") or "")
    stderr = str(result.get("stderr") or "")
    exit_code = int(result.get("exit_code") or 0)
    timed_out = bool(result.get("timed_out"))
    err = result.get("error")

    ok = (exit_code == 0) and (not timed_out) and (not err)

    parts: list[str] = [f"language: {language}", f"exit_code: {exit_code}"]
    if timed_out:
        parts.append(f"⚠ timed out after {timeout}s")
    if err:
        parts.append(f"error: {err}")
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    if not stdout and not stderr and not err:
        parts.append("(no output)")

    data = "\n\n".join(parts)
    if len(data) > _OUTPUT_CAP_FOR_MODEL:
        data = data[:_OUTPUT_CAP_FOR_MODEL] + "\n\n…[truncated for model context]"

    _log.info(
        "code_exec  language=%s exit=%d timed_out=%s elapsed=%.2fs",
        language, exit_code, timed_out, time.time() - t0,
    )

    return ToolResult(
        tool=ToolName.CODE_EXEC, action_index=0, ok=ok,
        data=data, elapsed_s=round(time.time() - t0, 2),
    )
