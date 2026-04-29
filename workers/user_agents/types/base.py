"""Shared types and the single tool loop used by every agent type.

The tool loop:
  1. compose messages (system + user)
  2. call model
  3. if response contains tool calls (JSON block), execute via tool_queue
  4. append observations and loop, up to budgets.max_iterations
  5. return final text + token totals + event log
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import requests

from infra.nocodb_client import NocodbClient

_log = logging.getLogger("agents.types.base")

TOOL_CALL_BLOCK_RE = re.compile(r"```tool\s*(\{.*?\})\s*```", re.DOTALL)
JSON_BLOCK_RE = re.compile(r"\{[\s\S]*?\}")


@dataclass
class Budgets:
    max_iterations: int = 5
    max_runtime_seconds: int = 300
    max_tokens_per_run: int = 0  # 0 = unbounded
    started_at: float = field(default_factory=time.time)

    def time_left(self) -> float:
        return self.max_runtime_seconds - (time.time() - self.started_at)


@dataclass
class RunContext:
    agent: dict
    assignment: dict
    db: NocodbClient
    budgets: Budgets
    events: list[dict] = field(default_factory=list)
    forbidden_tables: set[str] = field(default_factory=set)
    dry_run: bool = False
    test_mode: bool = False
    worker_id: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    iterations: int = 0

    def log(self, kind: str, **fields):
        evt = {"t": datetime.now(timezone.utc).isoformat(), "kind": kind, **fields}
        self.events.append(evt)


@dataclass
class RunResult:
    output: str
    refs: dict = field(default_factory=dict)
    summary: str = ""


# ---------- model call ----------

def call_model(ctx: RunContext, messages: list[dict]) -> tuple[str, dict]:
    """Synchronous model call honoring budgets; returns (text, usage)."""
    agent = ctx.agent
    from infra.config import get_model_url
    model_key = (agent.get("model") or "").lower()
    url = get_model_url(model_key) or ""
    if not url:
        raise RuntimeError(f"no URL for model: {model_key}")

    body = {
        "model": agent.get("model"),
        "messages": messages,
        "temperature": agent.get("temperature", 0.4),
        "max_tokens": agent.get("max_tokens", 1500),
    }
    t0 = time.time()
    r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=600)
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {}) or {}
    ctx.tokens_in += int(usage.get("prompt_tokens") or 0)
    ctx.tokens_out += int(usage.get("completion_tokens") or 0)
    ctx.log(
        "llm_call",
        model=agent.get("model"),
        tokens_in=usage.get("prompt_tokens"),
        tokens_out=usage.get("completion_tokens"),
        ms=int((time.time() - t0) * 1000),
    )
    return text, usage


# ---------- tool calls ----------

def parse_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from model output. Format: ```tool {"name":..,"args":..}```"""
    calls: list[dict] = []
    for m in TOOL_CALL_BLOCK_RE.finditer(text):
        try:
            calls.append(json.loads(m.group(1)))
        except Exception:
            continue
    return calls


def run_tool_calls(ctx: RunContext, calls: list[dict]) -> list[dict]:
    """Dispatch via tool_queue if available, else direct synchronous call.

    For the MVP we call known tool entry points directly (web_search, rag_lookup,
    url_scraper). HTTP_REQUEST and SEND_EMAIL slot in here when implemented.
    """
    out: list[dict] = []
    allowed = set([t.strip() for t in (ctx.agent.get("allowed_tools") or "").split(",") if t.strip()])
    for call in calls:
        name = call.get("name") or call.get("tool")
        args = call.get("args") or call.get("params") or {}
        if allowed and name not in allowed:
            out.append({"name": name, "ok": False, "error": "tool not allowed"})
            ctx.log("tool_denied", name=name)
            continue
        t0 = time.time()
        try:
            result = _direct_tool_call(name, args, ctx)
            out.append({"name": name, "ok": True, "result": result})
            ctx.log("tool_ok", name=name, ms=int((time.time() - t0) * 1000))
        except Exception as e:
            out.append({"name": name, "ok": False, "error": f"{type(e).__name__}: {e}"})
            ctx.log("tool_err", name=name, error=str(e)[:200])
    return out


def _direct_tool_call(name: str, args: dict, ctx: RunContext) -> Any:
    if name == "web_search":
        from tools.search.basic import basic_search  # type: ignore
        return basic_search(args.get("query", ""), args.get("max_results", 5))
    if name == "rag_lookup":
        from infra.rag import retrieve
        return retrieve(
            query=args.get("query", ""),
            org_id=int(ctx.agent.get("org_id") or 1),
            collection_name=args.get("collection") or ctx.agent.get("rag_collection", "agent_outputs"),
            n_results=args.get("n_results", 8),
            top_k=args.get("top_k", 3),
        )
    if name == "url_scraper":
        from tools.url_viewer import view_url  # type: ignore
        return view_url(args.get("url", ""))
    if name == "http_request":
        return _http_request(args, ctx)
    if name == "nocodb_query":
        table = args.get("table")
        where = args.get("where", "")
        limit = int(args.get("limit", 10))
        if table not in ctx.db.tables:
            raise ValueError(f"unknown table: {table}")
        return ctx.db._get_paginated(table, params={"where": where, "limit": limit})
    raise NotImplementedError(f"tool not wired: {name}")


def _http_request(args: dict, ctx: RunContext) -> dict:
    """Minimal HTTP_REQUEST tool. Resolves api_connections by name."""
    import re as _re
    method = (args.get("method") or "GET").upper()
    path = args.get("path", "")
    conn_name = args.get("connection")
    headers: dict = dict(args.get("headers") or {})
    params = args.get("params") or {}
    body = args.get("body")

    base = ""
    if conn_name and "api_connections" in ctx.db.tables:
        rows = ctx.db._get_paginated("api_connections", params={
            "where": f"(name,eq,{conn_name})",
            "limit": 1,
        })
        if rows:
            conn = rows[0]
            base = (conn.get("base_url") or "").rstrip("/")
            allowed_methods = [m.strip().upper() for m in (conn.get("allowed_methods") or "GET").split(",")]
            if method not in allowed_methods:
                raise PermissionError(f"method {method} not allowed for {conn_name}")
            paths_re = conn.get("allowed_paths_regex")
            if paths_re and not _re.search(paths_re, path):
                raise PermissionError(f"path {path} not allowed for {conn_name}")
            try:
                headers.update(json.loads(conn.get("default_headers_json") or "{}"))
            except Exception:
                pass

    hosts_re = ctx.agent.get("allowed_outbound_hosts_regex")
    full_url = (base + path) if base else path
    if hosts_re and not _re.search(hosts_re, full_url):
        raise PermissionError(f"host blocked by allowed_outbound_hosts_regex")

    if ctx.dry_run:
        return {"dry_run": True, "method": method, "url": full_url}

    r = requests.request(method, full_url, headers=headers, params=params,
                         json=body if isinstance(body, (dict, list)) else None,
                         data=body if isinstance(body, (str, bytes)) else None,
                         timeout=30)
    return {
        "status": r.status_code,
        "headers": dict(r.headers),
        "body": r.text[:8000],
    }


# ---------- the loop ----------

TOOL_USAGE_HINT = (
    "When you need a tool, emit a fenced block exactly like:\n"
    "```tool\n{\"name\":\"web_search\",\"args\":{\"query\":\"...\"}}\n```\n"
    "Multiple blocks are run in parallel. After observations come back, continue or finish.\n"
    "When done, write the final answer with no tool blocks."
)


def tool_loop(ctx: RunContext, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt + "\n\n" + TOOL_USAGE_HINT},
        {"role": "user", "content": user_prompt},
    ]
    last_text = ""
    for i in range(ctx.budgets.max_iterations):
        ctx.iterations = i + 1
        if ctx.budgets.time_left() <= 0:
            ctx.log("budget_runtime_exceeded")
            break
        text, _usage = call_model(ctx, messages)
        last_text = text
        calls = parse_tool_calls(text)
        if not calls:
            return _strip_tool_blocks(text)
        results = run_tool_calls(ctx, calls)
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": json.dumps({"observations": results})[:20000]})
    return _strip_tool_blocks(last_text)


def _strip_tool_blocks(text: str) -> str:
    return TOOL_CALL_BLOCK_RE.sub("", text).strip()


# ---------- output validation + reflection ----------

def validate_json(text: str, schema: dict | None) -> tuple[bool, dict | None, str]:
    """Best-effort JSON extraction + schema-presence check (not full jsonschema)."""
    if not schema:
        return True, None, ""
    m = JSON_BLOCK_RE.search(text)
    if not m:
        return False, None, "no JSON object found"
    try:
        obj = json.loads(m.group(0))
    except Exception as e:
        return False, None, f"invalid JSON: {e}"
    required = schema.get("required") or []
    for r in required:
        if r not in obj:
            return False, obj, f"missing required key: {r}"
    return True, obj, ""


def reflect(ctx: RunContext, output: str, original_task: str) -> str:
    """Pass output through self-critique. Returns possibly-revised output."""
    crit_messages = [
        {"role": "system", "content": "You are a strict reviewer. Read the task and the draft. If the draft is good, reply with the draft unchanged. If not, output a revised draft. Output only the (possibly-revised) draft text."},
        {"role": "user", "content": f"TASK:\n{original_task}\n\nDRAFT:\n{output}"},
    ]
    try:
        revised, _ = call_model(ctx, crit_messages)
        ctx.log("reflect_ok", changed=(revised.strip() != output.strip()))
        return revised
    except Exception as e:
        ctx.log("reflect_err", error=str(e)[:200])
        return output
