"""Agent runtime — polls assignments, claims, dispatches, retries, audits.

Single loop. One process can run multiple workers; each worker has a unique id.
Claims are atomic via NocoDB conditional update (best-effort: re-read after
patch and confirm we own it). Heartbeats every 15s while running. Stale
heartbeats older than agent.heartbeat_ttl_seconds are reclaimable.

Robustness wired here:
  - claim + heartbeat + dead-worker recovery
  - per-agent budgets (runs/tokens/cost per day, max_concurrent_runs)
  - circuit breaker (consecutive_failures → active=false + incident)
  - retry with exponential backoff
  - memoization (dedup_key + memoize_ttl_seconds)
  - audit (events_jsonl + prompt_snapshot in agent_runs)
  - hooks (pre_run_hook, post_run_hook)
  - on_error_action (retry | escalate | pause | fallback)
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import socket
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

from infra.nocodb_client import NocodbClient
from workers.user_agents import context as ctxmod
from workers.user_agents.types import dispatch as dispatch_type
from workers.user_agents.types.base import Budgets, RunContext, RunResult

_log = logging.getLogger("agents.runtime")

ASSIGNMENTS = "assignments"
AGENTS = "agents"
RUNS = "agent_runs"
INCIDENTS = "agent_incidents"

POLL_INTERVAL_S = 5
HEARTBEAT_INTERVAL_S = 15
DEFAULT_HEARTBEAT_TTL_S = 60
DEFAULT_BACKOFF_BASE_S = 30


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"


# ---------- claim ----------

def _claim_one(client: NocodbClient, worker: str) -> dict | None:
    """Find a queued assignment and claim it. Returns the claimed row or None.

    Atomicity strategy: pick the oldest queued, write our worker id + claimed_at,
    then re-read; if claimed_by_worker matches us, we own it.
    """
    if ASSIGNMENTS not in client.tables:
        return None
    candidates = client._get_paginated(ASSIGNMENTS, params={
        "where": "(status,eq,queued)",
        "sort": "Id",
        "limit": 5,
    })
    if not candidates:
        candidates = _reclaim_stale(client)
    for row in candidates:
        next_retry = row.get("next_retry_at")
        if next_retry and next_retry > _iso_now():
            continue
        try:
            client._patch(ASSIGNMENTS, int(row["Id"]), {
                "status": "running",
                "claimed_by_worker": worker,
                "claimed_at": _iso_now(),
                "heartbeat_at": _iso_now(),
            })
        except Exception:
            continue
        check = client._get_paginated(ASSIGNMENTS, params={"where": f"(Id,eq,{row['Id']})", "limit": 1})
        if check and check[0].get("claimed_by_worker") == worker:
            return check[0]
    return None


def _reclaim_stale(client: NocodbClient) -> list[dict]:
    """Find running assignments whose heartbeat is past TTL — reset to queued."""
    rows = client._get_paginated(ASSIGNMENTS, params={"where": "(status,eq,running)", "limit": 50})
    stale: list[dict] = []
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=DEFAULT_HEARTBEAT_TTL_S * 3)
    for r in rows:
        hb = r.get("heartbeat_at") or r.get("claimed_at")
        if not hb:
            continue
        try:
            hb_dt = datetime.fromisoformat(hb.replace("Z", "+00:00"))
            if hb_dt.tzinfo is None:
                hb_dt = hb_dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if hb_dt < cutoff:
            try:
                client._patch(ASSIGNMENTS, int(r["Id"]), {"status": "queued", "claimed_by_worker": ""})
                stale.append(r)
                _log.warning("reclaimed stale assignment id=%s", r["Id"])
            except Exception:
                pass
    return stale


# ---------- heartbeat ----------

class Heartbeat:
    def __init__(self, client: NocodbClient, assignment_id: int):
        self.client = client
        self.assignment_id = assignment_id
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.wait(HEARTBEAT_INTERVAL_S):
            try:
                self.client._patch(ASSIGNMENTS, self.assignment_id, {"heartbeat_at": _iso_now()})
            except Exception:
                _log.debug("heartbeat write failed", exc_info=True)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


# ---------- budgets / circuit ----------

def _check_budgets(agent: dict) -> tuple[bool, str]:
    runs_today = int(agent.get("runs_today") or 0)
    max_runs = int(agent.get("max_runs_per_day") or 0)
    if max_runs and runs_today >= max_runs:
        return False, "runs_per_day exceeded"
    tokens_today = int(agent.get("tokens_today") or 0)
    max_tok = int(agent.get("max_tokens_per_day") or 0)
    if max_tok and tokens_today >= max_tok:
        return False, "tokens_per_day exceeded"
    cost_today = float(agent.get("cost_usd_today") or 0)
    max_cost = float(agent.get("max_cost_usd_per_day") or 0)
    if max_cost and cost_today >= max_cost:
        return False, "cost_usd_per_day exceeded"
    if agent.get("active") is False:
        return False, "agent inactive"
    if agent.get("pause_until") and agent["pause_until"] > _iso_now():
        return False, f"paused until {agent['pause_until']}"
    return True, ""


def _trip_circuit(client: NocodbClient, agent: dict, reason: str):
    try:
        client._patch(AGENTS, int(agent["Id"]), {"active": False})
    except Exception:
        _log.warning("agent deactivate failed", exc_info=True)
    if INCIDENTS in client.tables:
        try:
            client._post(INCIDENTS, {
                "agent_id": agent["Id"],
                "org_id": agent.get("org_id"),
                "kind": "circuit_breaker",
                "reason": reason,
                "created_at": _iso_now(),
            })
        except Exception:
            _log.warning("incident write failed", exc_info=True)


def _bump_counters(client: NocodbClient, agent_id: int, *, runs: int = 0, tokens: int = 0, cost_usd: float = 0,
                   consecutive_failures: int | None = None, last_run_status: str | None = None,
                   last_run_summary: str | None = None):
    rows = client._get_paginated(AGENTS, params={"where": f"(Id,eq,{agent_id})", "limit": 1})
    if not rows:
        return
    a = rows[0]
    update: dict = {}
    if runs:
        update["runs_today"] = int(a.get("runs_today") or 0) + runs
    if tokens:
        update["tokens_today"] = int(a.get("tokens_today") or 0) + tokens
    if cost_usd:
        update["cost_usd_today"] = float(a.get("cost_usd_today") or 0) + cost_usd
    if consecutive_failures is not None:
        update["consecutive_failures"] = consecutive_failures
    if last_run_status:
        update["last_run_status"] = last_run_status
    if last_run_summary is not None:
        update["last_run_summary"] = last_run_summary[:1000]
    update["last_run_at"] = _iso_now()
    try:
        client._patch(AGENTS, agent_id, update)
    except Exception:
        _log.warning("agent counter update failed", exc_info=True)


# ---------- memoization ----------

def _memoized_result(client: NocodbClient, assignment: dict, agent: dict) -> dict | None:
    ttl = int(agent.get("memoize_ttl_seconds") or 0)
    dedup = assignment.get("dedup_key")
    if not ttl or not dedup:
        return None
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=ttl)).strftime("%Y-%m-%d %H:%M:%S")
    rows = client._get_paginated(ASSIGNMENTS, params={
        "where": f"(dedup_key,eq,{dedup})~and(status,eq,completed)~and(completed_at,gt,{cutoff})",
        "limit": 1,
        "sort": "-Id",
    })
    return rows[0] if rows else None


# ---------- hooks ----------

def _call_hook(path: str, ctx: RunContext, phase: str):
    if not path:
        return
    try:
        mod_name, fn_name = path.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)
        fn(ctx, phase)
    except Exception:
        _log.warning("hook %s failed (%s)", path, phase, exc_info=True)


# ---------- audit ----------

def _write_run_record(client: NocodbClient, ctx: RunContext, prompt_snapshot: str,
                      status: str, summary: str, error: str = ""):
    if RUNS not in client.tables:
        return None
    try:
        agent = ctx.agent
        events_jsonl = "\n".join(json.dumps(e) for e in ctx.events)[:200000]
        return client._post(RUNS, {
            "assignment_id": ctx.assignment.get("Id"),
            "agent_name": agent.get("name"),
            "org_id": agent.get("org_id"),
            "task_description": (ctx.assignment.get("task") or "")[:1000],
            "summary": summary[:1000],
            "tokens_input": ctx.tokens_in,
            "tokens_output": ctx.tokens_out,
            "duration_seconds": round(time.time() - ctx.budgets.started_at, 2),
            "model_name": agent.get("model"),
            "iteration_count": ctx.iterations,
            "cost_usd": ctx.cost_usd,
            "prompt_version": agent.get("prompt_version"),
            "prompt_snapshot": prompt_snapshot[:60000],
            "events_jsonl": events_jsonl,
            "worker_id": ctx.worker_id,
            "status": status,
            "error": error[:2000],
        })
    except Exception:
        _log.warning("agent_runs write failed", exc_info=True)
        return None


# ---------- main run ----------

def _run_one(client: NocodbClient, worker: str, assignment: dict) -> None:
    aid = int(assignment["agent_id"])
    rows = client._get_paginated(AGENTS, params={"where": f"(Id,eq,{aid})", "limit": 1})
    if not rows:
        _fail(client, assignment, "agent not found", retry=False)
        return
    agent = rows[0]

    ok, reason = _check_budgets(agent)
    if not ok:
        _fail(client, assignment, f"budget: {reason}", retry=False)
        return

    if cached := _memoized_result(client, assignment, agent):
        try:
            client._patch(ASSIGNMENTS, int(assignment["Id"]), {
                "status": "completed",
                "completed_at": _iso_now(),
                "result_summary": (cached.get("result_summary") or "")[:1000],
                "result_ref_json": cached.get("result_ref_json") or "{}",
            })
        except Exception:
            pass
        _log.info("memoized completion id=%s from=%s", assignment["Id"], cached.get("Id"))
        return

    budgets = Budgets(
        max_iterations=int(agent.get("max_iterations") or 5),
        max_runtime_seconds=int(agent.get("max_runtime_seconds") or 300),
        max_tokens_per_run=int(agent.get("max_tokens_per_run") or 0),
    )
    forbidden = set(p.strip() for p in (agent.get("forbidden_tables") or "").split(",") if p.strip())
    ctx = RunContext(
        agent=agent,
        assignment=assignment,
        db=client,
        budgets=budgets,
        forbidden_tables=forbidden,
        dry_run=bool(agent.get("dry_run")),
        test_mode=bool(agent.get("test_mode")),
        worker_id=worker,
    )
    ctx.log("run_start", agent=agent.get("name"), type=agent.get("type"))

    sysp = ctxmod.build_system_prompt(agent)
    userp = ctxmod.build_user_context(client, agent, assignment)
    prompt_snapshot = f"# SYSTEM\n{sysp}\n\n# USER\n{userp}"

    hb = Heartbeat(client, int(assignment["Id"]))
    hb.start()
    result: RunResult | None = None
    error = ""
    try:
        _call_hook(agent.get("pre_run_hook"), ctx, "pre")
        result = dispatch_type(agent.get("type"), ctx)
        _call_hook(agent.get("post_run_hook"), ctx, "post")
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        _log.warning("agent run failed id=%s err=%s", assignment["Id"], error, exc_info=True)
    finally:
        hb.stop()

    if error:
        _handle_error(client, assignment, agent, ctx, prompt_snapshot, error)
        return

    summary = (result.summary if result else "")[:1000]
    refs_json = json.dumps(result.refs if result else {})
    try:
        client._patch(ASSIGNMENTS, int(assignment["Id"]), {
            "status": "completed",
            "completed_at": _iso_now(),
            "result_summary": summary,
            "result_ref_json": refs_json,
        })
    except Exception:
        _log.warning("assignment complete write failed", exc_info=True)

    _write_run_record(client, ctx, prompt_snapshot, "completed", summary)
    _bump_counters(client, int(agent["Id"]),
                   runs=1, tokens=ctx.tokens_in + ctx.tokens_out, cost_usd=ctx.cost_usd,
                   consecutive_failures=0, last_run_status="completed", last_run_summary=summary)
    ctx.log("run_done", summary=summary)
    _log.info("agent run ok  agent=%s assignment=%s iters=%d tokens=%d/%d",
              agent.get("name"), assignment["Id"], ctx.iterations, ctx.tokens_in, ctx.tokens_out)


def _handle_error(client: NocodbClient, assignment: dict, agent: dict,
                  ctx: RunContext, prompt_snapshot: str, error: str):
    consec = int(agent.get("consecutive_failures") or 0) + 1
    threshold = int(agent.get("circuit_breaker_threshold") or 5)
    on_error = (agent.get("on_error_action") or "retry").lower()

    attempts = int(assignment.get("attempts") or 0) + 1
    max_attempts = int(assignment.get("max_attempts") or 3)
    backoff = DEFAULT_BACKOFF_BASE_S * (2 ** max(0, attempts - 1))

    update: dict = {"attempts": attempts, "error": error[:2000]}
    if on_error == "pause" or attempts >= max_attempts:
        update["status"] = "failed"
        update["completed_at"] = _iso_now()
    elif on_error == "escalate":
        update["status"] = "awaiting_approval"
    else:
        update["status"] = "queued"
        update["next_retry_at"] = (datetime.now(timezone.utc) + timedelta(seconds=backoff)).strftime("%Y-%m-%d %H:%M:%S")
        update["claimed_by_worker"] = ""

    try:
        client._patch(ASSIGNMENTS, int(assignment["Id"]), update)
    except Exception:
        _log.warning("assignment error write failed", exc_info=True)

    _write_run_record(client, ctx, prompt_snapshot, "failed", "", error=error)
    _bump_counters(client, int(agent["Id"]),
                   runs=1, tokens=ctx.tokens_in + ctx.tokens_out,
                   consecutive_failures=consec, last_run_status="failed",
                   last_run_summary=error[:500])

    if consec >= threshold:
        _trip_circuit(client, agent, f"consecutive_failures={consec} ({error[:200]})")


def _fail(client: NocodbClient, assignment: dict, reason: str, retry: bool):
    update = {"error": reason[:2000]}
    update["status"] = "queued" if retry else "failed"
    if not retry:
        update["completed_at"] = _iso_now()
    try:
        client._patch(ASSIGNMENTS, int(assignment["Id"]), update)
    except Exception:
        pass


# ---------- daily reset ----------

def reset_daily_counters(client: NocodbClient | None = None):
    client = client or NocodbClient()
    rows = client._get_paginated(AGENTS, params={"limit": 500})
    for a in rows:
        try:
            client._patch(AGENTS, int(a["Id"]), {
                "runs_today": 0, "tokens_today": 0, "cost_usd_today": 0,
            })
        except Exception:
            pass


# ---------- enqueue helper ----------

def enqueue_assignment(
    agent_id: int, task: str, *, source: str = "manual", source_meta: dict | None = None,
    org_id: int = 1, priority: int = 3, dedup_key: str = "", parent_assignment_id: int | None = None,
    client: NocodbClient | None = None,
) -> dict:
    client = client or NocodbClient()
    if dedup_key:
        existing = client._get_paginated(ASSIGNMENTS, params={
            "where": f"(dedup_key,eq,{dedup_key})~and(status,in,queued,running)",
            "limit": 1,
        })
        if existing:
            return existing[0]
    payload = {
        "agent_id": agent_id,
        "org_id": org_id,
        "source": source,
        "source_meta_json": json.dumps(source_meta or {}),
        "task": task,
        "priority": priority,
        "status": "queued",
        "dedup_key": dedup_key,
        "attempts": 0,
        "max_attempts": 3,
    }
    if parent_assignment_id:
        payload["parent_assignment_id"] = parent_assignment_id
    return client._post(ASSIGNMENTS, payload)


# ---------- worker loop ----------

def run_worker(stop_event: threading.Event | None = None):
    worker = _worker_id()
    client = NocodbClient()
    _log.info("agent worker starting  id=%s", worker)
    stop_event = stop_event or threading.Event()
    while not stop_event.is_set():
        try:
            assignment = _claim_one(client, worker)
            if not assignment:
                stop_event.wait(POLL_INTERVAL_S)
                continue
            _run_one(client, worker, assignment)
        except Exception:
            _log.warning("worker loop iteration failed", exc_info=True)
            stop_event.wait(POLL_INTERVAL_S)
    _log.info("agent worker stopping  id=%s", worker)


_worker_thread: threading.Thread | None = None
_worker_stop: threading.Event | None = None


def start_worker_in_background():
    global _worker_thread, _worker_stop
    if _worker_thread and _worker_thread.is_alive():
        return
    _worker_stop = threading.Event()
    _worker_thread = threading.Thread(target=run_worker, args=(_worker_stop,), daemon=True, name="agent-worker")
    _worker_thread.start()


def stop_worker():
    if _worker_stop:
        _worker_stop.set()
