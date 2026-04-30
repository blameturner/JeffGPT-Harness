"""Harvest API — trigger / inspect / control harvest runs.

Endpoints:
  GET    /harvest/policies
  POST   /harvest/{policy}                  body: {seed, params, org_id?}
  POST   /harvest/scrape-now                body: {url, org_id?}
  POST   /harvest/bulk-upload               body: {urls, org_id?}
  GET    /harvest/runs                      ?policy=&status=&limit=
  GET    /harvest/runs/{run_id}
  GET    /harvest/runs/{run_id}/artifacts
  POST   /harvest/runs/{run_id}/cancel
  POST   /harvest/runs/{run_id}/retry

The DB only needs ONE new table: ``harvest_runs`` (see
docs/scraper-pathfinder-refactor.md for the schema).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from infra.nocodb_client import NocodbClient
from tools._org import resolve_org_id
from tools.harvest.policy import REGISTRY, list_policies, get_policy
from workers.tool_queue import get_tool_queue

_log = logging.getLogger("main.harvest")

router = APIRouter(prefix="/harvest")

_TABLE = "harvest_runs"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


# ── request models ────────────────────────────────────────────────────────

class HarvestRunRequest(BaseModel):
    seed: str | list | None = None
    params: dict = {}
    org_id: int | None = None


class ScrapeNowRequest(BaseModel):
    url: str
    org_id: int | None = None


class BulkUploadRequest(BaseModel):
    urls: list[str]
    org_id: int | None = None


# ── helpers ───────────────────────────────────────────────────────────────

def _enqueue_run(policy_name: str, seed, params: dict, org_id: int) -> dict:
    """Insert harvest_runs row and queue a `harvest_run` job.

    A list-shaped seed is stored in ``params_json["urls"]`` (not the seed
    column) — the seed column is a SingleLineText with a length cap, so a
    list of URLs would be silently truncated otherwise. The seed column
    then carries a human-readable summary like ``<bulk: N urls>``.
    """
    if get_policy(policy_name) is None:
        raise HTTPException(status_code=400, detail=f"unknown policy: {policy_name}")

    params = dict(params or {})
    if isinstance(seed, list):
        # Move full list into params; seed column gets a summary.
        params.setdefault("urls", list(seed))
        seed_repr = f"<bulk: {len(seed)} urls>"
    elif isinstance(seed, str):
        seed_repr = seed
    else:
        seed_repr = ""
    seed_repr = seed_repr[:500]

    client = NocodbClient()
    payload = {
        "policy": policy_name,
        "seed": seed_repr,
        "params_json": json.dumps(params),
        "status": "queued",
        "urls_planned": 0,
        "urls_fetched": 0,
        "urls_persisted": 0,
        "urls_unchanged": 0,
        "urls_skipped": 0,
        "urls_failed": 0,
        "artifacts_json": "{}",
        "cost_usd": 0.0,
        "org_id": org_id,
    }
    try:
        row = client._post(_TABLE, payload)
        run_id = row.get("Id")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"harvest_runs insert failed: {str(e)[:300]}")
    if not run_id:
        raise HTTPException(status_code=500, detail="harvest_runs insert returned no Id")

    tq = get_tool_queue()
    if tq is None:
        client._patch(_TABLE, run_id, {
            "status": "failed",
            "error_message": "tool_queue_unavailable",
        })
        raise HTTPException(status_code=503, detail="tool_queue_unavailable")

    job_id = tq.submit(
        "harvest_run",
        {"run_id": run_id, "org_id": org_id, "policy": policy_name},
        source=f"harvest_{policy_name}",
        priority=4,
        org_id=org_id,
    )
    return {"status": "queued", "run_id": run_id, "job_id": job_id, "policy": policy_name}


# ── policy catalog ────────────────────────────────────────────────────────

@router.get("/policies")
def policies():
    return {"policies": list_policies()}


# ── trigger (literal routes; policy-named route is /run/{policy} below) ─

@router.post("/scrape-now")
def scrape_now(body: ScrapeNowRequest):
    """One-off URL ingestion. Uses the ``single_url`` policy."""
    if not body.url or not body.url.startswith("http"):
        raise HTTPException(status_code=400, detail="url required and must be http(s)")
    org_id = int(body.org_id or resolve_org_id(0) or 1)
    return _enqueue_run("single_url", body.url, {}, org_id)


@router.post("/bulk-upload")
def bulk_upload(body: BulkUploadRequest):
    """Bulk URL ingestion in a single harvest run via the ``single_url`` policy
    with url_list seed strategy. Stores the list in params_json (not the seed
    column, which has a length cap), so any list size up to 500 URLs works."""
    urls = [u for u in (body.urls or []) if isinstance(u, str) and u.startswith("http")]
    if not urls:
        raise HTTPException(status_code=400, detail="urls list empty or invalid")
    if len(urls) > 500:
        urls = urls[:500]
    org_id = int(body.org_id or resolve_org_id(0) or 1)
    # Seed left as a count summary so the column is informative; the runner
    # reads URLs from params_json["urls"] for url_list seed_strategy.
    seed_summary = f"<bulk: {len(urls)} urls>"
    return _enqueue_run("single_url", seed_summary, {"urls": urls}, org_id)


# Path-param trigger LAST so literal routes above (scrape-now, bulk-upload,
# policies, runs, hosts) match first. FastAPI checks routes in registration
# order — declaring /{policy} earlier would capture every literal subpath.
@router.post("/run/{policy}")
def trigger(policy: str, body: HarvestRunRequest | None = None):
    body = body or HarvestRunRequest()
    org_id = int(body.org_id or resolve_org_id(0) or 1)
    return _enqueue_run(policy, body.seed, body.params or {}, org_id)


# ── inspection ────────────────────────────────────────────────────────────

@router.get("/runs")
def list_runs(policy: str | None = None, status: str | None = None,
              limit: int = 50, org_id: int | None = None):
    limit = max(1, min(int(limit), 500))
    where_parts = []
    if policy:
        where_parts.append(f"(policy,eq,{policy})")
    if status:
        where_parts.append(f"(status,eq,{status})")
    if org_id:
        where_parts.append(f"(org_id,eq,{int(org_id)})")
    params = {"limit": limit, "sort": "-CreatedAt"}
    if where_parts:
        params["where"] = "~and".join(where_parts)
    client = NocodbClient()
    try:
        rows = client._get(_TABLE, params=params).get("list", [])
    except Exception:
        raise HTTPException(status_code=500, detail="harvest_runs query failed")
    return {"runs": rows, "count": len(rows)}


_NON_TERMINAL_STATUSES = ("queued", "planning", "fetching", "extracting", "persisting")


@router.get("/active")
def active_runs(org_id: int | None = None, limit: int = 50):
    """Runs that haven't reached a terminal state. The Live UI polls this
    every few seconds while any are non-terminal."""
    limit = max(1, min(int(limit), 200))
    where_parts = ["(status,in," + ",".join(_NON_TERMINAL_STATUSES) + ")"]
    if org_id:
        where_parts.append(f"(org_id,eq,{int(org_id)})")
    params = {"where": "~and".join(where_parts), "sort": "-CreatedAt", "limit": limit}
    client = NocodbClient()
    try:
        rows = client._get(_TABLE, params=params).get("list", [])
    except Exception:
        raise HTTPException(status_code=500, detail="harvest_runs query failed")
    return {"runs": rows, "count": len(rows)}


@router.get("/runs/{run_id}/log")
def get_run_log(run_id: int, tail: int = 100):
    """Per-URL event tail for the Live drawer. Sourced from
    artifacts_json["events"] which the runner maintains as a rolling
    buffer (capped at 200 events)."""
    tail = max(1, min(int(tail), 500))
    client = NocodbClient()
    try:
        rows = client._get(_TABLE, params={"where": f"(Id,eq,{run_id})", "limit": 1}).get("list", [])
    except Exception:
        raise HTTPException(status_code=500, detail="harvest_runs query failed")
    if not rows:
        raise HTTPException(status_code=404, detail="run not found")
    row = rows[0]
    raw = row.get("artifacts_json") or "{}"
    try:
        arts = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except (json.JSONDecodeError, TypeError):
        arts = {}
    events = arts.get("events") if isinstance(arts, dict) else []
    if not isinstance(events, list):
        events = []
    return {
        "run_id": run_id,
        "status": row.get("status"),
        "urls_planned": row.get("urls_planned") or 0,
        "urls_fetched": row.get("urls_fetched") or 0,
        "urls_persisted": row.get("urls_persisted") or 0,
        "urls_failed": row.get("urls_failed") or 0,
        "events": events[-tail:],
    }


@router.get("/runs/{run_id}")
def get_run(run_id: int):
    client = NocodbClient()
    try:
        rows = client._get(_TABLE, params={"where": f"(Id,eq,{run_id})", "limit": 1}).get("list", [])
    except Exception:
        raise HTTPException(status_code=500, detail="harvest_runs query failed")
    if not rows:
        raise HTTPException(status_code=404, detail="run not found")
    return {"run": rows[0]}


@router.get("/runs/{run_id}/artifacts")
def get_artifacts(run_id: int):
    client = NocodbClient()
    try:
        rows = client._get(_TABLE, params={"where": f"(Id,eq,{run_id})", "limit": 1}).get("list", [])
    except Exception:
        raise HTTPException(status_code=500, detail="harvest_runs query failed")
    if not rows:
        raise HTTPException(status_code=404, detail="run not found")
    raw = rows[0].get("artifacts_json") or "{}"
    try:
        arts = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except (json.JSONDecodeError, TypeError):
        arts = {}
    return {"run_id": run_id, "artifacts": arts}


# ── lifecycle ─────────────────────────────────────────────────────────────

@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: int):
    client = NocodbClient()
    try:
        rows = client._get(_TABLE, params={"where": f"(Id,eq,{run_id})", "limit": 1}).get("list", [])
    except Exception:
        raise HTTPException(status_code=500, detail="harvest_runs query failed")
    if not rows:
        raise HTTPException(status_code=404, detail="run not found")
    row = rows[0]
    if row.get("status") in ("completed", "failed", "cancelled"):
        return {"status": row.get("status"), "run_id": run_id, "note": "already terminal"}
    try:
        client._patch(_TABLE, run_id, {
            "status": "cancelled",
            "finished_at": _now_iso(),
            "error_message": "cancelled by user",
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"cancel failed: {str(e)[:300]}")
    return {"status": "cancelled", "run_id": run_id}


# ── per-host config ──────────────────────────────────────────────────────


class HostPolicyPatch(BaseModel):
    rate_limit_per_host_s: float | None = None
    respect_robots: bool | None = None
    headless_fallback: bool | None = None
    connection_id: int | None = None
    notes: str | None = None


@router.get("/hosts")
def hosts_list():
    """Snapshot of per-host config (file-backed). Use to audit overrides."""
    from tools.harvest import host_config
    return {"hosts": host_config.all_hosts()}


@router.get("/hosts/{host}")
def host_get(host: str):
    """Resolve effective per-host config (file → features → defaults)."""
    from tools.harvest import host_config, rate_limit
    cfg = host_config.get(host)
    return {
        "host": host.lower(),
        "config": cfg,
        "rate_limit_status": rate_limit.status(host.lower()),
    }


@router.patch("/hosts/{host}")
def host_patch(host: str, body: HostPolicyPatch):
    from tools.harvest import host_config
    fields = body.model_dump(exclude_none=True)
    if not fields:
        raise HTTPException(status_code=400, detail="no fields to update")
    merged = host_config.set_host(host, fields)
    return {"host": host.lower(), "config": merged}


@router.delete("/hosts/{host}")
def host_delete(host: str):
    from tools.harvest import host_config
    ok = host_config.delete_host(host)
    if not ok:
        raise HTTPException(status_code=404, detail="host not found")
    return {"host": host.lower(), "deleted": True}


@router.post("/hosts/reload")
def hosts_reload():
    """Force reload of host_config from disk (in case the JSON was edited)."""
    from tools.harvest import host_config
    host_config.reload()
    return {"hosts": host_config.all_hosts()}


# ── retry / chain ────────────────────────────────────────────────────────

@router.post("/runs/{run_id}/retry")
def retry_run(run_id: int):
    client = NocodbClient()
    try:
        rows = client._get(_TABLE, params={"where": f"(Id,eq,{run_id})", "limit": 1}).get("list", [])
    except Exception:
        raise HTTPException(status_code=500, detail="harvest_runs query failed")
    if not rows:
        raise HTTPException(status_code=404, detail="run not found")
    prior = rows[0]
    seed = prior.get("seed") or ""
    try:
        params = json.loads(prior.get("params_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        params = {}
    org_id = int(prior.get("org_id") or 1)
    out = _enqueue_run(prior.get("policy") or "", seed, params, org_id)
    out["parent_run_id"] = run_id
    # Best-effort link via parent_run_id column if present
    try:
        client._patch(_TABLE, out["run_id"], {"parent_run_id": run_id})
    except Exception:
        pass
    return out
