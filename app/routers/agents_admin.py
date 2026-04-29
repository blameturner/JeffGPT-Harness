"""Agents administration router — CRUD, run-now, pause/resume, clone,
test-prompt, runs/assignments/versions sub-resources, plus the cross-agent
inboxes (assignments, approvals, incidents, templates, artifact-versions).

Mounted at root so `/agents/{id}/*`, `/assignments`, `/approvals`,
`/incidents`, `/templates`, `/artifact-versions` are all reachable.
The pre-existing `/agents` (list) and `/run` routes in app/routers/agents.py
remain unchanged; this router adds the rest without conflict.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from infra.nocodb_client import NocodbClient
from tools._org import resolve_org_id
from workers.user_agents import context as ctxmod
from workers.user_agents.runtime import enqueue_assignment

_log = logging.getLogger("main.agents_admin")
router = APIRouter(tags=["agents_admin"])

AGENTS = "agents"
ASSIGNMENTS = "assignments"
RUNS = "agent_runs"
APPROVALS = "agent_approvals"
INCIDENTS = "agent_incidents"
TEMPLATES = "agent_templates"
ARTIFACT_VERSIONS = "artifact_versions"


def _client() -> NocodbClient:
    return NocodbClient()


def _require_table(c: NocodbClient, table: str):
    if table not in c.tables:
        raise HTTPException(status_code=503, detail=f"{table} table missing — see docs/new-tables.md")


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _delete_row(c: NocodbClient, table: str, row_id: int):
    requests.delete(f"{c.url}/{c.tables[table]}/{row_id}", headers=c.headers, timeout=10).raise_for_status()


def _get_row(c: NocodbClient, table: str, row_id: int) -> dict | None:
    rows = c._get_paginated(table, params={"where": f"(Id,eq,{row_id})", "limit": 1})
    return rows[0] if rows else None


# ============================================================
# Agent CRUD (single — list endpoint already exists in agents.py)
# ============================================================

class AgentCreate(BaseModel):
    name: str
    org_id: int = 1
    type: str = "document"
    model: str = ""
    description: str = ""
    persona: str = ""
    system_prompt_template: str = ""
    brief: str = ""
    pinned_context: str = ""
    temperature: float = 0.4
    max_tokens: int = 1500
    max_iterations: int = 5
    max_runtime_seconds: int = 300
    allowed_tools: str = ""
    connected_apis: str = ""
    connected_smtp: str = ""
    type_config_json: dict = Field(default_factory=dict)
    active: bool = True
    extras: dict = Field(default_factory=dict)


@router.post("/agents")
def create_agent(payload: AgentCreate):
    c = _client()
    _require_table(c, AGENTS)
    body = payload.model_dump(exclude={"extras"})
    body["org_id"] = resolve_org_id(body.get("org_id"))
    if isinstance(body.get("type_config_json"), (dict, list)):
        body["type_config_json"] = json.dumps(body["type_config_json"])
    body.update(payload.extras or {})
    body.setdefault("prompt_version", 1)
    return c._post(AGENTS, body)


@router.get("/agents/{agent_id}")
def get_agent(agent_id: int):
    c = _client()
    _require_table(c, AGENTS)
    row = _get_row(c, AGENTS, agent_id)
    if not row:
        raise HTTPException(status_code=404, detail="agent not found")
    return row


@router.patch("/agents/{agent_id}")
def patch_agent(agent_id: int, payload: dict):
    c = _client()
    _require_table(c, AGENTS)
    current = _get_row(c, AGENTS, agent_id)
    if not current:
        raise HTTPException(status_code=404, detail="agent not found")
    blocked = {"Id", "id", "CreatedAt", "UpdatedAt",
               "runs_today", "tokens_today", "cost_usd_today",
               "consecutive_failures", "last_run_at", "last_run_status", "last_run_summary"}
    update = {k: v for k, v in payload.items() if k not in blocked}
    for k in ("type_config_json", "tool_config_json", "prompt_variables_json",
              "output_schema_json", "trigger_table_watch_json",
              "notify_on_complete_json", "notify_on_error_json"):
        if k in update and isinstance(update[k], (dict, list)):
            update[k] = json.dumps(update[k])
    if any(k in update for k in ("system_prompt_template", "persona", "pinned_context")):
        update["prompt_version"] = int(current.get("prompt_version") or 1) + 1
    return c._patch(AGENTS, agent_id, update)


@router.delete("/agents/{agent_id}")
def delete_agent(agent_id: int):
    c = _client()
    _require_table(c, AGENTS)
    if not _get_row(c, AGENTS, agent_id):
        raise HTTPException(status_code=404, detail="agent not found")
    _delete_row(c, AGENTS, agent_id)
    return {"ok": True, "deleted": agent_id}


# ============================================================
# Agent operations
# ============================================================

class RunNowRequest(BaseModel):
    task: str = ""
    priority: int = 3
    dedup_key: str = ""
    source: str = "manual"
    source_meta: dict = Field(default_factory=dict)


@router.post("/agents/{agent_id}/run")
def run_now(agent_id: int, payload: RunNowRequest):
    c = _client()
    agent = _get_row(c, AGENTS, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="agent not found")
    task = payload.task or agent.get("brief") or ""
    row = enqueue_assignment(
        agent_id=agent_id,
        task=task,
        source=payload.source,
        source_meta=payload.source_meta,
        org_id=resolve_org_id(agent.get("org_id")),
        priority=payload.priority,
        dedup_key=payload.dedup_key,
        client=c,
    )
    return {"assignment": row}


class PauseRequest(BaseModel):
    until: str = ""


@router.post("/agents/{agent_id}/pause")
def pause_agent(agent_id: int, payload: PauseRequest):
    c = _client()
    update: dict = {"active": False}
    if payload.until:
        update["pause_until"] = payload.until
    return c._patch(AGENTS, agent_id, update)


@router.post("/agents/{agent_id}/resume")
def resume_agent(agent_id: int):
    c = _client()
    return c._patch(AGENTS, agent_id, {"active": True, "pause_until": ""})


@router.post("/agents/{agent_id}/reset-circuit")
def reset_circuit(agent_id: int):
    c = _client()
    return c._patch(AGENTS, agent_id, {"consecutive_failures": 0, "active": True})


@router.post("/agents/{agent_id}/reset-counters")
def reset_counters(agent_id: int):
    c = _client()
    return c._patch(AGENTS, agent_id, {"runs_today": 0, "tokens_today": 0, "cost_usd_today": 0})


class CloneRequest(BaseModel):
    name: str
    overrides: dict = Field(default_factory=dict)


@router.post("/agents/{agent_id}/clone")
def clone_agent(agent_id: int, payload: CloneRequest):
    c = _client()
    src = _get_row(c, AGENTS, agent_id)
    if not src:
        raise HTTPException(status_code=404, detail="agent not found")
    body = dict(src)
    for k in ("Id", "id", "CreatedAt", "UpdatedAt",
              "runs_today", "tokens_today", "cost_usd_today",
              "consecutive_failures", "last_run_at", "last_run_status", "last_run_summary"):
        body.pop(k, None)
    body["name"] = payload.name
    body["prompt_version"] = 1
    body.update(payload.overrides or {})
    for k in ("type_config_json", "tool_config_json", "prompt_variables_json",
              "output_schema_json", "trigger_table_watch_json",
              "notify_on_complete_json", "notify_on_error_json"):
        if k in body and isinstance(body[k], (dict, list)):
            body[k] = json.dumps(body[k])
    return c._post(AGENTS, body)


class TestPromptRequest(BaseModel):
    task: str


@router.post("/agents/{agent_id}/test-prompt")
def test_prompt(agent_id: int, payload: TestPromptRequest):
    """Build the prompt as it would appear at runtime, without enqueuing
    or running anything. Lets the UI show what the agent would see."""
    c = _client()
    agent = _get_row(c, AGENTS, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="agent not found")
    fake_assignment = {"task": payload.task, "Id": 0, "source_meta_json": "{}"}
    sysp = ctxmod.build_system_prompt(agent)
    userp = ctxmod.build_user_context(c, agent, fake_assignment)
    return {
        "system_prompt": sysp,
        "user_prompt": userp,
        "estimated_chars": len(sysp) + len(userp),
        "agent_type": agent.get("type"),
        "model": agent.get("model"),
    }


# ============================================================
# Per-agent sub-resources
# ============================================================

@router.get("/agents/{agent_id}/runs")
def list_agent_runs(agent_id: int, limit: int = 50, offset: int = 0):
    c = _client()
    _require_table(c, RUNS)
    agent = _get_row(c, AGENTS, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="agent not found")
    rows = c._get_paginated(RUNS, params={
        "where": f"(agent_name,eq,{agent.get('name')})~and(org_id,eq,{resolve_org_id(agent.get('org_id'))})",
        "limit": limit,
        "offset": offset,
        "sort": "-Id",
    })
    return {"runs": rows, "count": len(rows)}


@router.get("/agents/{agent_id}/runs/{run_id}")
def get_agent_run(agent_id: int, run_id: int):
    c = _client()
    _require_table(c, RUNS)
    row = _get_row(c, RUNS, run_id)
    if not row:
        raise HTTPException(status_code=404, detail="run not found")
    return row


@router.get("/agents/{agent_id}/assignments")
def list_agent_assignments(agent_id: int, status: str = "", limit: int = 50, offset: int = 0):
    c = _client()
    _require_table(c, ASSIGNMENTS)
    where = f"(agent_id,eq,{agent_id})"
    if status:
        where += f"~and(status,eq,{status})"
    rows = c._get_paginated(ASSIGNMENTS, params={"where": where, "limit": limit, "offset": offset, "sort": "-Id"})
    return {"assignments": rows, "count": len(rows)}


@router.get("/agents/{agent_id}/artifacts/versions")
def list_artifact_versions(agent_id: int, limit: int = 50, offset: int = 0):
    c = _client()
    _require_table(c, ARTIFACT_VERSIONS)
    rows = c._get_paginated(ARTIFACT_VERSIONS, params={
        "where": f"(agent_id,eq,{agent_id})",
        "limit": limit, "offset": offset, "sort": "-Id",
    })
    return {"versions": rows, "count": len(rows)}


# ============================================================
# Cross-agent: assignments inbox
# ============================================================

class AssignmentCreate(BaseModel):
    agent_id: int
    task: str
    org_id: int = 1
    source: str = "manual"
    source_meta: dict = Field(default_factory=dict)
    priority: int = 3
    dedup_key: str = ""
    parent_assignment_id: int | None = None


@router.get("/assignments")
def list_assignments(
    org_id: int = 1, status: str = "", source: str = "",
    agent_id: int | None = None, q: str = "", limit: int = 100, offset: int = 0,
):
    c = _client()
    _require_table(c, ASSIGNMENTS)
    parts = [f"(org_id,eq,{resolve_org_id(org_id)})"]
    if status:
        parts.append(f"(status,eq,{status})")
    if source:
        parts.append(f"(source,eq,{source})")
    if agent_id is not None:
        parts.append(f"(agent_id,eq,{agent_id})")
    if q:
        parts.append(f"(task,like,%{q}%)")
    rows = c._get_paginated(ASSIGNMENTS, params={
        "where": "~and".join(parts), "limit": limit, "offset": offset, "sort": "-Id",
    })
    return {"assignments": rows, "count": len(rows)}


@router.post("/assignments")
def create_assignment(payload: AssignmentCreate):
    return enqueue_assignment(
        agent_id=payload.agent_id,
        task=payload.task,
        source=payload.source,
        source_meta=payload.source_meta,
        org_id=resolve_org_id(payload.org_id),
        priority=payload.priority,
        dedup_key=payload.dedup_key,
        parent_assignment_id=payload.parent_assignment_id,
    )


@router.get("/assignments/{assignment_id}")
def get_assignment(assignment_id: int):
    c = _client()
    _require_table(c, ASSIGNMENTS)
    row = _get_row(c, ASSIGNMENTS, assignment_id)
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    return row


@router.patch("/assignments/{assignment_id}")
def patch_assignment(assignment_id: int, payload: dict):
    c = _client()
    _require_table(c, ASSIGNMENTS)
    blocked = {"Id", "id", "CreatedAt", "UpdatedAt", "claimed_by_worker", "claimed_at", "heartbeat_at"}
    update = {k: v for k, v in payload.items() if k not in blocked}
    return c._patch(ASSIGNMENTS, assignment_id, update)


@router.post("/assignments/{assignment_id}/cancel")
def cancel_assignment(assignment_id: int):
    c = _client()
    _require_table(c, ASSIGNMENTS)
    row = _get_row(c, ASSIGNMENTS, assignment_id)
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    if row.get("status") not in ("queued", "awaiting_approval"):
        raise HTTPException(status_code=409, detail=f"cannot cancel from status {row.get('status')}")
    return c._patch(ASSIGNMENTS, assignment_id, {"status": "failed", "error": "cancelled by user", "completed_at": _iso_now()})


@router.post("/assignments/{assignment_id}/retry")
def retry_assignment(assignment_id: int):
    c = _client()
    _require_table(c, ASSIGNMENTS)
    row = _get_row(c, ASSIGNMENTS, assignment_id)
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    if row.get("status") not in ("failed", "completed"):
        raise HTTPException(status_code=409, detail=f"cannot retry from status {row.get('status')}")
    return c._patch(ASSIGNMENTS, assignment_id, {
        "status": "queued",
        "error": "",
        "claimed_by_worker": "",
        "next_retry_at": "",
    })


# ============================================================
# Cross-agent: approvals
# ============================================================

class ApprovalDecision(BaseModel):
    note: str = ""


@router.get("/approvals")
def list_approvals(org_id: int = 1, status: str = "pending", limit: int = 100):
    c = _client()
    _require_table(c, APPROVALS)
    where = f"(org_id,eq,{resolve_org_id(org_id)})"
    if status:
        where += f"~and(status,eq,{status})"
    rows = c._get_paginated(APPROVALS, params={"where": where, "limit": limit, "sort": "-Id"})
    return {"approvals": rows, "count": len(rows)}


@router.post("/approvals/{approval_id}/approve")
def approve(approval_id: int, payload: ApprovalDecision):
    c = _client()
    _require_table(c, APPROVALS)
    row = _get_row(c, APPROVALS, approval_id)
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    c._patch(APPROVALS, approval_id, {
        "status": "approved",
        "decided_at": _iso_now(),
        "note": payload.note,
    })
    if (aid := row.get("assignment_id")) and ASSIGNMENTS in c.tables:
        try:
            assn = _get_row(c, ASSIGNMENTS, int(aid))
            if assn and assn.get("status") == "awaiting_approval":
                c._patch(ASSIGNMENTS, int(aid), {"status": "queued", "claimed_by_worker": ""})
        except Exception:
            _log.warning("failed to resume assignment %s after approval", aid, exc_info=True)
    return {"ok": True, "approval_id": approval_id}


@router.post("/approvals/{approval_id}/reject")
def reject(approval_id: int, payload: ApprovalDecision):
    c = _client()
    _require_table(c, APPROVALS)
    row = _get_row(c, APPROVALS, approval_id)
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    c._patch(APPROVALS, approval_id, {
        "status": "rejected",
        "decided_at": _iso_now(),
        "note": payload.note,
    })
    if (aid := row.get("assignment_id")) and ASSIGNMENTS in c.tables:
        try:
            c._patch(ASSIGNMENTS, int(aid), {
                "status": "failed", "error": f"approval rejected: {payload.note}",
                "completed_at": _iso_now(),
            })
        except Exception:
            _log.warning("failed to fail assignment %s after rejection", aid, exc_info=True)
    return {"ok": True, "approval_id": approval_id}


# ============================================================
# Cross-agent: incidents
# ============================================================

class IncidentResolve(BaseModel):
    note: str = ""


@router.get("/incidents")
def list_incidents(org_id: int = 1, resolved: bool | None = None, limit: int = 100):
    c = _client()
    _require_table(c, INCIDENTS)
    parts = [f"(org_id,eq,{resolve_org_id(org_id)})"]
    if resolved is True:
        parts.append("(resolved_at,isnot,null)")
    elif resolved is False:
        parts.append("(resolved_at,is,null)")
    rows = c._get_paginated(INCIDENTS, params={"where": "~and".join(parts), "limit": limit, "sort": "-Id"})
    return {"incidents": rows, "count": len(rows)}


@router.post("/incidents/{incident_id}/resolve")
def resolve_incident(incident_id: int, payload: IncidentResolve):
    c = _client()
    _require_table(c, INCIDENTS)
    return c._patch(INCIDENTS, incident_id, {
        "resolved_at": _iso_now(),
        "resolved_by": "user",
        "reason": (payload.note or ""),
    })


# ============================================================
# Templates
# ============================================================

class TemplateInstantiate(BaseModel):
    name: str
    org_id: int = 1
    overrides: dict = Field(default_factory=dict)


@router.get("/templates")
def list_templates():
    c = _client()
    _require_table(c, TEMPLATES)
    rows = c._get_paginated(TEMPLATES, params={"limit": 200})
    return {"templates": rows, "count": len(rows)}


@router.post("/templates/{template_id}/instantiate")
def instantiate_template(template_id: int, payload: TemplateInstantiate):
    c = _client()
    _require_table(c, TEMPLATES)
    _require_table(c, AGENTS)
    tpl = _get_row(c, TEMPLATES, template_id)
    if not tpl:
        raise HTTPException(status_code=404, detail="template not found")
    try:
        defaults = json.loads(tpl.get("defaults_json") or "{}")
    except Exception:
        defaults = {}
    body = {**defaults, **payload.overrides, "name": payload.name, "org_id": resolve_org_id(payload.org_id)}
    body.setdefault("prompt_version", 1)
    body.setdefault("active", True)
    for k in ("type_config_json", "tool_config_json", "prompt_variables_json",
              "output_schema_json", "trigger_table_watch_json",
              "notify_on_complete_json", "notify_on_error_json"):
        if k in body and isinstance(body[k], (dict, list)):
            body[k] = json.dumps(body[k])
    return c._post(AGENTS, body)


# ============================================================
# Artifact versions
# ============================================================

@router.get("/artifact-versions")
def list_versions(agent_id: int | None = None, table_name: str = "", row_id: int | None = None,
                  limit: int = 100, offset: int = 0):
    c = _client()
    _require_table(c, ARTIFACT_VERSIONS)
    parts = []
    if agent_id is not None:
        parts.append(f"(agent_id,eq,{agent_id})")
    if table_name:
        parts.append(f"(table_name,eq,{table_name})")
    if row_id is not None:
        parts.append(f"(row_id,eq,{row_id})")
    where = "~and".join(parts) if parts else ""
    rows = c._get_paginated(ARTIFACT_VERSIONS, params={
        "where": where, "limit": limit, "offset": offset, "sort": "-Id",
    })
    return {"versions": rows, "count": len(rows)}


@router.get("/artifact-versions/{version_id}")
def get_version(version_id: int):
    c = _client()
    _require_table(c, ARTIFACT_VERSIONS)
    row = _get_row(c, ARTIFACT_VERSIONS, version_id)
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    return row


@router.post("/artifact-versions/{version_id}/rollback")
def rollback_version(version_id: int):
    c = _client()
    _require_table(c, ARTIFACT_VERSIONS)
    v = _get_row(c, ARTIFACT_VERSIONS, version_id)
    if not v:
        raise HTTPException(status_code=404, detail="version not found")
    target_table = v.get("table_name")
    target_row = int(v.get("row_id") or 0)
    target_col = v.get("column_name")
    if not target_table or not target_row or not target_col:
        raise HTTPException(status_code=409, detail="version row missing target")
    if target_table not in c.tables:
        raise HTTPException(status_code=503, detail=f"target table missing: {target_table}")

    current = _get_row(c, target_table, target_row)
    current_text = (current or {}).get(target_col, "")
    c._patch(target_table, target_row, {target_col: v.get("before_text") or ""})

    try:
        c._post(ARTIFACT_VERSIONS, {
            "agent_id": v.get("agent_id"),
            "assignment_id": v.get("assignment_id"),
            "table_name": target_table,
            "row_id": target_row,
            "column_name": target_col,
            "before_text": current_text,
            "after_text": v.get("before_text") or "",
            "created_at": _iso_now(),
        })
    except Exception:
        _log.warning("rollback version write failed", exc_info=True)
    return {"ok": True, "rolled_back_to_version": version_id}
