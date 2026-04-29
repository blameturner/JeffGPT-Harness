"""Supervisor agent — delegates to team members by inserting child assignments."""
from __future__ import annotations

import logging

from workers.user_agents import context as ctxmod
from workers.user_agents.types.base import RunContext, RunResult, tool_loop

_log = logging.getLogger("agents.types.supervisor")
ASSIGNMENTS_TABLE = "assignments"


def run(ctx: RunContext) -> RunResult:
    cfg = ctxmod.parse_json(ctx.agent.get("type_config_json"), {})
    team_ids = [int(x) for x in (cfg.get("team_agent_ids") or []) if x]
    if not team_ids:
        raise ValueError("supervisor missing team_agent_ids")

    team_rows: list[dict] = []
    for tid in team_ids:
        rows = ctx.db._get_paginated("agents", params={"where": f"(Id,eq,{tid})", "limit": 1})
        if rows:
            team_rows.append(rows[0])

    sysp = ctxmod.build_system_prompt(ctx.agent)
    catalog = "\n".join(
        f"- id={a['Id']} name={a.get('name')} type={a.get('type')} brief={(a.get('brief') or '')[:120]}"
        for a in team_rows
    )
    type_input = f"\n# TEAM\n{catalog}\n\nReturn ONE JSON object: {{\"delegate_to\": [agent_id, ...], \"task\": \"...\", \"rationale\": \"...\"}}"
    userp = ctxmod.build_user_context(ctx.db, ctx.agent, ctx.assignment, type_input)

    raw = tool_loop(ctx, sysp, userp)
    plan = _extract_json(raw) or {}
    targets = [int(x) for x in (plan.get("delegate_to") or []) if x in team_ids]
    sub_task = plan.get("task") or ctx.assignment.get("task") or ""

    refs: dict = {"delegated_to": [], "rationale": plan.get("rationale", "")}
    if not targets:
        ctx.log("supervisor_no_targets")
        return RunResult(output=raw, refs=refs, summary="no delegation chosen")

    if ctx.dry_run or ctx.test_mode:
        return RunResult(output=raw, refs={**refs, "delegated_to": targets}, summary=f"would delegate to {targets}")

    for tid in targets:
        try:
            new_row = ctx.db._post(ASSIGNMENTS_TABLE, {
                "agent_id": tid,
                "org_id": ctx.agent.get("org_id"),
                "source": "supervisor",
                "source_meta_json": __import__("json").dumps({"supervisor_id": ctx.agent.get("Id")}),
                "task": sub_task,
                "priority": ctx.assignment.get("priority") or 3,
                "status": "queued",
                "parent_assignment_id": ctx.assignment.get("Id"),
            })
            refs["delegated_to"].append({"agent_id": tid, "assignment_id": new_row.get("Id")})
            ctx.log("delegated", to_agent=tid, assignment=new_row.get("Id"))
        except Exception as e:
            _log.warning("delegation write failed agent=%d: %s", tid, e)

    return RunResult(output=raw, refs=refs, summary=f"delegated to {len(refs['delegated_to'])} agent(s)")


def _extract_json(text: str):
    import json, re
    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None
