"""Document agent — owns one row, edits its long-text body iteratively."""
from __future__ import annotations

import logging

from workers.user_agents import artifact, context as ctxmod
from workers.user_agents.types.base import RunContext, RunResult, tool_loop, reflect

_log = logging.getLogger("agents.types.document")


def run(ctx: RunContext) -> RunResult:
    cfg = ctxmod.parse_json(ctx.agent.get("type_config_json"), {})
    table = cfg.get("target_table")
    row_id = int(cfg.get("target_row_id") or 0)
    column = cfg.get("target_column", "body")
    edit_mode = cfg.get("edit_mode", "replace")
    if not table or not row_id:
        raise ValueError("document agent missing target_table/target_row_id")

    current = artifact.read(ctx.db, table, row_id, column)

    sysp = ctxmod.build_system_prompt(ctx.agent)
    type_input = f"\n# CURRENT DOCUMENT ({table}#{row_id}.{column}, mode={edit_mode}):\n{current or '(empty)'}"
    userp = ctxmod.build_user_context(ctx.db, ctx.agent, ctx.assignment, type_input)

    output = tool_loop(ctx, sysp, userp)

    if ctx.agent.get("reflect"):
        output = reflect(ctx, output, ctx.assignment.get("task") or "")

    diff = artifact.write(
        ctx.db,
        agent_id=int(ctx.agent.get("Id")),
        assignment_id=int(ctx.assignment.get("Id") or 0),
        table=table,
        row_id=row_id,
        column=column,
        new_text=output,
        edit_mode=edit_mode,
        forbidden_tables=ctx.forbidden_tables,
        dry_run=ctx.dry_run or ctx.test_mode,
    )
    ctx.log("artifact_write", table=table, row=row_id, col=column, mode=edit_mode,
            before_len=len(diff["before"] or ""), after_len=len(diff["after"] or ""))
    return RunResult(
        output=output,
        refs={"table": table, "row_id": row_id, "column": column},
        summary=f"updated {table}#{row_id}.{column} ({edit_mode})",
    )
