"""Queue agent — processes N rows matching a filter, one at a time."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from workers.user_agents import artifact, context as ctxmod
from workers.user_agents.types.base import RunContext, RunResult, tool_loop

_log = logging.getLogger("agents.types.queue")


def _iso_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def run(ctx: RunContext) -> RunResult:
    cfg = ctxmod.parse_json(ctx.agent.get("type_config_json"), {})
    table = cfg.get("target_table")
    where = cfg.get("filter") or ""
    output_column = cfg.get("output_column")
    done_column = cfg.get("done_column")
    batch_size = int(cfg.get("batch_size") or 5)
    if not table or not output_column:
        raise ValueError("queue agent missing target_table/output_column")
    if table not in ctx.db.tables:
        raise ValueError(f"unknown table: {table}")

    rows = ctx.db._get_paginated(table, params={"where": where, "limit": batch_size})
    if not rows:
        return RunResult(output="", refs={"processed": 0}, summary="no rows match filter")

    sysp = ctxmod.build_system_prompt(ctx.agent)
    processed = []
    for row in rows:
        if ctx.budgets.time_left() <= 0:
            ctx.log("budget_runtime_exceeded")
            break
        type_input = f"\n# CURRENT ROW ({table}#{row.get('Id')}):\n{row}"
        userp = ctxmod.build_user_context(ctx.db, ctx.agent, ctx.assignment, type_input)
        output = tool_loop(ctx, sysp, userp)
        artifact.write(
            ctx.db,
            agent_id=int(ctx.agent.get("Id")),
            assignment_id=int(ctx.assignment.get("Id") or 0),
            table=table,
            row_id=int(row["Id"]),
            column=output_column,
            new_text=output,
            edit_mode="replace",
            forbidden_tables=ctx.forbidden_tables,
            dry_run=ctx.dry_run or ctx.test_mode,
        )
        if done_column and not (ctx.dry_run or ctx.test_mode):
            try:
                ctx.db._patch(table, int(row["Id"]), {done_column: _iso_now()})
            except Exception:
                _log.warning("done_column write failed", exc_info=True)
        processed.append(int(row["Id"]))
        ctx.log("queue_item_done", table=table, row=row["Id"])

    return RunResult(
        output=f"processed {len(processed)} rows",
        refs={"table": table, "row_ids": processed},
        summary=f"queue: {len(processed)} rows in {table}",
    )
