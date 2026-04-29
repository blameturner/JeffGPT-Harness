"""Producer agent — creates one new row per run."""
from __future__ import annotations

import json
import logging
import re

from workers.user_agents import artifact, context as ctxmod
from workers.user_agents.types.base import RunContext, RunResult, tool_loop, validate_json

_log = logging.getLogger("agents.types.producer")
JSON_RE = re.compile(r"\{[\s\S]*\}")


def run(ctx: RunContext) -> RunResult:
    cfg = ctxmod.parse_json(ctx.agent.get("type_config_json"), {})
    table = cfg.get("target_table")
    column_map = cfg.get("column_map") or {}
    if not table or not column_map:
        raise ValueError("producer agent missing target_table/column_map")

    sysp = ctxmod.build_system_prompt(ctx.agent)
    userp = ctxmod.build_user_context(ctx.db, ctx.agent, ctx.assignment, "")

    schema = ctxmod.parse_json(ctx.agent.get("output_schema_json"), {})
    max_retries = int(ctx.agent.get("max_validation_retries") or 2)

    output = ""
    parsed: dict = {}
    for attempt in range(max_retries + 1):
        output = tool_loop(ctx, sysp, userp)
        if not schema:
            try:
                m = JSON_RE.search(output)
                parsed = json.loads(m.group(0)) if m else {"body": output}
            except Exception:
                parsed = {"body": output}
            break
        ok, obj, err = validate_json(output, schema)
        if ok:
            parsed = obj or {}
            break
        ctx.log("output_invalid", attempt=attempt, error=err)
        userp = userp + f"\n\nPREVIOUS OUTPUT FAILED VALIDATION: {err}\nReturn valid JSON matching the schema."

    payload = {}
    for col, expr in column_map.items():
        if isinstance(expr, str) and expr.startswith("<llm.") and expr.endswith(">"):
            key = expr[5:-1]
            payload[col] = parsed.get(key, "")
        else:
            payload[col] = expr

    if "org_id" not in payload and ctx.agent.get("org_id"):
        payload["org_id"] = ctx.agent["org_id"]

    inserted = artifact.insert(
        ctx.db,
        table=table,
        payload=payload,
        forbidden_tables=ctx.forbidden_tables,
        dry_run=ctx.dry_run or ctx.test_mode,
    )
    ctx.log("artifact_insert", table=table, id=inserted.get("Id"))
    return RunResult(
        output=output,
        refs={"table": table, "row_id": inserted.get("Id")},
        summary=f"inserted {table}#{inserted.get('Id')}",
    )
