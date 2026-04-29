"""Responder agent — replies to an inbound message.

Reply channels:
  - email   : via configured smtp_account (deferred until SEND_EMAIL tool wired)
  - api     : returned in result_summary; HTTP trigger returns this
  - none    : just log

Approval mode writes to agent_approvals; the actual send happens after approve.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from workers.user_agents import context as ctxmod
from workers.user_agents.types.base import RunContext, RunResult, tool_loop

_log = logging.getLogger("agents.types.responder")
APPROVALS_TABLE = "agent_approvals"


def _iso_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def run(ctx: RunContext) -> RunResult:
    cfg = ctxmod.parse_json(ctx.agent.get("type_config_json"), {})
    inbox_kind = cfg.get("inbox_kind", "api")
    reply_mode = cfg.get("reply_mode", "approval")
    log_table = cfg.get("log_table")

    source_meta = ctxmod.parse_json(ctx.assignment.get("source_meta_json"), {})
    incoming = source_meta.get("body") or ctx.assignment.get("task", "")
    sender = source_meta.get("from") or source_meta.get("sender") or ""

    sysp = ctxmod.build_system_prompt(ctx.agent)
    type_input = (
        f"\n# INBOUND ({inbox_kind})\n"
        f"From: {sender}\n"
        f"Subject: {source_meta.get('subject','')}\n\n"
        f"{incoming}\n"
    )
    userp = ctxmod.build_user_context(ctx.db, ctx.agent, ctx.assignment, type_input)
    draft = tool_loop(ctx, sysp, userp)

    refs: dict = {"channel": inbox_kind, "to": sender, "draft": draft}

    if log_table and log_table in ctx.db.tables and not (ctx.dry_run or ctx.test_mode):
        try:
            ctx.db._post(log_table, {
                "from": sender,
                "subject": source_meta.get("subject", ""),
                "summary": (incoming or "")[:500],
                "suggested_reply_body": draft,
                "status": "drafted",
                "org_id": ctx.agent.get("org_id"),
            })
        except Exception:
            _log.warning("log_table write failed", exc_info=True)

    if reply_mode == "approval" and APPROVALS_TABLE in ctx.db.tables and not (ctx.dry_run or ctx.test_mode):
        try:
            ctx.db._post(APPROVALS_TABLE, {
                "assignment_id": ctx.assignment.get("Id"),
                "agent_id": ctx.agent.get("Id"),
                "org_id": ctx.agent.get("org_id"),
                "action_kind": f"reply_{inbox_kind}",
                "action_payload_json": __import__("json").dumps({"to": sender, "body": draft}),
                "status": "pending",
            })
            ctx.log("approval_queued", to=sender)
        except Exception:
            _log.warning("approval write failed", exc_info=True)
        return RunResult(output=draft, refs=refs, summary=f"draft awaiting approval (to={sender})")

    if reply_mode == "auto":
        ctx.log("auto_reply_pending_send_tool", to=sender, channel=inbox_kind)
        # actual send wired when SEND_EMAIL / outbound api tool lands

    return RunResult(output=draft, refs=refs, summary=f"drafted reply to {sender}")
