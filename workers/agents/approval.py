"""Approval handling for pending deep-search and research plans.

Extracted from chat_agent.py — runs *before* the gate check so that
approval messages ("Approved", "Go ahead") that don't trigger any gate
hints are still handled correctly.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

from tools.framework.contract import ToolAction, ToolContext, ToolName, ToolPlan
from tools.framework.dispatcher import execute_plan

_log = logging.getLogger(__name__)

# Modes that represent a user-approved plan execution.
APPROVAL_MODES = ("deep_approved", "research_approved")

# All modes related to deep/research (planning + approved).
_RELATED_MODES = ("deep_approved", "deep", "research_approved", "research")


@dataclass
class ApprovalResult:
    """Return value from :func:`handle_approvals`."""

    handled: bool = False
    """True if an approval was processed (caller should skip hints)."""

    tool_context: ToolContext = field(default_factory=ToolContext)
    """Tool execution results, if any."""

    search_status: str = ""
    """e.g. ``"queued"`` when background jobs were submitted."""

    search_context: str = ""
    """Serialised tool-result data for the LLM context window."""

    search_confidence: str = ""
    """e.g. ``"pending"`` while background jobs are in flight."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mark_plan_approved(db, conversation_id: int, plan_model: str) -> None:
    """Find the most recent plan message and flip its status to *approved*."""
    try:
        msgs = db.list_messages(conversation_id)
        for m in reversed(msgs):
            if m.get("model") == plan_model and m.get("search_status") == "awaiting_approval":
                db._patch("messages", m["Id"], {
                    "Id": m["Id"],
                    "search_status": "approved",
                })
                _log.info("approval conv=%s  marked %s message as approved", conversation_id, plan_model)
                break
    except Exception:
        _log.warning("approval conv=%s  failed to mark %s as approved", conversation_id, plan_model, exc_info=True)


def _clear_stale_plans(db, conversation_id: int, pending_ds: str, pending_rs: str) -> tuple[str, str]:
    """Clear pending plans when the user has moved on to a different topic.

    Returns the (possibly cleared) ``(pending_ds, pending_rs)`` values.
    """
    stale_clear: dict[str, str] = {}
    if pending_ds:
        stale_clear["pending_deep_search"] = ""
        pending_ds = ""
    if pending_rs:
        stale_clear["pending_research"] = ""
        pending_rs = ""
    if stale_clear:
        try:
            db.update_conversation(conversation_id, stale_clear)
            _log.info("approval conv=%s  cleared stale pending plans: %s", conversation_id, sorted(stale_clear.keys()))
        except Exception:
            pass
    return pending_ds, pending_rs


def _execute_approval(
    db,
    conversation_id: int,
    org_id: int,
    raw_plan: str,
    tool_name: ToolName,
    plan_model: str,
    emit,
) -> ApprovalResult:
    """Run the approved plan and return an :class:`ApprovalResult`."""
    try:
        plan_dict = json.loads(raw_plan)
    except Exception:
        plan_dict = {}

    tool_label = tool_name.value  # e.g. "deep_search" or "research"

    if not plan_dict:
        _log.warning("approval conv=%s  %s_approved but plan was empty", conversation_id, tool_label)
        return ApprovalResult(handled=True)

    if tool_label == "deep_search":
        _log.info(
            "approval conv=%s  deep search approved — executing %d queries, %d urls",
            conversation_id,
            len(plan_dict.get("queries", [])),
            len(plan_dict.get("urls", [])),
        )
    else:
        _log.info(
            "approval conv=%s  research approved — executing %d queries",
            conversation_id,
            len(plan_dict.get("queries", [])),
        )

    emit({"type": "plan_approved", "tool": tool_label})
    _mark_plan_approved(db, conversation_id, plan_model)

    actions = [ToolAction(
        tool=tool_name,
        params={
            "_phase": "execute",
            "_plan": plan_dict,
            "_org_id": org_id,
            "_conversation_id": conversation_id,
        },
        reason=f"{tool_label} (approved, executing)",
    )]
    plan = ToolPlan(actions=actions, summary=f"Executing approved {tool_label.replace('_', ' ')}")

    try:
        tool_context = asyncio.run(execute_plan(plan, emit))
        _log.info("approval conv=%s  %s execution dispatched", conversation_id, tool_label)
    except Exception:
        _log.error("approval conv=%s  %s execution failed", conversation_id, tool_label, exc_info=True)
        tool_context = ToolContext()

    result = ApprovalResult(handled=True, tool_context=tool_context)

    for r in tool_context.results:
        if r.tool.value == tool_label and r.ok:
            result.search_context = r.data
            result.search_status = "queued"
            result.search_confidence = "pending"

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def handle_approvals(
    db,
    convo: dict,
    conversation_id: int,
    org_id: int,
    search_mode: str,
    emit,
) -> ApprovalResult:
    """Process any pending deep-search / research approvals.

    This must run **before** the gate check in the chat pipeline.

    Parameters
    ----------
    db:
        Database accessor (must expose ``list_messages``, ``update_conversation``,
        and ``_patch``).
    convo:
        The conversation dict (must contain ``pending_deep_search`` and
        ``pending_research`` keys).
    conversation_id:
        Numeric conversation ID.
    org_id:
        Organisation ID for tool dispatch.
    search_mode:
        The current search mode string (e.g. ``"deep_approved"``).
    emit:
        SSE emitter callable.

    Returns
    -------
    ApprovalResult
        Contains everything the caller needs to update ``search_result`` and
        decide whether to skip the gate-check hints.
    """
    pending_ds = convo.get("pending_deep_search") or ""
    pending_rs = convo.get("pending_research") or ""

    # Clear stale plans when user moves on to something else.
    if search_mode not in _RELATED_MODES:
        pending_ds, pending_rs = _clear_stale_plans(db, conversation_id, pending_ds, pending_rs)

    # Deep search approval
    if search_mode == "deep_approved":
        if pending_ds:
            return _execute_approval(
                db, conversation_id, org_id,
                raw_plan=pending_ds,
                tool_name=ToolName.DEEP_SEARCH,
                plan_model="deep_search_plan",
                emit=emit,
            )
        _log.warning("approval conv=%s  deep_approved but no pending plan — falling back to normal", conversation_id)
        return ApprovalResult(handled=True)

    # Research approval
    if search_mode == "research_approved":
        if pending_rs:
            return _execute_approval(
                db, conversation_id, org_id,
                raw_plan=pending_rs,
                tool_name=ToolName.RESEARCH,
                plan_model="research_plan",
                emit=emit,
            )
        _log.warning("approval conv=%s  research_approved but no pending plan — falling back to normal", conversation_id)
        return ApprovalResult(handled=True)

    # No approval to process.
    return ApprovalResult()
