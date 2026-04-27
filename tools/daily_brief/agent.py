"""Daily brief producer.

One pass per scheduled slot. Reads the recall layer, runs an LLM call with
the mode-appropriate prompt, applies a hard silence gate, and persists:

  1. an `insights` row with trigger="daily_brief" — the canonical artifact
  2. an assistant message into the home conversation — the chat surface

If the recall layer reports no signal, the producer skips the LLM call
entirely and returns empty. If the LLM returns content despite empty
recall, the post-check downgrades to empty (defence in depth).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from infra.config import get_feature, is_feature_enabled
from infra.nocodb_client import NocodbClient
from shared.home_conversation import get_or_create_home_conversation
from shared.insights import TRIGGER_CHAT_IDLE, create as create_insight
from shared.models import model_call
from shared.pa.recall import build_recall, RecallPayload

_log = logging.getLogger("daily_brief")

TRIGGER_DAILY_BRIEF = "daily_brief"


def _cfg(key: str, default):
    return get_feature("daily_brief", key, default)


def _parse_output(raw: str) -> dict | None:
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _silence_gate_passes(payload: RecallPayload) -> bool:
    """Defence in depth: if recall has no signal, the brief is silenced
    regardless of what the LLM returned."""
    return payload.has_signal


def _gather_sources(payload: RecallPayload) -> list[dict]:
    sources: list[dict] = []
    for l in payload.open_loops_user[:8]:
        sources.append({"kind": "loop", "id": l.id, "intent": l.intent})
    for l in payload.open_loops_assistant[:5]:
        sources.append({"kind": "assistant_loop", "id": l.id})
    for r in payload.completed_research[:5]:
        if r.get("plan_id"):
            sources.append({"kind": "research_plan", "id": r["plan_id"]})
    if payload.thread_of_day is not None:
        sources.append({
            "kind": "conversation",
            "id": payload.thread_of_day.conversation_id,
        })
    return sources


def _post_to_home(
    org_id: int,
    body_markdown: str,
    insight_id: int | None,
    summary: str,
) -> int | None:
    try:
        convo = get_or_create_home_conversation(int(org_id))
    except Exception:
        _log.warning("daily_brief: home conversation lookup failed  org=%d", org_id, exc_info=True)
        return None
    convo_id = int(convo.get("Id") or 0)
    if convo_id <= 0:
        return None
    footer = ""
    if insight_id:
        footer = f"\n\n_— briefing #{insight_id}_"
    content = body_markdown.rstrip() + footer
    try:
        client = NocodbClient()
        msg = client.add_message(
            conversation_id=convo_id,
            org_id=int(org_id),
            role="assistant",
            content=content,
            model="daily_brief",
            insight_id=insight_id,
            source="daily_brief",
            response_style=summary[:120] if summary else "",
        )
        return int(msg.get("Id") or 0) or None
    except Exception:
        _log.warning("daily_brief: home message post failed  org=%d", org_id, exc_info=True)
        return None


def run_daily_brief(org_id: int, now: datetime | None = None, force: bool = False) -> dict:
    """Single tick. Returns a status dict for logging."""
    out: dict = {
        "status": "ok",
        "org_id": int(org_id),
        "produced": False,
        "mode": "",
        "reason": "",
    }
    if not is_feature_enabled("pa"):
        out["status"] = "skipped"
        out["reason"] = "pa_disabled"
        return out
    if not _cfg("enabled", True):
        out["status"] = "skipped"
        out["reason"] = "daily_brief_disabled"
        return out
    if int(org_id) <= 0:
        out["status"] = "error"
        out["reason"] = "invalid_org_id"
        return out

    if now is None:
        now = datetime.now(timezone.utc)

    payload = build_recall(int(org_id), now=now)
    out["mode"] = payload.time_context.mode

    if not force and not _silence_gate_passes(payload):
        out["reason"] = "no_signal"
        return out

    # Lazy import so the module loads cleanly in test contexts that don't
    # have prompt-time deps.
    from tools.daily_brief.prompts import build_prompt
    prompt = build_prompt(payload)

    try:
        raw, _tokens = model_call("daily_brief", prompt)
    except Exception:
        _log.warning("daily_brief: model_call raised  org=%d", org_id, exc_info=True)
        out["status"] = "error"
        out["reason"] = "model_call_failed"
        return out

    parsed = _parse_output(raw or "")
    if parsed is None:
        out["status"] = "error"
        out["reason"] = "json_parse_failed"
        return out

    if parsed.get("empty") is True:
        out["reason"] = "model_silent"
        return out

    if not _silence_gate_passes(payload):
        # post-check: model spoke but recall had no signal — drop it
        out["reason"] = "post_silence_gate"
        return out

    body = (parsed.get("body_markdown") or "").strip()
    if not body:
        out["reason"] = "empty_body"
        return out
    summary = (parsed.get("summary") or "").strip()[:200]
    topic = (parsed.get("topic") or "").strip()[:200]
    if not topic and payload.thread_of_day is not None:
        topic = payload.thread_of_day.title or ""
    included_ask_ids = parsed.get("included_ask_ids") or []
    if not isinstance(included_ask_ids, list):
        included_ask_ids = []

    max_chars = int(_cfg("max_body_chars", 3500) or 3500)
    if len(body) > max_chars:
        body = body[:max_chars].rstrip() + "…"

    insight_id = create_insight(
        org_id=int(org_id),
        title=summary or (topic or "Daily brief"),
        body_markdown=body,
        topic=topic,
        summary=summary,
        trigger=TRIGGER_DAILY_BRIEF,
        sources=_gather_sources(payload),
    )
    out["insight_id"] = insight_id

    msg_id = _post_to_home(int(org_id), body, insight_id, summary)
    out["home_message_id"] = msg_id

    out["produced"] = True
    out["included_ask_ids"] = [int(x) for x in included_ask_ids if isinstance(x, (int, str)) and str(x).isdigit()]
    out["body_chars"] = len(body)
    _log.info(
        "daily_brief produced  org=%d mode=%s insight=%s msg=%s body_chars=%d",
        org_id, out["mode"], insight_id, msg_id, len(body),
    )
    return out


# Re-export for symmetry with other producers
__all__ = ["run_daily_brief", "TRIGGER_DAILY_BRIEF"]
