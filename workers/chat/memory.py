"""Persistent per-conversation chat memory.

Three categories of structured items live in the `chat_memory` table:

  - fact      — slow-changing context ("Altitude Group acquires 4-6 sites/year")
  - decision  — point-in-time conclusion ("agreed on capability statement library")
  - thread    — open question still on the table ("year-to-year roadmap not yet drafted")

Items have a status: ``proposed`` (awaiting user review, written by the summariser),
``active`` (accepted), or ``rejected``. Pinned items are prepended verbatim to every
turn's system prompt — the model is forbidden to forget them.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

_log = logging.getLogger("chat.memory")

NOCODB_TABLE = "chat_memory"

CATEGORIES = ("fact", "decision", "thread")
STATUS_PROPOSED = "proposed"
STATUS_ACTIVE = "active"
STATUS_REJECTED = "rejected"

DEFAULT_TOKEN_BUDGET = 800
DEFAULT_EXTRACT_EVERY_N_TURNS = 6


def _db():
    from infra.nocodb_client import NocodbClient
    return NocodbClient()


def _table_present(db) -> bool:
    try:
        return NOCODB_TABLE in db.tables
    except Exception:
        return False


def list_items(
    conversation_id: int,
    org_id: int,
    *,
    status: str | None = None,
    category: str | None = None,
    pinned_only: bool = False,
    limit: int = 200,
) -> list[dict]:
    db = _db()
    if not _table_present(db):
        return []
    parts = [f"(conversation_id,eq,{int(conversation_id)})", f"(org_id,eq,{int(org_id)})"]
    if status:
        parts.append(f"(status,eq,{status})")
    if category:
        parts.append(f"(category,eq,{category})")
    if pinned_only:
        parts.append("(pinned,eq,1)")
    try:
        rows = db._get_paginated(NOCODB_TABLE, params={
            "where": "~and".join(parts),
            "sort": "-CreatedAt",
            "limit": limit,
        })
    except Exception:
        _log.warning("chat_memory list failed conv=%d", conversation_id, exc_info=True)
        return []
    return [_normalize_row(r) for r in rows]


def _normalize_row(r: dict) -> dict:
    row_id = r.get("Id")
    return {
        "id": row_id,
        "Id": row_id,
        "conversation_id": r.get("conversation_id"),
        "org_id": r.get("org_id"),
        "category": r.get("category") or "fact",
        "text": r.get("text") or "",
        "pinned": bool(r.get("pinned")),
        "status": r.get("status") or STATUS_ACTIVE,
        "source_message_id": r.get("source_message_id"),
        "confidence": int(r.get("confidence") or 0),
        "created_at": r.get("CreatedAt") or r.get("created_at"),
        "updated_at": r.get("UpdatedAt") or r.get("updated_at"),
        "last_seen_at": r.get("last_seen_at"),
    }


def add_item(
    conversation_id: int,
    org_id: int,
    category: str,
    text: str,
    *,
    pinned: bool = False,
    status: str = STATUS_ACTIVE,
    source_message_id: int | None = None,
    confidence: int = 0,
) -> dict | None:
    if category not in CATEGORIES:
        raise ValueError(f"invalid category: {category}")
    text = (text or "").strip()
    if not text:
        return None
    db = _db()
    if not _table_present(db):
        _log.warning("chat_memory add skipped — table missing")
        return None
    payload: dict[str, Any] = {
        "conversation_id": int(conversation_id),
        "org_id": int(org_id),
        "category": category,
        "text": text[:4000],
        "pinned": 1 if pinned else 0,
        "status": status,
        "confidence": max(0, min(100, int(confidence))),
    }
    if source_message_id is not None:
        payload["source_message_id"] = int(source_message_id)
    try:
        row = db._post(NOCODB_TABLE, payload)
        return _normalize_row(row)
    except Exception:
        _log.error("chat_memory add failed conv=%d", conversation_id, exc_info=True)
        return None


def update_item(item_id: int, **changes) -> dict | None:
    db = _db()
    if not _table_present(db):
        return None
    patch: dict[str, Any] = {"Id": int(item_id)}
    if "text" in changes and changes["text"] is not None:
        patch["text"] = str(changes["text"])[:4000]
    if "category" in changes and changes["category"] in CATEGORIES:
        patch["category"] = changes["category"]
    if "pinned" in changes and changes["pinned"] is not None:
        patch["pinned"] = 1 if changes["pinned"] else 0
    if "status" in changes and changes["status"] in (STATUS_PROPOSED, STATUS_ACTIVE, STATUS_REJECTED):
        patch["status"] = changes["status"]
    if "confidence" in changes and changes["confidence"] is not None:
        patch["confidence"] = max(0, min(100, int(changes["confidence"])))
    try:
        row = db._patch(NOCODB_TABLE, int(item_id), patch)
        return _normalize_row(row) if row else None
    except Exception:
        _log.error("chat_memory update failed id=%d", item_id, exc_info=True)
        return None


def delete_item(item_id: int) -> bool:
    db = _db()
    if not _table_present(db):
        return False
    try:
        db._delete(NOCODB_TABLE, int(item_id))
        return True
    except Exception:
        _log.error("chat_memory delete failed id=%d", item_id, exc_info=True)
        return False


def format_for_prompt(items: list[dict], *, token_budget: int = DEFAULT_TOKEN_BUDGET) -> str:
    """Render pinned memory items as a single system block.

    Token budget is approximate (chars/4). Items are ordered: facts first, then
    decisions, then threads — matches how a reader would skim. If the budget is
    exceeded, lower-priority items (newer threads first) are dropped silently.
    """
    if not items:
        return ""

    by_cat = {"fact": [], "decision": [], "thread": []}
    for it in items:
        cat = it.get("category") or "fact"
        if cat in by_cat:
            by_cat[cat].append(it)

    char_budget = max(200, token_budget * 4)
    lines: list[str] = ["[chat_memory] These items are established context for this conversation. Treat them as ground truth; refer to them by name when relevant; do not contradict them without flagging."]

    def _emit(category_label: str, rows: list[dict]) -> int:
        if not rows:
            return 0
        used = 0
        section = [f"\n{category_label}:"]
        for it in rows:
            line = f"- {it.get('text', '').strip()}"
            section.append(line)
            used += len(line) + 1
        block = "\n".join(section)
        lines.append(block)
        return used

    spent = 0
    spent += _emit("Facts", by_cat["fact"])
    if spent < char_budget:
        spent += _emit("Decisions", by_cat["decision"])
    if spent < char_budget:
        spent += _emit("Open threads", by_cat["thread"])

    out = "\n".join(lines)
    if len(out) > char_budget:
        out = out[:char_budget] + "\n[...memory truncated to fit token budget...]"
    return out


# ---- structured extraction --------------------------------------------------

_EXTRACTION_PROMPT = """\
You are extracting durable context from a conversation. Read the messages below \
and produce a STRUCTURED JSON object capturing what should be remembered for \
future turns.

Three categories:
- "facts": stable, slow-changing context about the user, their work, people, \
projects, constraints, or preferences. Each item is one sentence, factual, \
free of opinion. Examples: "User's company is Altitude Group, a property \
group acquiring 4-6 sites per year." / "Owner is anti-systems and has \
abandoned Monday.com adoption attempts."
- "decisions": specific conclusions reached together in the conversation. \
Include date if present in the text. Example: "Agreed to focus on \
capability-statement library as productisation play."
- "threads": questions or topics raised but not resolved. Example: \
"Year-to-year roadmap structure not yet drafted."

OUTPUT CONTRACT:
Return ONLY a single JSON object with this exact shape, no prose:
{
  "facts": [{"text": "...", "confidence": 0-100}],
  "decisions": [{"text": "...", "confidence": 0-100}],
  "threads": [{"text": "...", "confidence": 0-100}]
}

Rules:
- Only include items with concrete grounding in the messages.
- Skip transient details (greetings, single-turn questions already answered).
- Skip items already covered by EXISTING MEMORY below — only return new or \
materially-changed items.
- Keep each item under 200 characters.
- Maximum 6 items per category.
- If a category has nothing, return an empty array.
- Do not invent facts not in the conversation.

EXISTING MEMORY (do not duplicate):
{existing}

CONVERSATION TO EXTRACT FROM:
{conversation}
"""


def _strip_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = s.rstrip("`").strip()
    return s


def _extract_json_object(s: str) -> str:
    s = _strip_fence(s)
    start = s.find("{")
    if start < 0:
        return ""
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return ""


def extract_structured_delta(
    older_text: str,
    existing_items: list[dict],
) -> dict | None:
    """Run the structured-extraction LLM call. Returns dict with three lists or None."""
    if not older_text or not older_text.strip():
        return None

    existing_summary = ""
    if existing_items:
        bullets = []
        for it in existing_items[-30:]:
            bullets.append(f"- [{it.get('category')}] {it.get('text', '')[:200]}")
        existing_summary = "\n".join(bullets)
    else:
        existing_summary = "(none)"

    prompt = _EXTRACTION_PROMPT.replace("{existing}", existing_summary)
    prompt = prompt.replace("{conversation}", older_text[:12000])

    try:
        from shared.models import model_call
        raw, _tokens = model_call("chat_memory_extract", prompt, temperature=0.2)
    except Exception:
        _log.warning("chat_memory extract: model_call failed", exc_info=True)
        return None

    if not raw:
        return None

    candidate = _extract_json_object(raw)
    if not candidate:
        _log.info("chat_memory extract: no JSON object in response")
        return None
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        _log.info("chat_memory extract: JSON parse failed")
        return None

    out: dict[str, list[dict]] = {"fact": [], "decision": [], "thread": []}
    for src_key, dst_key in (("facts", "fact"), ("decisions", "decision"), ("threads", "thread")):
        items = data.get(src_key) or []
        if not isinstance(items, list):
            continue
        for it in items[:6]:
            if not isinstance(it, dict):
                continue
            text = str(it.get("text") or "").strip()
            if not text or len(text) > 400:
                continue
            try:
                conf = max(0, min(100, int(it.get("confidence") or 0)))
            except (ValueError, TypeError):
                conf = 0
            out[dst_key].append({"text": text, "confidence": conf})

    return out


def persist_extracted_delta(
    conversation_id: int,
    org_id: int,
    delta: dict,
    *,
    source_message_id: int | None = None,
) -> int:
    """Write extracted items as ``status="proposed"`` so the user can review.

    Returns the count persisted.
    """
    if not delta:
        return 0
    persisted = 0
    for category, items in delta.items():
        if category not in CATEGORIES:
            continue
        for it in items:
            row = add_item(
                conversation_id=conversation_id,
                org_id=org_id,
                category=category,
                text=it.get("text", ""),
                pinned=False,
                status=STATUS_PROPOSED,
                source_message_id=source_message_id,
                confidence=it.get("confidence", 0),
            )
            if row:
                persisted += 1
    return persisted


def get_pinned_for_prompt(
    conversation_id: int,
    org_id: int,
    *,
    token_budget: int | None = None,
) -> str:
    """Convenience helper used at payload build time. Returns empty string if
    the table doesn't exist or there are no pinned active items."""
    items = list_items(
        conversation_id=conversation_id,
        org_id=org_id,
        status=STATUS_ACTIVE,
        pinned_only=True,
    )
    if not items:
        return ""
    return format_for_prompt(items, token_budget=token_budget or DEFAULT_TOKEN_BUDGET)
