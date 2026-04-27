"""Extract first-person commitments from the assistant's reply.

Called after each chat turn alongside the user-side extractor. Mines
phrases like "I'll check…", "let me dig into…", "I'll have that for you
tomorrow" and writes them as `pa_open_loops` rows tagged with a
`source_ref` of ``assistant_commitment:msg_<id>`` so the recall layer
can surface them separately as "things I owe you".

Lives in its own module so the user-side extractor remains tightly
scoped on user-stated intent — we don't want to conflate "user said
they'll do X" with "assistant said it'll do X".
"""
from __future__ import annotations

import json
import logging

from infra.config import is_feature_enabled
from shared.models import model_call
from shared.pa.memory import (
    LOOP_INTENT_TODO,
    create_open_loop,
)

_log = logging.getLogger("pa.assistant_extractor")

_MAX_CHARS = 4000

_PROMPT = """You read an ASSISTANT reply and extract first-person commitments
the assistant made. Things the assistant said *it* would do for the user.

Output STRICT JSON only, no prose, no markdown fences:

{{
  "commitments": [
    {{"text": "<commitment in <= 12 words, third person from the assistant>",
      "when_hint": "<freeform: tonight|by tomorrow|this week|>"}}
  ]
}}

RULES:
- Only first-person assistant commitments: "I'll …", "let me …", "I'll have
  …", "I'll check / look into / draft / pull / verify / find …".
- Skip rhetorical "let me think" or "let me know if …" (the second is a
  request to the user, not a commitment).
- Skip generic acknowledgements ("I understand", "got it").
- Empty array is valid and common — most replies have no commitments.
- Capture at most 3 commitments per reply.

ASSISTANT REPLY:
{assistant_reply}
"""


def _parse(raw: str) -> list[dict]:
    if not raw:
        return []
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return []
    try:
        obj = json.loads(raw[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    items = obj.get("commitments") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return []
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        text = str(it.get("text", "")).strip()
        if not text:
            continue
        when = str(it.get("when_hint", "") or "").strip()
        out.append({"text": text, "when_hint": when})
    return out[:3]


def extract_assistant_commitments(
    org_id: int,
    assistant_reply: str,
    source_message_id: int | None,
) -> dict:
    out: dict = {"commitments_written": 0, "error": ""}
    try:
        if not is_feature_enabled("pa"):
            return out
        reply = (assistant_reply or "").strip()
        if len(reply) < 10:
            return out
        prompt = _PROMPT.format(assistant_reply=reply[:_MAX_CHARS])
        raw, _tokens = model_call("pa_assistant_commitments", prompt)
        items = _parse(raw or "")
        if not items:
            return out
        for it in items:
            text = it["text"]
            when = it["when_hint"]
            source_ref = (
                f"assistant_commitment:msg_{source_message_id}"
                if source_message_id else "assistant_commitment:unknown"
            )
            try:
                row = create_open_loop(
                    int(org_id),
                    text=text,
                    intent=LOOP_INTENT_TODO,
                    when_hint=when,
                    source_message_id=source_message_id,
                )
                if row:
                    # tag the row with our source_ref via memory.upsert pattern —
                    # create_open_loop doesn't accept source_ref directly, so we
                    # patch immediately. Best-effort.
                    _patch_source_ref(row, source_ref)
                    out["commitments_written"] += 1
            except Exception:
                _log.debug("assistant_extractor: create_open_loop failed", exc_info=True)
        return out
    except Exception as e:
        _log.warning("assistant_extractor: unexpected failure", exc_info=True)
        out["error"] = str(e)
        return out


def _patch_source_ref(row: dict, source_ref: str) -> None:
    """Stamp ``source_ref`` on a freshly-created loop row."""
    try:
        from infra.config import NOCODB_TABLE_PA_OPEN_LOOPS
        from infra.nocodb_client import NocodbClient
        client = NocodbClient()
        loop_id = int(row.get("Id") or 0)
        if loop_id <= 0:
            return
        client._patch(NOCODB_TABLE_PA_OPEN_LOOPS, loop_id, {
            "Id": loop_id,
            "source_ref": source_ref[:200],
        })
    except Exception:
        _log.debug("assistant_extractor: source_ref patch failed", exc_info=True)
