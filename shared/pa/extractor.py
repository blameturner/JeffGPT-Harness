from __future__ import annotations

import json
import logging

from infra.config import is_feature_enabled
from shared.models import model_call
from shared.pa.memory import (
    create_open_loop,
    find_loop_by_text,
    resolve_loop,
    upsert_warm_topic,
    upsert_user_fact,
    LOOP_INTENT_TODO,
    LOOP_INTENT_EVENT,
    LOOP_INTENT_DECISION,
    LOOP_INTENT_WAITING,
    LOOP_INTENT_WORRY,
    TOPIC_KIND_TASK,
    TOPIC_KIND_INTEREST,
    TOPIC_KIND_STATED,
    FACT_CONFIDENCE_OBSERVED,
    FACT_CONFIDENCE_STATED,
)

# Expected config.json role for "pa_extractor":
#   role: exp_rwkv_r, temperature: 0.2, max_tokens: 500, max_input_chars: 6000

_log = logging.getLogger("pa.extractor")

_MAX_CHARS = 4000

_INTENT_MAP: dict[str, str] = {
    "todo": LOOP_INTENT_TODO,
    "event": LOOP_INTENT_EVENT,
    "decision_pending": LOOP_INTENT_DECISION,
    "waiting_on_other": LOOP_INTENT_WAITING,
    "worry": LOOP_INTENT_WORRY,
}

_TOPIC_KIND_MAP: dict[str, str] = {
    "task": TOPIC_KIND_TASK,
    "interest": TOPIC_KIND_INTEREST,
    "user_stated": TOPIC_KIND_STATED,
}

_CONFIDENCE_MAP: dict[str, str] = {
    "stated": FACT_CONFIDENCE_STATED,
    "observed": FACT_CONFIDENCE_OBSERVED,
}

_PROMPT_TEMPLATE = """You track personal-assistant memory from a chat between USER and ASSISTANT.
Output STRICT JSON only, no prose, no markdown fences:

{{
  "new_loops": [
    {{"text": "<what the user committed to in <= 12 words>",
     "intent": "todo|event|decision_pending|waiting_on_other|worry",
     "when_hint": "<freeform: today|this arvo|before Fri|next week|>"}}
  ],
  "resolved_loops": [
    {{"match_text": "<phrase identifying the prior loop>",
     "note": "<short resolution note>"}}
  ],
  "new_facts": [
    {{"kind": "routine|preference|project|relationship|constraint|interest",
     "key": "<short slug, e.g. morning_routine, project:jeff, prefers_brief>",
     "value": "<the fact itself>",
     "confidence": "stated|observed"}}
  ],
  "topics": [
    {{"phrase": "<named entity or concrete topic, <= 5 words>",
     "kind": "task|interest|user_stated"}}
  ]
}}

RULES:
- Do NOT invent commitments the user did not make. Prefer empty arrays.
- Skip greetings, thanks, generic chatter.
- A fact must be persistent (not "I'm tired today"). Skip ephemeral.
- Topics: concrete nouns/projects only; skip pronouns, generic words.
- All arrays may be empty.

USER:
{user_message}

A:
{assistant_reply}
"""


def _parse_json(raw: str) -> dict | None:
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
    if not isinstance(obj, dict):
        return None
    return obj


def _as_list(obj: dict, key: str) -> list[dict]:
    val = obj.get(key)
    if not isinstance(val, list):
        return []
    return [x for x in val if isinstance(x, dict)]


def extract_and_persist(
    org_id: int,
    user_message: str,
    assistant_reply: str,
    source_message_id: int | None = None,
) -> dict:
    summary: dict = {
        "loops_created": 0,
        "loops_resolved": 0,
        "facts_written": 0,
        "topics_boosted": 0,
        "new_topic_ids": [],
        "error": "",
    }
    try:
        if not is_feature_enabled("pa"):
            return summary
        if user_message is None or len(user_message.strip()) < 3:
            return summary
        if not assistant_reply:
            return summary

        user_trimmed = user_message[:_MAX_CHARS]
        reply_trimmed = assistant_reply[:_MAX_CHARS]
        prompt = _PROMPT_TEMPLATE.format(
            user_message=user_trimmed,
            assistant_reply=reply_trimmed,
        )

        raw, _tokens = model_call("pa_extractor", prompt)
        parsed = _parse_json(raw or "")
        if parsed is None:
            _log.warning("pa_extractor: failed to parse JSON output")
            summary["error"] = "json_parse_failed"
            return summary

        for loop in _as_list(parsed, "new_loops"):
            text = str(loop.get("text", "")).strip()
            if not text:
                continue
            intent = _INTENT_MAP.get(str(loop.get("intent", "")).strip(), LOOP_INTENT_TODO)
            when_hint = str(loop.get("when_hint", "") or "")
            res = create_open_loop(
                org_id,
                text,
                intent=intent,
                when_hint=when_hint,
                source_message_id=source_message_id,
            )
            if res:
                summary["loops_created"] += 1

        for item in _as_list(parsed, "resolved_loops"):
            match_text = str(item.get("match_text", "")).strip()
            if not match_text:
                continue
            note = str(item.get("note", "") or "")
            existing = find_loop_by_text(org_id, match_text)
            if not existing:
                continue
            loop_id = existing.get("Id") or existing.get("id")
            if loop_id is None:
                continue
            try:
                resolve_loop(int(loop_id), note=note)
                summary["loops_resolved"] += 1
            except (ValueError, TypeError):
                continue

        for fact in _as_list(parsed, "new_facts"):
            kind = str(fact.get("kind", "")).strip()
            key = str(fact.get("key", "")).strip()
            value = str(fact.get("value", "")).strip()
            if not kind or not key or not value:
                continue
            confidence = _CONFIDENCE_MAP.get(
                str(fact.get("confidence", "")).strip(),
                FACT_CONFIDENCE_OBSERVED,
            )
            source_ref = f"msg:{source_message_id}" if source_message_id else ""
            res = upsert_user_fact(
                org_id,
                kind=kind,
                key=key,
                value=value,
                confidence=confidence,
                source_ref=source_ref,
            )
            if res:
                summary["facts_written"] += 1

        for topic in _as_list(parsed, "topics"):
            phrase = str(topic.get("phrase", "")).strip()
            if not phrase:
                continue
            kind = _TOPIC_KIND_MAP.get(
                str(topic.get("kind", "")).strip(),
                TOPIC_KIND_INTEREST,
            )
            res = upsert_warm_topic(org_id, phrase, kind=kind)
            if res:
                summary["topics_boosted"] += 1
                tid = res.get("Id") or res.get("id")
                if tid is not None:
                    try:
                        summary["new_topic_ids"].append(int(tid))
                    except (TypeError, ValueError):
                        pass

        return summary
    except Exception as e:
        _log.exception("pa_extractor: unexpected failure")
        summary["error"] = str(e)
        return summary
