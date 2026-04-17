import json
import logging
import re
from typing import Any
from infra.config import get_feature
from shared.models import model_call

_log = logging.getLogger("research.critic")

DEFAULT_CONFIDENCE_THRESHOLD = 80


def _strip_fence(raw: str) -> str:
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = s.rstrip("`").strip()
    return s


def _extract_json_object(raw: str) -> str:
    s = _strip_fence(raw)
    start = s.find("{")
    if start < 0:
        return ""
    obj_depth = 0
    arr_depth = 0
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
            obj_depth += 1
        elif ch == "}":
            obj_depth -= 1
            if obj_depth == 0 and arr_depth == 0:
                return s[start:i + 1]
        elif ch == "[":
            arr_depth += 1
        elif ch == "]":
            arr_depth -= 1
    return ""


def _coerce_gaps(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, str]] = []
    allowed_status = {"missing", "insufficient", "superficial"}
    for item in value:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or "").strip()
        status = str(item.get("status") or "").strip().lower()
        needed = str(item.get("needed") or "").strip()
        if not field:
            continue
        if status not in allowed_status:
            status = "insufficient"
        out.append({
            "field": field,
            "status": status,
            "needed": needed or "additional evidence required",
        })
    return out


def _coerce_queries(value: Any) -> list[str]:
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        q = str(item or "").strip()
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out[:8]


def _fallback_response(topic: str, reason: str) -> dict:
    return {
        "gaps_found": [],
        "new_search_requirements": [f"{topic} latest primary sources"],
        "confidence_score": 25,
        "ready_for_completion": False,
        "notes": reason[:300],
    }


def _normalize_critic_output(topic: str, data: dict[str, Any]) -> dict:
    gaps = _coerce_gaps(data.get("gaps_found"))
    new_queries = _coerce_queries(data.get("new_search_requirements"))

    try:
        confidence = int(data.get("confidence_score", 0))
    except Exception:
        confidence = 0
    confidence = max(0, min(100, confidence))

    ready_raw = data.get("ready_for_completion")
    if isinstance(ready_raw, bool):
        ready = ready_raw
    elif isinstance(ready_raw, str):
        ready = ready_raw.strip().lower() in {"true", "1", "yes"}
    else:
        ready = False

    notes = str(data.get("notes") or "").strip()
    if not notes:
        notes = "Critic analysis completed"

    # Conservative guard: if the critic reports notable gaps or no confidence,
    # never allow completion even if ready flag was malformed/optimistic.
    if gaps or confidence < 60:
        ready = False

    if not new_queries and gaps:
        for g in gaps[:3]:
            new_queries.append(f"{topic} {g.get('field', '')} latest evidence")
    if not new_queries and not ready:
        new_queries.append(f"{topic} latest statistics and primary sources")

    return {
        "gaps_found": gaps,
        "new_search_requirements": _coerce_queries(new_queries),
        "confidence_score": confidence,
        "ready_for_completion": ready,
        "notes": notes[:500],
    }


def analyze_gaps(topic: str, content: str, schema: dict, context: str = "") -> dict:
    prompt = f"""You are a Research Critic. Analyze the draft research content for gaps and sufficiency.

TOPIC: {topic}
SCHEMA (data to extract): {json.dumps(schema)}

DRAFT CONTENT:
{content[:20000]}

CONTEXT (additional):
{context[:5000] if context else "No additional context."}

Return ONLY one valid JSON object with EXACTLY these keys:
- "gaps_found": array of objects with keys {"field","status","needed"}; status must be one of "missing","insufficient","superficial"
- "new_search_requirements": array of specific follow-up search queries
- "confidence_score": integer 0..100
- "ready_for_completion": boolean
- "notes": concise assessment string

Rules:
- No markdown, no code fences, no prose outside JSON.
- If evidence is weak or incomplete, set ready_for_completion to false.
- If gaps exist, include actionable new_search_requirements.
- Prefer conservative scoring over optimistic scoring.
"""

    try:
        function_name = str(get_feature("research", "critic_model", "research_agent") or "research_agent")
        result, _ = model_call(function_name, prompt, max_tokens=1000, temperature=0.2)
        if not result:
            return _fallback_response(topic, "critic empty response")

        candidate = _extract_json_object(result)
        if not candidate:
            return _fallback_response(topic, "critic returned no JSON object")
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            return _fallback_response(topic, "critic returned non-object JSON")
        return _normalize_critic_output(topic, parsed)
    except json.JSONDecodeError as e:
        _log.warning("critic parse failed  topic=%s  error=%s", topic[:40], e)
        return _fallback_response(topic, f"critic parse error: {e}")
    except Exception as e:
        _log.warning("critic failed  topic=%s  error=%s", topic[:40], e)
        return _fallback_response(topic, f"critic error: {e}")


def get_confidence_threshold() -> int:
    return get_feature("research", "confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)