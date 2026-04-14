import json
import logging
import re

from infra.config import is_feature_enabled
from shared.models import model_call

_log = logging.getLogger("research_planner")

DEFAULT_MAX_QUERIES = 20


def _strip_fence(raw: str) -> str:
    s = raw.strip()
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
    # truncated — best-effort close
    tail = s[start:]
    if in_str:
        tail += '"'
    tail = re.sub(r",\s*$", "", tail)
    tail += "]" * arr_depth
    tail += "}" * obj_depth
    return tail


def _generate_plan(topic: str, max_queries: int = DEFAULT_MAX_QUERIES) -> dict:
    prompt = f"""Given this research topic: "{topic}"

Generate a JSON research plan with:
1. "hypotheses": 2-3 specific hypotheses to investigate
2. "sub_topics": 4-6 specific sub-topics to research
3. "queries": {max_queries} specific, search-engine-friendly queries (max {max_queries})
4. "schema": JSON schema defining what data to extract (field name + type: numeric, text, date, percent)

Respond with ONLY valid JSON, no explanation, no markdown."""

    try:
        result, _ = model_call("research_planner", prompt)
    except Exception as e:
        _log.warning("plan generation failed  topic=%s  error=%s", topic[:40], e)
        return {"error": str(e)[:200]}

    if not result:
        return {"error": "empty model response"}

    candidate = _extract_json_object(result)
    if not candidate:
        _log.warning("plan parse failed  topic=%s  no JSON object", topic[:40])
        return {"error": "no JSON object in response", "raw": result[:500]}

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        _log.warning(
            "plan parse failed  topic=%s  error=%s  chars=%d",
            topic[:40], e, len(result),
        )
        return {"error": str(e)[:200], "raw": result[:500]}


def create_research_plan(topic: str, org_id: int = 0) -> dict:
    from infra.nocodb_client import NocodbClient

    if not is_feature_enabled("research_planner"):
        return {"status": "disabled", "error": "research_planner feature disabled"}

    max_queries = is_feature_enabled("research_max_queries") or DEFAULT_MAX_QUERIES

    plan = _generate_plan(topic, max_queries)
    if "error" in plan:
        return {"status": "failed", "error": plan["error"]}

    client = NocodbClient()
    try:
        row = client._post("research_plans", {
            "org_id": org_id,
            "topic": topic,
            "hypotheses": json.dumps(plan.get("hypotheses", [])),
            "sub_topics": json.dumps(plan.get("sub_topics", [])),
            "queries": json.dumps(plan.get("queries", [])),
            "schema": json.dumps(plan.get("schema", {})),
            "status": "generating"
        })
        return {"status": "created", "plan_id": row.get("Id")}
    except Exception as e:
        _log.warning("save plan failed  topic=%s  error=%s", topic[:40], e)
        return {"status": "failed", "error": str(e)[:200]}


def get_next_plan() -> dict | None:
    from infra.nocodb_client import NocodbClient

    client = NocodbClient()
    try:
        data = client._get("research_plans", params={
            "where": "(status,eq,generating)",
            "limit": 1
        })
        rows = data.get("list", [])
        return rows[0] if rows else None
    except Exception:
        return None


def complete_plan(plan_id: int, status: str = "complete") -> None:
    from infra.nocodb_client import NocodbClient

    client = NocodbClient()
    try:
        client._patch("research_plans", plan_id, {"status": status})
    except Exception:
        _log.warning("complete plan failed  id=%d", plan_id)