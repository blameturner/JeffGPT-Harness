import json
import logging

from infra.config import is_feature_enabled
from shared.models import model_call

_log = logging.getLogger("research_planner")

DEFAULT_MAX_QUERIES = 20


def _generate_plan(topic: str, max_queries: int = DEFAULT_MAX_QUERIES) -> dict:
    prompt = f"""Given this research topic: "{topic}"

Generate a JSON research plan with:
1. "hypotheses": 2-3 specific hypotheses to investigate
2. "sub_topics": 4-6 specific sub-topics to research
3. "queries": {max_queries} specific, search-engine-friendly queries (max {max_queries})
4. "schema": JSON schema defining what data to extract (field name + type: numeric, text, date, percent)

Respond with ONLY valid JSON, no explanation."""

    try:
        result, _ = model_call("tool_planner", prompt, max_tokens=800, temperature=0.3)
        plan = json.loads(result)
        return plan
    except json.JSONDecodeError as e:
        _log.warning("plan parse failed  topic=%s  error=%s", topic[:40], e)
        return {"error": str(e)[:200]}
    except Exception as e:
        _log.warning("plan generation failed  topic=%s  error=%s", topic[:40], e)
        return {"error": str(e)[:200]}


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