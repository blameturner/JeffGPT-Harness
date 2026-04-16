import json
import logging
import re
import concurrent.futures as _futures

from infra.config import get_feature
from shared.models import model_call

_log = logging.getLogger("research_planner")

DEFAULT_MAX_QUERIES = 20
DEFAULT_PLANNER_TIMEOUT_S = 1200


def _planner_timeout_s() -> int:
    raw = get_feature("research", "planner_timeout_s", DEFAULT_PLANNER_TIMEOUT_S)
    try:
        val = int(raw)
        return val if val > 0 else DEFAULT_PLANNER_TIMEOUT_S
    except Exception:
        return DEFAULT_PLANNER_TIMEOUT_S


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

    timeout_s = _planner_timeout_s()

    def _run():
        return model_call("research_planner", prompt)

    ex = _futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="research-plan")
    try:
        fut = ex.submit(_run)
        try:
            result, _ = fut.result(timeout=timeout_s)
        except _futures.TimeoutError:
            _log.warning("planner timeout after %ds  topic=%s", timeout_s, topic[:40])
            return {"error": f"planner timeout after {timeout_s}s"}
        except Exception as e:
            _log.warning("plan generation failed  topic=%s  error=%s", topic[:40], e)
            return {"error": str(e)[:200]}
    finally:
        ex.shutdown(wait=False)

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
    """Create a shell research_plans row and queue the planner job.
    Returns immediately with the plan_id so the UI can poll for status.
    """
    from infra.nocodb_client import NocodbClient
    from workers.tool_queue import get_tool_queue

    if not get_feature("research", "planner_enabled", True):
        return {"status": "disabled", "error": "research_planner feature disabled"}

    client = NocodbClient()
    try:
        row = client._post("research_plans", {
            "org_id": org_id,
            "topic": topic,
            "hypotheses": "[]",
            "sub_topics": "[]",
            "queries": "[]",
            "schema": "{}",
            "iterations": 0,
            "status": "pending",
        })
        plan_id = row.get("Id")
    except Exception as e:
        _log.warning("shell plan save failed  topic=%s  error=%s", topic[:40], e)
        return {"status": "failed", "error": str(e)[:200]}

    tq = get_tool_queue()
    if tq:
        job_id = tq.submit(
            "research_planner",
            {"plan_id": plan_id, "org_id": org_id},
            source="research_api",
            priority=3,
            org_id=org_id,
        )
        _log.info("Queued research planner job %s for plan_id %d", job_id, plan_id)
        return {"status": "queued", "plan_id": plan_id, "job_id": job_id}

    _log.warning("Tool queue not available, running planner synchronously for plan_id %d", plan_id)
    return run_research_planner_job(plan_id)


def run_research_planner_job(plan_id: int) -> dict:
    """Planner tool-queue handler: generate queries/schema for an existing row, then queue the agent."""
    from infra.nocodb_client import NocodbClient
    from workers.tool_queue import get_tool_queue

    if not get_feature("research", "planner_enabled", True):
        return {"status": "disabled", "error": "research_planner feature disabled"}

    max_queries = get_feature("research", "max_queries", DEFAULT_MAX_QUERIES)

    client = NocodbClient()
    plan_row = client._get("research_plans", params={"where": f"(Id,eq,{plan_id})", "limit": 1})
    plan = plan_row.get("list", [])[0] if plan_row.get("list") else None
    if not plan:
        return {"status": "not_found", "plan_id": plan_id}

    topic = plan.get("topic", "")
    if not topic:
        client._patch("research_plans", plan_id, {"status": "failed", "error_message": "no topic"})
        return {"status": "failed", "error": "no topic", "plan_id": plan_id}

    generated = _generate_plan(topic, max_queries)
    if "error" in generated:
        client._patch("research_plans", plan_id, {
            "status": "failed",
            "error_message": str(generated.get("error"))[:500],
        })
        return {"status": "failed", "error": generated["error"], "plan_id": plan_id}

    queries = (generated.get("queries") or [])[:max_queries]

    try:
        client._patch("research_plans", plan_id, {
            "hypotheses": json.dumps(generated.get("hypotheses", [])),
            "sub_topics": json.dumps(generated.get("sub_topics", [])),
            "queries": json.dumps(queries),
            "schema": json.dumps(generated.get("schema", {})),
            "status": "generating",
        })
    except Exception as e:
        _log.warning("plan patch failed  id=%d  error=%s", plan_id, e)
        client._patch("research_plans", plan_id, {"status": "failed", "error_message": str(e)[:500]})
        return {"status": "failed", "error": str(e)[:200], "plan_id": plan_id}

    tq = get_tool_queue()
    if tq:
        plan_org_id = int(plan.get("org_id") or 0)
        job_id = tq.submit(
            "research_agent",
            {"plan_id": plan_id, "org_id": plan_org_id},
            source="research_planner",
            priority=3,
            org_id=plan_org_id,
        )
        _log.info("Queued research agent job %s for plan_id %d", job_id, plan_id)
    else:
        _log.warning("Tool queue not available, research agent will not run automatically for plan_id %d", plan_id)

    return {"status": "generating", "plan_id": plan_id, "query_count": len(queries)}


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


def complete_plan(plan_id: int, status: str = "completed") -> None:
    from infra.nocodb_client import NocodbClient

    client = NocodbClient()
    try:
        client._patch("research_plans", plan_id, {"status": status})
    except Exception:
        _log.warning("complete plan failed  id=%d", plan_id)