import json
import logging
import re
import time
import concurrent.futures as _futures

from infra.config import get_feature
from shared.models import model_call

_log = logging.getLogger("research_planner")

DEFAULT_MAX_QUERIES = 20
DEFAULT_PLANNER_TIMEOUT_S = 1200
DEFAULT_PLANNER_RETRY_ATTEMPTS = 3
DEFAULT_PLANNER_RETRY_BACKOFF_S = 4


def _planner_timeout_s() -> int:
    raw = get_feature("research", "planner_timeout_s", DEFAULT_PLANNER_TIMEOUT_S)
    try:
        val = int(raw)
        return val if val > 0 else DEFAULT_PLANNER_TIMEOUT_S
    except Exception:
        return DEFAULT_PLANNER_TIMEOUT_S


def _planner_retry_attempts() -> int:
    raw = get_feature("research", "planner_retry_attempts", DEFAULT_PLANNER_RETRY_ATTEMPTS)
    try:
        val = int(raw)
        return val if val > 0 else DEFAULT_PLANNER_RETRY_ATTEMPTS
    except Exception:
        return DEFAULT_PLANNER_RETRY_ATTEMPTS


def _planner_retry_backoff_s() -> float:
    raw = get_feature("research", "planner_retry_backoff_s", DEFAULT_PLANNER_RETRY_BACKOFF_S)
    try:
        val = float(raw)
        return val if val >= 0 else DEFAULT_PLANNER_RETRY_BACKOFF_S
    except Exception:
        return DEFAULT_PLANNER_RETRY_BACKOFF_S


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


def _as_string_list(value) -> list[str]:
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        s = str(item or "").strip()
        if s:
            out.append(s)
    return out


def _normalize_plan_payload(data: dict, max_queries: int) -> dict:
    hypotheses = _as_string_list(data.get("hypotheses") or [])
    sub_topics = _as_string_list(data.get("sub_topics") or [])
    queries = _as_string_list(data.get("queries") or [])[:max_queries]
    schema = data.get("schema")
    if not isinstance(schema, dict):
        schema = {}
    if not queries:
        return {"error": "planner produced no queries"}
    return {
        "hypotheses": hypotheses,
        "sub_topics": sub_topics,
        "queries": queries,
        "schema": schema,
    }


def _generate_plan(topic: str, max_queries: int = DEFAULT_MAX_QUERIES) -> dict:
    min_queries = 10 if max_queries >= 10 else max_queries
    prompt = f"""You are a research planning engine.

TOPIC:
{topic}

Return ONLY one valid JSON object with EXACTLY these top-level keys:
- "hypotheses"
- "sub_topics"
- "queries"
- "schema"

Required output contract:
1) "hypotheses": array of 2-4 concise, testable hypotheses.
2) "sub_topics": array of 4-8 specific research sub-topics.
3) "queries": array of {min_queries}-{max_queries} unique, high-signal search queries.
   - No generic queries; each should include concrete entities/metrics/angles.
   - Prefer query phrasing that can find primary sources, statistics, and recent evidence.
   - Never return an empty queries array.
4) "schema": object where each key is a field to extract and each value is one of:
   "numeric", "text", "date", "percent".
   - Include 6-12 fields that are useful to evaluate the hypotheses.

Formatting rules:
- Output raw JSON only (no markdown, no backticks, no prose).
- No trailing commas, comments, or extra keys.
- If uncertain, still return best-effort concrete hypotheses, sub-topics, and queries.

Example shape (structure only, not content):
{{
  "hypotheses": ["...", "..."],
  "sub_topics": ["...", "..."],
  "queries": ["...", "..."],
  "schema": {{"field_name": "text"}}
}}"""

    timeout_s = _planner_timeout_s()
    attempts = _planner_retry_attempts()
    backoff_s = _planner_retry_backoff_s()

    def _run():
        return model_call("research_planner", prompt)

    last_error = "unknown planner error"
    last_raw = ""

    for attempt in range(1, attempts + 1):
        ex = _futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="research-plan")
        try:
            fut = ex.submit(_run)
            try:
                result, _ = fut.result(timeout=timeout_s)
            except _futures.TimeoutError:
                last_error = f"planner timeout after {timeout_s}s"
                _log.warning(
                    "planner timeout  attempt=%d/%d  topic=%s",
                    attempt, attempts, topic[:40],
                )
                continue
            except Exception as e:
                last_error = str(e)[:200]
                _log.warning(
                    "plan generation failed  attempt=%d/%d  topic=%s  error=%s",
                    attempt, attempts, topic[:40], e,
                )
                continue
        finally:
            ex.shutdown(wait=False)

        if not result:
            last_error = "empty model response"
            _log.warning(
                "planner empty response  attempt=%d/%d  topic=%s",
                attempt, attempts, topic[:40],
            )
        else:
            candidate = _extract_json_object(result)
            last_raw = result[:500]
            if not candidate:
                last_error = "no JSON object in response"
                _log.warning(
                    "planner parse failed (no JSON object)  attempt=%d/%d  topic=%s",
                    attempt, attempts, topic[:40],
                )
            else:
                try:
                    parsed = json.loads(candidate)
                    normalized = _normalize_plan_payload(parsed, max_queries)
                    if "error" not in normalized:
                        return normalized
                    last_error = str(normalized.get("error") or "invalid planner payload")
                    _log.warning(
                        "planner payload invalid  attempt=%d/%d  topic=%s  error=%s",
                        attempt, attempts, topic[:40], last_error,
                    )
                except json.JSONDecodeError as e:
                    last_error = f"invalid JSON: {str(e)[:160]}"
                    _log.warning(
                        "plan parse failed  attempt=%d/%d  topic=%s  error=%s  chars=%d",
                        attempt, attempts, topic[:40], e, len(result),
                    )

        if attempt < attempts and backoff_s > 0:
            time.sleep(backoff_s)

    out = {"error": f"planner failed after {attempts} attempts: {last_error}"}
    if last_raw:
        out["raw"] = last_raw
    return out


def create_research_plan(topic: str, org_id: int = 0) -> dict:
    """Create a shell research_plans row and queue the planner job.
    Returns immediately with the plan_id so the UI can poll for status.
    """
    from infra.nocodb_client import NocodbClient
    from workers.tool_queue import get_tool_queue

    if not get_feature("research", "planner_enabled", True):
        return {"status": "disabled", "error": "research_planner feature disabled"}
    if int(org_id or 0) <= 0:
        return {"status": "failed", "error": "invalid_org_id"}

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

    _log.error("Tool queue unavailable for research planner plan_id=%d", plan_id)
    client._patch("research_plans", plan_id, {
        "status": "failed",
        "error_message": "tool_queue_unavailable",
    })
    return {"status": "failed", "error": "tool_queue_unavailable", "plan_id": plan_id}


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
        _log.error("Tool queue unavailable for research agent enqueue plan_id=%d", plan_id)
        client._patch("research_plans", plan_id, {
            "status": "failed",
            "error_message": "tool_queue_unavailable",
        })
        return {"status": "failed", "error": "tool_queue_unavailable", "plan_id": plan_id}

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