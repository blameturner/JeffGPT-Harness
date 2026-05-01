import json
import logging
import re
import time
import concurrent.futures as _futures

from infra.config import get_feature
from shared.models import model_call

_log = logging.getLogger("research_planner")

DEFAULT_MAX_QUERIES = 20
DEFAULT_PLANNER_TIMEOUT_S = 3600
DEFAULT_PLANNER_RETRY_ATTEMPTS = 5
DEFAULT_PLANNER_RETRY_BACKOFF_S = 8


def _fallback_plan(topic: str, max_queries: int) -> dict:
    """Deterministic plan used when the LLM planner fails or times out.

    Produces a usable {hypotheses, sub_topics, queries, schema} from the
    topic itself so the research agent can still go to search instead of
    wasting the cycle. Quality is lower than an LLM plan, but signal > 0
    is much better than the previous outcome (a hard fail).
    """
    topic_clean = (topic or "").strip()
    if not topic_clean:
        return {"error": "empty topic for fallback plan"}
    n = max(4, min(max_queries, 10))
    queries = [
        f'"{topic_clean}"',
        f'{topic_clean} overview',
        f'{topic_clean} 2026',
        f'{topic_clean} alternatives',
        f'{topic_clean} comparison',
        f'{topic_clean} pricing',
        f'{topic_clean} pros and cons',
        f'{topic_clean} recent news',
        f'{topic_clean} tradeoffs',
        f'{topic_clean} architecture',
    ][:n]
    return {
        "hypotheses": [
            f"{topic_clean} has clear strengths and weaknesses worth surfacing.",
            f"There are concrete alternatives to {topic_clean} the user should know.",
            f"{topic_clean} has notable recent developments in the last 12 months.",
        ],
        "sub_topics": [
            f"What {topic_clean} actually is",
            f"How {topic_clean} compares to alternatives",
            f"Recent changes to {topic_clean}",
            f"Practical tradeoffs of {topic_clean}",
        ],
        "queries": queries,
        "schema": {
            "name": "text",
            "vendor": "text",
            "release_year": "numeric",
            "pricing_model": "text",
            "primary_use_case": "text",
            "competitor_count": "numeric",
            "recent_change_summary": "text",
        },
        "_fallback": True,
    }


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


def _clean_json_text(s: str) -> str:
    """Forgiving cleanup for local-model JSON output.

    Local CPU-bound models routinely emit:
      - trailing commas before } or ]
      - smart quotes (curly) instead of straight quotes
      - single-quoted strings instead of double-quoted
      - unescaped newlines inside strings (we leave alone — handled by parser)
    This pass catches the easy ones.
    """
    if not s:
        return s
    # smart quotes → straight
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("‘", "'").replace("’", "'")
    # trailing commas: ,} or ,]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def _salvage_queries(raw: str) -> list[str]:
    """Last-resort: pull just the queries array out of malformed model output.

    Triggered when full JSON parse fails. Looks for ``"queries"`` followed by
    a [ ... ] block and extracts the string elements. Even a half-formed
    response usually has the queries section intact, and queries are the
    only thing the research agent strictly needs to proceed.
    """
    if not raw:
        return []
    s = _clean_json_text(_strip_fence(raw))
    # Find the queries array — try a few common spellings local models produce.
    for key_re in (r'"queries"\s*:\s*\[', r"'queries'\s*:\s*\[", r"queries\s*:\s*\["):
        m = re.search(key_re, s, re.IGNORECASE)
        if not m:
            continue
        start = m.end()
        depth = 1
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
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    inner = s[start:i]
                    # Pull every quoted string out of the array
                    items = re.findall(r'"((?:[^"\\]|\\.)*)"', inner)
                    if not items:
                        items = re.findall(r"'((?:[^'\\]|\\.)*)'", inner)
                    return [it.strip() for it in items if it.strip()]
        # unclosed — try whatever we got so far
        inner = s[start:]
        items = re.findall(r'"((?:[^"\\]|\\.)*)"', inner)
        return [it.strip() for it in items if it.strip()][:20]
    return []


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


def _topic_keywords(topic: str) -> set[str]:
    kws = re.findall(r"[a-zA-Z0-9][\w\-.]*", topic.lower())
    return {k for k in kws if len(k) > 2}


def _is_low_signal_query(query: str, topic_kws: set[str]) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return True
    toks = re.findall(r"[a-zA-Z0-9][\w\-.]*", q)
    if len(toks) < 3:
        return True
    generic_starts = (
        "what is ", "overview of ", "introduction to ", "about ",
        "latest news", "news about", "definition of ",
    )
    if any(q.startswith(gs) for gs in generic_starts):
        return True
    if topic_kws and not any(t in topic_kws for t in toks):
        return True
    return False


def _normalize_plan_payload(data: dict, max_queries: int, topic: str = "") -> dict:
    hypotheses = _as_string_list(data.get("hypotheses") or [])
    sub_topics = _as_string_list(data.get("sub_topics") or [])
    raw_queries = _as_string_list(data.get("queries") or [])
    topic_kws = _topic_keywords(topic)

    queries: list[str] = []
    seen: set[str] = set()
    for q in raw_queries:
        key = q.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        if _is_low_signal_query(q, topic_kws):
            continue
        queries.append(q)
        if len(queries) >= max_queries:
            break

    schema = data.get("schema")
    if not isinstance(schema, dict):
        schema = {}
    # Validate value types — anything outside the allowed set coerces to "text"
    allowed_types = {"numeric", "text", "date", "percent"}
    schema = {
        str(k)[:60]: (str(v).lower() if str(v).lower() in allowed_types else "text")
        for k, v in schema.items()
        if k and isinstance(k, str)
    }
    # Backfill a default schema if the LLM omitted one. Without this, the
    # synthesis prompt's "Comparison" section gets nothing to render and
    # the briefing reads thin.
    if not schema:
        schema = _default_schema_for_topic(topic)
        _log.info("planner: schema empty, applied default for topic=%s", topic[:60])
    if not queries:
        return {"error": "planner produced no valid high-signal queries"}
    return {
        "hypotheses": hypotheses,
        "sub_topics": sub_topics,
        "queries": queries,
        "schema": schema,
    }


def _default_schema_for_topic(topic: str) -> dict:
    """Generic but useful default — works for almost any product/topic the
    user is comparing or evaluating. Keys land in the briefing's comparison
    table; we want enough columns that the table is informative."""
    return {
        "name": "text",
        "vendor": "text",
        "primary_use_case": "text",
        "pricing_model": "text",
        "license": "text",
        "release_year": "numeric",
        "recent_change": "text",
        "key_strength": "text",
        "key_weakness": "text",
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
   - Use exact domain names/technologies/brands from the topic when present.
   - Never substitute similar-sounding words (e.g. product names must stay exact).
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
            last_raw = result[:500]
            candidate = _extract_json_object(result)
            normalized = None
            if candidate:
                cleaned = _clean_json_text(candidate)
                try:
                    parsed = json.loads(cleaned)
                    normalized = _normalize_plan_payload(parsed, max_queries, topic=topic)
                except json.JSONDecodeError as e:
                    _log.warning(
                        "plan parse failed (will try salvage)  attempt=%d/%d  topic=%s  error=%s",
                        attempt, attempts, topic[:40], str(e)[:120],
                    )
            else:
                _log.warning(
                    "planner: no JSON object  attempt=%d/%d  topic=%s  raw_head=%s",
                    attempt, attempts, topic[:40], last_raw[:200],
                )

            if normalized is None or "error" in (normalized or {}):
                # Salvage: even half-formed model output usually has the
                # queries array. Pull it, backfill the rest.
                salvaged_queries = _salvage_queries(result)
                if salvaged_queries:
                    _log.info(
                        "planner: salvaged %d queries from malformed response  topic=%s",
                        len(salvaged_queries), topic[:40],
                    )
                    normalized = _normalize_plan_payload({
                        "queries": salvaged_queries,
                        "hypotheses": [],
                        "sub_topics": [],
                        "schema": {},
                    }, max_queries, topic=topic)
                    if normalized and "error" not in normalized:
                        normalized["_salvaged"] = True

            if normalized and "error" not in normalized:
                return normalized
            if normalized and "error" in normalized:
                last_error = str(normalized.get("error") or "invalid planner payload")
                _log.warning(
                    "planner payload invalid  attempt=%d/%d  topic=%s  error=%s",
                    attempt, attempts, topic[:40], last_error,
                )
            elif candidate is None or not candidate:
                last_error = "no JSON object in response"
            else:
                last_error = "invalid JSON, salvage produced no queries"

        if attempt < attempts and backoff_s > 0:
            time.sleep(backoff_s)

    if get_feature("research", "planner_fallback_enabled", True):
        _log.warning(
            "planner failed %d attempts, using deterministic fallback  topic=%s  last_error=%s",
            attempts, topic[:60], last_error,
        )
        fb = _fallback_plan(topic, max_queries)
        if "error" not in fb:
            return fb

    out = {"error": f"planner failed after {attempts} attempts: {last_error}"}
    if last_raw:
        out["raw"] = last_raw
    return out


def create_research_plan(
    topic: str,
    org_id: int = 0,
    parent_insight_id: int | None = None,
    focus: str = "",
    defer_run: bool = False,
) -> dict:
    """Create a shell research_plans row and (optionally) queue the planner job.
    Returns immediately with the plan_id so the UI can poll for status.

    If ``parent_insight_id`` is set, the completed paper will be appended to
    that insight (see ``shared.insights.append_research``).

    If ``defer_run`` is True, the row is created with ``type="hidden"`` and
    NO planner job is queued. The plan sits as a draft until a caller invokes
    ``start_research_plan(plan_id)``. This is the path used by auto-spawned
    insight follow-ups so the UI can hide them until the user opts in.
    """
    from infra.nocodb_client import NocodbClient
    from workers.tool_queue import get_tool_queue

    if not get_feature("research", "planner_enabled", True):
        return {"status": "disabled", "error": "research_planner feature disabled"}
    if int(org_id or 0) <= 0:
        from tools._org import resolve_org_id
        org_id = resolve_org_id(org_id)

    client = NocodbClient()
    try:
        payload = {
            "org_id": org_id,
            "topic": topic,
            "hypotheses": "[]",
            "sub_topics": "[]",
            "queries": "[]",
            "schema": "{}",
            "iterations": 0,
            "status": "pending",
        }
        if parent_insight_id:
            payload["parent_insight_id"] = int(parent_insight_id)
        # Two distinct provenance tags:
        #   - "pending insight" → auto-seeded follow-up drafts (defer_run=True
        #     from tools.insight.agent). The UI hides these from the main
        #     research feed; they only surface on the parent insight's page.
        #   - "insight deep-dive" → user explicitly clicked Deep Dive on an
        #     insight (parent_insight_id set, planner runs immediately). These
        #     ARE visible in the main feed because the user asked for the
        #     research to happen — hiding them would lose the result.
        if defer_run:
            payload["type"] = "pending insight"
        elif parent_insight_id:
            payload["type"] = "insight deep-dive"
        if focus:
            payload["focus"] = focus[:500]
        row = client._post("research_plans", payload)
        plan_id = row.get("Id")
    except Exception as e:
        _log.warning("shell plan save failed  topic=%s  error=%s", topic[:40], e)
        return {"status": "failed", "error": str(e)[:200]}

    if defer_run:
        return {"status": "deferred", "plan_id": plan_id}

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

    # Run the agent inline in this same job — avoids a second queue hop where
    # the agent job would sit pending behind unrelated work.
    from tools.research.agent import run_research_agent
    _log.info("Running research agent inline for plan_id %d (queries=%d)", plan_id, len(queries))
    return run_research_agent(plan_id)


def start_research_plan(plan_id: int) -> dict:
    """Invoke a deferred (hidden) research plan: clear the hidden type and
    queue the planner job that will generate queries and run the agent."""
    from infra.nocodb_client import NocodbClient
    from workers.tool_queue import get_tool_queue
    from tools._org import resolve_org_id

    client = NocodbClient()
    try:
        plan_row = client._get("research_plans", params={"where": f"(Id,eq,{plan_id})", "limit": 1})
        plan = plan_row.get("list", [])[0] if plan_row.get("list") else None
    except Exception as e:
        return {"status": "failed", "error": str(e)[:200], "plan_id": plan_id}
    if not plan:
        return {"status": "not_found", "plan_id": plan_id}

    org_id = resolve_org_id(plan.get("org_id"))
    try:
        # Use None (NULL) rather than "" — if the column is a SingleSelect
        # without an empty option, an empty string would 400.
        client._patch("research_plans", plan_id, {"type": None, "status": "pending"})
    except Exception as e:
        _log.warning("start_research_plan: clear type failed  plan_id=%d  error=%s", plan_id, e)

    tq = get_tool_queue()
    if not tq:
        client._patch("research_plans", plan_id, {
            "status": "failed",
            "error_message": "tool_queue_unavailable",
        })
        return {"status": "failed", "error": "tool_queue_unavailable", "plan_id": plan_id}

    job_id = tq.submit(
        "research_planner",
        {"plan_id": plan_id, "org_id": org_id},
        source="research_api",
        priority=3,
        org_id=org_id,
    )
    _log.info("Started deferred research plan plan_id=%d job_id=%s", plan_id, job_id)
    return {"status": "queued", "plan_id": plan_id, "job_id": job_id}


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


def reap_stale_plans() -> dict:
    """Mark research_plans rows as failed only when they're truly wedged:
    in a transient state (generating/searching/synthesizing), with no
    activity (UpdatedAt) for ``research.stale_plan_hours``, AND no inflight
    tool_job_queue row for the plan. This leaves legitimate long-running
    syntheses alone while cleaning up rows orphaned by worker crashes."""
    from datetime import datetime, timedelta, timezone
    from infra.nocodb_client import NocodbClient

    stale_hours = float(get_feature("research", "stale_plan_hours", 24) or 24)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=stale_hours)

    client = NocodbClient()

    # Collect plan_ids referenced by any non-terminal tool_job_queue row.
    active_plan_ids: set[int] = set()
    try:
        tool_rows = client._get_paginated("tool_job_queue", params={
            "where": "(type,in,research_agent,research_planner)"
                     "~and(status,in,queued,running,retry_scheduled)",
            "limit": 200,
        })
        for tr in tool_rows:
            raw = tr.get("payload") or ""
            try:
                pid = json.loads(raw).get("plan_id") if isinstance(raw, str) else raw.get("plan_id")
                if pid:
                    active_plan_ids.add(int(pid))
            except Exception:
                continue
    except Exception:
        _log.debug("reap: tool_job_queue scan skipped", exc_info=True)

    reaped = 0
    for state in ("generating", "searching", "synthesizing"):
        try:
            rows = client._get_paginated("research_plans", params={
                "where": f"(status,eq,{state})",
                "limit": 100,
            })
        except Exception:
            _log.warning("reap: scan failed  state=%s", state, exc_info=True)
            continue
        for row in rows:
            plan_id = row.get("Id")
            if not plan_id or int(plan_id) in active_plan_ids:
                continue
            updated = row.get("UpdatedAt") or row.get("CreatedAt") or ""
            try:
                ts = datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if ts >= cutoff:
                continue
            try:
                client._patch("research_plans", plan_id, {
                    "status": "failed",
                    "error_message": f"reaped: stuck in {state} {stale_hours:g}h+ with no inflight job",
                })
                reaped += 1
                _log.info("research plan reaped  id=%s state=%s age_h=%.1f",
                          plan_id, state, (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0)
            except Exception:
                _log.warning("reap: patch failed  id=%s", plan_id, exc_info=True)
    return {"status": "ok", "reaped": reaped, "skipped_inflight": len(active_plan_ids)}