import concurrent.futures as _futures
import json
import logging

from infra.config import get_feature
from infra.memory import recall
from infra.nocodb_client import NocodbClient
from tools.research.critic import analyze_gaps, get_confidence_threshold
from tools.search.intent import (
    CHAT_INTENT_RESEARCH,
    INTENT_RESPONSE_TEMPLATE,
    INTENT_ROUTE_CHAT,
    SEARCH_POLICY_FULL,
)
from tools.search.orchestrator import run_web_search
from shared.models import model_call

_log = logging.getLogger("research.agent")

DEFAULT_MAX_ITERATIONS = 3
# Per-query ceiling for run_web_search. SEARCH_POLICY_FULL's internal budget is
# 60s but the orchestrator only enforces that cap for CONTEXTUAL; FULL runs
# uncapped. Without this outer timeout, one slow LLM call inside web_search
# hangs the whole research iteration and the plan sits stuck on "synthesizing".
_WEB_SEARCH_PER_QUERY_TIMEOUT_S = 90
# Ceiling for the synthesis LLM call. model_call has no internal timeout.
_SYNTHESIS_TIMEOUT_S = 300
# Ceiling for the critic (analyze_gaps) LLM call. Same rationale as synthesis
# — critic runs another model_call after synthesis and could hang just the
# same. Shorter cap because the critic prompt is smaller.
_CRITIC_TIMEOUT_S = 120


def _research_intent_dict(topic: str, entities: list[str] | None = None) -> dict:
    return {
        "route": INTENT_ROUTE_CHAT,
        "intent": CHAT_INTENT_RESEARCH,
        "secondary_intent": None,
        "entities": ([topic] if topic else []) + (entities or []),
        "location_hint": None,
        "time_sensitive": False,
        "temporal_anchor": None,
        "confidence": "high",
        "search_policy": SEARCH_POLICY_FULL,
        "response_template": INTENT_RESPONSE_TEMPLATE[CHAT_INTENT_RESEARCH],
    }


def _run_web_search_with_timeout(query: str, org_id: int, intent: dict) -> tuple[str, list[dict], str] | None:
    """Wrap run_web_search in a thread + hard timeout. Returns None on timeout
    or exception so the caller can skip this query instead of hanging forever.
    The orchestrator only enforces hard_cap_s for SEARCH_POLICY_CONTEXTUAL;
    research uses SEARCH_POLICY_FULL which otherwise has no outer cap.

    Important: we do NOT use `with ThreadPoolExecutor(...)` because its
    __exit__ calls shutdown(wait=True), which would block waiting for the
    timed-out thread to finish — defeating the timeout. Python threads
    cannot be force-killed, so we let the hung thread leak in the
    background (shutdown wait=False) and return control to the handler."""
    ex = _futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="research-ws")
    try:
        fut = ex.submit(run_web_search, query, org_id=org_id, intent_dict=intent)
        try:
            return fut.result(timeout=_WEB_SEARCH_PER_QUERY_TIMEOUT_S)
        except _futures.TimeoutError:
            _log.warning("web search timeout after %ds  query=%s", _WEB_SEARCH_PER_QUERY_TIMEOUT_S, query[:60])
            return None
        except Exception as e:
            _log.warning("web search failed for query: %s  error=%s", query[:40], e)
            return None
    finally:
        ex.shutdown(wait=False)


def _fetch_fresh_context(topic: str, queries: list, org_id: int) -> tuple[str, list[dict]]:
    """Run web_search for each query with a per-query hard timeout; return
    aggregated context block and collected sources. A hanging query is skipped
    rather than blocking the whole research iteration."""
    context_parts: list[str] = []
    all_sources: list[dict] = []
    intent = _research_intent_dict(topic)
    for q in queries:
        res = _run_web_search_with_timeout(q, org_id, intent)
        if res is None:
            continue
        context_block, sources, confidence = res
        if context_block:
            context_parts.append(f"\n--- Query: {q} (confidence={confidence}) ---")
            context_parts.append(context_block)
        if sources:
            all_sources.extend(sources)
    return "\n\n".join(context_parts), all_sources


def _recall_accumulated(queries: list, org_id: int, n_per_query: int = 4) -> str:
    """Pull prior web_search results from RAG across earlier iterations, deduped by URL."""
    seen: set[str] = set()
    blocks: list[str] = []
    for q in queries:
        try:
            hits = recall(q, org_id=org_id, collection_name="web_search", n_results=n_per_query)
        except Exception as e:
            _log.warning("recall web_search failed for query: %s  error=%s", q[:40], e)
            continue
        for h in hits:
            meta = h.get("metadata") or {}
            url = meta.get("url") or ""
            if url in seen:
                continue
            seen.add(url)
            title = meta.get("title") or url or "unknown"
            text = (h.get("text") or "")[:800]
            blocks.append(f"[Source: {title} ({url})]\n{text}")
    return "\n\n".join(blocks)


def _build_context(topic: str, fresh_queries: list, prior_queries: list, org_id: int) -> str:
    fresh_text, _sources = _fetch_fresh_context(topic, fresh_queries, org_id)
    parts = [f"Research Topic: {topic}\n"]
    if prior_queries:
        accumulated = _recall_accumulated(prior_queries, org_id)
        if accumulated:
            parts.append("=== ACCUMULATED PRIOR FINDINGS ===")
            parts.append(accumulated)
    if fresh_text:
        parts.append("=== NEW FINDINGS (this iteration) ===" if prior_queries else "=== FINDINGS ===")
        parts.append(fresh_text)
    return "\n\n".join(parts)


def _synthesize(topic: str, context: str, schema: dict, iteration: int) -> dict:
    prompt = f"""You are a Research Synthesis Agent. Write a comprehensive research paper on the topic.

TOPIC: {topic}
ITERATION: {iteration + 1}
SCHEMA (data structure to produce): {json.dumps(schema)}

AVAILABLE CONTEXT:
{context[:25000]}

INSTRUCTIONS:
1. Write a comprehensive research paper using the context above.
2. Every claim must cite a source using [Source: URL] format.
3. Structure the output according to the schema.
4. If information is missing or insufficient, note it as "INCOMPLETE".
5. Output ONLY valid JSON matching the schema structure."""

    def _run():
        return model_call("research_agent", prompt, max_tokens=4000, temperature=0.3)

    # Same rationale as _run_web_search_with_timeout: avoid `with` so the
    # executor doesn't block on shutdown when the LLM call hangs.
    ex = _futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="research-syn")
    try:
        fut = ex.submit(_run)
        try:
            result, _ = fut.result(timeout=_SYNTHESIS_TIMEOUT_S)
            return {"status": "ok", "content": result}
        except _futures.TimeoutError:
            _log.warning("synthesis timeout after %ds  topic=%s", _SYNTHESIS_TIMEOUT_S, topic[:40])
            return {"status": "failed", "error": f"synthesis timeout after {_SYNTHESIS_TIMEOUT_S}s"}
        except Exception as e:
            _log.warning("synthesis failed  topic=%s  error=%s", topic[:40], e)
            return {"status": "failed", "error": str(e)[:200]}
    finally:
        ex.shutdown(wait=False)


def _call_with_timeout(fn, args: tuple, timeout_s: float, label: str):
    """Generic thread+timeout wrapper. Returns the function result, or None if
    it timed out or raised. Uses shutdown(wait=False) so a hung thread doesn't
    block the calling handler."""
    ex = _futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"research-{label}")
    try:
        fut = ex.submit(fn, *args)
        try:
            return fut.result(timeout=timeout_s)
        except _futures.TimeoutError:
            _log.warning("%s timeout after %ds", label, timeout_s)
            return None
        except Exception as e:
            _log.warning("%s failed  error=%s", label, e)
            return None
    finally:
        ex.shutdown(wait=False)


def _safe_json_loads(raw, fallback):
    """json.loads that never raises — returns `fallback` on any parse error.
    Corrupt queries/schema JSON in the DB row should NOT crash the handler."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return fallback


def _patch_or_log(client, plan_id: int, patch: dict, label: str) -> None:
    """Best-effort row update. If the DB blips mid-run we still want the
    handler to continue and return cleanly — the final except-path will
    retry the terminal patch."""
    try:
        client._patch("research_plans", plan_id, patch)
    except Exception:
        _log.debug("research_plans patch failed  plan_id=%d  label=%s", plan_id, label, exc_info=True)


def run_research_agent(plan_id: int) -> dict:
    if not get_feature("research", "agent_enabled", True):
        return {"status": "disabled", "error": "research_agent feature disabled"}

    client = NocodbClient()

    # Whole-handler guard: ANY exception — including DB errors before the status
    # flip or bad JSON in plan fields — lands in the except path and flips the
    # row to "failed". The plan can never be left stuck in a non-terminal state.
    try:
        plan_row = client._get("research_plans", params={"where": f"(Id,eq,{plan_id})", "limit": 1})
        plan = plan_row.get("list", [])[0] if plan_row.get("list") else None

        if not plan:
            return {"status": "not_found", "plan_id": plan_id}

        topic = plan.get("topic", "")
        queries = _safe_json_loads(plan.get("queries", "[]"), [])
        schema = _safe_json_loads(plan.get("schema", "{}"), {})
        iterations = plan.get("iterations", 0)
        org_id = plan.get("org_id", 0)

        max_iterations = plan.get("max_iterations") or get_feature("research", "max_iterations", DEFAULT_MAX_ITERATIONS)
        confidence_threshold = plan.get("confidence_threshold") or get_confidence_threshold()

        # Status lifecycle (so the UI can see WHERE a run is stuck):
        #   searching     -> inside _fetch_fresh_context (web_search per query)
        #   synthesizing  -> inside _synthesize (LLM synthesis)
        #   completed/failed/generating -> terminal or next-iteration state
        # Previously this code flipped to "synthesizing" at entry, so every hang —
        # even in web_search — presented as "stuck on synthesizing".
        _patch_or_log(client, plan_id, {"status": "searching"}, "searching")

        return _run_research_agent_inner(
            client, plan_id, topic, queries, schema, iterations,
            org_id, max_iterations, confidence_threshold, plan,
        )
    except Exception as e:
        _log.error("research_agent uncaught error  plan_id=%d", plan_id, exc_info=True)
        _patch_or_log(client, plan_id, {
            "status": "failed",
            "error_message": f"uncaught: {str(e)[:300]}",
        }, "failed-uncaught")
        return {"status": "failed", "plan_id": plan_id, "error": str(e)[:300]}


def _run_research_agent_inner(
    client,
    plan_id: int,
    topic: str,
    queries: list,
    schema: dict,
    iterations: int,
    org_id: int,
    max_iterations: int,
    confidence_threshold: int,
    plan: dict,
) -> dict:
    if iterations == 0:
        fresh_queries = queries
        prior_queries: list = []
    else:
        fresh_queries = []
        prev_report_raw = plan.get("gap_report") or ""
        if prev_report_raw:
            try:
                prev_report = json.loads(prev_report_raw)
                fresh_queries = prev_report.get("new_search_requirements", []) or []
            except (json.JSONDecodeError, TypeError):
                _log.warning("gap_report parse failed  plan_id=%d, searching all queries", plan_id)
        if not fresh_queries:
            fresh_queries = queries
            prior_queries = []
        else:
            fresh_set = set(fresh_queries)
            prior_queries = [q for q in queries if q not in fresh_set]

    context = _build_context(topic, fresh_queries, prior_queries, org_id)

    # We're done searching — flip to "synthesizing" right before the LLM call so
    # the UI status accurately reflects which phase is running.
    _patch_or_log(client, plan_id, {"status": "synthesizing"}, "synthesizing")

    synthesis = _synthesize(topic, context, schema, iterations)

    if synthesis.get("status") == "failed":
        client._patch("research_plans", plan_id, {"status": "failed", "error_message": synthesis.get("error")})
        return {"status": "failed", **synthesis}

    # Bound the critic call the same way — without a timeout wrapper a hung
    # model_call inside analyze_gaps would stall the handler AFTER synthesis
    # already succeeded, wasting the synthesis and sitting the row on
    # "synthesizing" indefinitely.
    gap_analysis = _call_with_timeout(
        analyze_gaps,
        (topic, synthesis.get("content", ""), schema, context),
        _CRITIC_TIMEOUT_S,
        "critic",
    )
    if gap_analysis is None:
        # Timed out or raised. Bail with a failure marker rather than retrying
        # — if the model is stuck, another call won't help this iteration.
        client._patch("research_plans", plan_id, {
            "status": "failed",
            "error_message": f"critic timeout/error after {_CRITIC_TIMEOUT_S}s",
        })
        return {"status": "failed", "plan_id": plan_id, "error": "critic timeout"}

    gap_report = json.dumps(gap_analysis)
    confidence = gap_analysis.get("confidence_score", 0)
    ready = gap_analysis.get("ready_for_completion", False)
    # `or []` guards against the model returning null for this key — `None + list`
    # would raise TypeError on line `new_queries_list = queries + new_queries`.
    new_queries = gap_analysis.get("new_search_requirements") or []

    new_queries_list = queries + new_queries
    updated_queries = json.dumps(new_queries_list)

    if ready or confidence >= confidence_threshold or iterations + 1 >= max_iterations:
        from datetime import datetime, timezone
        paper_content = synthesis.get("content", "")
        client._patch("research_plans", plan_id, {
            "status": "completed",
            "paper_content": paper_content,
            "gap_report": gap_report,
            "confidence_score": confidence,
            "iterations": iterations + 1,
            # NocoDB v1 DateTime columns reject isoformat()'s microseconds+tz suffix
            "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        })

        if paper_content:
            try:
                from workers.post_turn import ingest_output
                ingest_output(
                    output=paper_content,
                    user_text=topic,
                    org_id=org_id,
                    conversation_id=0,
                    model="research_agent",
                    rag_collection="research",
                    knowledge_collection="research_knowledge",
                    source="research",
                    extra_metadata={
                        "plan_id": plan_id,
                        "topic": topic,
                        "confidence_score": confidence,
                        "iteration": iterations + 1,
                    },
                )
            except Exception:
                _log.warning("research ingest_output failed  plan_id=%d", plan_id, exc_info=True)

        return {"status": "completed", "confidence": confidence, "plan_id": plan_id}
    elif new_queries:
        client._patch("research_plans", plan_id, {
            "status": "generating",
            "gap_report": gap_report,
            "confidence_score": confidence,
            "iterations": iterations + 1,
            "queries": updated_queries
        })

        from workers.tool_queue import get_tool_queue
        tq = get_tool_queue()
        if tq:
            job_id = tq.submit(
                "research_agent",
                {"plan_id": plan_id},
                source="research_agent_iteration",
                priority=3,
            )
            _log.info("Re-queued research agent job %s for plan_id %d (iter=%d)", job_id, plan_id, iterations + 1)
        else:
            _log.warning("Tool queue not available, next iteration will not run for plan_id %d", plan_id)

        return {"status": "needs_more_research", "confidence": confidence, "new_queries": new_queries, "plan_id": plan_id}
    else:
        client._patch("research_plans", plan_id, {
            "status": "failed",
            "gap_report": gap_report,
            "confidence_score": confidence,
            "error_message": "No new queries but not ready"
        })
        return {"status": "failed", "confidence": confidence, "plan_id": plan_id}


def get_next_research() -> dict | None:
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