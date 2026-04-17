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
DEFAULT_WEB_SEARCH_PER_QUERY_TIMEOUT_S = 180
DEFAULT_SYNTHESIS_TIMEOUT_S = 1200
DEFAULT_CRITIC_TIMEOUT_S = 480


def _research_timeout(primary_key: str, default_s: int, *legacy_keys: str) -> int:
    for key in (primary_key, *legacy_keys):
        raw = get_feature("research", key, None)
        if raw in (None, ""):
            continue
        try:
            val = int(raw)
            if val > 0:
                return val
        except Exception:
            continue
    return default_s


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
    timeout_s = _research_timeout(
        "web_search_per_query_timeout_s",
        DEFAULT_WEB_SEARCH_PER_QUERY_TIMEOUT_S,
        "web_search_timeout_s",
    )
    ex = _futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="research-ws")
    try:
        extraction_function_name = str(
            get_feature("research", "search_extraction_model", "research_search_extraction")
            or "research_search_extraction"
        )
        fut = ex.submit(
            run_web_search,
            query,
            org_id=org_id,
            intent_dict=intent,
            extraction_function_name=extraction_function_name,
        )
        try:
            return fut.result(timeout=timeout_s)
        except _futures.TimeoutError:
            _log.warning("web search timeout after %ds  query=%s", timeout_s, query[:60])
            return None
        except Exception as e:
            _log.warning("web search failed for query: %s  error=%s", query[:40], e)
            return None
    finally:
        ex.shutdown(wait=False)


def _fetch_fresh_context(topic: str, queries: list, org_id: int) -> tuple[str, list[dict]]:
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
    timeout_s = _research_timeout("synthesis_timeout_s", DEFAULT_SYNTHESIS_TIMEOUT_S)

    # Build a human-readable section guide from the schema keys so the model knows
    # what to cover, without constraining the output to JSON.
    schema_guide = ""
    if schema:
        fields = list(schema.keys()) if isinstance(schema, dict) else []
        if fields:
            schema_guide = "Cover the following topics as sections:\n" + "\n".join(f"- {f}" for f in fields)

    prompt = f"""You are a Research Synthesis Agent. Write a comprehensive, well-structured research paper in Markdown format.

TOPIC: {topic}
ITERATION: {iteration + 1}
{schema_guide}

AVAILABLE CONTEXT:
{context[:25000]}

INSTRUCTIONS:
1. Write a comprehensive research paper using the context above.
2. Use Markdown formatting with ## section headers and ### subsections.
3. Every factual claim must cite a source inline as [Source: URL].
4. If information for a section is missing or insufficient, note it as _Information unavailable_.
5. End with a ## Sources section listing all cited URLs.
6. Output ONLY the Markdown document — no JSON, no code fences, no preamble."""

    def _run():
        return model_call("research_agent", prompt, max_tokens=4000, temperature=0.3)

    ex = _futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="research-syn")
    try:
        fut = ex.submit(_run)
        try:
            result, _ = fut.result(timeout=timeout_s)
            return {"status": "ok", "content": result}
        except _futures.TimeoutError:
            _log.warning("synthesis timeout after %ds  topic=%s", timeout_s, topic[:40])
            return {"status": "failed", "error": f"synthesis timeout after {timeout_s}s"}
        except Exception as e:
            _log.warning("synthesis failed  topic=%s  error=%s", topic[:40], e)
            return {"status": "failed", "error": str(e)[:200]}
    finally:
        ex.shutdown(wait=False)


def _call_with_timeout(fn, args: tuple, timeout_s: float, label: str):
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
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return fallback


def _patch_or_log(client, plan_id: int, patch: dict, label: str) -> None:
    try:
        client._patch("research_plans", plan_id, patch)
    except Exception:
        _log.debug("research_plans patch failed  plan_id=%d  label=%s", plan_id, label, exc_info=True)


def _coerce_query_list(value) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        s = str(item or "").strip()
        if s:
            out.append(s)
    return out


def _dedupe_queries(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in items:
        key = q.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def _fallback_queries_from_gaps(topic: str, gap_analysis: dict) -> list[str]:
    out: list[str] = []
    for g in (gap_analysis.get("gaps_found") or []):
        if not isinstance(g, dict):
            continue
        field = str(g.get("field") or "").strip()
        needed = str(g.get("needed") or "").strip()
        if field and needed:
            out.append(f"{topic} {field} {needed}")
        elif field:
            out.append(f"{topic} {field} latest evidence")
    if not out and topic:
        out = [
            f"{topic} latest statistics",
            f"{topic} primary sources",
            f"{topic} peer reviewed evidence",
        ]
    return _dedupe_queries(out)[:5]


def run_research_agent(plan_id: int) -> dict:
    if not get_feature("research", "agent_enabled", True):
        return {"status": "disabled", "error": "research_agent feature disabled"}

    client = NocodbClient()


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
    # already succeeded, wasting the synthesis.
    critic_timeout_s = _research_timeout("critic_timeout_s", DEFAULT_CRITIC_TIMEOUT_S)
    gap_analysis = _call_with_timeout(
        analyze_gaps,
        (topic, synthesis.get("content", ""), schema, context),
        critic_timeout_s,
        "critic",
    )
    if gap_analysis is None:
        # Critic timed out or raised. We have a valid synthesis — save it and
        # complete rather than discarding the work. Mark confidence as 50
        # (unknown) so the UI can show it was an imperfect run.
        _log.warning(
            "research critic timeout/error — completing with synthesis result  plan_id=%d",
            plan_id,
        )
        from datetime import datetime, timezone
        paper_content = synthesis.get("content", "")
        client._patch("research_plans", plan_id, {
            "status": "completed",
            "paper_content": paper_content,
            "gap_report": "{}",
            "confidence_score": 50,
            "iterations": iterations + 1,
            "error_message": f"critic timeout after {critic_timeout_s}s — completed best-effort",
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
                    extra_metadata={"plan_id": plan_id, "topic": topic, "iteration": iterations + 1},
                )
            except Exception:
                _log.warning("research ingest_output failed (critic timeout path)  plan_id=%d", plan_id, exc_info=True)
        return {"status": "completed", "confidence": 50, "plan_id": plan_id, "note": "critic_timeout_best_effort"}

    gap_report = json.dumps(gap_analysis)
    confidence = gap_analysis.get("confidence_score", 0)
    ready = gap_analysis.get("ready_for_completion", False)
    # Normalize model output so odd shapes (null/string/mixed) don't derail iteration.
    new_queries = _coerce_query_list(gap_analysis.get("new_search_requirements") or [])

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
                {"plan_id": plan_id, "org_id": org_id},
                source="research_agent_iteration",
                priority=3,
                org_id=org_id,
            )
            _log.info("Re-queued research agent job %s for plan_id %d (iter=%d)", job_id, plan_id, iterations + 1)
        else:
            _log.error("Tool queue unavailable during research iteration  plan_id=%d", plan_id)
            client._patch("research_plans", plan_id, {
                "status": "failed",
                "error_message": "tool_queue_unavailable",
            })
            return {"status": "failed", "error": "tool_queue_unavailable", "plan_id": plan_id}

        return {"status": "needs_more_research", "confidence": confidence, "new_queries": new_queries, "plan_id": plan_id}
    else:
        if iterations + 1 < max_iterations:
            fallback_queries = _fallback_queries_from_gaps(topic, gap_analysis)
            if fallback_queries:
                merged = _dedupe_queries(queries + fallback_queries)
                client._patch("research_plans", plan_id, {
                    "status": "generating",
                    "gap_report": gap_report,
                    "confidence_score": confidence,
                    "iterations": iterations + 1,
                    "queries": json.dumps(merged),
                    "error_message": "critic produced no new_search_requirements; generated fallback queries",
                })
                from workers.tool_queue import get_tool_queue
                tq = get_tool_queue()
                if tq:
                    job_id = tq.submit(
                        "research_agent",
                        {"plan_id": plan_id, "org_id": org_id},
                        source="research_agent_fallback_iteration",
                        priority=3,
                        org_id=org_id,
                    )
                    _log.info("Re-queued research agent fallback job %s for plan_id %d (iter=%d)", job_id, plan_id, iterations + 1)
                    return {
                        "status": "needs_more_research",
                        "confidence": confidence,
                        "new_queries": fallback_queries,
                        "plan_id": plan_id,
                        "note": "fallback_queries_generated",
                    }
                _log.error("Tool queue unavailable for fallback iteration  plan_id=%d", plan_id)
                client._patch("research_plans", plan_id, {
                    "status": "failed",
                    "error_message": "tool_queue_unavailable",
                })
                return {"status": "failed", "error": "tool_queue_unavailable", "plan_id": plan_id}

        # Final fallback: preserve synthesized markdown as best-effort completion
        # instead of failing hard with an unusable row.
        from datetime import datetime, timezone
        paper_content = synthesis.get("content", "")
        client._patch("research_plans", plan_id, {
            "status": "completed",
            "paper_content": paper_content,
            "gap_report": gap_report,
            "confidence_score": confidence,
            "iterations": iterations + 1,
            "error_message": "No new queries produced before max iterations; completed best-effort",
            "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        })
        return {"status": "completed", "confidence": confidence, "plan_id": plan_id, "note": "best_effort_no_new_queries"}


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