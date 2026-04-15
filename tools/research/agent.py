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


def _fetch_fresh_context(topic: str, queries: list, org_id: int) -> tuple[str, list[dict]]:
    """Run web_search for each query; return aggregated context block and collected sources."""
    context_parts: list[str] = []
    all_sources: list[dict] = []
    intent = _research_intent_dict(topic)
    for q in queries:
        try:
            context_block, sources, confidence = run_web_search(q, org_id=org_id, intent_dict=intent)
        except Exception as e:
            _log.warning("web search failed for query: %s  error=%s", q[:40], e)
            continue
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

    try:
        result, _ = model_call("research_agent", prompt, max_tokens=4000, temperature=0.3)
        return {"status": "ok", "content": result}
    except Exception as e:
        _log.warning("synthesis failed  topic=%s  error=%s", topic[:40], e)
        return {"status": "failed", "error": str(e)[:200]}


def run_research_agent(plan_id: int) -> dict:
    if not get_feature("research", "agent_enabled", True):
        return {"status": "disabled", "error": "research_agent feature disabled"}

    max_iterations = get_feature("research", "max_iterations", DEFAULT_MAX_ITERATIONS)
    confidence_threshold = get_confidence_threshold()

    client = NocodbClient()
    plan_row = client._get("research_plans", params={"where": f"(Id,eq,{plan_id})", "limit": 1})
    plan = plan_row.get("list", [])[0] if plan_row.get("list") else None

    if not plan:
        return {"status": "not_found", "plan_id": plan_id}

    topic = plan.get("topic", "")
    queries = json.loads(plan.get("queries", "[]"))
    schema = json.loads(plan.get("schema", "{}"))
    iterations = plan.get("iterations", 0)
    org_id = plan.get("org_id", 0)

    client._patch("research_plans", plan_id, {"status": "synthesizing"})

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

    synthesis = _synthesize(topic, context, schema, iterations)

    if synthesis.get("status") == "failed":
        client._patch("research_plans", plan_id, {"status": "failed", "error_message": synthesis.get("error")})
        return {"status": "failed", **synthesis}

    gap_analysis = analyze_gaps(topic, synthesis.get("content", ""), schema, context)

    gap_report = json.dumps(gap_analysis)
    confidence = gap_analysis.get("confidence_score", 0)
    ready = gap_analysis.get("ready_for_completion", False)
    new_queries = gap_analysis.get("new_search_requirements", [])

    new_queries_list = queries + new_queries
    updated_queries = json.dumps(new_queries_list)

    if ready or confidence >= confidence_threshold or iterations + 1 >= max_iterations:
        paper_content = synthesis.get("content", "")
        client._patch("research_plans", plan_id, {
            "status": "completed",
            "paper_content": paper_content,
            "gap_report": gap_report,
            "confidence_score": confidence,
            "iterations": iterations + 1
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