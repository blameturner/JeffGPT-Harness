import json
import logging

from infra.config import is_feature_enabled
from infra.memory import recall
from infra.nocodb_client import NocodbClient
from tools.research.critic import analyze_gaps, get_confidence_threshold
from shared.models import model_call

_log = logging.getLogger("research.agent")

DEFAULT_MAX_ITERATIONS = 3


def _get_retrieval_context(topic: str, queries: list, org_id: int, collection: str = "discovery") -> str:
    context_parts = [f"Research Topic: {topic}\n"]
    for q in queries[:10]:
        try:
            results = recall(q, org_id=org_id, collection_name=collection, n_results=5)
            if results:
                context_parts.append(f"\n--- Query: {q} ---")
                for r in results:
                    text = r.get("text", "")[:500]
                    meta = r.get("metadata", {})
                    source = meta.get("url", "unknown")
                    context_parts.append(f"[Source: {source}]\n{text}")
        except Exception as e:
            _log.warning("recall failed for query: %s  error=%s", q[:40], e)
    return "\n\n".join(context_parts)


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
    if not is_feature_enabled("research_agent"):
        return {"status": "disabled", "error": "research_agent feature disabled"}

    max_iterations = is_feature_enabled("research_max_iterations") or DEFAULT_MAX_ITERATIONS
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

    context = _get_retrieval_context(topic, queries, org_id)

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

    if ready or confidence >= confidence_threshold or iterations >= max_iterations:
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