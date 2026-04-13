"""Research executor — iterative deep investigation with approval flow.

Two phases:
  1. T3 generates research plan (queries, objectives, assessment criteria)
     → returned for user approval.
  2. After approval, background job runs iterative loop:
     a. Search + scrape + summarise (per-source, thorough)
     b. T3 assesses coverage gaps
     c. If gaps: T3 generates refined queries → repeat from (a)
     d. Max 3 iterations
     e. T1 synthesises final research report
     f. Stores to ChromaDB + FalkorDB, delivers to conversation
"""

from __future__ import annotations

import asyncio
import json
import logging

from tools.framework.contract import ToolName, ToolResult
from tools.framework.dispatcher import register_executor
from tools.framework.executors.search.pipeline import (
    parse_llm_json,
    store_pending_plan,
    clear_pending_plan,
)
from workers.enrichment.models import model_call

_log = logging.getLogger("tools.research")

MAX_ITERATIONS = 3
MAX_SOURCES_PER_ITERATION = 8
MAX_TOTAL_SOURCES = 20

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PLAN_PROMPT = """You are a research strategist. Design a thorough research plan for the question below.

OUTPUT FORMAT: Return ONLY a JSON object. No prose before or after. No markdown fences.
The first character of your response MUST be {{ and the last MUST be }}.

REQUIRED KEYS (all four are mandatory):

1. "objective" (string): One sentence — what the research aims to determine or answer.

2. "queries" (array of 6-10 strings): Search queries targeting DIFFERENT angles:
   - At least 1 technical/definitional query
   - At least 1 comparison query (X vs Y)
   - At least 1 query with quoted phrases for exact matching
   - At least 1 query targeting recent data (include year or "latest" or "2024")
   - At least 1 query seeking contrary evidence or criticism
   - Each query must be a specific search string, NOT a description of what to search

3. "lookout" (array of 3-5 strings): Specific evidence to look for, e.g.:
   - "peer-reviewed data on X"
   - "cost comparisons between A and B with actual numbers"
   - "expert criticism or dissent on Y"

4. "completion_criteria" (array of 2-3 strings): Conditions that mean research is thorough enough, e.g.:
   - "found at least 3 independent sources with quantitative data"
   - "identified both supporting and contrary evidence"

USER QUESTION:
{question}

{context_section}"""

_ASSESS_PROMPT = """You are evaluating whether a research investigation has gathered enough evidence.

OBJECTIVE: {objective}
LOOKING FOR: {lookout}
COMPLETION CRITERIA: {completion_criteria}

EVIDENCE GATHERED ({source_count} sources):
{evidence_summary}

YOUR TASK: Determine if the completion criteria are met.

OUTPUT FORMAT: Return ONLY a JSON object. No prose before or after. No markdown fences.
The first character of your response MUST be {{ and the last MUST be }}.

REQUIRED KEYS:

1. "complete" — JSON boolean: true if ALL completion criteria are met, false otherwise.
   IMPORTANT: Use the JSON boolean true or false, NOT the strings "true" or "false".

2. "met_criteria" — array of strings: Which completion criteria are satisfied. Empty array if none.

3. "gaps" — array of strings: Specific evidence gaps remaining. Be precise — name what is missing.
   Example: ["no quantitative pricing data found", "only 1 source on competitor comparison"]

4. "new_queries" — array of 3-5 search query strings targeting the gaps. Empty array if complete.
   Each query must be a specific search string that would find the missing evidence.
   Do NOT repeat queries that were already used."""

_SYNTHESISE_PROMPT = """You are a research analyst. Write a formal research report answering the question below.

RESEARCH QUESTION: {question}
RESEARCH OBJECTIVE: {objective}

You have {source_count} sources gathered across {iterations} round(s) of investigation.

EVIDENCE:
{evidence_block}

REPORT STRUCTURE (follow this exactly):

## Executive Summary
3-4 sentences. State the key finding, the confidence level (high/medium/low), and the basis for that confidence.

## Key Findings
Numbered list. Each finding MUST cite at least one source by number [1], [2] etc.
Include specific data points: numbers, percentages, dates, prices, names.
Do NOT make vague statements like "sources suggest" — state the specific fact and cite it.

## Analysis
Synthesise across sources. Identify patterns, trends, and consensus.
Where sources disagree, explain the disagreement and which position has stronger evidence.
Distinguish primary sources (official data, research papers, documentation) from secondary (blogs, forums, opinions).

## Contrary Evidence & Limitations
What evidence contradicts the main findings? What important questions remain unanswered?
What are the confidence caveats? (e.g. "limited to English-language sources", "most data from 2024")
If no contrary evidence was found, say so explicitly.

## Conclusion
Direct answer to the research question. Use hedging language appropriate to the confidence level.
State clearly what is established fact vs. what is the analyst's assessment.

## Sources
Numbered list of all source URLs used, matching the citation numbers in the report.

RULES:
- Every factual claim MUST have a citation [N].
- Use numbers, dates, and specifics — never "some sources say" or "it appears that".
- If evidence is contradictory, present BOTH sides with citations.
- State hypotheses as hypotheses, facts as facts. Never confuse the two."""


# ---------------------------------------------------------------------------
# T3 plan generation
# ---------------------------------------------------------------------------

def _generate_plan(question: str, conversation_topics: list[str] | None = None) -> dict | None:
    """Use T3 to generate a structured research plan."""
    context_section = ""
    if conversation_topics:
        context_section = f"CONVERSATION CONTEXT:\n{', '.join(conversation_topics[:10])}"

    prompt = _PLAN_PROMPT.format(
        question=question[:2000],
        context_section=context_section,
    )

    raw, tokens = model_call("research_queries", prompt)
    _log.info("plan generation  tokens=%d", tokens)

    if not raw:
        _log.warning("plan generation returned empty")
        return None

    plan = parse_llm_json(raw)
    if not isinstance(plan, dict) or "queries" not in plan:
        _log.warning("plan missing required fields: %s",
                     sorted(plan.keys()) if isinstance(plan, dict) else type(plan))
        return None

    return plan


def assess_progress(objective: str, lookout: list[str], criteria: list[str],
                    summaries: list[dict]) -> dict:
    """Use T3 to assess whether research is complete or needs more work."""
    evidence = "\n\n".join(
        f"[{i+1}] {s['url']}\n{s['summary'][:600]}"
        for i, s in enumerate(summaries)
    )

    prompt = _ASSESS_PROMPT.format(
        objective=objective,
        lookout=", ".join(lookout),
        completion_criteria=", ".join(criteria),
        source_count=len(summaries),
        evidence_summary=evidence[:10000],
    )

    raw, tokens = model_call("research_assess", prompt)
    _log.info("assessment  tokens=%d", tokens)

    _INCOMPLETE = {"complete": False, "gaps": ["assessment failed — continuing"], "new_queries": [], "met_criteria": []}

    if not raw:
        _log.warning("assessment returned empty — treating as incomplete")
        return _INCOMPLETE

    result = parse_llm_json(raw)
    if not isinstance(result, dict):
        _log.warning("assessment unparseable — treating as incomplete")
        return _INCOMPLETE

    raw_complete = result.get("complete")
    result["complete"] = raw_complete is True or str(raw_complete).lower() == "true"
    return result


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

@register_executor(ToolName.RESEARCH)
async def execute(params: dict, emit) -> ToolResult:
    """Phase 1: generate plan. Phase 2: submit background job."""
    phase = params.get("_phase", "plan")
    org_id = params.get("_org_id") or 0
    conversation_id = params.get("_conversation_id")

    if not org_id:
        return ToolResult(tool=ToolName.RESEARCH, action_index=0, ok=False,
                          data="research missing org context")

    if phase == "execute":
        return await _execute_approved(params, emit)

    # ---- Phase 1: Generate plan for approval ----
    user_message = params.get("_user_message") or ""
    conversation_topics = params.get("_conversation_topics") or []

    if not user_message:
        return ToolResult(tool=ToolName.RESEARCH, action_index=0, ok=False,
                          data="No question provided for research.")

    emit({"type": "tool_status", "phase": "planning", "message": "Designing research approach..."})

    plan = await asyncio.get_running_loop().run_in_executor(
        None, _generate_plan, user_message, conversation_topics,
    )

    if not plan:
        return ToolResult(tool=ToolName.RESEARCH, action_index=0, ok=False,
                          data="Failed to generate research plan.")

    full_plan = {
        "question": user_message,
        "objective": plan.get("objective", ""),
        "queries": plan.get("queries", []),
        "lookout": plan.get("lookout", []),
        "completion_criteria": plan.get("completion_criteria", []),
    }

    if conversation_id:
        try:
            store_pending_plan(conversation_id, "research", full_plan)
            _log.info("plan stored  conv=%s queries=%d", conversation_id, len(full_plan["queries"]))
        except Exception:
            _log.error("plan storage failed  conv=%s", conversation_id, exc_info=True)

    query_list = "\n".join(f"- {q}" for q in full_plan["queries"])
    lookout_list = "\n".join(f"- {l}" for l in full_plan["lookout"])
    criteria_list = "\n".join(f"- {c}" for c in full_plan["completion_criteria"])

    plan_summary = (
        f"I've designed a research plan for your question.\n\n"
        f"**Objective:** {full_plan['objective']}\n\n"
        f"**Search queries ({len(full_plan['queries'])}):**\n{query_list}\n\n"
        f"**What I'll look for:**\n{lookout_list}\n\n"
        f"**Completion criteria:**\n{criteria_list}\n\n"
        f"The research will run iteratively — searching, assessing gaps, and refining "
        f"until the criteria are met (up to {MAX_ITERATIONS} rounds). "
        f"Please review and approve this plan, or suggest changes."
    )

    return ToolResult(tool=ToolName.RESEARCH, action_index=0, ok=True, data=plan_summary)


async def _execute_approved(params: dict, emit) -> ToolResult:
    """Phase 2: Submit background job to tool queue."""
    org_id = int(params.get("_org_id") or 0)
    conversation_id = params.get("_conversation_id")
    plan = params.get("_plan") or {}

    if not plan.get("queries"):
        return ToolResult(tool=ToolName.RESEARCH, action_index=0, ok=False,
                          data="No queries in the approved research plan.")

    from workers.tool_queue import get_tool_queue
    tq = get_tool_queue()
    if not tq:
        return ToolResult(tool=ToolName.RESEARCH, action_index=0, ok=False,
                          data="Tool job queue not available.")

    job_id = tq.submit(
        job_type="research",
        payload={"plan": plan, "org_id": org_id, "conversation_id": conversation_id},
        source="research", org_id=org_id, priority=1,
    )
    _log.info("job queued  conv=%s job=%s queries=%d", conversation_id, job_id, len(plan["queries"]))

    if conversation_id:
        try:
            clear_pending_plan(conversation_id, "research")
        except Exception:
            _log.warning("failed to clear pending plan  conv=%s", conversation_id, exc_info=True)

    msg = (f"Research queued with {len(plan['queries'])} queries. "
           f"Will iterate up to {MAX_ITERATIONS} rounds until completion criteria are met.")
    emit({"type": "jobs_queued", "tool": "research", "message": msg, "status": "running"})

    return ToolResult(tool=ToolName.RESEARCH, action_index=0, ok=True, data=msg)
