"""
Research executor — iterative deep investigation with approval flow.

Unlike deep search (single pass: query → scrape → summarise → deliver),
research runs an iterative loop:

  1. T3 generates research plan (queries, objectives, assessment criteria)
  2. User approves or revises the plan
  3. Queued execution on T1 secondary:
     a. Search + scrape + summarise (per-source, thorough)
     b. T3 assesses coverage gaps in gathered evidence
     c. If gaps found: T3 generates refined queries → repeat from (a)
     d. Max 3 iterations to bound compute
  4. T1 secondary synthesises final research report
  5. Stores to ChromaDB + FalkorDB, delivers to conversation

The whole execution phase runs as a single tool_queue job at priority 1
(highest background priority, 120s backoff from chat).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time

from tools.framework.contract import ToolName, ToolResult
from tools.framework.dispatcher import register_executor
from tools.framework.executors.web_search import _search_all
from workers.enrichment.models import model_call

_log = logging.getLogger("tools.research")

MAX_ITERATIONS = 3
MAX_SOURCES_PER_ITERATION = 8
MAX_TOTAL_SOURCES = 20

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PLAN_PROMPT = """You are a research strategist. Given a user's research question and conversation context, design a thorough research plan.

Your plan must include:
1. **objective**: One sentence stating what the research aims to determine
2. **queries**: 6-10 specific search queries targeting different angles (technical, historical, comparative, expert opinion, data/statistics, contrary evidence)
3. **lookout**: 3-5 specific things to look for in sources (e.g. "peer-reviewed data on X", "cost comparisons between A and B", "expert dissent on Y")
4. **completion_criteria**: 2-3 conditions that mean the research is thorough enough to conclude (e.g. "found at least 3 independent sources agreeing on cost trends", "identified both supporting and contrary evidence")

Return ONLY a JSON object with these four keys. No prose, no markdown fences.
First character must be `{{`, last must be `}}`.

USER QUESTION:
{question}

{context_section}"""

_ASSESS_PROMPT = """You are evaluating research progress. Given the research objective, what we're looking for, completion criteria, and evidence gathered so far, determine if the research is complete or needs more investigation.

OBJECTIVE: {objective}

LOOKING FOR: {lookout}

COMPLETION CRITERIA: {completion_criteria}

EVIDENCE GATHERED SO FAR ({source_count} sources):
{evidence_summary}

Assess:
1. Which completion criteria are met?
2. What gaps remain?
3. Should we search for more? If yes, provide 3-5 NEW search queries targeting the gaps.

Return a JSON object:
{{
  "complete": true/false,
  "met_criteria": ["criteria that are satisfied"],
  "gaps": ["specific gaps remaining"],
  "new_queries": ["query1", "query2", ...] (empty array if complete)
}}

No prose, no markdown. First character `{{`, last `}}`."""

_SYNTHESISE_PROMPT = """You are a research analyst producing a formal research report.

RESEARCH QUESTION: {question}

RESEARCH OBJECTIVE: {objective}

You have {source_count} sources gathered across {iterations} rounds of investigation.

EVIDENCE:
{evidence_block}

Write a comprehensive research report with this structure:

1. **Executive Summary** (3-4 sentences — the key finding and confidence level)
2. **Key Findings** (numbered, each citing source numbers [1], [2] etc.)
3. **Analysis** (synthesise across sources, identify patterns, resolve contradictions)
4. **Contrary Evidence & Limitations** (what disagrees, what's missing, confidence caveats)
5. **Conclusion** (direct answer to the research question with appropriate hedging)
6. **Sources** (numbered list of URLs)

Rules:
- Cite sources by number [1], [2] etc.
- Distinguish primary sources (papers, official docs, data) from secondary (blogs, forums)
- State hypotheses clearly as hypotheses, not facts
- If evidence is contradictory, present both sides and explain which is stronger and why
- Be specific — include numbers, dates, names, not vague summaries"""


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
    _log.info("research plan generation  tokens=%d", tokens)

    if not raw:
        _log.warning("research plan generation returned empty")
        return None

    try:
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip()).strip()
        plan = json.loads(cleaned)
    except json.JSONDecodeError:
        _log.warning("research plan unparseable: %s", raw[:300])
        return None

    if not isinstance(plan, dict) or "queries" not in plan:
        _log.warning("research plan missing required fields: %s", sorted(plan.keys()) if isinstance(plan, dict) else type(plan))
        return None

    return plan


def _assess_progress(objective: str, lookout: list[str], criteria: list[str],
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
        evidence_summary=evidence[:3800],
    )

    raw, tokens = model_call("research_assess", prompt)
    _log.info("research assessment  tokens=%d", tokens)

    # On failure, default to incomplete so the loop continues gathering
    # evidence rather than stopping prematurely on a T3 hiccup.
    _INCOMPLETE = {"complete": False, "gaps": ["assessment failed — continuing"], "new_queries": [], "met_criteria": []}

    if not raw:
        _log.warning("research assessment returned empty — treating as incomplete")
        return _INCOMPLETE

    try:
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip()).strip()
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        _log.warning("research assessment unparseable — treating as incomplete")
        return _INCOMPLETE

    return result


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

@register_executor(ToolName.RESEARCH)
async def execute(params: dict, emit) -> ToolResult:
    """
    Research has two phases:

    Phase 1 (plan): T3 generates research plan → return for user approval
    Phase 2 (execute): Iterative search loop → T1 synthesise → deliver
    """
    phase = params.get("_phase", "plan")
    org_id = params.get("_org_id") or 0
    conversation_id = params.get("_conversation_id")

    if not org_id:
        return ToolResult(
            tool=ToolName.RESEARCH, action_index=0, ok=False,
            data="research missing org context",
        )

    if phase == "execute":
        return await _execute_research(params, emit)

    # ---- Phase 1: Generate plan for approval ----
    user_message = params.get("_user_message") or ""
    conversation_topics = params.get("_conversation_topics") or []

    if not user_message:
        return ToolResult(
            tool=ToolName.RESEARCH, action_index=0, ok=False,
            data="No question provided for research.",
        )

    emit({"type": "tool_status", "phase": "planning", "message": "Designing research approach..."})

    plan = await asyncio.get_event_loop().run_in_executor(
        None, _generate_plan, user_message, conversation_topics,
    )

    if not plan:
        return ToolResult(
            tool=ToolName.RESEARCH, action_index=0, ok=False,
            data="Failed to generate research plan.",
        )

    # Store plan on conversation for approval
    full_plan = {
        "question": user_message,
        "objective": plan.get("objective", ""),
        "queries": plan.get("queries", []),
        "lookout": plan.get("lookout", []),
        "completion_criteria": plan.get("completion_criteria", []),
    }

    if conversation_id:
        try:
            from nocodb_client import NocodbClient
            db = NocodbClient()
            db.update_conversation(int(conversation_id), {
                "pending_research": json.dumps(full_plan),
            })
            _log.info("research plan stored  conv=%s queries=%d", conversation_id, len(full_plan["queries"]))
        except Exception:
            _log.error("research plan storage failed  conv=%s", conversation_id, exc_info=True)

    # Build plan summary for the chat model
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

    emit({
        "type": "research_plan",
        "tool": "research",
        "message": plan_summary,
        "status": "awaiting_approval",
        "plan": full_plan,
    })

    return ToolResult(
        tool=ToolName.RESEARCH, action_index=0, ok=True,
        data=plan_summary,
    )


async def _execute_research(params: dict, emit) -> ToolResult:
    """Phase 2: Queue the research job for background execution on T1."""
    org_id = int(params.get("_org_id") or 0)
    conversation_id = params.get("_conversation_id")
    plan = params.get("_plan") or {}

    if not plan.get("queries"):
        return ToolResult(
            tool=ToolName.RESEARCH, action_index=0, ok=False,
            data="No queries in the approved research plan.",
        )

    # Submit as a single research job to the tool queue.
    # The job handler runs the iterative loop.
    from workers.tool_queue import get_tool_queue
    tq = get_tool_queue()
    if not tq:
        return ToolResult(
            tool=ToolName.RESEARCH, action_index=0, ok=False,
            data="Tool job queue not available.",
        )

    job_id = tq.submit(
        job_type="research",
        payload={
            "plan": plan,
            "org_id": org_id,
            "conversation_id": conversation_id,
        },
        source="research",
        org_id=org_id,
        priority=1,
    )

    _log.info("research job queued  conv=%s job=%s queries=%d",
              conversation_id, job_id, len(plan["queries"]))

    # Clear the pending plan
    if conversation_id:
        try:
            from nocodb_client import NocodbClient
            db = NocodbClient()
            db.update_conversation(int(conversation_id), {"pending_research": ""})
        except Exception:
            pass

    emit({
        "type": "jobs_queued",
        "tool": "research",
        "message": f"Research queued with {len(plan['queries'])} queries. "
                   f"Will iterate up to {MAX_ITERATIONS} rounds until completion criteria are met.",
        "status": "running",
    })

    return ToolResult(
        tool=ToolName.RESEARCH, action_index=0, ok=True,
        data=f"Research approved and queued. The investigation will run iteratively "
             f"with up to {MAX_ITERATIONS} rounds of search and analysis. "
             f"A comprehensive report will be delivered to this conversation when complete.",
    )