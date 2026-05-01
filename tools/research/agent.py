"""Research agent — long-form, doc-type-aware, section-by-section synthesis.

Pipeline:
  1. Resolve doc_type (planner-supplied or inferred from topic)
  2. Fetch source corpus by running each planner query through web search
  3. Write the paper section-by-section (each LLM call is bounded)
  4. Save paper_content; ingest into RAG; append to insight if scoped

Review is an explicit second pass invoked by the user (review_research_paper):
a big-model reviewer reads the paper and emits per-section revision
instructions; the writer re-runs the affected sections with those
instructions appended; the new paper replaces the old.

There is no iterative critic loop. Model calls inherit the global httpx
timeout from `shared.models` (default 2400s) and the model-pool slot
queue serialises concurrent calls — research does not impose a tighter
cap on top of that.
"""
import concurrent.futures as _futures
import json
import logging
from datetime import datetime, timezone

from infra.config import get_feature
from infra.nocodb_client import NocodbClient
from tools.search.intent import (
    CHAT_INTENT_RESEARCH,
    INTENT_RESPONSE_TEMPLATE,
    INTENT_ROUTE_CHAT,
    SEARCH_POLICY_FULL,
)
from tools.search.orchestrator import run_web_search
from shared.models import model_call
from tools._org import resolve_org_id

_log = logging.getLogger("research.agent")

DEFAULT_WEB_SEARCH_PER_QUERY_TIMEOUT_S = 240
DEFAULT_DOC_TYPE_TIMEOUT_S = 300
DEFAULT_SECTION_TIMEOUT_S = 1800
DEFAULT_REVIEWER_TIMEOUT_S = 2400


# Each doc type defines the bookend section titles, the prose register, and
# what the executive summary should accomplish. The body sections come from
# the planner's sub_topics — so these define the spine, not the whole paper.
DOC_TYPES = {
    "research_report": {
        "opener": "Background",
        "closer": "Discussion and Conclusions",
        "tone": "academic, neutral, evidence-led prose. Cite sources for every concrete claim.",
        "summary_role": "Summarise the question investigated, the strongest findings, and the conclusions reached.",
    },
    "business_plan": {
        "opener": "Market Opportunity",
        "closer": "Roadmap and Operating Plan",
        "tone": "decisive operator voice — concrete, numbers-first, written for a founder or investor audience.",
        "summary_role": "State the venture proposition, market size, customer, business model, and the funding/effort ask.",
    },
    "market_analysis": {
        "opener": "Market Overview",
        "closer": "Outlook and Implications",
        "tone": "analyst voice — quantified, comparative, written for a strategy or investment reader.",
        "summary_role": "Headline the market size, growth, structure, and the most important shifts underway.",
    },
    "technical_brief": {
        "opener": "Context and Constraints",
        "closer": "Recommendation and Tradeoffs",
        "tone": "engineering voice — precise, comparative, opinionated where the evidence permits.",
        "summary_role": "State the problem, the candidate approaches, and the recommended approach with the load-bearing reason.",
    },
    "comparison": {
        "opener": "Evaluation Criteria",
        "closer": "Recommendation",
        "tone": "fair-minded comparative reviewer — every option treated symmetrically before any recommendation.",
        "summary_role": "Name the options compared, the criteria used, and the leading recommendation.",
    },
    "how_to": {
        "opener": "Prerequisites and Setup",
        "closer": "Common Pitfalls and Troubleshooting",
        "tone": "practical instructor voice — second-person, concrete, sequential.",
        "summary_role": "State what the reader will accomplish, the prerequisites, and the rough effort/time required.",
    },
    "policy_brief": {
        "opener": "Issue and Stakeholders",
        "closer": "Recommended Policy Direction",
        "tone": "policy advisor voice — neutral framing, balanced presentation of options.",
        "summary_role": "State the policy question, who is affected, the options considered, and the recommendation.",
    },
    "feasibility_study": {
        "opener": "Problem Statement",
        "closer": "Recommendation and Next Steps",
        "tone": "evaluator voice — rigorous, hedged where evidence is thin, decisive where it is not.",
        "summary_role": "State what was assessed, the verdict (feasible / conditionally / not), and the load-bearing reasons.",
    },
    "deep_dive": {
        "opener": "Background and Stakes",
        "closer": "Implications and Open Questions",
        "tone": "long-form explanatory voice — patient, layered, narrative-driven, but rigorously sourced.",
        "summary_role": "Frame why this matters, the core finding, and what the reader will understand by the end.",
    },
    "white_paper": {
        "opener": "Executive Position",
        "closer": "Implications and Call to Action",
        "tone": "authoritative industry voice — vendor-neutral but persuasive, written for a decision-making readership.",
        "summary_role": "State the position taken, the central evidence, and the call to action.",
    },
    "literature_review": {
        "opener": "Scope and Methodology",
        "closer": "Synthesis and Open Questions",
        "tone": "academic synthesis voice — careful attribution to prior work, neutral, theme-driven not chronological.",
        "summary_role": "State the question, the body of literature reviewed, and the dominant findings and gaps.",
    },
    "competitive_analysis": {
        "opener": "Competitive Landscape",
        "closer": "Strategic Implications",
        "tone": "competitive intelligence voice — fact-led, comparative, written for a strategy or product reader.",
        "summary_role": "Identify the competitors, the dimensions compared, and the strategic implication.",
    },
    "due_diligence": {
        "opener": "Investment Thesis and Scope",
        "closer": "Findings and Recommendation",
        "tone": "investor diligence voice — skeptical, evidence-grading, surface every red flag.",
        "summary_role": "State the target, the diligence scope, the top findings, and the go/no-go recommendation with conditions.",
    },
    "product_spec": {
        "opener": "Problem and Goals",
        "closer": "Open Questions and Sequencing",
        "tone": "PRD voice — declarative, structured, written for engineers and designers to act on.",
        "summary_role": "State the problem, target user, success metrics, and the proposed solution at a glance.",
    },
    "architecture_decision": {
        "opener": "Context and Forces",
        "closer": "Decision and Consequences",
        "tone": "ADR voice — terse, decision-focused, every claim defensible.",
        "summary_role": "State the decision, the alternatives considered, and the load-bearing reason.",
    },
    "case_study": {
        "opener": "Situation and Background",
        "closer": "Outcomes and Lessons",
        "tone": "narrative analytical voice — storyline-driven but rigorously sourced.",
        "summary_role": "State the subject, the situation studied, and the key lessons drawn.",
    },
    "swot_analysis": {
        "opener": "Subject and Frame",
        "closer": "Strategic Implications",
        "tone": "strategist voice — concise, comparative, structured around the SWOT quadrants.",
        "summary_role": "State the subject and the most consequential strength, weakness, opportunity, and threat.",
    },
    "industry_report": {
        "opener": "Industry Definition and Scope",
        "closer": "Outlook and Watch List",
        "tone": "industry analyst voice — quantified, structural, written for a sector reader.",
        "summary_role": "State the industry, its size, its structure, and the most consequential dynamics.",
    },
    "risk_assessment": {
        "opener": "Scope and Methodology",
        "closer": "Mitigation Recommendations",
        "tone": "risk officer voice — neutral, structured by likelihood × impact, hedged where evidence is thin.",
        "summary_role": "State the assessment scope, the top risks, and the recommended mitigations.",
    },
    "investment_memo": {
        "opener": "Thesis",
        "closer": "Decision and Conditions",
        "tone": "IC memo voice — opinionated, thesis-led, every claim backed.",
        "summary_role": "State the investment thesis, the key risks, and the recommended action.",
    },
    "trend_report": {
        "opener": "Trend Landscape",
        "closer": "Implications and Watch Items",
        "tone": "futurist analyst voice — pattern-led, hedged on timing, evidenced by signals.",
        "summary_role": "State the trends covered, their drivers, and the implications for the reader.",
    },
    "retrospective": {
        "opener": "What Happened",
        "closer": "Lessons and Forward Actions",
        "tone": "post-mortem voice — blameless, factual, lesson-extracting.",
        "summary_role": "State what was undertaken, the outcome, and the most important lessons.",
    },
    "forecast": {
        "opener": "Forecast Question and Method",
        "closer": "Scenarios and Confidence",
        "tone": "forecaster voice — probabilistic, scenario-based, calibrated uncertainty.",
        "summary_role": "State the forecast question, the scenarios, the most likely outcome, and the confidence.",
    },
    "explainer": {
        "opener": "Why This Matters",
        "closer": "What to Take Away",
        "tone": "patient explainer voice — accessible, no jargon without unpacking, readable end to end.",
        "summary_role": "State what the reader will understand by the end and why it matters.",
    },
}
DEFAULT_DOC_TYPE = "research_report"

# Sections that are regenerated wholesale on every paper build/review pass —
# the reviewer should not propose targeted revisions to these because they
# are derived from the body. (Comparison and Recommendation are user-meaningful
# and CAN be revised, so they are not in this set.)
_PROTECTED_SECTIONS = {"Executive Summary", "Key Takeaways", "Sources"}


# ── small utilities ──────────────────────────────────────────────────────────

def _research_timeout(key: str, default_s: int) -> int:
    raw = get_feature("research", key, None)
    if raw in (None, ""):
        return default_s
    try:
        v = int(raw)
        return v if v > 0 else default_s
    except Exception:
        return default_s


def _safe_json_loads(raw, fallback):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return fallback


def _patch_or_log(client, plan_id: int, patch: dict, label: str) -> None:
    try:
        client._patch("research_plans", plan_id, patch)
    except Exception:
        # WARNING (not DEBUG) so a silent NocoDB rejection during status
        # transitions actually surfaces in logs.
        _log.warning(
            "research_plans patch failed  plan_id=%d  label=%s  fields=%s",
            plan_id, label, list(patch.keys()), exc_info=True,
        )


def _safe_call(fn, _timeout_unused: float, label: str):
    """Invoke a model/search function inline, log boundaries, swallow any
    exception.

    The earlier version of this helper wrapped `fn` in a `ThreadPoolExecutor`
    with `fut.result(timeout=...)`. That timeout was tighter than
    `model_call`'s own httpx timeout (set globally via `models.http_timeout_s`
    in `config.json`, default 2400s) and would fire mid-stream — orphaning
    the in-flight request and surfacing as 'all body sections failed' even
    when the model was about to return. Nothing else in the codebase wraps
    `model_call` like that, and nothing else hits these timeouts.

    `model_call` and `run_web_search` already:
      - acquire a slot from the model pool (queues if busy);
      - bound a single HTTP call with the global httpx timeout;
      - return `("", 0)` on any failure.

    So this helper just runs `fn` inline. The legacy `timeout_s` argument is
    accepted but ignored to keep the callsites unchanged — the surrounding
    `_research_timeout(...)` lookups still work, they just no longer enforce
    an outer cap.
    """
    import time as _time
    _log.info("call %s INVOKE", label)
    t0 = _time.time()
    try:
        result = fn()
        elapsed = round(_time.time() - t0, 1)
        size_hint = ""
        if isinstance(result, tuple) and result and isinstance(result[0], str):
            size_hint = f"  out_chars={len(result[0])}"
        _log.info("call %s RETURN  %.1fs%s", label, elapsed, size_hint)
        return result
    except Exception as e:
        elapsed = round(_time.time() - t0, 1)
        _log.warning("call %s ERROR  %.1fs  err=%s", label, elapsed, str(e)[:200])
        return None


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


# ── corpus fetch ─────────────────────────────────────────────────────────────

def _fetch_corpus(topic: str, queries: list[str], org_id: int) -> tuple[str, list[dict]]:
    intent = _research_intent_dict(topic)
    extraction_function_name = str(
        get_feature("research", "search_extraction_model", "research_search_extraction")
        or "research_search_extraction"
    )
    timeout_s = _research_timeout(
        "web_search_per_query_timeout_s", DEFAULT_WEB_SEARCH_PER_QUERY_TIMEOUT_S,
    )
    blocks: list[str] = []
    sources: list[dict] = []
    seen_urls: set[str] = set()
    _log.info("corpus FETCH START  topic=%r  n_queries=%d", topic[:80], len(queries))

    # Run queries in parallel. The previous sequential loop multiplied
    # per-query latency by N — with N=20 and a slow extractor, a single
    # research run could spend 20×timeout on this stage. Cap concurrency
    # at 4 to keep us from hammering searxng / the extraction model. The
    # model_pool gate inside extraction calls already serialises against
    # the chat path, so chat priority is unaffected.
    from workers.tool_queue import (
        report_progress, is_job_cancelled, JobCancelled,
        current_job_id, bind_job_id,
    )
    import concurrent.futures
    import threading
    # Capture the parent contextvar — ThreadPool workers do NOT inherit it
    # by default, so without bind_job_id() the report_progress and
    # is_job_cancelled calls inside _one would be no-ops.
    parent_job_id = current_job_id()
    max_workers = min(4, max(1, len(queries)))
    completed_count = [0]
    completed_lock = threading.Lock()

    def _one(idx_q):
        idx, q = idx_q
        with bind_job_id(parent_job_id):
            if is_job_cancelled():
                return idx, q, None
            report_progress(f"search start [{idx}/{len(queries)}]: {q[:60]}")
            res = _safe_call(
                lambda q=q: run_web_search(
                    q, org_id=org_id, intent_dict=intent,
                    extraction_function_name=extraction_function_name,
                ),
                timeout_s,
                f"search[{idx}/{len(queries)}]:{q[:40]}",
            )
            with completed_lock:
                completed_count[0] += 1
                done_n = completed_count[0]
            report_progress(
                f"search done [{done_n}/{len(queries)}]: {q[:60]}"
            )
            return idx, q, res

    # Manual pool management (not `with`) so we can pass cancel_futures=True
    # on shutdown — otherwise a JobCancelled raise inside the loop would
    # block on every queued search to finish before propagating, wasting
    # 10+ minutes of CPU after the user already terminated the job.
    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="research-search",
    )
    try:
        futures = [pool.submit(_one, item) for item in enumerate(queries, start=1)]
        for fut in concurrent.futures.as_completed(futures):
            if is_job_cancelled():
                # Cancel every still-queued future; running ones finish.
                for f in futures:
                    f.cancel()
                raise JobCancelled("cancelled mid-search")
            try:
                idx, q, res = fut.result()
            except concurrent.futures.CancelledError:
                continue
            if not res:
                continue
            try:
                ctx, src, conf = res
            except (TypeError, ValueError):
                continue
            if ctx:
                blocks.append(f"--- Query: {q} (confidence={conf}) ---\n{ctx}")
            for s in (src or []):
                if not isinstance(s, dict):
                    continue
                url = (s.get("url") or "").strip()
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                sources.append(s)
    finally:
        # Don't wait for queued tasks on cancel/exception; cancel them so a
        # JobCancelled raise propagates promptly instead of blocking on
        # every remaining LLM call.
        pool.shutdown(wait=False, cancel_futures=True)
    out_corpus = "\n\n".join(blocks)

    # Fallback: if the orchestrator's LLM rerank/extract chain returned
    # nothing for every query (very common when the local extraction model
    # is overloaded or searxng's reranker is too strict), salvage raw page
    # text directly via searxng + scrape_page. Better to write from messy
    # source text than to fail the whole research run.
    if not out_corpus.strip():
        _log.warning("corpus empty after extraction; falling back to raw scrape")
        out_corpus, sources = _fetch_corpus_raw(topic, queries, sources)

    _log.info(
        "corpus FETCH DONE  topic=%r  blocks=%d  sources=%d  chars=%d",
        topic[:80], len(blocks) or 1, len(sources), len(out_corpus),
    )
    return out_corpus, sources


def _fetch_corpus_raw(topic: str, queries: list[str],
                      prior_sources: list[dict]) -> tuple[str, list[dict]]:
    """Last-resort corpus fetch: searxng URLs → raw scrape, no LLM in the loop.

    Used when the full orchestrator returns nothing across all queries. The
    output is messier (no per-page summarisation) but the section writer can
    still synthesise from it.
    """
    try:
        from tools.search.engine import searxng_search, _dedupe
        from tools.search.scraping import scrape_page
    except Exception:
        _log.warning("raw corpus fallback unavailable (search/scrape import failed)")
        return "", prior_sources

    seen_urls: set[str] = {(s.get("url") or "").strip() for s in (prior_sources or []) if s}
    raw_results: list[dict] = []
    for q in queries[:3]:  # cap — if 3 queries don't yield anything, more won't help
        try:
            raw_results.extend(searxng_search(q, max_results=8))
        except Exception:
            continue
    raw_results = _dedupe(raw_results)[:10]
    if not raw_results:
        return "", prior_sources

    blocks: list[str] = []
    new_sources: list[dict] = list(prior_sources or [])
    for r in raw_results:
        url = (r.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        try:
            text = scrape_page(url, r.get("snippet", ""))
        except Exception:
            text = ""
        if not text:
            continue
        title = r.get("title") or url
        # Cap each page so the corpus stays in a reasonable LLM context budget.
        blocks.append(f"--- {title} ({url}) ---\n{text[:4000]}")
        new_sources.append(r)
        if len(blocks) >= 6:
            break

    return "\n\n".join(blocks), new_sources


# ── doc-type detection ──────────────────────────────────────────────────────

def _infer_doc_type(topic: str, planned_doc_type: str | None = None) -> str:
    if planned_doc_type and planned_doc_type in DOC_TYPES:
        return planned_doc_type
    timeout_s = _research_timeout("doc_type_timeout_s", DEFAULT_DOC_TYPE_TIMEOUT_S)
    options = ", ".join(DOC_TYPES.keys())
    prompt = f"""Classify this request into ONE document type from this list:
{options}

REQUEST:
{topic}

Rules:
- If the request explicitly names the format ("write me a business plan", "give me a market analysis", "feasibility study", "research report", "comparison of X vs Y"), use that.
- If the request is just a topic with no format hint, infer from intent ("which X should I use" → comparison; "everything about X" or "explain X" → deep_dive; vendor/category landscape → market_analysis).
- Output ONLY the chosen type as a single lowercase word, nothing else."""
    res = _safe_call(
        lambda: model_call("research_doc_type", prompt, temperature=0.1, max_tokens=60),
        timeout_s,
        "doc_type",
    )
    if not res:
        return DEFAULT_DOC_TYPE
    try:
        raw, _ = res
    except (TypeError, ValueError):
        return DEFAULT_DOC_TYPE
    line = (raw or "").strip().splitlines()[:1]
    token = (line[0] if line else "").strip().lower().split()[:1]
    cand = (token[0] if token else "").strip(".:,") if token else ""
    if cand in DOC_TYPES:
        return cand
    return DEFAULT_DOC_TYPE


# ── section writers ─────────────────────────────────────────────────────────

def _section_prompt(*, topic: str, doc_type: str, section_title: str, section_role: str,
                    corpus: str, hypotheses: list[str], target_words: int,
                    revision_note: str | None = None) -> str:
    spec = DOC_TYPES[doc_type]
    hyp_block = ""
    if hypotheses:
        hyp_block = "HYPOTHESES TO CONSIDER:\n" + "\n".join(f"- {h}" for h in hypotheses) + "\n\n"
    rev_block = ""
    if revision_note:
        rev_block = f"\nREVISION INSTRUCTIONS (apply on top of the section spec above):\n{revision_note}\n"
    return f"""You are writing ONE section of a {doc_type.replace('_', ' ')} on the topic below.

TOPIC: {topic}

SECTION HEADING: ## {section_title}
SECTION GOAL: {section_role}

DOCUMENT TONE: {spec['tone']}

TARGET LENGTH: ~{target_words} words of substantive prose. Stay close to the target — too short reads thin, too long reads padded.

{hyp_block}AVAILABLE SOURCE MATERIAL (use ONLY this; never fabricate facts or URLs):
{corpus[:24000]}
{rev_block}
RULES:
- Output Markdown starting with `## {section_title}`. Do NOT output the document title, executive summary, or any other section — just this one.
- Write in flowing prose paragraphs. Use `###` subsections only when the material genuinely splits into distinct angles.
- Every concrete claim (number, date, named entity, attribution, comparison) MUST carry an inline `[Source: URL]` citation drawn from the source material above.
- Do not use bullet points unless the section is intrinsically a list (e.g., a numbered procedure in a how-to). Default to paragraphs.
- Never write "Information unavailable" or similar boilerplate. If something is unknown, omit it or briefly note the gap in prose.
- Synthesise across sources — draw contrasts, note agreement, flag contradictions. Do not restate sources one by one.
- No preamble, no "Here is the section:", no closing summary. Output raw section markdown only."""


def _write_section(*, topic: str, doc_type: str, section_title: str, section_role: str,
                   corpus: str, hypotheses: list[str], target_words: int,
                   revision_note: str | None = None,
                   attempts: int = 3) -> str | None:
    """Write one section with progressive shrinkage on failure.

    Attempt 1: full target_words, corpus[:30000], no max_tokens cap.
    Attempt 2: ~2/3 target, corpus[:16000], max_tokens=3000.
    Attempt 3: ~1/3 target, corpus[:8000],  max_tokens=1500.

    The third attempt is a small-context, small-output 'last try' that
    almost always fits inside an overloaded local model's window. Worth
    the extra round-trip when the alternative is dropping the section.
    """
    timeout_s = _research_timeout("section_timeout_s", DEFAULT_SECTION_TIMEOUT_S)
    last_err = "unknown"
    for n in range(1, attempts + 1):
        if n == 1:
            this_target = target_words
            this_corpus = corpus[:30000]
            this_max_tokens: int | None = None
        elif n == 2:
            this_target = max(400, (target_words * 2) // 3)
            this_corpus = corpus[:16000]
            this_max_tokens = 3000
            _log.warning(
                "section %r retry %d/%d — shrinking (target=%d words, corpus=%d chars, max_tokens=%d)",
                section_title[:40], n, attempts, this_target, len(this_corpus), this_max_tokens,
            )
        else:
            this_target = max(250, target_words // 3)
            this_corpus = corpus[:8000]
            this_max_tokens = 1500
            _log.warning(
                "section %r LAST retry %d/%d — minimal (target=%d words, corpus=%d chars, max_tokens=%d)",
                section_title[:40], n, attempts, this_target, len(this_corpus), this_max_tokens,
            )
        prompt = _section_prompt(
            topic=topic, doc_type=doc_type, section_title=section_title,
            section_role=section_role, corpus=this_corpus, hypotheses=hypotheses,
            target_words=this_target, revision_note=revision_note,
        )
        kwargs: dict = {"temperature": 0.3}
        if this_max_tokens:
            kwargs["max_tokens"] = this_max_tokens
        res = _safe_call(
            lambda: model_call("research_section_writer", prompt, **kwargs),
            timeout_s,
            f"section[{n}/{attempts}]:{section_title[:30]}",
        )
        if not res:
            last_err = "timeout_or_error"
            continue
        try:
            text, _ = res
        except (TypeError, ValueError):
            last_err = "bad_tuple"
            continue
        text = (text or "").strip()
        if text:
            return text
        last_err = "empty"
    _log.warning("section %r FAILED after %d attempts  last_err=%s",
                 section_title[:40], attempts, last_err)
    return None


def _write_executive_summary(*, topic: str, doc_type: str, body_md: str,
                             target_words: int = 400) -> str | None:
    spec = DOC_TYPES[doc_type]
    timeout_s = _research_timeout("section_timeout_s", DEFAULT_SECTION_TIMEOUT_S)
    prompt = f"""You are writing the Executive Summary of a {doc_type.replace('_', ' ')}.

TOPIC: {topic}

SUMMARY GOAL: {spec['summary_role']}

DOCUMENT TONE: {spec['tone']}

TARGET LENGTH: ~{target_words} words across 2-4 paragraphs of prose.

THE SECTIONS YOU ARE SUMMARISING:
{body_md[:18000]}

RULES:
- Output starts with `## Executive Summary` and contains nothing but the summary.
- No bullets. Prose only.
- Cite the most important figures/claims with `[Source: URL]` drawn from the body.
- Do not repeat the body verbatim — distil the headline findings, conclusions, and recommendation."""
    res = _safe_call(
        lambda: model_call("research_section_writer", prompt, temperature=0.25),
        timeout_s,
        "exec_summary",
    )
    if not res:
        return None
    try:
        text, _ = res
    except (TypeError, ValueError):
        return None
    return (text or "").strip() or None


def _write_comparison(topic: str, schema: dict, corpus: str) -> str | None:
    if not isinstance(schema, dict) or not schema:
        return None
    fields = [k for k in schema.keys() if not str(k).startswith("_")]
    if not fields:
        return None
    timeout_s = _research_timeout("section_timeout_s", DEFAULT_SECTION_TIMEOUT_S)
    prompt = f"""Build the comparison table for this document.

TOPIC: {topic}

COLUMNS (in order): {", ".join(fields)}

SOURCE MATERIAL:
{corpus[:25000]}

RULES:
- Output starts with `## Comparison` and contains ONE markdown table.
- Rows are the resources/entities/options compared in the source material.
- Cells must be sourced from the material. Where a value is missing, write `—` (em dash). Never write "Information unavailable".
- Below the table, add 1-2 sentences of prose flagging any cross-row pattern worth noticing. No bullets.
- Output only the section. No preamble, no closing remarks."""
    res = _safe_call(
        lambda: model_call("research_section_writer", prompt, temperature=0.2),
        timeout_s,
        "comparison",
    )
    if not res:
        return None
    try:
        text, _ = res
    except (TypeError, ValueError):
        return None
    return (text or "").strip() or None


def _write_takeaways_and_recommendation(*, topic: str, doc_type: str, body_md: str) -> str | None:
    spec = DOC_TYPES[doc_type]
    timeout_s = _research_timeout("section_timeout_s", DEFAULT_SECTION_TIMEOUT_S)
    prompt = f"""Write the closing two sections of a {doc_type.replace('_', ' ')}.

TOPIC: {topic}
DOCUMENT TONE: {spec['tone']}

THE BODY YOU ARE CLOSING:
{body_md[:18000]}

OUTPUT:
1. `## Key Takeaways` — 4 to 7 crisp bullets, each one sentence. This is the only place bullets are allowed in the closing.
2. `## Recommendation` — 1 to 2 paragraphs of prose. Concrete guidance for the reader given the evidence. If evidence is genuinely insufficient for a recommendation, say so and explain what would be needed.

Output the two sections one after the other in Markdown. No preamble, no outer bullets, no closing summary."""
    res = _safe_call(
        lambda: model_call("research_section_writer", prompt, temperature=0.3),
        timeout_s,
        "takeaways",
    )
    if not res:
        return None
    try:
        text, _ = res
    except (TypeError, ValueError):
        return None
    return (text or "").strip() or None


def _splice_section(paper_md: str, section_title: str, new_section_md: str) -> str:
    """Replace the body of a `## <section_title>` section with `new_section_md`.

    Case-insensitive match on the heading text. If no match, appends. The new
    section text is expected to start with `## <section_title>` already.
    """
    import re as _re
    if not paper_md:
        return new_section_md.strip()
    pattern = _re.compile(
        r"(^##\s+" + _re.escape(section_title.strip()) + r"\s*$)(.*?)(?=^##\s|\Z)",
        flags=_re.IGNORECASE | _re.MULTILINE | _re.DOTALL,
    )
    if pattern.search(paper_md):
        return pattern.sub(new_section_md.strip() + "\n\n", paper_md, count=1).rstrip() + "\n"
    return paper_md.rstrip() + "\n\n" + new_section_md.strip() + "\n"


def _build_sources(sources: list[dict]) -> str:
    if not sources:
        return ""
    seen: set[str] = set()
    lines: list[str] = ["## Sources"]
    for s in sources:
        if not isinstance(s, dict):
            continue
        url = (s.get("url") or "").strip()
        title = (s.get("title") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        lines.append(f"- [{title}]({url})" if title else f"- {url}")
    return "\n".join(lines) if len(lines) > 1 else ""


# ── paper assembly ──────────────────────────────────────────────────────────

def _build_generation_notes(*, body_failed: list[str], opener_ok: bool, opener_title: str,
                            closer_ok: bool, closer_title: str,
                            takeaways_ok: bool, exec_summary_ok: bool) -> str:
    """Render a short transparency footer when one or more sections couldn't
    be generated. Empty string when everything worked — the paper looks
    clean by default."""
    missing: list[str] = []
    if not opener_ok:
        missing.append(opener_title)
    if not closer_ok:
        missing.append(closer_title)
    missing.extend(body_failed)
    if not takeaways_ok:
        missing.append("Takeaways")
    if not exec_summary_ok:
        missing.append("Executive Summary")
    if not missing:
        return ""
    bullets = "\n".join(f"- {s}" for s in missing)
    return (
        "## Generation notes\n\n"
        "_The local model could not produce the following section(s) on this "
        "run. The rest of the paper is still based on the retrieved sources; "
        "use **Review** or **Expand section** to regenerate when the model is "
        "available._\n\n"
        f"{bullets}"
    )


def _build_paper(*, topic: str, doc_type: str, queries: list[str], schema: dict,
                 hypotheses: list[str], sub_topics: list[str], org_id: int,
                 revision_notes: dict[str, str] | None = None) -> tuple[str, list[dict]]:
    """Section-by-section synthesis. Each LLM call is bounded and retried.

    Pipeline:
      1. Verify we have queries and a non-empty corpus (raises RuntimeError if not).
      2. Write opener, body sections, comparison, closer — each with retries.
      3. Decide if the paper is good enough to save:
            - Body must have ≥1 substantive section (cannot be all empty).
            - If both opener and closer failed, abort.
         If neither holds, raise RuntimeError so the caller marks failed
         rather than save junk.
      4. Write takeaways + executive summary using whatever body content
         actually succeeded.
      5. Assemble. Sections that failed are simply omitted; the paper still
         flows because each section was independent prose.
    """
    from workers.tool_queue import report_progress as _rp
    revision_notes = revision_notes or {}
    spec = DOC_TYPES.get(doc_type) or DOC_TYPES[DEFAULT_DOC_TYPE]

    if not queries:
        raise RuntimeError("no queries on plan; cannot synthesise")
    _rp(f"fetching corpus: {len(queries)} queries")
    corpus, sources = _fetch_corpus(topic, queries, org_id)
    if not corpus.strip():
        raise RuntimeError("no source material retrieved (web search failed for all queries)")
    _rp(f"corpus ready: {len(sources)} sources, {len(corpus)} chars")

    # 1. Opener
    _rp(f"writing opener: {spec['opener']}")
    opener = _write_section(
        topic=topic, doc_type=doc_type, section_title=spec["opener"],
        section_role=f"Establish the {spec['opener'].lower()} for the rest of the document.",
        corpus=corpus, hypotheses=hypotheses, target_words=500,
        revision_note=revision_notes.get(spec["opener"]),
    ) or ""

    # 2. Body — one section per sub_topic. Run in parallel: section writes
    # are independent (each only reads `corpus` + the section's own role).
    # Cap at 3 concurrent so we don't completely starve other research jobs
    # of the t1_secondary slots; the model_pool semaphore handles the rest.
    from workers.tool_queue import (
        report_progress as _report,
        is_job_cancelled as _cancelled,
        JobCancelled as _Cancelled,
        current_job_id as _current_job_id_get,
        bind_job_id as _bind_job_id,
    )
    import concurrent.futures
    import threading
    body_pieces_ordered: list[str | None] = [None] * len(sub_topics or [])
    body_failed: list[str] = []

    section_done = [0]
    section_lock = threading.Lock()
    # Capture parent contextvar; thread-pool workers don't inherit it.
    parent_job_id = _current_job_id_get()

    def _write_body(idx_sub):
        idx, sub = idx_sub
        with _bind_job_id(parent_job_id):
            if _cancelled():
                return idx, sub, None
            _report(f"section start [{idx + 1}/{len(sub_topics or [])}]: {sub[:80]}")
            sec = _write_section(
                topic=topic, doc_type=doc_type, section_title=sub,
                section_role=f"Cover '{sub}' as a substantive body section of the document.",
                corpus=corpus, hypotheses=hypotheses, target_words=700,
                revision_note=revision_notes.get(sub),
            )
            with section_lock:
                section_done[0] += 1
                done_n = section_done[0]
            _report(
                f"section done [{done_n}/{len(sub_topics or [])}]: {sub[:80]}"
                f" {'OK' if sec else 'EMPTY'}"
            )
            return idx, sub, sec

    if sub_topics:
        max_workers = min(3, len(sub_topics))
        _report(f"writing body: {len(sub_topics)} sections (parallel x{max_workers})")
        # Manual pool (not `with`) so cancel can stop queued sections
        # promptly instead of waiting for every remaining LLM call.
        pool_s = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="research-section",
        )
        try:
            futures = [pool_s.submit(_write_body, item) for item in enumerate(sub_topics)]
            for fut in concurrent.futures.as_completed(futures):
                if _cancelled():
                    for f in futures:
                        f.cancel()
                    raise _Cancelled("cancelled mid-body")
                try:
                    idx, sub, sec = fut.result()
                except concurrent.futures.CancelledError:
                    continue
                if sec:
                    body_pieces_ordered[idx] = sec
                else:
                    body_failed.append(sub)
        finally:
            pool_s.shutdown(wait=False, cancel_futures=True)
    body_pieces = [p for p in body_pieces_ordered if p]
    body = "\n\n".join(body_pieces)

    # 3. Comparison (if schema provided) — never critical, fine if it drops
    if schema:
        _rp("writing comparison block")
    comparison = _write_comparison(topic, schema, corpus) or ""

    # 4. Closer (gets opener+body so it can synthesise the spine)
    _rp(f"writing closer: {spec['closer']}")
    closer_corpus = corpus + "\n\n=== CURRENT BODY ===\n" + (opener + "\n\n" + body)[:8000]
    closer = _write_section(
        topic=topic, doc_type=doc_type, section_title=spec["closer"],
        section_role=f"Synthesise the body into the {spec['closer'].lower()}.",
        corpus=closer_corpus, hypotheses=hypotheses, target_words=600,
        revision_note=revision_notes.get(spec["closer"]),
    ) or ""

    # ── Sanity gate: only refuse if NOTHING worked ────────────────────────
    # Save what we have when the local model is patchy. Earlier the gate
    # threw out the whole paper if all body sections failed, which lost the
    # opener+closer+sources too — material that's still useful and was
    # already paid for. Operator gets a transparent 'Generation notes'
    # footer (added below) listing what dropped, plus a partial paper.
    n_body_total = len(sub_topics or [])
    n_body_ok = len(body_pieces)
    nothing_worked = (not opener and not closer and n_body_ok == 0)
    if nothing_worked:
        raise RuntimeError(
            f"synthesis produced no sections (sub_topics={n_body_total}, "
            "opener+closer also failed) — local model unavailable or overloaded"
        )

    # Log what dropped so the user has visibility on partial success
    if body_failed:
        _log.warning(
            "build_paper: body sections that failed and will be omitted: %s",
            ", ".join(repr(s) for s in body_failed),
        )
    if not opener:
        _log.warning("build_paper: opener section %r failed and will be omitted", spec["opener"])
    if not closer:
        _log.warning("build_paper: closer section %r failed and will be omitted", spec["closer"])

    # 5. Takeaways + Recommendation (closing)
    _rp("writing takeaways + recommendation")
    full_body = "\n\n".join(p for p in (opener, body, comparison, closer) if p)
    takeaways = _write_takeaways_and_recommendation(
        topic=topic, doc_type=doc_type, body_md=full_body,
    ) or ""

    # 6. Executive summary — last, so it summarises real content
    _rp("writing executive summary")
    exec_summary = _write_executive_summary(
        topic=topic, doc_type=doc_type, body_md=full_body,
    ) or ""
    _rp("paper assembled")

    # 7. Sources
    sources_md = _build_sources(sources)

    # 8. Generation notes — surfaced only when something dropped, so a
    # partial paper carries its own caveat instead of looking complete.
    gen_notes = _build_generation_notes(
        body_failed=body_failed,
        opener_ok=bool(opener), opener_title=spec["opener"],
        closer_ok=bool(closer), closer_title=spec["closer"],
        takeaways_ok=bool(takeaways), exec_summary_ok=bool(exec_summary),
    )

    parts = [f"# {topic}".strip()]
    for piece in (exec_summary, opener, body, comparison, closer, takeaways, sources_md, gen_notes):
        if piece and piece.strip():
            parts.append(piece.strip())
    paper = "\n\n".join(parts)
    _log.info(
        "build_paper DONE  body_ok=%d/%d  opener=%s  closer=%s  takeaways=%s  exec_summary=%s  comparison=%s  total_chars=%d",
        n_body_ok, n_body_total,
        "ok" if opener else "MISSING",
        "ok" if closer else "MISSING",
        "ok" if takeaways else "MISSING",
        "ok" if exec_summary else "MISSING",
        "ok" if comparison else "skipped",
        len(paper),
    )
    return paper, sources


# ── public entry points ─────────────────────────────────────────────────────

def run_research_agent(plan_id: int) -> dict:
    """Tool-queue handler: produce the paper for an existing plan row."""
    if not get_feature("research", "agent_enabled", True):
        return {"status": "disabled", "error": "research_agent feature disabled"}

    from workers.tool_queue import report_progress, is_job_cancelled, JobCancelled

    client = NocodbClient()
    try:
        report_progress(f"loading plan {plan_id}")
        plan_row = client._get("research_plans", params={"where": f"(Id,eq,{plan_id})", "limit": 1})
        plan = plan_row.get("list", [])[0] if plan_row.get("list") else None
        if not plan:
            return {"status": "not_found", "plan_id": plan_id}

        topic = plan.get("topic", "")
        queries = _safe_json_loads(plan.get("queries", "[]"), [])
        schema = _safe_json_loads(plan.get("schema", "{}"), {}) or {}
        hypotheses = _safe_json_loads(plan.get("hypotheses", "[]"), [])
        sub_topics = _safe_json_loads(plan.get("sub_topics", "[]"), [])
        iterations = plan.get("iterations", 0) or 0
        org_id = resolve_org_id(plan.get("org_id"))

        # doc_type may have been stashed by an earlier run; otherwise infer.
        planned_doc_type = schema.pop("_doc_type", None) if isinstance(schema, dict) else None
        doc_type = _infer_doc_type(topic, planned_doc_type=planned_doc_type)

        report_progress(
            f"plan loaded: {len(queries)} queries, {len(sub_topics)} sub-topics, doc_type={doc_type}"
        )
        if is_job_cancelled():
            raise JobCancelled("cancelled before search")

        _patch_or_log(client, plan_id, {"status": "searching"}, "searching")
        report_progress(f"searching: {len(queries)} queries → web search + extract")

        paper, _sources = _build_paper(
            topic=topic, doc_type=doc_type, queries=queries, schema=schema,
            hypotheses=hypotheses, sub_topics=sub_topics, org_id=org_id,
        )

        if not paper or not paper.strip():
            client._patch("research_plans", plan_id, {
                "status": "failed",
                "error_message": "synthesis produced empty paper",
            })
            return {"status": "failed", "plan_id": plan_id, "error": "empty paper"}

        # Save paper_content FIRST in its own patch — if any later metadata
        # field is rejected (column missing, type mismatch, oversize, etc.) the
        # paper is still durably persisted and the user does not lose it.
        try:
            client._patch("research_plans", plan_id, {"paper_content": paper})
            _log.info("research paper saved  plan_id=%d  chars=%d", plan_id, len(paper))
        except Exception as save_exc:
            _log.error(
                "research paper save failed  plan_id=%d  paper_chars=%d  err=%s",
                plan_id, len(paper), save_exc, exc_info=True,
            )
            # Try a fallback minimal patch with just the error
            try:
                client._patch("research_plans", plan_id, {
                    "status": "failed",
                    "error_message": f"paper save failed: {str(save_exc)[:300]}",
                })
            except Exception:
                pass
            return {"status": "failed", "plan_id": plan_id, "error": f"paper save failed: {str(save_exc)[:200]}"}

        # Now metadata. Each piece in its own patch so a single rejection does
        # not lose the rest. Failures here are warnings, not fatal — the paper
        # is already saved.
        schema_to_save = dict(schema or {})
        schema_to_save["_doc_type"] = doc_type
        for label, fields in (
            ("status_complete", {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            }),
            ("iterations", {"iterations": iterations + 1}),
            ("schema", {"schema": json.dumps(schema_to_save)}),
        ):
            try:
                client._patch("research_plans", plan_id, fields)
            except Exception:
                _log.warning(
                    "research metadata patch failed  plan_id=%d  label=%s",
                    plan_id, label, exc_info=True,
                )

        try:
            from workers.post_turn import ingest_output
            ingest_output(
                output=paper, user_text=topic, org_id=org_id, conversation_id=0,
                model="research_agent", rag_collection="research",
                knowledge_collection="research_knowledge", source="research",
                extra_metadata={"plan_id": plan_id, "topic": topic, "doc_type": doc_type},
            )
        except Exception:
            _log.warning("research ingest_output failed  plan_id=%d", plan_id, exc_info=True)
        try:
            from shared.insights import append_research
            focus = str(plan.get("focus") or "").strip()
            append_research(plan_id, paper, focus=focus)
        except Exception:
            _log.warning("research append_to_insight failed  plan_id=%d", plan_id, exc_info=True)

        return {"status": "completed", "plan_id": plan_id, "doc_type": doc_type}
    except Exception as e:
        _log.error("research_agent uncaught error  plan_id=%d", plan_id, exc_info=True)
        _patch_or_log(client, plan_id, {
            "status": "failed",
            "error_message": f"uncaught: {str(e)[:300]}",
        }, "failed-uncaught")
        return {"status": "failed", "plan_id": plan_id, "error": str(e)[:300]}


def review_research_paper(plan_id: int, user_instructions: str = "") -> dict:
    """Explicit user-triggered review pass.

    Step 1: a big-model reviewer reads the existing paper and emits per-section
    revision instructions.
    Step 2: the writer rebuilds the paper with those instructions appended to
    the affected sections. The new paper replaces the old.
    """
    client = NocodbClient()
    try:
        plan_row = client._get("research_plans", params={"where": f"(Id,eq,{plan_id})", "limit": 1})
        plan = plan_row.get("list", [])[0] if plan_row.get("list") else None
        if not plan:
            return {"status": "not_found", "plan_id": plan_id}

        topic = plan.get("topic", "")
        queries = _safe_json_loads(plan.get("queries", "[]"), [])
        schema = _safe_json_loads(plan.get("schema", "{}"), {}) or {}
        hypotheses = _safe_json_loads(plan.get("hypotheses", "[]"), [])
        sub_topics = _safe_json_loads(plan.get("sub_topics", "[]"), [])
        iterations = plan.get("iterations", 0) or 0
        org_id = resolve_org_id(plan.get("org_id"))
        prior_paper = (plan.get("paper_content") or "").strip()

        if not prior_paper:
            return {"status": "failed", "plan_id": plan_id, "error": "no paper to review"}

        planned_doc_type = schema.pop("_doc_type", None) if isinstance(schema, dict) else None
        doc_type = _infer_doc_type(topic, planned_doc_type=planned_doc_type)

        _patch_or_log(client, plan_id, {"status": "reviewing"}, "reviewing")

        revision_notes = _generate_revision_notes(
            topic=topic, doc_type=doc_type, paper=prior_paper,
            user_instructions=user_instructions, sub_topics=sub_topics,
        )

        if not revision_notes:
            # Reviewer found nothing actionable. Preserve prior paper, mark complete.
            schema_to_save = dict(schema or {})
            schema_to_save["_doc_type"] = doc_type
            client._patch("research_plans", plan_id, {
                "status": "completed",
                "schema": json.dumps(schema_to_save),
                "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                "error_message": "review found no revisions to apply",
            })
            return {"status": "completed", "plan_id": plan_id, "note": "no_revisions"}

        _patch_or_log(client, plan_id, {"status": "revising"}, "revising")

        # Targeted splice: re-run only the sections the reviewer flagged, then
        # replace them in the prior paper. Avoids re-doing the whole 8-section
        # build on every review (which was effectively as expensive as the
        # initial synthesis).
        spec = DOC_TYPES.get(doc_type) or DOC_TYPES[DEFAULT_DOC_TYPE]
        corpus, _src = _fetch_corpus(topic, queries, org_id)
        new_paper = prior_paper
        revised_count = 0
        for sec_title, note in revision_notes.items():
            target_words = 600 if sec_title == spec["opener"] else (700 if sec_title == spec["closer"] else 900)
            section_role = (
                f"Rewrite the '{sec_title}' section per the revision instructions, "
                "keeping the same heading. Match the document tone."
            )
            sec_md = _write_section(
                topic=topic, doc_type=doc_type, section_title=sec_title,
                section_role=section_role, corpus=corpus, hypotheses=hypotheses,
                target_words=target_words, revision_note=note,
            )
            if not sec_md:
                continue
            new_paper = _splice_section(new_paper, sec_title, sec_md)
            revised_count += 1

        if revised_count == 0 or not new_paper or not new_paper.strip():
            client._patch("research_plans", plan_id, {
                "status": "completed",
                "error_message": "all section rewrites failed; prior paper preserved",
            })
            return {"status": "failed", "plan_id": plan_id, "error": "no sections revised"}

        schema_to_save = dict(schema or {})
        schema_to_save["_doc_type"] = doc_type
        client._patch("research_plans", plan_id, {
            "status": "completed",
            "paper_content": new_paper,
            "schema": json.dumps(schema_to_save),
            "iterations": iterations + 1,
            "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        })

        try:
            from shared.insights import append_research
            focus = str(plan.get("focus") or "").strip()
            append_research(plan_id, new_paper, focus=focus)
        except Exception:
            _log.warning("review append_to_insight failed  plan_id=%d", plan_id, exc_info=True)

        return {
            "status": "completed", "plan_id": plan_id, "doc_type": doc_type,
            "revisions_applied": len(revision_notes),
        }
    except Exception as e:
        _log.error("research review uncaught error  plan_id=%d", plan_id, exc_info=True)
        _patch_or_log(client, plan_id, {
            "status": "failed",
            "error_message": f"review uncaught: {str(e)[:300]}",
        }, "review-uncaught")
        return {"status": "failed", "plan_id": plan_id, "error": str(e)[:300]}


# ── reviewer ────────────────────────────────────────────────────────────────

def _generate_revision_notes(*, topic: str, doc_type: str, paper: str,
                             user_instructions: str, sub_topics: list[str]) -> dict[str, str]:
    """Big-model review pass. Returns {section_title: instruction_text}.

    Empty dict means the reviewer signalled the paper is fine as-is (or the
    response failed to parse — we treat that as 'no actionable revisions'
    rather than blocking the user).
    """
    timeout_s = _research_timeout("reviewer_timeout_s", DEFAULT_REVIEWER_TIMEOUT_S)
    spec = DOC_TYPES.get(doc_type) or DOC_TYPES[DEFAULT_DOC_TYPE]

    user_block = ""
    if user_instructions and user_instructions.strip():
        user_block = f"\nUSER REVIEW NOTES (apply where relevant):\n{user_instructions.strip()}\n"

    section_list = [spec["opener"], *(sub_topics or []), spec["closer"]]
    section_block = "\n".join(f"- {s}" for s in section_list)

    prompt = f"""You are reviewing a {doc_type.replace('_', ' ')} for accuracy, depth, coherence, and tone match.

TOPIC: {topic}

DOCUMENT TONE TARGET: {spec['tone']}

REVISABLE SECTIONS (you may emit revision notes for any of these by exact name):
{section_block}
{user_block}
THE FULL PAPER:
{paper[:60000]}

Return ONLY a JSON object with this shape:
{{
  "overall_assessment": "<2-3 sentence summary of the paper's strengths and weaknesses>",
  "revisions": [
    {{"section": "<exact section title from the list above>", "instructions": "<concrete revision instruction — what to add, remove, sharpen, or correct, including which sources to lean on>"}}
  ]
}}

Rules:
- ONLY include sections that genuinely need work. If a section is fine, omit it.
- If the paper is solid as-is, return {{"overall_assessment": "...", "revisions": []}}.
- Be specific in instructions: name the claim that's missing or weak, the angle to add, the citation to add or remove. Generic instructions ("expand more", "improve flow") are not allowed.
- Output raw JSON only — no markdown fences, no preamble, no trailing prose."""

    res = _safe_call(
        lambda: model_call("research_reviewer", prompt, temperature=0.2),
        timeout_s,
        "reviewer",
    )
    if not res:
        return {}
    try:
        raw, _ = res
    except (TypeError, ValueError):
        return {}
    raw = (raw or "").strip()
    if not raw:
        return {}
    if raw.startswith("```"):
        raw = raw.lstrip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip().rstrip("`").strip()
    # Forgiving cleanup for local-model JSON output: smart quotes → straight,
    # trailing commas before } or ] dropped. Same fix the planner applies.
    import re as _re
    raw = raw.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    raw = _re.sub(r",\s*([}\]])", r"\1", raw)
    parsed = None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start:end + 1])
            except (json.JSONDecodeError, TypeError):
                parsed = None
    if not isinstance(parsed, dict):
        return {}
    revisions = parsed.get("revisions") or []
    notes: dict[str, str] = {}
    for r in revisions:
        if not isinstance(r, dict):
            continue
        sec = (r.get("section") or "").strip()
        ins = (r.get("instructions") or "").strip()
        if sec and ins and sec not in _PROTECTED_SECTIONS:
            notes[sec] = ins
    return notes


# ── tool-queue compatibility ────────────────────────────────────────────────

def get_next_research() -> dict | None:
    client = NocodbClient()
    try:
        data = client._get("research_plans", params={
            "where": "(status,eq,generating)", "limit": 1,
        })
        rows = data.get("list", [])
        return rows[0] if rows else None
    except Exception:
        return None
