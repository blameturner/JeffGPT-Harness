"""Post-build operations on a completed research_plans paper.

All operations are user-invoked (no auto-runs). They are dispatched through
a single tool-queue handler ``research_op`` (see ``run_research_op`` below).
Two SYNC ops (``citation_audit``, ``chat_with_paper``) are cheap and run
inline from the API endpoint.

Operations either:
  - mutate ``paper_content`` on the plan row (rewrites)
  - write to ``artifacts_json`` on the plan row (derived/report content as
    a single JSON dict keyed by op name — keeps DB column footprint small)
  - return inline (chat_with_paper creates a conversation row)
"""
import json
import logging
import re
from datetime import datetime, timezone

from infra.nocodb_client import NocodbClient
from shared.models import model_call
from tools._org import resolve_org_id

from tools.research.agent import (
    DOC_TYPES,
    DEFAULT_DOC_TYPE,
    DEFAULT_REVIEWER_TIMEOUT_S,
    DEFAULT_SECTION_TIMEOUT_S,
    _call_with_timeout,
    _fetch_corpus,
    _infer_doc_type,
    _research_timeout,
    _safe_json_loads,
    _write_section,
)

_log = logging.getLogger("research.ops")


# ── small utilities ─────────────────────────────────────────────────────────

def _load_plan(plan_id: int):
    client = NocodbClient()
    row = client._get("research_plans", params={"where": f"(Id,eq,{plan_id})", "limit": 1})
    plan = row.get("list", [])[0] if row.get("list") else None
    return client, plan


def _doc_type_of(plan: dict, topic: str) -> str:
    schema = _safe_json_loads(plan.get("schema", "{}"), {}) or {}
    planned = schema.get("_doc_type") if isinstance(schema, dict) else None
    return _infer_doc_type(topic, planned_doc_type=planned)


def _load_schema(plan: dict) -> dict:
    raw = plan.get("schema") or "{}"
    val = _safe_json_loads(raw, {})
    return val if isinstance(val, dict) else {}


def _load_artifacts(plan: dict) -> dict:
    """Artifacts are stashed inside the existing ``schema`` JSON column under
    the reserved ``_artifacts`` key so we don't need a new DB column.
    """
    schema = _load_schema(plan)
    arts = schema.get("_artifacts") if isinstance(schema, dict) else None
    return arts if isinstance(arts, dict) else {}


def _save_artifact(client, plan_id: int, plan: dict, kind: str, text: str) -> None:
    schema = _load_schema(plan)
    arts = schema.get("_artifacts") if isinstance(schema.get("_artifacts"), dict) else {}
    arts[kind] = {
        "text": text,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    }
    schema["_artifacts"] = arts
    try:
        client._patch("research_plans", plan_id, {"schema": json.dumps(schema)})
    except Exception:
        _log.warning("save_artifact: patch failed plan_id=%d kind=%s", plan_id, kind, exc_info=True)


def _section_split(paper_md: str):
    """Return [(heading_full_line, heading_text, body), ...] for top-level ## sections.

    `# Title` is treated as preamble. Each section runs from `## H` to just
    before the next `## ` (or end of document). `###` subsections stay inside
    their parent's body.
    """
    if not paper_md:
        return []
    out = []
    cur_head = None
    cur_text = ""
    cur_body: list[str] = []
    for line in paper_md.splitlines():
        m = re.match(r"^##\s+(.+)$", line)
        if m and not line.startswith("###"):
            if cur_head is not None:
                out.append((cur_head, cur_text, "\n".join(cur_body).rstrip()))
            cur_head = line
            cur_text = m.group(1).strip()
            cur_body = []
        else:
            if cur_head is None:
                continue
            cur_body.append(line)
    if cur_head is not None:
        out.append((cur_head, cur_text, "\n".join(cur_body).rstrip()))
    return out


def _preamble(paper_md: str, sections) -> str:
    if not sections:
        return paper_md
    first_idx = paper_md.find(sections[0][0])
    return paper_md[:first_idx] if first_idx > 0 else ""


def _replace_section(paper_md: str, target_title: str, new_section_md: str) -> str:
    sections = _section_split(paper_md)
    if not sections:
        return (paper_md.rstrip() + "\n\n" + new_section_md.strip()).strip()
    pre = _preamble(paper_md, sections).rstrip()
    out = [pre] if pre else []
    replaced = False
    for full_head, text, body in sections:
        if (not replaced) and text.strip().lower() == target_title.strip().lower():
            out.append(new_section_md.strip())
            replaced = True
        else:
            piece = full_head + (("\n" + body) if body else "")
            out.append(piece)
    if not replaced:
        out.append(new_section_md.strip())
    return "\n\n".join(p for p in out if p and p.strip())


def _insert_section_before(paper_md: str, before_title: str, new_section_md: str) -> str:
    sections = _section_split(paper_md)
    if not sections:
        return (paper_md.rstrip() + "\n\n" + new_section_md.strip()).strip()
    pre = _preamble(paper_md, sections).rstrip()
    out = [pre] if pre else []
    inserted = False
    for full_head, text, body in sections:
        if (not inserted) and before_title and text.strip().lower() == before_title.strip().lower():
            out.append(new_section_md.strip())
            inserted = True
        out.append(full_head + (("\n" + body) if body else ""))
    if not inserted:
        out.append(new_section_md.strip())
    return "\n\n".join(p for p in out if p and p.strip())


def _insert_section_after(paper_md: str, after_title: str, new_section_md: str) -> str:
    sections = _section_split(paper_md)
    if not sections:
        return (paper_md.rstrip() + "\n\n" + new_section_md.strip()).strip()
    pre = _preamble(paper_md, sections).rstrip()
    out = [pre] if pre else []
    inserted = False
    for full_head, text, body in sections:
        out.append(full_head + (("\n" + body) if body else ""))
        if (not inserted) and after_title and text.strip().lower() == after_title.strip().lower():
            out.append(new_section_md.strip())
            inserted = True
    if not inserted:
        out.append(new_section_md.strip())
    return "\n\n".join(p for p in out if p and p.strip())


def _build_corpus(plan: dict, topic: str, org_id: int):
    queries = _safe_json_loads(plan.get("queries", "[]"), []) or []
    return _fetch_corpus(topic, queries, org_id)


# ── operations ──────────────────────────────────────────────────────────────

def fact_check_paper(plan_id: int) -> dict:
    client, plan = _load_plan(plan_id)
    if not plan:
        return {"status": "not_found", "plan_id": plan_id}
    paper = (plan.get("paper_content") or "").strip()
    if not paper:
        return {"status": "failed", "error": "no paper to fact-check"}
    topic = plan.get("topic", "")
    org_id = resolve_org_id(plan.get("org_id"))
    corpus, _src = _build_corpus(plan, topic, org_id)
    timeout_s = _research_timeout("reviewer_timeout_s", DEFAULT_REVIEWER_TIMEOUT_S)
    prompt = f"""You are fact-checking a research paper against its source corpus.

TOPIC: {topic}

SOURCE CORPUS (the only ground truth):
{corpus[:50000]}

PAPER:
{paper[:50000]}

For every concrete factual claim in the PAPER (numbers, dates, attributions, named entities, comparisons), classify it as one of:
- supported — the source corpus directly supports it
- weak — only partially or implicitly supported
- unsupported — not present in the source corpus
- contradicted — the source corpus says otherwise

Output a Markdown report:
# Fact-Check Report

## Summary
<2-3 sentences with the supported / weak / unsupported / contradicted counts and headline issues>

## Findings
For each non-supported claim emit a bullet:
- [<status>] "<claim quoted from paper>" — <one-line evidence note + source URL>

Supported claims may be summarised in aggregate at the top of Findings. Output raw Markdown only, no preamble."""
    res = _call_with_timeout(
        lambda: model_call("research_reviewer", prompt, temperature=0.1),
        timeout_s, "fact_check",
    )
    if not res:
        return {"status": "failed", "error": "fact_check timeout/error"}
    try:
        text, _ = res
    except (TypeError, ValueError):
        return {"status": "failed", "error": "bad model response"}
    text = (text or "").strip()
    if not text:
        return {"status": "failed", "error": "empty"}
    _save_artifact(client, plan_id, plan, "fact_check", text)
    return {"status": "completed", "plan_id": plan_id, "kind": "fact_check"}


def citation_audit_paper(plan_id: int) -> dict:
    """Cheap, no-LLM. Parses [Source: URL] inline citations and the ## Sources block."""
    client, plan = _load_plan(plan_id)
    if not plan:
        return {"status": "not_found", "plan_id": plan_id}
    paper = (plan.get("paper_content") or "").strip()
    if not paper:
        return {"status": "failed", "error": "no paper"}
    inline_urls: set[str] = set()
    for s in re.findall(r"\[Source:\s*([^\]]+)\]", paper):
        token = s.strip().split()[0].strip(",.;:") if s.strip() else ""
        if token:
            inline_urls.add(token)
    src_match = re.search(r"##\s*Sources\s*\n(.*?)(?:\n##\s|\Z)", paper, flags=re.DOTALL | re.IGNORECASE)
    bib_urls: set[str] = set()
    if src_match:
        block = src_match.group(1)
        for url in re.findall(r"\((https?://[^)\s]+)\)", block):
            bib_urls.add(url)
        for url in re.findall(r"(https?://\S+)", block):
            bib_urls.add(url.strip(",.;:)"))
    orphan = sorted(u for u in inline_urls if u not in bib_urls)
    unused = sorted(u for u in bib_urls if u not in inline_urls)
    lines = [
        "# Citation Audit",
        "",
        f"- Inline citations: **{len(inline_urls)}** unique URLs",
        f"- Bibliography entries: **{len(bib_urls)}** URLs",
        f"- Orphan inline (cited but not in bibliography): **{len(orphan)}**",
        f"- Unused bibliography (in sources but never cited): **{len(unused)}**",
    ]
    if orphan:
        lines += ["", "## Orphan inline citations", *(f"- {u}" for u in orphan)]
    if unused:
        lines += ["", "## Unused bibliography entries", *(f"- {u}" for u in unused)]
    report = "\n".join(lines)
    _save_artifact(client, plan_id, plan, "citation_audit", report)
    return {
        "status": "completed", "plan_id": plan_id, "kind": "citation_audit",
        "orphan_count": len(orphan), "unused_count": len(unused),
        "inline_count": len(inline_urls), "bibliography_count": len(bib_urls),
    }


_SPINE_PROTECTED = {"executive summary", "key takeaways", "sources", "comparison"}


def expand_section(plan_id: int, section_title: str, target_words: int = 1800) -> dict:
    if not section_title:
        return {"status": "failed", "error": "section_title required"}
    if section_title.strip().lower() in _SPINE_PROTECTED:
        return {"status": "failed", "error": f"'{section_title}' is auto-generated; use review or resize instead"}
    client, plan = _load_plan(plan_id)
    if not plan:
        return {"status": "not_found", "plan_id": plan_id}
    paper = (plan.get("paper_content") or "").strip()
    if not paper:
        return {"status": "failed", "error": "no paper"}
    topic = plan.get("topic", "")
    org_id = resolve_org_id(plan.get("org_id"))
    doc_type = _doc_type_of(plan, topic)
    hypotheses = _safe_json_loads(plan.get("hypotheses", "[]"), []) or []
    corpus, _src = _build_corpus(plan, topic, org_id)
    sec = _write_section(
        topic=topic, doc_type=doc_type, section_title=section_title,
        section_role=(
            f"Deepen the '{section_title}' section to roughly {target_words} words. "
            "Add evidence, analytical depth, and contrast across sources. Do not pad."
        ),
        corpus=corpus, hypotheses=hypotheses, target_words=target_words,
    )
    if not sec:
        return {"status": "failed", "error": "writer timeout/error"}
    new_paper = _replace_section(paper, section_title, sec)
    client._patch("research_plans", plan_id, {"paper_content": new_paper})
    return {"status": "completed", "plan_id": plan_id, "kind": "expand_section", "section": section_title}


def add_new_section(plan_id: int, heading: str, brief: str = "",
                    after_section: str | None = None, target_words: int = 1000) -> dict:
    if not heading:
        return {"status": "failed", "error": "heading required"}
    if heading.strip().lower() in _SPINE_PROTECTED:
        return {"status": "failed", "error": f"'{heading}' is reserved; pick a different heading"}
    client, plan = _load_plan(plan_id)
    if not plan:
        return {"status": "not_found", "plan_id": plan_id}
    paper = (plan.get("paper_content") or "").strip()
    if not paper:
        return {"status": "failed", "error": "no paper"}
    topic = plan.get("topic", "")
    org_id = resolve_org_id(plan.get("org_id"))
    doc_type = _doc_type_of(plan, topic)
    hypotheses = _safe_json_loads(plan.get("hypotheses", "[]"), []) or []
    corpus, _src = _build_corpus(plan, topic, org_id)
    role = brief or f"New body section: {heading}. Cover this angle of the topic substantively."
    sec = _write_section(
        topic=topic, doc_type=doc_type, section_title=heading,
        section_role=role, corpus=corpus, hypotheses=hypotheses,
        target_words=target_words,
    )
    if not sec:
        return {"status": "failed", "error": "writer timeout/error"}
    if after_section:
        new_paper = _insert_section_after(paper, after_section, sec)
    else:
        # Default: insert just before ## Sources, or append if no Sources block
        new_paper = _insert_section_before(paper, "Sources", sec)
    client._patch("research_plans", plan_id, {"paper_content": new_paper})
    return {"status": "completed", "plan_id": plan_id, "kind": "add_section", "heading": heading}


def add_counter_arguments(plan_id: int, target_words: int = 900) -> dict:
    return add_new_section(
        plan_id,
        heading="Counter-arguments and Limitations",
        brief=(
            "Steel-man the strongest opposing view to this paper's conclusions, and surface the "
            "limitations of the evidence used. Use the same source corpus; cite where possible. "
            "Be intellectually honest — do not strawman."
        ),
        target_words=target_words,
    )


def add_fresh_sources(plan_id: int, queries: list[str] | None = None) -> dict:
    client, plan = _load_plan(plan_id)
    if not plan:
        return {"status": "not_found", "plan_id": plan_id}
    paper = (plan.get("paper_content") or "").strip()
    if not paper:
        return {"status": "failed", "error": "no paper"}
    topic = plan.get("topic", "")
    org_id = resolve_org_id(plan.get("org_id"))
    new_queries = list(queries or [])
    existing = _safe_json_loads(plan.get("queries", "[]"), []) or []
    if not new_queries:
        prompt = f"""Generate 4 fresh, high-signal search queries to deepen this research paper. Avoid topics already covered. Output one query per line, plain text only.

TOPIC: {topic}

EXISTING QUERIES:
{chr(10).join(existing[:20])}

PAPER (excerpt):
{paper[:8000]}"""
        res = _call_with_timeout(
            lambda: model_call("research_planner", prompt, temperature=0.3, max_tokens=400),
            300, "fresh_query_gen",
        )
        if res:
            try:
                txt, _ = res
            except (TypeError, ValueError):
                txt = ""
            for ln in (txt or "").splitlines():
                cand = ln.strip(" -*\t").strip()
                cand = re.sub(r"^\d+[\).\s]+", "", cand).strip()
                if cand and cand.lower() not in {q.lower() for q in existing + new_queries}:
                    new_queries.append(cand)
                if len(new_queries) >= 4:
                    break
    if not new_queries:
        return {"status": "failed", "error": "no fresh queries"}
    fresh_corpus, _fresh_sources = _fetch_corpus(topic, new_queries, org_id)
    if not fresh_corpus.strip():
        return {"status": "failed", "error": "fresh queries returned no material"}
    doc_type = _doc_type_of(plan, topic)
    hypotheses = _safe_json_loads(plan.get("hypotheses", "[]"), []) or []
    sec = _write_section(
        topic=topic, doc_type=doc_type, section_title="Recent Additions",
        section_role=(
            "Synthesise the fresh source material below into a substantive section that extends "
            "the paper. Cite every claim. Do not duplicate what's already covered earlier."
        ),
        corpus=fresh_corpus, hypotheses=hypotheses, target_words=900,
    )
    if not sec:
        return {"status": "failed", "error": "writer error"}
    new_paper = _insert_section_before(paper, "Sources", sec)
    merged = list(existing) + [q for q in new_queries if q not in existing]
    client._patch("research_plans", plan_id, {
        "paper_content": new_paper,
        "queries": json.dumps(merged),
    })
    return {"status": "completed", "plan_id": plan_id, "kind": "add_fresh_sources",
            "new_queries": new_queries}


def refresh_for_recency(plan_id: int, since_date: str = "") -> dict:
    client, plan = _load_plan(plan_id)
    if not plan:
        return {"status": "not_found", "plan_id": plan_id}
    base_queries = (_safe_json_loads(plan.get("queries", "[]"), []) or [])[:6]
    if not since_date:
        since_date = datetime.now(timezone.utc).strftime("%Y")
    # Plain "since X" fares poorly with web search; appending the year as a
    # bare token surfaces date-tagged results far more reliably.
    fresh = [f"{q} {since_date}" for q in base_queries]
    return add_fresh_sources(plan_id, queries=fresh)


def reframe_for_audience(plan_id: int, audience: str = "non-technical executive") -> dict:
    client, plan = _load_plan(plan_id)
    if not plan:
        return {"status": "not_found", "plan_id": plan_id}
    paper = (plan.get("paper_content") or "").strip()
    if not paper:
        return {"status": "failed", "error": "no paper"}
    topic = plan.get("topic", "")
    timeout_s = _research_timeout("reviewer_timeout_s", DEFAULT_REVIEWER_TIMEOUT_S)
    prompt = f"""Rewrite this paper for a {audience} audience.

REQUIREMENTS:
- Preserve every fact, citation, and quantitative claim. Do NOT remove sources.
- Adjust register, sentence length, vocabulary, and emphasis to suit the audience.
- Keep the same section headings and section order.
- Keep the inline `[Source: URL]` format and the same ## Sources block at the end.

TOPIC: {topic}
TARGET AUDIENCE: {audience}

PAPER:
{paper[:60000]}

Output the reframed paper in full as raw Markdown. No preamble, no closing summary."""
    # Whole-paper rewrite — needs more output budget than the reviewer's
    # default 12k tokens for very long papers. 24k tokens covers ~17k words.
    res = _call_with_timeout(
        lambda: model_call("research_reviewer", prompt, temperature=0.25, max_tokens=24000),
        timeout_s, "reframe",
    )
    if not res:
        return {"status": "failed", "error": "reframe timeout/error"}
    try:
        text, _ = res
    except (TypeError, ValueError):
        return {"status": "failed", "error": "bad model response"}
    new_paper = (text or "").strip()
    if not new_paper:
        return {"status": "failed", "error": "empty"}
    client._patch("research_plans", plan_id, {"paper_content": new_paper})
    return {"status": "completed", "plan_id": plan_id, "kind": "reframe", "audience": audience}


def resize_paper(plan_id: int, target_words: int = 3000) -> dict:
    client, plan = _load_plan(plan_id)
    if not plan:
        return {"status": "not_found", "plan_id": plan_id}
    paper = (plan.get("paper_content") or "").strip()
    if not paper:
        return {"status": "failed", "error": "no paper"}
    topic = plan.get("topic", "")
    timeout_s = _research_timeout("reviewer_timeout_s", DEFAULT_REVIEWER_TIMEOUT_S)
    direction = "tighten" if target_words < len(paper.split()) else "expand"
    prompt = f"""Resize this paper to ~{target_words} total words by {direction}ing.

REQUIREMENTS:
- Preserve every cited fact and the same ## Sources block.
- Keep the same section headings.
- If tightening: cut filler and repetition.
- If expanding: add evidentiary depth from the existing citations only — do NOT invent new sources.
- Keep inline `[Source: URL]` citations.

TOPIC: {topic}

PAPER:
{paper[:60000]}

Output the resized paper in full as raw Markdown."""
    # Resize emits the whole paper too; size budget to the target word count
    # (rough rule: 1 word ≈ 1.4 tokens) plus headroom.
    rs_max_tokens = max(4000, min(int(target_words * 1.6), 24000))
    res = _call_with_timeout(
        lambda: model_call("research_reviewer", prompt, temperature=0.25, max_tokens=rs_max_tokens),
        timeout_s, "resize",
    )
    if not res:
        return {"status": "failed", "error": "resize timeout/error"}
    try:
        text, _ = res
    except (TypeError, ValueError):
        return {"status": "failed", "error": "bad model response"}
    new_paper = (text or "").strip()
    if not new_paper:
        return {"status": "failed", "error": "empty"}
    client._patch("research_plans", plan_id, {"paper_content": new_paper})
    return {"status": "completed", "plan_id": plan_id, "kind": "resize", "target_words": target_words}


def _generic_artifact(plan_id: int, kind: str, prompt_builder, max_tokens: int = 2500,
                      role: str = "research_section_writer") -> dict:
    client, plan = _load_plan(plan_id)
    if not plan:
        return {"status": "not_found", "plan_id": plan_id}
    paper = (plan.get("paper_content") or "").strip()
    if not paper:
        return {"status": "failed", "error": "no paper"}
    topic = plan.get("topic", "")
    prompt = prompt_builder(topic, paper)
    timeout_s = _research_timeout("section_timeout_s", DEFAULT_SECTION_TIMEOUT_S)
    res = _call_with_timeout(
        lambda: model_call(role, prompt, temperature=0.3, max_tokens=max_tokens),
        timeout_s, kind,
    )
    if not res:
        return {"status": "failed", "error": f"{kind} timeout/error"}
    try:
        text, _ = res
    except (TypeError, ValueError):
        return {"status": "failed", "error": "bad model response"}
    text = (text or "").strip()
    if not text:
        return {"status": "failed", "error": "empty"}
    _save_artifact(client, plan_id, plan, kind, text)
    return {"status": "completed", "plan_id": plan_id, "kind": kind}


def generate_slide_deck(plan_id: int) -> dict:
    def build(topic: str, paper: str) -> str:
        return f"""Turn this research paper into a slide deck outline.

TOPIC: {topic}

PAPER:
{paper[:50000]}

Output 8-12 slides in Markdown. For each slide:
- `## Slide N: <Title>`
- 3-6 concise bullet points
- A `Source: <URL>` line listing the 1-2 most load-bearing citations on that slide

Slide flow: hook / context / 3-6 substantive content slides / synthesis / recommendation / takeaways. Output the deck only, no preamble."""
    return _generic_artifact(plan_id, "slide_deck", build, max_tokens=3500)


def generate_email_tldr(plan_id: int) -> dict:
    def build(topic: str, paper: str) -> str:
        return f"""Write a 200-300 word email digest of this paper for a colleague who hasn't read it.

TOPIC: {topic}

PAPER:
{paper[:50000]}

Format:
- `Subject: <one-line>`
- 2-3 short paragraphs of plain prose. Carry the 2-3 most load-bearing citations inline as `[Source: URL]`.
- Close with one line: the key takeaway or recommended next step.

Output the email body only, no preamble."""
    return _generic_artifact(plan_id, "email_tldr", build, max_tokens=900)


def generate_qa_pack(plan_id: int) -> dict:
    def build(topic: str, paper: str) -> str:
        return f"""Generate the 10 most likely questions a reader will ask about this paper, and answer each strictly from the paper's content.

TOPIC: {topic}

PAPER:
{paper[:50000]}

Format as Markdown:
## Q1: <question>
<2-4 sentence answer drawn from the paper, with `[Source: URL]` citations carried over>

…continue through Q10.

If the paper does not answer a question, say so explicitly rather than inventing. Output only the Q&A pack, no preamble."""
    return _generic_artifact(plan_id, "qa_pack", build, max_tokens=3500)


def generate_action_plan(plan_id: int) -> dict:
    def build(topic: str, paper: str) -> str:
        return f"""Extract a concrete action plan from this paper.

TOPIC: {topic}

PAPER:
{paper[:50000]}

Format as a Markdown checklist grouped by horizon:
## Now (next 2 weeks)
- [ ] <action> — <why, with `[Source: URL]` if grounded in paper>

## Next (1-3 months)
- [ ] …

## Later (3-12 months)
- [ ] …

Only include actions the paper genuinely supports. If a horizon has nothing, omit it. No preamble."""
    return _generic_artifact(plan_id, "action_plan", build, max_tokens=2200)


def chat_with_paper(plan_id: int, org_id: int | None = None) -> dict:
    """Create a fresh conversation linked to this paper. Returns conversation_id.

    Only writes columns documented to exist on ``conversations``. The
    plan_id linkage is encoded in ``system_note`` so the chat agent can
    detect and ground in the paper.
    """
    client, plan = _load_plan(plan_id)
    if not plan:
        return {"status": "not_found", "plan_id": plan_id}
    paper = (plan.get("paper_content") or "").strip()
    if not paper:
        return {"status": "failed", "error": "no paper"}
    topic = plan.get("topic", "")
    org_id = int(org_id or resolve_org_id(plan.get("org_id") or 0) or 1)
    system_note = (
        f"[research_plan_id={plan_id}] This conversation is grounded ONLY in the "
        f"research paper for plan {plan_id} (topic: {topic}). Do not bring in "
        "outside facts. If asked something not covered by the paper, say so."
    )
    payload = {
        "org_id": org_id,
        "title": f"Chat: {topic[:60]}",
        "system_note": system_note,
    }
    try:
        row = client._post("conversations", payload)
        conv_id = row.get("Id") or row.get("id")
    except Exception as e:
        return {"status": "failed", "error": str(e)[:200]}
    return {"status": "ok", "conversation_id": conv_id, "plan_id": plan_id}


# ── single dispatcher ───────────────────────────────────────────────────────

ASYNC_OPS = {
    "fact_check":         lambda pid, p: fact_check_paper(pid),
    "expand_section":     lambda pid, p: expand_section(pid, p.get("section_title") or p.get("section"), int(p.get("target_words", 1800))),
    "add_section":        lambda pid, p: add_new_section(pid, p.get("heading"), p.get("brief", ""), p.get("after_section"), int(p.get("target_words", 1000))),
    "counter_arguments":  lambda pid, p: add_counter_arguments(pid, int(p.get("target_words", 900))),
    "add_fresh_sources":  lambda pid, p: add_fresh_sources(pid, p.get("queries")),
    "refresh_recency":    lambda pid, p: refresh_for_recency(pid, p.get("since_date", "")),
    "reframe":            lambda pid, p: reframe_for_audience(pid, p.get("audience", "non-technical executive")),
    "resize":             lambda pid, p: resize_paper(pid, int(p.get("target_words", 3000))),
    "slide_deck":         lambda pid, p: generate_slide_deck(pid),
    "email_tldr":         lambda pid, p: generate_email_tldr(pid),
    "qa_pack":            lambda pid, p: generate_qa_pack(pid),
    "action_plan":        lambda pid, p: generate_action_plan(pid),
}

SYNC_OPS = {
    "citation_audit":     lambda pid, p: citation_audit_paper(pid),
    "chat_with_paper":    lambda pid, p: chat_with_paper(pid, p.get("org_id")),
}


def run_research_op(payload: dict) -> dict:
    """Tool-queue handler. payload: {plan_id, kind, params}"""
    plan_id = payload.get("plan_id")
    kind = payload.get("kind")
    params = payload.get("params") or {}
    if not plan_id or not kind:
        return {"status": "failed", "error": "missing plan_id or kind"}
    fn = ASYNC_OPS.get(kind) or SYNC_OPS.get(kind)
    if not fn:
        return {"status": "failed", "error": f"unknown op kind: {kind}"}
    try:
        return fn(int(plan_id), params)
    except Exception as e:
        _log.error("research_op uncaught error  plan_id=%s kind=%s", plan_id, kind, exc_info=True)
        return {"status": "failed", "error": str(e)[:300], "plan_id": plan_id, "kind": kind}
