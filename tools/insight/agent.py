"""Insight producer — the home dashboard's headline artefact.

When the user has been idle for the configured window (default 2h after the
last chat/code activity) OR the fallback twice-daily window fires, the
dispatcher enqueues one ``insight_produce`` job per org. That job:

1. Picks a topic worth writing about (see :mod:`tools.insight.topic_picker`).
2. Gathers substantial source material:
   - Graph neighbourhood for the topic + related entities.
   - Chroma RAG hits across ``agent_outputs``, ``chat_knowledge``, and the
     org's digest collection.
   - Recent completed ``scrape_targets`` summaries mentioning the topic.
3. Runs a long-form synthesis prompt and produces a 800-1500 word briefing
   in markdown.
4. Writes an :mod:`shared.insights` row, surfacing it on the dashboard.
5. Posts a short assistant message into the home conversation so the user's
   feed shows the insight inline alongside their chat history.
6. (Fire-and-forget) enqueues a research plan for the same topic so the
   next cycle has fresher web-scraped material to synthesise from.

Runs as a tool-queue (Huey-backed) handler, so it inherits the existing
chat-idle back-off and priority plumbing.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from infra.config import get_feature, is_feature_enabled
from infra.memory import recall
from infra.nocodb_client import NocodbClient
from shared import insights as insights_mod
from shared.home_conversation import get_or_create_home_conversation
from shared.models import model_call
from tools._org import resolve_org_id
from tools.insight.topic_picker import pick_topic

_log = logging.getLogger("insight.agent")


def _note_backoff(org_id: int, status: str) -> None:
    """Tell the dispatcher to park this org for a while after a dry run."""
    try:
        from tools.insight.dispatcher import note_producer_result
        note_producer_result(org_id, status)
    except Exception:
        _log.debug("note_producer_result failed", exc_info=True)


_SYNTHESIS_PROMPT = """You are writing a long-form research briefing for the user's home dashboard. The user has been idle and asked the system to go deeper on the topics they've been working on. Deliver a genuinely useful document — not a summary of summaries, not "did you know" trivia.

TOPIC: {topic}
ANGLE: {angle}
WHY THIS TOPIC WAS CHOSEN: {rationale}
RELATED ENTITIES (contrast against these where relevant): {related_entities}

SOURCE MATERIAL (verbatim excerpts from the user's own knowledge base and prior research):

{material}

WRITE THE BRIEFING AS MARKDOWN. Required structure:

1. `# <title>` — a specific, useful title. NOT "{topic}" verbatim — a title that promises value (e.g. "Duck Creek in the 2026 P&C Core Platform Landscape").
2. `## The short version` — 3-5 sentences of prose. Lead with the single most useful takeaway. Name 2-3 comparison points that matter. End with a concrete "what this means for you" sentence.
3. `## Context — why this matters now` — one paragraph grounding the topic in the user's recent activity using the rationale above. Do not just repeat the rationale verbatim.
4. `## The landscape` — 2-4 paragraphs of synthesis. Contrast alternatives, cite named competitors/approaches, surface tradeoffs. Inline citations for concrete factual claims: `[source: <url or filename or graph>]`.
5. `## What's new or moving` — 1-2 paragraphs focused on recent developments, direction of travel, momentum. If the source material doesn't cover recency, say so explicitly in one sentence and move on — do NOT fabricate.
6. `## Practical angles` — 2-4 paragraphs tied to the user's apparent use case. What decisions does this inform? What hidden gotchas exist? Be opinionated where the evidence supports it.
7. `## Open questions worth pushing on` — 3-6 bullet points, each a specific follow-up the user could usefully send the system (e.g. "Compare X's pricing model to Y's at enterprise tier"). This is where bullets ARE allowed.
8. `## Sources` — deduped bulleted list of every URL cited above. Include graph-derived facts as `graph: <relationship>`.

HARD RULES:
- Default to flowing prose. Bullets ONLY in `## Open questions worth pushing on` and `## Sources`.
- Target 800-1500 words of body content. Shorter only if the material is genuinely thin (and say so explicitly).
- Every concrete factual claim (numbers, names of competing products, dated events, prices) MUST carry an inline citation. General analysis and framing does NOT need one.
- NEVER fabricate URLs, prices, dates, or features. If the material doesn't contain it, either omit or call out the gap.
- Write for a technical, time-constrained reader. Compress. No filler, no "In conclusion", no "It is important to note".
- Output raw markdown only — no JSON, no wrapping code fences.
"""

_ACK_MESSAGE_PROMPT = """Write a 1-2 sentence note announcing a new insight briefing to the user on their home dashboard.

The briefing topic: {topic}
Why it was picked: {rationale}
The briefing's opening line: {lead}

Rules:
- Reference the rationale naturally (e.g. "Since we've been discussing X, I pulled together…").
- Mention the topic and that a full briefing is on their home dashboard.
- No markdown, no bullet points, no "I hope you find this helpful".
- ONE sentence preferred, two if it genuinely improves the message.

Output only the note, nothing else."""


# ---- gathering source material ----------------------------------------------

def _chroma_hits(org_id: int, query: str, collection: str, n: int) -> list[dict]:
    try:
        return recall(query, org_id, collection_name=collection, n_results=n)
    except Exception:
        # Missing collections are common on fresh installs; warn but don't fail
        # the whole job.
        _log.warning("chroma recall failed  coll=%s q=%s", collection, query[:60], exc_info=True)
        return []


def _graph_neighbourhood(org_id: int, topic: str, related: list[str]) -> tuple[list[str], list[str]]:
    """Alias-aware weighted 2-hop walk.

    Returns ``(edge_lines, source_chunk_ids)``: the formatted edge lines
    ready to paste into the synthesis prompt, plus a deduped list of
    ``source_chunks`` referenced by those edges so the caller can pull
    original text from the entity-mentions Chroma collection.
    """
    try:
        from infra.graph import get_weighted_neighbourhood
    except Exception:
        _log.error("insight: graph import failed", exc_info=True)
        return [], []

    seeds = [topic] + list(related or [])
    edges = get_weighted_neighbourhood(org_id, seeds, max_hops=2, edge_limit=80)
    if not edges:
        return [], []

    lines: list[str] = []
    chunk_ids: set[str] = set()
    for e in edges:
        lines.append(
            f"- ({e['from']}:{e['from_type']}) -[{e['relationship']} hits={e['hits']}]-> "
            f"({e['to']}:{e['to_type']})"
        )
        for c in e.get("source_chunks") or []:
            if c:
                chunk_ids.add(str(c))
    return lines[:60], sorted(chunk_ids)[:12]


def _source_chunks_text(org_id: int, chunk_ids: list[str]) -> list[str]:
    """Pull the raw chat-turn text for each chunk id so the synthesiser
    can cite real text, not just graph edges."""
    if not chunk_ids:
        return []
    try:
        from infra.config import scoped_collection
        from infra.memory import client as chroma_client
        scoped = scoped_collection(org_id, "chat_entity_mentions")
        coll = chroma_client.get_or_create_collection(scoped)
        result = coll.get(ids=list(chunk_ids))
    except Exception:
        _log.warning("source chunk fetch failed  chunks=%d", len(chunk_ids), exc_info=True)
        return []
    docs = result.get("documents") or []
    out: list[str] = []
    for i, doc in enumerate(docs):
        if not doc:
            continue
        out.append(f"- [source: chat:{chunk_ids[i] if i < len(chunk_ids) else '?'}]\n  {doc[:800]}")
    return out


def _recent_scrape_summaries(client: NocodbClient, org_id: int, topic: str, related: list[str]) -> list[str]:
    # NocoDB textual filters are limited; pull the last N completed scrapes
    # and keyword-filter in Python.
    try:
        rows = client._get_paginated("scrape_targets", params={
            "where": f"(org_id,eq,{org_id})~and(status,eq,ok)",
            "sort": "-last_scraped_at",
            "limit": 150,
        })
    except Exception:
        return []
    needles = {s.lower() for s in ([topic] + list(related or [])) if s}
    out: list[str] = []
    for r in rows:
        summary = (r.get("summary") or "").strip()
        if not summary:
            continue
        blob = (summary + " " + (r.get("url") or "") + " " + (r.get("title") or "")).lower()
        if not any(n in blob for n in needles):
            continue
        url = (r.get("url") or "").strip()
        out.append(f"- [{url}]\n  {summary[:800]}")
        if len(out) >= 12:
            break
    return out


def _format_chroma_block(hits: list[dict], label: str) -> str:
    if not hits:
        return ""
    parts = [f"{label} ({len(hits)} hits):"]
    for h in hits[:6]:
        meta = h.get("metadata") or {}
        # Prefer a real URL if the chunk carries one; fall back to a semantic
        # tag. The synthesis prompt cites whatever appears in `[source: ...]`
        # so URL-first matters for hyperlinked output.
        url = meta.get("url") or meta.get("source_url")
        src = url or meta.get("source") or meta.get("kind") or label.lower().replace(" ", "_")
        parts.append(f"- [source: {src}]\n  {(h.get('text') or '')[:600]}")
    return "\n".join(parts)


def _gather_material(org_id: int, topic: str, related: list[str]) -> str:
    client = NocodbClient()
    sections: list[str] = []

    # Weighted 2-hop graph walk + reverse lookup of source chunks
    graph_lines, chunk_ids = _graph_neighbourhood(org_id, topic, related)
    if graph_lines:
        sections.append("GRAPH NEIGHBOURHOOD (alias-aware, 2-hop, ordered by reinforcement hits):\n" + "\n".join(graph_lines))
    chunk_texts = _source_chunks_text(org_id, chunk_ids)
    if chunk_texts:
        sections.append("SOURCE CHAT TURNS (the text these edges were extracted from):\n" + "\n".join(chunk_texts))

    # Recent scrapes mentioning the topic
    scrapes = _recent_scrape_summaries(client, org_id, topic, related)
    if scrapes:
        sections.append("RECENT SCRAPE SUMMARIES:\n" + "\n".join(scrapes))

    # Chroma across three collections
    for coll, label in (
        ("agent_outputs", "PRIOR AGENT OUTPUTS"),
        ("chat_knowledge", "CHAT KNOWLEDGE"),
        ("daily_digests", "DIGEST CONTEXT"),
    ):
        hits = _chroma_hits(org_id, topic, coll, n=5)
        block = _format_chroma_block(hits, label)
        if block:
            sections.append(block)

    return "\n\n".join(sections) if sections else "(no material found — synthesise from general knowledge, but note the gap explicitly)"


# ---- synthesis ---------------------------------------------------------------

def _synthesize(topic_info: dict, material: str) -> str:
    prompt = _SYNTHESIS_PROMPT.format(
        topic=topic_info.get("topic") or "",
        angle=topic_info.get("angle") or "general overview",
        rationale=topic_info.get("rationale") or "",
        related_entities=", ".join(topic_info.get("related_entities") or []) or "(none)",
        material=material[:25000],
    )
    try:
        raw, tokens = model_call("insight_synthesis", prompt)
    except Exception:
        _log.error("insight synthesis model_call failed  topic=%s", topic_info.get("topic"), exc_info=True)
        return ""
    _log.info("insight synthesis done  topic=%s tokens=%s chars=%d",
              topic_info.get("topic"), tokens, len(raw or ""))
    return (raw or "").strip()


def _title_and_lead(briefing: str, fallback_topic: str) -> tuple[str, str]:
    lines = [ln.rstrip() for ln in briefing.splitlines()]
    title = fallback_topic
    lead = ""
    for ln in lines:
        if ln.startswith("# ") and title == fallback_topic:
            title = ln.lstrip("# ").strip()
        elif ln and not ln.startswith("#"):
            lead = ln.strip()
            if lead:
                break
    return title[:200], lead[:400]


def _post_to_home_conversation(org_id: int, topic_info: dict, title: str, lead: str, insight_id: int | None) -> None:
    try:
        convo = get_or_create_home_conversation(org_id)
    except Exception:
        _log.warning("home convo lookup failed  org=%d", org_id, exc_info=True)
        return

    try:
        prompt = _ACK_MESSAGE_PROMPT.format(
            topic=topic_info.get("topic") or "",
            rationale=topic_info.get("rationale") or "",
            lead=lead or title,
        )
        raw, _tokens = model_call("insight_ack", prompt)
        note = (raw or "").strip()
    except Exception:
        _log.warning("insight ack model failed, using template", exc_info=True)
        note = (
            f"{topic_info.get('rationale') or 'Follow-up briefing'}. "
            f"I've pushed a full briefing titled '{title}' to your home dashboard."
        )

    if not note:
        note = f"New insight on your dashboard: {title}"
    if insight_id is not None:
        note += f"\n\n_(insight #{insight_id})_"

    try:
        client = NocodbClient()
        client.add_message(
            conversation_id=convo["Id"],
            org_id=org_id,
            role="assistant",
            content=note,
            model="insight_synthesis",
            insight_id=insight_id,
            source="insight_producer",
        )
    except Exception:
        _log.warning("insight home-convo post failed  insight=%s", insight_id, exc_info=True)


def _maybe_queue_research(topic: str, org_id: int) -> dict:
    """Queue a research plan so the next insight cycle has fresher material.

    Returns a status dict — logged visibly so a failing pipeline isn't silent.
    """
    if not get_feature("insights", "auto_research", True):
        return {"status": "disabled"}
    try:
        from tools.research.research_planner import create_research_plan
    except Exception:
        _log.error("insight: research_planner import failed — auto-research skipped", exc_info=True)
        return {"status": "import_failed"}
    try:
        result = create_research_plan(topic, org_id=org_id)
    except Exception as e:
        _log.error("insight: create_research_plan failed  topic=%s err=%s", topic[:80], e, exc_info=True)
        return {"status": "exception", "error": str(e)[:200]}
    if isinstance(result, dict) and result.get("status") == "queued":
        _log.info("insight: research plan queued  plan_id=%s job=%s topic=%s",
                  result.get("plan_id"), result.get("job_id"), topic[:60])
    else:
        _log.warning("insight: research plan not queued  topic=%s result=%s", topic[:60], result)
    return result if isinstance(result, dict) else {"status": "unknown"}


# ---- Huey handler ------------------------------------------------------------

def insight_produce_job(payload: dict | None = None) -> dict:
    """Tool-queue handler. One org per tick."""
    payload = payload or {}
    if not is_feature_enabled("insights"):
        return {"status": "disabled"}

    org_id = resolve_org_id(payload.get("org_id"))
    trigger = str(payload.get("trigger") or insights_mod.TRIGGER_FALLBACK)
    topic_hint = (payload.get("topic_hint") or "").strip()

    if topic_hint:
        topic_info = {
            "topic": topic_hint[:120],
            "related_entities": [],
            "rationale": f"You asked for a briefing on {topic_hint}",
            "angle": "user-requested overview",
        }
    else:
        topic_info = pick_topic(org_id)
    if not topic_info:
        _log.info("insight: no topic  org=%d", org_id)
        _note_backoff(org_id, "no_topic")
        return {"status": "no_topic", "org_id": org_id}

    topic = topic_info["topic"]
    _log.info("insight producing  org=%d topic=%s trigger=%s", org_id, topic[:60], trigger)

    material = _gather_material(org_id, topic, topic_info.get("related_entities") or [])
    briefing = _synthesize(topic_info, material)
    if not briefing or len(briefing) < 400:
        _log.warning("insight synthesis too thin  org=%d topic=%s chars=%d",
                     org_id, topic, len(briefing or ""))
        _note_backoff(org_id, "thin_synthesis")
        return {"status": "thin_synthesis", "org_id": org_id, "topic": topic}

    title, lead = _title_and_lead(briefing, topic)

    insight_id = insights_mod.create(
        org_id=org_id,
        title=title,
        body_markdown=briefing,
        topic=topic,
        summary=lead,
        trigger=trigger,
        related_entities=topic_info.get("related_entities") or [],
        sources=[],
    )

    _post_to_home_conversation(org_id, topic_info, title, lead, insight_id)
    research_result = _maybe_queue_research(topic, org_id)

    try:
        from shared.surfacing import push_insight
        push_insight(org_id, title, lead, insight_id)
    except Exception:
        _log.warning("push_insight failed (non-fatal)", exc_info=True)

    # Drop the cached digest preface so the next chat turn picks up the new context.
    try:
        from workers.chat.home import invalidate_digest_preface
        invalidate_digest_preface(org_id)
    except Exception:
        pass

    _note_backoff(org_id, "ok")
    return {
        "status": "ok",
        "org_id": org_id,
        "topic": topic,
        "insight_id": insight_id,
        "trigger": trigger,
        "chars": len(briefing),
        "research": research_result,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
