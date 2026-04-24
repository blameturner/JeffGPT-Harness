"""PA background brief handler for warm topics.

Model config note: add `pa_topic_research` to config.json bound to a mid-tier
model, e.g. `"role": "t1_secondary"`, temperature 0.3, max_tokens 260,
max_input_chars 4000.
"""
from __future__ import annotations

import logging
from typing import Any

from infra.config import is_feature_enabled
from infra.memory import recall
from shared.models import model_call
from shared.pa.memory import find_topic_by_phrase, set_topic_brief
from tools.search.engine import searxng_search

log = logging.getLogger("pa.background")

_MAX_CONTEXT_CHARS = 3000
_MIN_BRIEF_CHARS = 40
_SOURCE_CAP = 5
_RAG_MERGE_CAP = 6


def _merge_rag_hits(hits_a: list[dict], hits_b: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for hit in list(hits_a or []) + list(hits_b or []):
        doc = (hit.get("document") or hit.get("text") or "") if isinstance(hit, dict) else ""
        key = doc[:100].strip()
        if not key:
            continue
        score = hit.get("relevance") or hit.get("score") or 0.0
        existing = seen.get(key)
        if existing is None:
            seen[key] = hit
        else:
            existing_score = existing.get("relevance") or existing.get("score") or 0.0
            if score > existing_score:
                seen[key] = hit
    merged = list(seen.values())
    merged.sort(
        key=lambda h: (h.get("relevance") or h.get("score") or 0.0),
        reverse=True,
    )
    return merged[:_RAG_MERGE_CAP]


def _format_rag_context(hits: list[dict], max_chars: int) -> str:
    if not hits:
        return "(none)"
    lines: list[str] = []
    used = 0
    for h in hits:
        doc = (h.get("document") or h.get("text") or "").strip()
        if not doc:
            continue
        snippet = doc[:400].replace("\n", " ")
        line = f"- {snippet}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines) if lines else "(none)"


def _format_web_context(results: list[dict], max_chars: int) -> str:
    if not results:
        return "(none)"
    lines: list[str] = []
    used = 0
    for r in results:
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        snippet = (r.get("snippet") or "").strip().replace("\n", " ")
        line = f"- {title} [{url}]: {snippet[:300]}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines) if lines else "(none)"


def _collect_sources(
    web_results: list[dict], rag_hits: list[dict], cap: int = _SOURCE_CAP
) -> list[dict]:
    out: list[dict] = []
    seen_urls: set[str] = set()
    for r in web_results or []:
        url = (r.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        out.append({
            "url": url,
            "title": (r.get("title") or "").strip(),
            "origin": "web",
        })
        seen_urls.add(url)
        if len(out) >= cap:
            return out
    for h in rag_hits or []:
        meta = h.get("metadata") or h.get("meta") or {}
        if not isinstance(meta, dict):
            continue
        url = (meta.get("url") or meta.get("source_url") or "").strip()
        if not url or url in seen_urls:
            continue
        out.append({
            "url": url,
            "title": (meta.get("title") or "").strip(),
            "origin": "rag",
        })
        seen_urls.add(url)
        if len(out) >= cap:
            break
    return out


def pa_topic_research_job(payload: dict) -> dict[str, Any]:
    try:
        if not is_feature_enabled("pa"):
            return {
                "status": "skipped",
                "topic_id": 0,
                "brief_chars": 0,
                "source_count": 0,
                "reason": "feature_disabled",
            }

        org_id = payload.get("org_id") if isinstance(payload, dict) else None
        topic_id = payload.get("topic_id") if isinstance(payload, dict) else None
        phrase = (payload.get("phrase") or "") if isinstance(payload, dict) else ""
        phrase = phrase.strip() if isinstance(phrase, str) else ""

        if not isinstance(org_id, int) or org_id <= 0 or (not topic_id and not phrase):
            return {
                "status": "error",
                "topic_id": 0,
                "brief_chars": 0,
                "source_count": 0,
                "reason": "invalid_payload",
            }

        if not topic_id:
            topic = find_topic_by_phrase(org_id, phrase)
            if not topic:
                return {
                    "status": "skipped",
                    "topic_id": 0,
                    "brief_chars": 0,
                    "source_count": 0,
                    "reason": "topic_not_found",
                }
            topic_id = topic.get("Id") or topic.get("id") or topic.get("topic_id")
        elif not phrase:
            # caller gave us the topic id only — hydrate phrase from the row
            try:
                from infra.nocodb_client import NocodbClient
                from infra.config import NOCODB_TABLE_PA_WARM_TOPICS
                client = NocodbClient()
                if NOCODB_TABLE_PA_WARM_TOPICS in client.tables:
                    rows = client._get(NOCODB_TABLE_PA_WARM_TOPICS, params={
                        "where": f"(Id,eq,{int(topic_id)})",
                        "limit": 1,
                    }).get("list", [])
                    if rows:
                        phrase = (rows[0].get("entity_or_phrase") or "").strip()
            except Exception as e:
                log.warning("pa.background: topic hydrate failed id=%s: %s", topic_id, e)

        try:
            topic_id = int(topic_id) if topic_id is not None else 0
        except (TypeError, ValueError):
            topic_id = 0

        if topic_id <= 0 or not phrase:
            return {
                "status": "error",
                "topic_id": topic_id,
                "brief_chars": 0,
                "source_count": 0,
                "reason": "invalid_payload",
            }

        rag_agent: list[dict] = []
        rag_web: list[dict] = []
        try:
            rag_agent = recall(phrase, org_id, collection_name="agent_outputs", n_results=5) or []
        except Exception as e:
            log.warning("pa.background: agent_outputs recall failed: %s", e)
        try:
            rag_web = recall(phrase, org_id, collection_name="web_search", n_results=5) or []
        except Exception as e:
            log.warning("pa.background: web_search recall failed: %s", e)

        rag_hits = _merge_rag_hits(rag_agent, rag_web)

        web_results: list[dict] = []
        try:
            web_results = searxng_search(phrase, max_results=5) or []
        except Exception as e:
            log.warning("pa.background: searxng_search failed: %s", e)

        half = _MAX_CONTEXT_CHARS // 2
        rag_context = _format_rag_context(rag_hits, max_chars=half)
        web_context = _format_web_context(web_results, max_chars=half)

        has_material = bool(rag_hits) or bool(web_results)

        brief = ""
        try:
            prompt = (
                "You're preparing a pocket brief on a topic the user cares about, to be "
                "used later if it becomes relevant in conversation. Write 2–4 sentences "
                "of CONCRETE, useful content — facts, numbers, names, current status. "
                "NOT an introduction, NOT generic background. Assume the reader already "
                "knows what the topic is.\n\n"
                f"TOPIC: {phrase}\n\n"
                f"RAG CONTEXT:\n{rag_context}\n\n"
                f"WEB RESULTS:\n{web_context}\n\n"
                "Output ONLY the brief text, no preamble, no 'Here's a brief'."
            )
            raw, _tokens = model_call("pa_topic_research", prompt)
            brief = (raw or "").strip()
        except Exception as e:
            log.warning("pa.background: model_call failed: %s", e)
            brief = ""

        if not has_material and len(brief) < _MIN_BRIEF_CHARS:
            return {
                "status": "skipped",
                "topic_id": topic_id,
                "brief_chars": 0,
                "source_count": 0,
                "reason": "no_material",
            }

        if len(brief) < _MIN_BRIEF_CHARS:
            return {
                "status": "skipped",
                "topic_id": topic_id,
                "brief_chars": len(brief),
                "source_count": 0,
                "reason": "no_material",
            }

        sources = _collect_sources(web_results, rag_hits, cap=_SOURCE_CAP)

        set_topic_brief(topic_id, brief=brief, sources=sources)

        return {
            "status": "ok",
            "topic_id": topic_id,
            "brief_chars": len(brief),
            "source_count": len(sources),
            "reason": "",
        }
    except Exception as e:
        log.warning("pa.background: unhandled exception", exc_info=True)
        return {
            "status": "error",
            "topic_id": 0,
            "brief_chars": 0,
            "source_count": 0,
            "reason": type(e).__name__,
        }
