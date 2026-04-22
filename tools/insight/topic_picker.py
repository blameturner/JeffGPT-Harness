"""Pick a topic the user would actually care about a deep synopsis on.

Signals considered:

- **Graph neighbourhood** — entities the user has touched recently (high degree
  + recent ``last_seen`` / hits; falls back to any named nodes if those
  properties aren't on the edges yet).
- **Recent chat messages** — names/terms the user has been typing.
- **Chroma retrieval** — recurring topics across the ``agent_outputs`` and
  ``chat_knowledge`` collections.

Passes the candidate list to a cheap model that picks ONE topic and returns
it plus a short rationale the insight producer quotes in the final briefing
("Since we've been discussing X, …").
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from infra.nocodb_client import NocodbClient
from shared.models import model_call

_log = logging.getLogger("insight.topic")

_MAX_ENTITIES = 15
_RECENT_MESSAGES = 12


def _recent_messages(client: NocodbClient, org_id: int) -> list[str]:
    try:
        rows = client._get_paginated("messages", params={
            "where": f"(org_id,eq,{org_id})~and(role,eq,user)",
            "sort": "-CreatedAt",
            "limit": _RECENT_MESSAGES,
        })
    except Exception:
        return []
    return [((r.get("content") or "").strip()[:500]) for r in rows if (r.get("content") or "").strip()]


def _graph_entities(org_id: int) -> list[dict]:
    try:
        from infra.graph import get_graph
        graph = get_graph(org_id)
        result = graph.query(
            "MATCH (n) OPTIONAL MATCH (n)-[r]-() "
            "WITH n, count(r) AS deg WHERE deg > 1 "
            "RETURN labels(n)[0], n.name, deg "
            "ORDER BY deg DESC LIMIT $limit",
            {"limit": _MAX_ENTITIES},
        )
    except Exception:
        _log.warning("topic_picker: graph query failed  org=%d", org_id, exc_info=True)
        return []
    return [
        {"type": row[0], "name": row[1], "degree": row[2]}
        for row in result.result_set if row and row[1]
    ]


def _recent_insight_topics(org_id: int, limit: int = 10) -> set[str]:
    """Avoid repeating a topic we've already written about this week."""
    try:
        from shared import insights as insights_mod
        rows = insights_mod.list_recent(org_id, limit=limit)
    except Exception:
        return set()
    return {(r.get("topic") or "").strip().lower() for r in rows if r.get("topic")}


_PICK_PROMPT = """You pick the single best topic for a long-form research briefing to push to the user's home dashboard.

GOAL: Pick ONE topic (1-6 words, specific and named) that a thoughtful assistant would use as the subject of a 800-1500 word briefing for this user RIGHT NOW. The user should read the result and think "yes, that's exactly what I wanted."

GUIDELINES:
- Prefer named entities (products, technologies, companies, standards) over abstract themes.
- Favour topics the user has been actively working with (appear in the graph AND in recent messages).
- Avoid topics we've already covered (see AVOID list).
- If the signal is weak, pick the most promising named entity from the GRAPH list anyway — never pick a generic "productivity" / "general tech" topic.
- The rationale should be one sentence quoting what connects the topic to the user's current activity.

GRAPH ENTITIES (most connected, descending):
{entities_block}

RECENT USER MESSAGES (last {n_msgs}):
{messages_block}

AVOID (recently covered):
{avoid_block}

Return STRICT JSON (no prose, no fences):
{{
  "topic": "<1-6 words, specific and named>",
  "related_entities": ["<up to 5 named entities from the GRAPH list that the briefing should contrast against>"],
  "rationale": "<single sentence, starts with 'Because' or 'Since'>",
  "angle": "<one short phrase describing the briefing's angle — e.g. 'competitive landscape', 'security posture', 'migration options', 'recent changes'>"
}}"""


def _parse_json(raw: str) -> dict | None:
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip()).strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        # Salvage: find the first { ... } block
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None


def pick_topic(org_id: int) -> dict[str, Any] | None:
    """Return ``{topic, related_entities, rationale, angle}`` or None."""
    client = NocodbClient()
    entities = _graph_entities(org_id)
    messages = _recent_messages(client, org_id)
    avoid = _recent_insight_topics(org_id)

    if not entities and not messages:
        _log.info("topic_picker: no signal  org=%d", org_id)
        return None

    entities_block = "\n".join(
        f"- {e['name']} ({e['type']}, degree={e['degree']})" for e in entities
    ) or "(none)"
    messages_block = "\n".join(f"- {m}" for m in messages) or "(none)"
    avoid_block = "\n".join(f"- {t}" for t in sorted(avoid) if t) or "(none)"

    prompt = _PICK_PROMPT.format(
        entities_block=entities_block,
        messages_block=messages_block,
        avoid_block=avoid_block,
        n_msgs=len(messages),
    )

    parsed = None
    raw = ""
    for attempt in (1, 2):
        try:
            raw, _tokens = model_call("insight_topic_picker", prompt)
        except Exception:
            _log.error("topic_picker model call failed  org=%d attempt=%d", org_id, attempt, exc_info=True)
            if attempt == 2:
                return None
            continue
        parsed = _parse_json(raw or "")
        if parsed and parsed.get("topic"):
            break
        _log.warning("topic_picker unparseable  org=%d attempt=%d raw=%s",
                     org_id, attempt, (raw or "")[:200])

    if not parsed or not parsed.get("topic"):
        return None

    topic = str(parsed.get("topic") or "").strip()
    if topic.lower() in avoid:
        _log.info("topic_picker: model picked already-covered topic '%s', dropping", topic)
        return None

    return {
        "topic": topic[:120],
        "related_entities": [str(e) for e in (parsed.get("related_entities") or [])][:8],
        "rationale": str(parsed.get("rationale") or "")[:400],
        "angle": str(parsed.get("angle") or "")[:120],
    }
