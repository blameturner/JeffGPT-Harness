"""Curiosity-driven moves the PA can surface into the Home conversation.

Each move is a pure function returning a :class:`MoveCandidate` or ``None``.
``None`` means "no fresh input right now — skip"; all failure modes collapse
to that same signal so the picker can treat moves uniformly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from shared.pa.memory import (
    MOVE_KIND_CLOSURE,
    MOVE_KIND_CONNECT,
    MOVE_KIND_NEWS,
    MOVE_KIND_SERENDIPITY,
    LOOP_STATUS_OPEN,
    LOOP_STATUS_NUDGED,
    TOPIC_KIND_INTEREST,
    TOPIC_KIND_STATED,
    drop_loop,
    list_open_loops,
    list_warm_topics,
    recent_moves,
)

_log = logging.getLogger("pa.moves")


@dataclass
class MoveCandidate:
    kind: str
    text: str
    why: str
    input_refs: dict = field(default_factory=dict)
    novelty: float = 0.5


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _hours_since(raw: str | None) -> float | None:
    ts = _parse_ts(raw)
    if ts is None:
        return None
    return (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0


def _clip(text: str, limit: int = 220) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# ── closure ───────────────────────────────────────────────────────────────────

def closure(org_id: int) -> MoveCandidate | None:
    try:
        rows = list_open_loops(org_id, status=None, limit=100)
    except Exception:
        _log.warning("closure: list_open_loops failed  org=%d", org_id, exc_info=True)
        return None

    candidates: list[tuple[float, dict]] = []
    for r in rows:
        status = r.get("status")
        if status not in (LOOP_STATUS_OPEN, LOOP_STATUS_NUDGED):
            continue
        age_h = _hours_since(r.get("CreatedAt") or r.get("created_at"))
        if age_h is None or age_h < 0.75:  # 45 minutes
            continue
        nudge_count = int(r.get("nudge_count") or 0)
        if nudge_count > 2:
            continue
        last_nudged_h = _hours_since(r.get("last_nudged_at")) if r.get("last_nudged_at") else None
        if status == LOOP_STATUS_NUDGED and (last_nudged_h is None or last_nudged_h < 6):
            continue
        # housekeeping: drop stale nudged loops
        if nudge_count >= 2 and age_h >= 72:
            try:
                drop_loop(int(r["Id"]), reason="auto_stale")
            except Exception:
                _log.debug("closure: drop_loop failed  id=%s", r.get("Id"), exc_info=True)
            continue
        # rank by (oldest last-nudge or oldest creation) first
        key = last_nudged_h if last_nudged_h is not None else age_h
        candidates.append((key, r))

    if not candidates:
        return None

    candidates.sort(key=lambda p: p[0], reverse=True)
    row = candidates[0][1]
    text = (row.get("text") or "").strip()
    if not text:
        return None
    nudge_count = int(row.get("nudge_count") or 0)

    if nudge_count == 0:
        line = f"how's the {text} coming along?"
    elif nudge_count == 1:
        line = f"still on {text}? anything blocking it?"
    else:
        line = f"been a few days on {text} — keep tracking or drop it?"

    novelty = 0.7 if nudge_count == 0 else 0.5
    return MoveCandidate(
        kind=MOVE_KIND_CLOSURE,
        text=_clip(line),
        why=f"open loop from earlier ({nudge_count} prior nudges)",
        input_refs={"loop_id": int(row["Id"])},
        novelty=novelty,
    )


# ── connect ───────────────────────────────────────────────────────────────────

def connect(org_id: int) -> MoveCandidate | None:
    try:
        topics = list_warm_topics(org_id, limit=6, min_warmth=0.4)
    except Exception:
        _log.warning("connect: list_warm_topics failed  org=%d", org_id, exc_info=True)
        return None
    if len(topics) < 2:
        return None

    try:
        from infra.graph import get_graph
        graph = get_graph(org_id)
    except Exception:
        _log.debug("connect: get_graph unavailable  org=%d", org_id, exc_info=True)
        return None
    if graph is None:
        return None

    names = [(t, (t.get("entity_or_phrase") or "").strip()) for t in topics]
    names = [(t, n) for t, n in names if n]

    cypher = (
        "MATCH (a {name: $a}), (b {name: $b}), "
        "p=shortestPath((a)-[*1..2]-(b)) RETURN p LIMIT 1"
    )

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            _, a = names[i]
            _, b = names[j]
            try:
                res = graph.query(cypher, {"a": a, "b": b})
            except Exception:
                _log.debug("connect: graph query failed  a=%r b=%r", a, b, exc_info=True)
                continue
            rows = getattr(res, "result_set", None) or []
            if not rows:
                continue
            path = rows[0][0] if rows[0] else None
            nodes = getattr(path, "nodes", None)
            if nodes is None and isinstance(path, dict):
                nodes = path.get("nodes")
            if not nodes or len(nodes) != 3:
                # length 2 == 3 nodes (a, bridge, b); direct links skipped
                continue
            bridge_node = nodes[1]
            bridge = None
            props = getattr(bridge_node, "properties", None)
            if isinstance(props, dict):
                bridge = props.get("name")
            if bridge is None and isinstance(bridge_node, dict):
                bridge = (bridge_node.get("properties") or {}).get("name") or bridge_node.get("name")
            if not bridge:
                continue
            line = (
                f"noticed {a} and {b} both come up in your chats — "
                f"they're linked through {bridge}. Worth exploring?"
            )
            return MoveCandidate(
                kind=MOVE_KIND_CONNECT,
                text=_clip(line),
                why=f"graph bridge between two warm topics via {bridge}",
                input_refs={"a": a, "b": b, "bridge": bridge},
                novelty=0.85,
            )
    return None


# ── news_watch ────────────────────────────────────────────────────────────────

def news_watch(org_id: int) -> MoveCandidate | None:
    try:
        topics = list_warm_topics(org_id, limit=10, min_warmth=0.4)
    except Exception:
        _log.warning("news_watch: list_warm_topics failed  org=%d", org_id, exc_info=True)
        return None
    topics = [
        t for t in topics
        if t.get("kind") in (TOPIC_KIND_INTEREST, TOPIC_KIND_STATED)
        and (t.get("entity_or_phrase") or "").strip()
    ]
    if not topics:
        return None

    # per-topic 12h cooldown
    try:
        recent = recent_moves(org_id, since_hours=12, limit=40)
    except Exception:
        recent = []
    recently_newsed: set[int] = set()
    for m in recent:
        if m.get("move_kind") != MOVE_KIND_NEWS:
            continue
        refs = m.get("input_refs")
        if isinstance(refs, str):
            import json
            try:
                refs = json.loads(refs)
            except (TypeError, ValueError):
                refs = {}
        if isinstance(refs, dict) and refs.get("topic_id"):
            try:
                recently_newsed.add(int(refs["topic_id"]))
            except (TypeError, ValueError):
                continue

    topic = None
    for t in sorted(topics, key=lambda x: float(x.get("warmth") or 0), reverse=True):
        try:
            tid = int(t.get("Id"))
        except (TypeError, ValueError):
            continue
        if tid in recently_newsed:
            continue
        topic = t
        break
    if topic is None:
        return None

    phrase = (topic.get("entity_or_phrase") or "").strip()
    year = datetime.now(timezone.utc).year
    query = f"{phrase} news {year}"

    try:
        from tools.search.engine import searxng_search
        results = searxng_search(query, max_results=3)
    except Exception:
        _log.debug("news_watch: searxng failed  org=%d q=%r", org_id, query, exc_info=True)
        return None
    if not results:
        return None

    pick = None
    for r in results:
        title = (r.get("title") or "").strip()
        if len(title) >= 20:
            pick = r
            break
    if pick is None:
        return None

    title = (pick.get("title") or "").strip()
    url = pick.get("url") or ""
    line = f'saw a fresh piece on {phrase}: "{title}" — want a quick read?'
    return MoveCandidate(
        kind=MOVE_KIND_NEWS,
        text=_clip(line),
        why=f"new search result for warm topic {phrase!r}",
        input_refs={"topic_id": int(topic["Id"]), "url": url, "title": title},
        novelty=0.75,
    )


# ── serendipity ───────────────────────────────────────────────────────────────

def serendipity(org_id: int) -> MoveCandidate | None:
    try:
        from infra.graph import get_sparse_concepts
        concepts = get_sparse_concepts(org_id, limit=10, max_degree=2)
    except Exception:
        _log.debug("serendipity: get_sparse_concepts failed  org=%d", org_id, exc_info=True)
        return None
    if not concepts:
        return None

    try:
        recent = recent_moves(org_id, since_hours=21 * 24, limit=100)
    except Exception:
        recent = []
    recently_seen: set[str] = set()
    for m in recent:
        if m.get("move_kind") != MOVE_KIND_SERENDIPITY:
            continue
        refs = m.get("input_refs")
        if isinstance(refs, str):
            import json
            try:
                refs = json.loads(refs)
            except (TypeError, ValueError):
                refs = {}
        if isinstance(refs, dict):
            ent = refs.get("entity")
            if isinstance(ent, str) and ent:
                recently_seen.add(ent.lower())

    for name in concepts:
        clean = (name or "").strip()
        if not clean or clean.lower() in recently_seen:
            continue
        line = (
            f"you mentioned {clean} a while back and never came back to it — "
            f"still in your head, or drop it?"
        )
        return MoveCandidate(
            kind=MOVE_KIND_SERENDIPITY,
            text=_clip(line),
            why=f"sparse concept not surfaced in 21d: {clean}",
            input_refs={"entity": clean},
            novelty=0.9,
        )
    return None
