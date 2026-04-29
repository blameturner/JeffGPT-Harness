"""Graph-expanded recall for chat turns.

Goal: when the user types something, look for entity names we already know
about in the graph, pull their immediate neighbourhood, and inject a
compact "Related graph facts" block into the system prompt. Gives the
model a ground truth to cite without round-tripping through Chroma.

Constraints (why this file is stingy):

- Must be cheap: the chat path is user-facing. We cache the set of known
  entity names per-org for 5 minutes so we don't re-scan the graph on
  every keystroke.
- Must degrade: any failure returns an empty string. Chat continues.
- Must be bounded: capped at 30 edges, 1200 chars in the payload.
"""
from __future__ import annotations

import logging
import re
import threading
import time

_log = logging.getLogger("graph.recall")

_ENTITY_CACHE_TTL_S = 300
_entity_cache: dict[int, tuple[float, set[str]]] = {}
_entity_lock = threading.Lock()
_refresh_inflight: set[int] = set()

_REFRESH_TIMEOUT_MS = 30_000
_ENTITY_LIMIT = 1000

_MAX_EDGES = 30
_MAX_CHARS = 1200
_QUERY_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{1,}")
_LOW_SIGNAL_SEEDS = {
    "a", "an", "and", "any", "app", "bot", "code", "data", "dev",
    "developer", "developers", "doc", "docs", "help", "home", "info",
    "issue", "question", "system", "test", "the", "tool", "tools",
}


def _fetch_entity_names(org_id: int, timeout_ms: int) -> set[str]:
    try:
        from infra.graph import list_entities_for_resolution
        entries = list_entities_for_resolution(
            org_id, limit=_ENTITY_LIMIT, min_degree=1, timeout_ms=timeout_ms,
        )
    except Exception:
        _log.warning("graph_recall: entity load failed  org=%d", org_id, exc_info=True)
        entries = []
    names: set[str] = set()
    for e in entries:
        n = (e.get("name") or "").strip()
        if n and len(n) >= 2:
            names.add(n)
        for a in (e.get("aliases") or []):
            a = str(a).strip()
            if a and len(a) >= 2:
                names.add(a)
    return names


def _background_refresh(org_id: int) -> None:
    with _entity_lock:
        if org_id in _refresh_inflight:
            return
        _refresh_inflight.add(org_id)

    def _worker():
        try:
            names = _fetch_entity_names(org_id, timeout_ms=_REFRESH_TIMEOUT_MS)
            with _entity_lock:
                _entity_cache[org_id] = (time.time(), names)
            _log.info("graph_recall refresh done  org=%d names=%d", org_id, len(names))
        finally:
            with _entity_lock:
                _refresh_inflight.discard(org_id)

    threading.Thread(target=_worker, daemon=True, name=f"graph-recall-refresh-{org_id}").start()


def _load_entity_names(org_id: int) -> set[str]:
    """Chat path is never blocking. Fresh hit → return. Stale hit → return
    stale, refresh in background. Cold miss → seed cache with an empty set
    and kick off the background fetch; the current turn skips graph context
    gracefully, the next turn has it. The background fetch gets the full
    30s window so it actually completes on a large graph."""
    now = time.time()
    with _entity_lock:
        hit = _entity_cache.get(org_id)
        if hit and now - hit[0] < _ENTITY_CACHE_TTL_S:
            return hit[1]
        if hit is None:
            # Seed an empty entry so subsequent calls hit the stale path
            # instead of repeatedly spawning cold-miss refreshes.
            _entity_cache[org_id] = (0.0, set())

    _background_refresh(org_id)
    with _entity_lock:
        return _entity_cache[org_id][1]


def invalidate_entity_cache(org_id: int | None = None) -> None:
    with _entity_lock:
        if org_id is None:
            _entity_cache.clear()
        else:
            _entity_cache.pop(org_id, None)


def _match_entities(text: str, known: set[str]) -> list[str]:
    if not text or not known:
        return []
    lower_known = {n.lower(): n for n in known}
    matched: list[str] = []
    seen: set[str] = set()
    # Word-boundary scan — cheap, avoids regex-compiling a huge alternation.
    for word in _QUERY_WORD_RE.findall(text):
        canonical = lower_known.get(word.lower())
        if canonical and canonical not in seen:
            matched.append(canonical)
            seen.add(canonical)
    # Also try multi-word phrase match for top-20 longest known names.
    if len(matched) < 5:
        low = text.lower()
        candidates = sorted(known, key=lambda s: -len(s))[:200]
        for n in candidates:
            if " " not in n:
                continue
            if n.lower() in low and n not in seen:
                matched.append(n)
                seen.add(n)
                if len(matched) >= 8:
                    break
    return matched[:8]


def _filter_graph_seeds(matches: list[str]) -> list[str]:
    filtered: list[str] = []
    seen: set[str] = set()
    for raw in matches:
        name = (raw or "").strip()
        if not name:
            continue
        low = name.lower()
        if low in _LOW_SIGNAL_SEEDS:
            continue
        if len(low) < 3 and " " not in low:
            continue
        if low in seen:
            continue
        seen.add(low)
        filtered.append(name)
    return filtered[:6]


def build_graph_context(org_id: int, query_text: str) -> str:
    """Returns a compact 'Related graph facts:' block, or '' if no matches."""
    if not query_text or not query_text.strip():
        return ""
    try:
        known = _load_entity_names(org_id)
    except Exception:
        _log.warning("graph_recall: cache load failed  org=%d", org_id, exc_info=True)
        return ""
    if not known:
        return ""

    matches = _filter_graph_seeds(_match_entities(query_text, known))
    if not matches:
        return ""

    try:
        from infra.graph import get_weighted_neighbourhood
        edges = get_weighted_neighbourhood(
            org_id, matches, max_hops=1, edge_limit=_MAX_EDGES,
        )
    except Exception:
        _log.warning("graph_recall: neighbourhood fetch failed  org=%d", org_id, exc_info=True)
        return ""

    if not edges:
        return ""

    lines = [f"Related graph facts (matched entities: {', '.join(matches)}):"]
    total = len(lines[0])
    for e in edges:
        line = (
            f"- ({e['from']}:{e['from_type']}) -[{e['relationship']} "
            f"hits={e['hits']}]-> ({e['to']}:{e['to_type']})"
        )
        if total + len(line) > _MAX_CHARS:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)
