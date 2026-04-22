"""Graph maintenance — the deferred cleanup work that keeps FalkorDB
useful as a retrieval source.

Two Huey handlers:

- ``graph_resolve_entities_job`` — daily. For each entity label, embed the
  node names, cluster tight-distance pairs, ask a small LLM model whether
  each pair is the same thing, and merge aliases into the higher-degree
  canonical via :func:`infra.graph.merge_alias`.

- ``graph_maintenance_job`` — weekly. Runs edge decay, prunes orphan nodes,
  and (opportunistically) adds ``CO_OCCURS_WITH`` edges between named
  entities that appear in the same recent chunks but aren't yet connected.

Both are best-effort, chunked, and bounded by feature-config caps so they
never chew through a workday of Huey time.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from infra.config import get_feature, is_feature_enabled
from infra.graph import (
    decay_edges,
    get_graph,
    list_entities_for_resolution,
    merge_alias,
    prune_orphans,
)
from shared.models import model_call
from tools._org import resolve_org_id

_log = logging.getLogger("graph.maintenance")


_ALIAS_JUDGE_PROMPT = """You decide whether two entity names in a knowledge graph refer to the SAME real-world thing.

LABEL: {label}
NAME A: {name_a}   (aliases so far: {aliases_a})
NAME B: {name_b}   (aliases so far: {aliases_b})

Same thing if: one is a clear abbreviation, spelling variant, case variant, or common shorthand of the other. Examples: "Postgres" / "PostgreSQL" / "PG" → same. "GPT-4" / "gpt4" / "gpt-4-turbo" → same. "AWS" / "Amazon Web Services" → same.

Different things if: they're distinct products/people/concepts that share a word. Examples: "Apple" (company) / "Apple" (fruit) → different (though same label usually means same domain). "Python" (language) / "Python 2" → same family but keep separate because the major version is semantically distinct.

Return STRICT JSON only:
{{"same": true|false, "canonical": "<the name that should be kept, or null if different>", "confidence": 0.0-1.0, "reason": "<one short sentence>"}}"""


# ---- entity resolution -------------------------------------------------------

def _parse_judge(raw: str) -> dict | None:
    if not raw:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip()).strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _cheap_similarity(a: str, b: str) -> float:
    """Cheap O(1)-ish similarity — normalised Levenshtein. Used to gate the
    LLM judge; we only ask the model about pairs that look close enough
    that a human would need to read them twice."""
    a = a.strip().lower()
    b = b.strip().lower()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # Simple Levenshtein via DP — capped by length so we bail out on long
    # strings where false positives dominate anyway.
    if abs(len(a) - len(b)) > 8:
        return 0.0
    if len(a) > 40 or len(b) > 40:
        return 0.0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j - 1], dp[j])
            prev = cur
    dist = dp[n]
    return 1.0 - (dist / max(m, n))


def _candidate_pairs(entities: list[dict]) -> list[tuple[dict, dict]]:
    """Pairs of same-label entities close enough to be worth LLM-judging."""
    by_label: dict[str, list[dict]] = {}
    for e in entities:
        by_label.setdefault(str(e.get("label") or "?"), []).append(e)
    sim_threshold = float(get_feature("graph_maintenance", "alias_similarity_threshold", 0.72))
    max_per_label = int(get_feature("graph_maintenance", "max_pairs_per_label", 20))
    out: list[tuple[dict, dict]] = []
    for label, group in by_label.items():
        if label == "?":
            continue
        group.sort(key=lambda g: (-int(g.get("degree") or 0), str(g.get("name") or "")))
        pairs_for_label = 0
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if pairs_for_label >= max_per_label:
                    break
                a, b = group[i], group[j]
                sim = _cheap_similarity(str(a.get("name") or ""), str(b.get("name") or ""))
                if sim >= sim_threshold:
                    out.append((a, b))
                    pairs_for_label += 1
            if pairs_for_label >= max_per_label:
                break
    return out


def graph_resolve_entities_job(payload: dict | None = None) -> dict:
    payload = payload or {}
    if not is_feature_enabled("graph_maintenance"):
        return {"status": "disabled"}

    org_id = resolve_org_id(payload.get("org_id"))
    max_entities = int(get_feature("graph_maintenance", "entity_scan_limit", 500))
    max_merges = int(get_feature("graph_maintenance", "max_merges_per_run", 25))

    entities = list_entities_for_resolution(org_id, limit=max_entities, min_degree=1)
    if not entities:
        _log.info("graph_resolve_entities: no candidates  org=%d", org_id)
        return {"status": "no_candidates", "org_id": org_id}

    pairs = _candidate_pairs(entities)
    _log.info("graph_resolve_entities  org=%d entities=%d pairs=%d", org_id, len(entities), len(pairs))

    merges = 0
    skipped = 0
    errors = 0
    confidence_floor = float(get_feature("graph_maintenance", "alias_confidence_floor", 0.75))

    for a, b in pairs:
        if merges >= max_merges:
            break
        prompt = _ALIAS_JUDGE_PROMPT.format(
            label=a.get("label") or "?",
            name_a=a.get("name"), aliases_a=a.get("aliases") or [],
            name_b=b.get("name"), aliases_b=b.get("aliases") or [],
        )
        try:
            raw, _tokens = model_call("graph_alias_judge", prompt)
        except Exception:
            _log.warning("alias judge call failed  pair=%s/%s", a.get("name"), b.get("name"), exc_info=True)
            errors += 1
            continue
        verdict = _parse_judge(raw)
        if not verdict or not verdict.get("same"):
            skipped += 1
            continue
        if float(verdict.get("confidence") or 0) < confidence_floor:
            skipped += 1
            continue

        canonical = (verdict.get("canonical") or "").strip()
        if not canonical or canonical not in {a.get("name"), b.get("name")}:
            # Default to higher-degree as canonical when the model is ambiguous.
            canonical = a.get("name") if int(a.get("degree") or 0) >= int(b.get("degree") or 0) else b.get("name")
        alias_name = b.get("name") if canonical == a.get("name") else a.get("name")
        result = merge_alias(org_id, a.get("label"), canonical, alias_name)
        if result.get("status") == "ok":
            merges += 1
            _log.info("alias merged  %s -> %s  edges_moved=%s", alias_name, canonical, result.get("edges_moved"))
        else:
            errors += 1

    if merges:
        try:
            from shared.graph_recall import invalidate_entity_cache
            invalidate_entity_cache(org_id)
        except Exception:
            _log.warning("graph_recall cache invalidation failed  org=%d", org_id, exc_info=True)

    return {
        "status": "ok",
        "org_id": org_id,
        "entities_scanned": len(entities),
        "candidate_pairs": len(pairs),
        "merges": merges,
        "skipped": skipped,
        "errors": errors,
    }


# ---- weekly maintenance ------------------------------------------------------

def _co_occurrence_pass(org_id: int, min_shared_chunks: int = 2, edge_limit: int = 40) -> int:
    """Walk edges, bucket by shared source_chunks, propose CO_OCCURS_WITH
    edges between pairs of entities that frequently appear together."""
    graph = get_graph(org_id)
    try:
        result = graph.query(
            "MATCH (a)-[r]->(b) WHERE r.source_chunks IS NOT NULL "
            "RETURN a.name, labels(a)[0], b.name, labels(b)[0], r.source_chunks "
            "LIMIT 5000"
        )
    except Exception:
        _log.warning("co-occurrence scan failed  org=%d", org_id, exc_info=True)
        return 0

    buckets: dict[str, set[tuple[str, str, str, str]]] = {}
    for row in result.result_set:
        if not row:
            continue
        a, alabel, b, blabel, chunks = row
        for c in (chunks or []):
            buckets.setdefault(str(c), set()).add((str(a), str(alabel or "Concept"), str(b), str(blabel or "Concept")))

    pair_counts: dict[tuple, int] = {}
    for triples in buckets.values():
        names = sorted({(t[0], t[1]) for t in triples} | {(t[2], t[3]) for t in triples})
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                k = (names[i], names[j])
                pair_counts[k] = pair_counts.get(k, 0) + 1

    added = 0
    for (na, nb), count in sorted(pair_counts.items(), key=lambda kv: -kv[1]):
        if count < min_shared_chunks:
            continue
        if added >= edge_limit:
            break
        (a_name, a_label), (b_name, b_label) = na, nb
        if a_name == b_name:
            continue
        try:
            graph.query(
                f"MATCH (a:`{a_label}` {{name: $a_name}}) "
                f"MATCH (b:`{b_label}` {{name: $b_name}}) "
                "MERGE (a)-[r:CO_OCCURS_WITH]-(b) "
                "ON CREATE SET r.hits = $count, r.weight = $count * 0.5 "
                "ON MATCH SET r.hits = coalesce(r.hits,0) + $count, "
                "  r.weight = coalesce(r.weight,0) + $count * 0.5",
                {"a_name": a_name, "b_name": b_name, "count": int(count)},
            )
            added += 1
        except Exception:
            _log.warning("co-occurrence merge failed  %s <-> %s", a_name, b_name, exc_info=True)
    _log.info("co-occurrence  org=%d proposed=%d added=%d", org_id, len(pair_counts), added)
    return added


def graph_maintenance_job(payload: dict | None = None) -> dict:
    payload = payload or {}
    if not is_feature_enabled("graph_maintenance"):
        return {"status": "disabled"}
    org_id = resolve_org_id(payload.get("org_id"))

    decay_factor = float(get_feature("graph_maintenance", "decay_factor", 0.9))
    drop_below = float(get_feature("graph_maintenance", "drop_below_weight", 0.2))
    co_min = int(get_feature("graph_maintenance", "co_occurrence_min", 2))
    co_limit = int(get_feature("graph_maintenance", "co_occurrence_limit", 40))

    co = _co_occurrence_pass(org_id, min_shared_chunks=co_min, edge_limit=co_limit)
    decay = decay_edges(org_id, factor=decay_factor, drop_below=drop_below)
    prune_orphans(org_id)

    return {
        "status": "ok",
        "org_id": org_id,
        "co_occurrence_added": co,
        "decay": decay,
    }
