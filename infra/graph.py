"""FalkorDB access layer.

Edges carry provenance so downstream jobs can reinforce, decay, and trace
back to source text:

- ``first_seen`` / ``last_seen`` — ISO timestamps (on create / on every match)
- ``hits``                       — counter incremented on every re-extraction
- ``weight``                     — float used by the decay job; default 1.0
- ``conversation_id``            — last conversation that mentioned this edge
- ``source_chunks``              — list of Chroma chunk ids the edge was
                                    extracted from (deduped, capped length)
- ``aliases`` on nodes           — list of alternate names merged in by the
                                    entity-resolution job

All helpers are best-effort: on any Cypher or connection failure they log
at WARNING or ERROR and return an inert value so callers don't crash.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import falkordb

from infra.config import FALKORDB_HOST, FALKORDB_PORT, scoped_graph

_log = logging.getLogger("graph")

client = falkordb.FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)

MAX_SOURCE_CHUNKS_PER_EDGE = 25


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_graph(org_id: int):
    name = scoped_graph(org_id)
    return client.select_graph(name)


def write_relationship(
    org_id: int,
    from_type: str,
    from_name: str,
    relationship: str,
    to_type: str,
    to_name: str,
    conversation_id: int | None = None,
    source_chunk_id: str | None = None,
) -> None:
    """MERGE a triple and update provenance.

    Re-mentions reinforce the edge (``hits += 1``, ``last_seen`` refreshed,
    chunk id appended to ``source_chunks`` without duplication).
    """
    graph_name = scoped_graph(org_id)
    graph = get_graph(org_id)
    now = _now_iso()

    params: dict = {
        "from_name": from_name,
        "to_name": to_name,
        "now": now,
        "chunk": str(source_chunk_id) if source_chunk_id else "",
        "conv": int(conversation_id) if conversation_id is not None else 0,
        "cap": MAX_SOURCE_CHUNKS_PER_EDGE,
    }
    # FalkorDB's Cypher supports list concatenation and coalesce. Append the
    # chunk id only when it's non-empty and not already present; cap the
    # list so a long-running org doesn't accumulate unbounded history on
    # a single edge.
    query = (
        f"MERGE (a:`{from_type}` {{name: $from_name}}) "
        f"MERGE (b:`{to_type}` {{name: $to_name}}) "
        f"MERGE (a)-[r:`{relationship}`]->(b) "
        "ON CREATE SET "
        "  r.first_seen = $now, "
        "  r.last_seen = $now, "
        "  r.hits = 1, "
        "  r.weight = 1.0, "
        "  r.source_chunks = CASE WHEN $chunk = '' THEN [] ELSE [$chunk] END, "
        "  r.conversation_id = $conv "
        "ON MATCH SET "
        "  r.last_seen = $now, "
        "  r.hits = coalesce(r.hits, 0) + 1, "
        "  r.weight = coalesce(r.weight, 1.0) + 0.1, "
        "  r.conversation_id = CASE WHEN $conv = 0 THEN r.conversation_id ELSE $conv END, "
        "  r.source_chunks = CASE "
        "    WHEN $chunk = '' THEN coalesce(r.source_chunks, []) "
        "    WHEN $chunk IN coalesce(r.source_chunks, []) THEN r.source_chunks "
        "    ELSE coalesce(r.source_chunks, [])[0..$cap-1] + [$chunk] "
        "  END"
    )

    _log.info(
        "write  (%s:%s)-[%s]->(%s:%s)  graph=%s conv=%s chunk=%s",
        from_name, from_type, relationship, to_name, to_type,
        graph_name, conversation_id, source_chunk_id,
    )
    try:
        result = graph.query(query, params)
        _log.info(
            "write ok  nodes_created=%d rels_created=%d props_set=%d graph=%s",
            result.nodes_created, result.relationships_created,
            result.properties_set, graph_name,
        )
    except Exception:
        _log.error(
            "write failed  (%s)-[%s]->(%s)  graph=%s",
            from_name, relationship, to_name, graph_name, exc_info=True,
        )
        raise


def get_sparse_concepts(org_id: int, limit: int = 10, max_degree: int = 3) -> list[str]:
    if limit <= 0:
        return []
    try:
        graph = get_graph(org_id)
        result = graph.query(
            "MATCH (n:Concept) "
            "OPTIONAL MATCH (n)-[r]-() "
            "WITH n, count(r) AS deg "
            "WHERE deg < $max_degree "
            "RETURN n.name, deg "
            "ORDER BY deg ASC "
            "LIMIT $limit",
            {"max_degree": int(max_degree), "limit": int(limit)},
        )
    except Exception:
        _log.warning("get_sparse_concepts failed org=%s", org_id, exc_info=True)
        return []
    concepts = [row[0] for row in result.result_set if row and row[0]]
    _log.debug("get_sparse_concepts  org=%s found=%d", org_id, len(concepts))
    return concepts


def get_connections(org_id: int, node_name: str) -> list[dict]:
    """Deprecated 1-hop getter kept for backwards compatibility."""
    graph = get_graph(org_id)
    try:
        result = graph.query(
            "MATCH (a {name: $name})-[r]->(b) RETURN a.name, type(r), b.name",
            {"name": node_name},
        )
    except Exception:
        _log.error("get_connections failed  node=%s", node_name, exc_info=True)
        return []
    return [
        {"from": row[0], "relationship": row[1], "to": row[2]}
        for row in result.result_set if row
    ]


def get_weighted_neighbourhood(
    org_id: int,
    seed_names: list[str],
    max_hops: int = 2,
    edge_limit: int = 80,
) -> list[dict]:
    """Weighted expansion from one or more seed nodes.

    Returns edges ordered by reinforcement strength (``hits`` then ``weight``).
    Alias-aware: matches on either ``name`` or membership in ``aliases``.
    """
    if not seed_names:
        return []
    graph = get_graph(org_id)
    seeds = [s for s in {str(n).strip() for n in seed_names} if s]
    if not seeds:
        return []

    if max_hops == 1:
        pattern = "(a)-[r]-(b)"
    else:
        # 2-hop expansion: seed → direct → second-degree. Direct hops are
        # emitted plus edges between neighbours themselves, giving a richer
        # local subgraph for synthesis.
        pattern = "(a)-[r*1..2]-(b)"

    try:
        result = graph.query(
            "MATCH (seed) WHERE seed.name IN $seeds OR any(alias IN coalesce(seed.aliases, []) WHERE alias IN $seeds) "
            f"MATCH {pattern} "
            "WHERE a = seed OR b = seed "
            "WITH a, b, last(r) AS edge "
            "RETURN a.name, labels(a)[0], type(edge), b.name, labels(b)[0], "
            "  coalesce(edge.hits, 1) AS hits, coalesce(edge.weight, 1.0) AS weight, "
            "  edge.last_seen AS last_seen, edge.source_chunks AS chunks "
            "ORDER BY hits DESC, weight DESC "
            "LIMIT $limit",
            {"seeds": seeds, "limit": int(edge_limit)},
        )
    except Exception:
        _log.warning(
            "get_weighted_neighbourhood failed  seeds=%s max_hops=%d",
            seeds[:3], max_hops, exc_info=True,
        )
        return []

    out: list[dict] = []
    seen: set[tuple] = set()
    for row in result.result_set:
        if not row or not row[0] or not row[3]:
            continue
        key = (str(row[0]), str(row[2]), str(row[3]))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "from": row[0],
            "from_type": row[1],
            "relationship": row[2],
            "to": row[3],
            "to_type": row[4],
            "hits": int(row[5] or 1),
            "weight": float(row[6] or 1.0),
            "last_seen": row[7],
            "source_chunks": list(row[8] or []),
        })
    return out


def list_entities_for_resolution(
    org_id: int,
    limit: int = 500,
    min_degree: int = 1,
) -> list[dict]:
    """Candidate list for the entity-resolution job: named nodes with at
    least one edge, returned with their label and degree."""
    graph = get_graph(org_id)
    try:
        result = graph.query(
            "MATCH (n) WHERE n.name IS NOT NULL "
            "OPTIONAL MATCH (n)-[r]-() "
            "WITH n, count(r) AS deg "
            "WHERE deg >= $min_degree "
            "RETURN labels(n)[0], n.name, coalesce(n.aliases, []) AS aliases, deg "
            "ORDER BY deg DESC LIMIT $limit",
            {"min_degree": int(min_degree), "limit": int(limit)},
        )
    except Exception:
        _log.warning("list_entities_for_resolution failed  org=%d", org_id, exc_info=True)
        return []
    return [
        {"label": row[0], "name": row[1], "aliases": list(row[2] or []), "degree": int(row[3] or 0)}
        for row in result.result_set if row and row[1]
    ]


def merge_alias(
    org_id: int,
    label: str,
    canonical: str,
    alias: str,
) -> dict:
    """Entity resolution primitive: rewire all edges from ``alias`` node onto
    ``canonical``, record the alias on the canonical node, then delete the
    alias node.
    """
    if not canonical or not alias or canonical == alias:
        return {"status": "noop"}
    graph = get_graph(org_id)
    # Rewire outgoing edges. Uses a two-step pattern because FalkorDB doesn't
    # support mid-query `SET r = ...` across changed endpoints reliably.
    rewire_out = (
        f"MATCH (x:`{label}` {{name: $alias}})-[r]->(y) "
        f"MATCH (c:`{label}` {{name: $canonical}}) "
        "MERGE (c)-[r2:REL_MIGRATED_PLACEHOLDER]->(y) "
        "DELETE r"
    )
    # We can't dynamically MERGE an edge whose type is taken from `type(r)`
    # in pure Cypher, so rewiring is done more carefully in Python below.
    # Fetch edges first, recreate, delete.
    try:
        out_edges = graph.query(
            f"MATCH (x:`{label}` {{name: $alias}})-[r]->(y) "
            "RETURN type(r), y.name, labels(y)[0], "
            "  coalesce(r.hits,1), coalesce(r.weight,1.0), "
            "  coalesce(r.source_chunks, []), r.first_seen, r.last_seen",
            {"alias": alias},
        ).result_set
        in_edges = graph.query(
            f"MATCH (x)-[r]->(y:`{label}` {{name: $alias}}) "
            "RETURN type(r), x.name, labels(x)[0], "
            "  coalesce(r.hits,1), coalesce(r.weight,1.0), "
            "  coalesce(r.source_chunks, []), r.first_seen, r.last_seen",
            {"alias": alias},
        ).result_set
    except Exception:
        _log.error("merge_alias: edge read failed  alias=%s -> %s", alias, canonical, exc_info=True)
        return {"status": "failed", "error": "edge read"}

    moved = 0
    for row in out_edges:
        rel, other, other_label, hits, weight, chunks, first_seen, last_seen = row
        try:
            graph.query(
                f"MATCH (c:`{label}` {{name: $canonical}}) "
                f"MATCH (y:`{other_label}` {{name: $other}}) "
                f"MERGE (c)-[r:`{rel}`]->(y) "
                "ON CREATE SET r.first_seen = $first_seen, r.last_seen = $last_seen, "
                "  r.hits = $hits, r.weight = $weight, r.source_chunks = $chunks "
                "ON MATCH SET r.hits = coalesce(r.hits,0) + $hits, "
                "  r.weight = coalesce(r.weight,0) + $weight, "
                "  r.last_seen = $last_seen, "
                "  r.source_chunks = coalesce(r.source_chunks,[]) + $chunks",
                {
                    "canonical": canonical, "other": other,
                    "first_seen": first_seen, "last_seen": last_seen,
                    "hits": int(hits), "weight": float(weight),
                    "chunks": list(chunks or []),
                },
            )
            moved += 1
        except Exception:
            _log.warning("merge_alias: out edge rewire failed  %s-[%s]->%s",
                         alias, rel, other, exc_info=True)
    for row in in_edges:
        rel, other, other_label, hits, weight, chunks, first_seen, last_seen = row
        try:
            graph.query(
                f"MATCH (x:`{other_label}` {{name: $other}}) "
                f"MATCH (c:`{label}` {{name: $canonical}}) "
                f"MERGE (x)-[r:`{rel}`]->(c) "
                "ON CREATE SET r.first_seen = $first_seen, r.last_seen = $last_seen, "
                "  r.hits = $hits, r.weight = $weight, r.source_chunks = $chunks "
                "ON MATCH SET r.hits = coalesce(r.hits,0) + $hits, "
                "  r.weight = coalesce(r.weight,0) + $weight, "
                "  r.last_seen = $last_seen, "
                "  r.source_chunks = coalesce(r.source_chunks,[]) + $chunks",
                {
                    "canonical": canonical, "other": other,
                    "first_seen": first_seen, "last_seen": last_seen,
                    "hits": int(hits), "weight": float(weight),
                    "chunks": list(chunks or []),
                },
            )
            moved += 1
        except Exception:
            _log.warning("merge_alias: in edge rewire failed  %s-[%s]->%s",
                         other, rel, alias, exc_info=True)

    # Update alias list on canonical node + delete alias node.
    try:
        graph.query(
            f"MATCH (c:`{label}` {{name: $canonical}}) "
            "SET c.aliases = CASE WHEN $alias IN coalesce(c.aliases, []) "
            "  THEN c.aliases ELSE coalesce(c.aliases, []) + [$alias] END",
            {"canonical": canonical, "alias": alias},
        )
        graph.query(
            f"MATCH (x:`{label}` {{name: $alias}}) DETACH DELETE x",
            {"alias": alias},
        )
    except Exception:
        _log.error("merge_alias: finalise failed  alias=%s -> %s", alias, canonical, exc_info=True)
        return {"status": "failed", "error": "finalise"}

    _log.info("merge_alias  label=%s %s -> %s  edges_moved=%d", label, alias, canonical, moved)
    return {"status": "ok", "edges_moved": moved}


def decay_edges(org_id: int, factor: float = 0.9, drop_below: float = 0.2) -> dict:
    """Multiply edge ``weight`` by ``factor``; drop edges whose weight falls
    below ``drop_below``. Run monthly by the maintenance job."""
    graph = get_graph(org_id)
    try:
        graph.query(
            "MATCH ()-[r]->() SET r.weight = coalesce(r.weight, 1.0) * $factor",
            {"factor": float(factor)},
        )
        dropped = graph.query(
            "MATCH ()-[r]->() WHERE coalesce(r.weight, 1.0) < $threshold DELETE r RETURN count(r)",
            {"threshold": float(drop_below)},
        )
    except Exception:
        _log.error("decay_edges failed  org=%d", org_id, exc_info=True)
        return {"status": "failed"}
    n_dropped = 0
    try:
        rs = dropped.result_set
        if rs and rs[0]:
            n_dropped = int(rs[0][0] or 0)
    except Exception:
        pass
    _log.info("decay_edges  org=%d factor=%.2f dropped=%d", org_id, factor, n_dropped)
    return {"status": "ok", "dropped": n_dropped}


def prune_orphans(org_id: int) -> dict:
    """Delete nodes with degree 0. Entity resolution often leaves these."""
    graph = get_graph(org_id)
    try:
        graph.query("MATCH (n) WHERE NOT (n)--() DELETE n")
    except Exception:
        _log.error("prune_orphans failed  org=%d", org_id, exc_info=True)
        return {"status": "failed"}
    return {"status": "ok"}
