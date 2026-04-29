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
from time import perf_counter

import falkordb

from infra.config import FALKORDB_HOST, FALKORDB_PORT, scoped_graph

_log = logging.getLogger("graph")

client = falkordb.FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)

MAX_SOURCE_CHUNKS_PER_EDGE = 25
DEFAULT_QUERY_TIMEOUT_MS = 5_000
CHAT_QUERY_TIMEOUT_MS = 1_200
MAINTENANCE_QUERY_TIMEOUT_MS = 30_000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_graph(org_id: int):
    name = scoped_graph(org_id)
    return client.select_graph(name)


def run_query(graph, query: str, params: dict | None = None, timeout_ms: int = DEFAULT_QUERY_TIMEOUT_MS):
    """Bound all graph calls so a slow planner decision cannot wedge callers."""
    return graph.query(query, params or {}, timeout=max(1, int(timeout_ms)))


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
        result = run_query(graph, query, params, timeout_ms=DEFAULT_QUERY_TIMEOUT_MS)
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
        return


def get_sparse_concepts(org_id: int, limit: int = 10, max_degree: int = 3) -> list[str]:
    if limit <= 0:
        return []
    try:
        graph = get_graph(org_id)
        result = run_query(
            graph,
            "MATCH (n:Concept) "
            "OPTIONAL MATCH (n)-[r]-() "
            "WITH n, count(r) AS deg "
            "WHERE deg < $max_degree "
            "RETURN n.name, deg "
            "ORDER BY deg ASC "
            "LIMIT $limit",
            {"max_degree": int(max_degree), "limit": int(limit)},
            timeout_ms=3_000,
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
        result = run_query(
            graph,
            "MATCH (a {name: $name})-[r]->(b) RETURN a.name, type(r), b.name",
            {"name": node_name},
            timeout_ms=2_000,
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
    timeout_ms: int = CHAT_QUERY_TIMEOUT_MS,
) -> list[dict]:
    """Weighted expansion from one or more seed nodes.

    Returns edges ordered by reinforcement strength (``hits`` then ``weight``).
    Alias-aware: matches on either ``name`` or membership in ``aliases``.
    """
    if not seed_names:
        return []
    graph = get_graph(org_id)
    seeds = list(dict.fromkeys(str(n).strip() for n in seed_names if str(n).strip()))
    if not seeds:
        return []

    if max_hops <= 1:
        query = (
            "MATCH (seed) "
            "WHERE seed.name IN $seeds OR any(alias IN coalesce(seed.aliases, []) WHERE alias IN $seeds) "
            "MATCH (seed)-[edge]-(nbr) "
            "RETURN seed.name, labels(seed)[0], type(edge), nbr.name, labels(nbr)[0], "
            "  coalesce(edge.hits, 1) AS hits, coalesce(edge.weight, 1.0) AS weight, "
            "  edge.last_seen AS last_seen, edge.source_chunks AS chunks "
            "ORDER BY hits DESC, weight DESC "
            "LIMIT $limit"
        )
    else:
        # 2-hop expansion stays seed-anchored, but emits the terminal edge so
        # callers get both direct and second-degree context.
        query = (
            "MATCH (seed) "
            "WHERE seed.name IN $seeds OR any(alias IN coalesce(seed.aliases, []) WHERE alias IN $seeds) "
            "MATCH p=(seed)-[r*1..2]-(other) "
            "WITH nodes(p) AS ns, last(r) AS edge "
            "RETURN ns[size(ns)-2].name, labels(ns[size(ns)-2])[0], "
            "  type(edge), ns[size(ns)-1].name, labels(ns[size(ns)-1])[0], "
            "  coalesce(edge.hits, 1) AS hits, coalesce(edge.weight, 1.0) AS weight, "
            "  edge.last_seen AS last_seen, edge.source_chunks AS chunks "
            "ORDER BY hits DESC, weight DESC "
            "LIMIT $limit"
        )

    def _run(active_seeds: list[str], limit: int):
        p = {"seeds": active_seeds, "limit": int(limit)}
        return run_query(graph, query, p, timeout_ms=timeout_ms)

    try:
        t0 = perf_counter()
        result = _run(seeds, edge_limit)
        elapsed_ms = int((perf_counter() - t0) * 1000)
        _log.debug(
            "get_weighted_neighbourhood ok  seeds=%d max_hops=%d edges=%d elapsed_ms=%d",
            len(seeds), max_hops, len(result.result_set), elapsed_ms,
        )
    except Exception:
        # Timeout fallback: retry a narrower query so chat can still proceed.
        fallback_seeds = seeds[:3]
        fallback_limit = min(int(edge_limit), 20)
        try:
            if len(fallback_seeds) != len(seeds) or fallback_limit != int(edge_limit):
                t0 = perf_counter()
                result = _run(fallback_seeds, fallback_limit)
                elapsed_ms = int((perf_counter() - t0) * 1000)
                _log.warning(
                    "get_weighted_neighbourhood fallback ok  seeds=%s -> %s max_hops=%d limit=%d elapsed_ms=%d",
                    seeds[:3], fallback_seeds, max_hops, fallback_limit, elapsed_ms,
                )
            else:
                raise
        except Exception:
            _log.warning(
                "get_weighted_neighbourhood failed  seeds=%s max_hops=%d timeout_ms=%d",
                seeds[:3], max_hops, timeout_ms, exc_info=True,
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
    timeout_ms: int = MAINTENANCE_QUERY_TIMEOUT_MS,
) -> list[dict]:
    """Candidate list for the entity-resolution job: named nodes with at
    least one edge, returned with their label and degree.

    Uses pattern-size (``size((n)--())``) instead of ``OPTIONAL MATCH +
    count``; the latter materialises every relationship twice across all
    named nodes and times out on non-trivial graphs. A FalkorDB query
    timeout bounds the call so a worst-case planner decision can't wedge
    the Redis connection.

    Callers on the chat hot path (``shared.graph_recall``) should pass a
    tight ``timeout_ms`` (e.g. 3000) so a cold cache doesn't stall a turn;
    background maintenance uses the default generous window."""
    graph = get_graph(org_id)
    # Don't filter by degree in Cypher: FalkorDB evaluates the predicate for
    # every matching node before applying LIMIT, which defeats the point.
    # ORDER BY deg DESC + LIMIT naturally drops isolated nodes when limit is
    # well below the total named-node count; the min_degree filter runs in
    # Python on the returned rows.
    try:
        result = run_query(
            graph,
            "MATCH (n) WHERE n.name IS NOT NULL "
            "RETURN labels(n)[0] AS label, n.name AS name, "
            "coalesce(n.aliases, []) AS aliases, size((n)--()) AS deg "
            "ORDER BY deg DESC LIMIT $limit",
            {"limit": int(limit)},
            timeout_ms=timeout_ms,
        )
    except Exception:
        _log.warning("list_entities_for_resolution failed  org=%d timeout_ms=%d",
                     org_id, timeout_ms, exc_info=True)
        return []
    out: list[dict] = []
    md = int(min_degree)
    for row in result.result_set:
        if not row or not row[1]:
            continue
        deg = int(row[3] or 0)
        if deg < md:
            continue
        out.append({"label": row[0], "name": row[1], "aliases": list(row[2] or []), "degree": deg})
    return out


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
        out_edges = run_query(
            graph,
            f"MATCH (x:`{label}` {{name: $alias}})-[r]->(y) "
            "RETURN type(r), y.name, labels(y)[0], "
            "  coalesce(r.hits,1), coalesce(r.weight,1.0), "
            "  coalesce(r.source_chunks, []), r.first_seen, r.last_seen",
            {"alias": alias},
            timeout_ms=DEFAULT_QUERY_TIMEOUT_MS,
        ).result_set
        in_edges = run_query(
            graph,
            f"MATCH (x)-[r]->(y:`{label}` {{name: $alias}}) "
            "RETURN type(r), x.name, labels(x)[0], "
            "  coalesce(r.hits,1), coalesce(r.weight,1.0), "
            "  coalesce(r.source_chunks, []), r.first_seen, r.last_seen",
            {"alias": alias},
            timeout_ms=DEFAULT_QUERY_TIMEOUT_MS,
        ).result_set
    except Exception:
        _log.error("merge_alias: edge read failed  alias=%s -> %s", alias, canonical, exc_info=True)
        return {"status": "failed", "error": "edge read"}

    moved = 0
    for row in out_edges:
        rel, other, other_label, hits, weight, chunks, first_seen, last_seen = row
        try:
            run_query(
                graph,
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
                timeout_ms=DEFAULT_QUERY_TIMEOUT_MS,
            )
            moved += 1
        except Exception:
            _log.warning("merge_alias: out edge rewire failed  %s-[%s]->%s",
                         alias, rel, other, exc_info=True)
    for row in in_edges:
        rel, other, other_label, hits, weight, chunks, first_seen, last_seen = row
        try:
            run_query(
                graph,
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
                timeout_ms=DEFAULT_QUERY_TIMEOUT_MS,
            )
            moved += 1
        except Exception:
            _log.warning("merge_alias: in edge rewire failed  %s-[%s]->%s",
                         other, rel, alias, exc_info=True)

    # Update alias list on canonical node + delete alias node.
    try:
        run_query(
            graph,
            f"MATCH (c:`{label}` {{name: $canonical}}) "
            "SET c.aliases = CASE WHEN $alias IN coalesce(c.aliases, []) "
            "  THEN c.aliases ELSE coalesce(c.aliases, []) + [$alias] END",
            {"canonical": canonical, "alias": alias},
            timeout_ms=DEFAULT_QUERY_TIMEOUT_MS,
        )
        run_query(
            graph,
            f"MATCH (x:`{label}` {{name: $alias}}) DETACH DELETE x",
            {"alias": alias},
            timeout_ms=DEFAULT_QUERY_TIMEOUT_MS,
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
        run_query(
            graph,
            "MATCH ()-[r]->() SET r.weight = coalesce(r.weight, 1.0) * $factor",
            {"factor": float(factor)},
            timeout_ms=MAINTENANCE_QUERY_TIMEOUT_MS,
        )
        dropped = run_query(
            graph,
            "MATCH ()-[r]->() WHERE coalesce(r.weight, 1.0) < $threshold DELETE r RETURN count(r)",
            {"threshold": float(drop_below)},
            timeout_ms=MAINTENANCE_QUERY_TIMEOUT_MS,
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
        run_query(graph, "MATCH (n) WHERE NOT (n)--() DELETE n", timeout_ms=MAINTENANCE_QUERY_TIMEOUT_MS)
    except Exception:
        _log.error("prune_orphans failed  org=%d", org_id, exc_info=True)
        return {"status": "failed"}
    return {"status": "ok"}
