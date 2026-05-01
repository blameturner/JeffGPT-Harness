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
# Shared timeout used by background graph maintenance jobs. Kept as a module
# constant for import compatibility with older callers.
MAINTENANCE_QUERY_TIMEOUT_MS = 15_000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_graph(org_id: int):
    name = scoped_graph(org_id)
    return client.select_graph(name)


def run_query(graph, query: str, params: dict | None = None, timeout_ms: int | None = None):
    """Compatibility query wrapper used by maintenance and PA graph callers.

    FalkorDB's Python client accepts `timeout` in milliseconds on newer
    versions. On older versions the kwarg may not exist, so we retry without
    it instead of failing import/startup paths.
    """
    params = params or {}
    if timeout_ms is None:
        return graph.query(query, params)
    try:
        return graph.query(query, params, timeout=int(timeout_ms))
    except TypeError:
        return graph.query(query, params)


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


def _is_timeout_error(exc: BaseException) -> bool:
    """FalkorDB raises redis.exceptions.ResponseError('Query timed out').
    We don't want to import redis here just for the type, so match by name
    + message."""
    name = type(exc).__name__
    if name in ("ResponseError", "RedisTimeoutError", "TimeoutError"):
        return True
    msg = str(exc).lower()
    return "timed out" in msg or "timeout" in msg


def _resolve_seed_names(graph, seeds: list[str], timeout_ms: int = 1500) -> list[str]:
    """Map raw seed strings → canonical node names (alias-aware).

    Cheap pre-pass: avoids putting an OR-on-alias predicate inside the
    bigger neighbourhood traversal where the planner mishandles it. Caps
    at 50 hits so a fuzzy match against a common phrase can't fan out.
    Falls back to the input list on any error.
    """
    try:
        result = run_query(
            graph,
            "MATCH (n) WHERE n.name IN $seeds "
            "  OR any(alias IN coalesce(n.aliases, []) WHERE alias IN $seeds) "
            "RETURN DISTINCT n.name LIMIT 50",
            {"seeds": list(seeds)},
            timeout_ms=int(timeout_ms),
        )
        return [row[0] for row in (result.result_set or []) if row and row[0]]
    except Exception:
        _log.debug("_resolve_seed_names: fall back to raw seeds", exc_info=True)
        return list(seeds)


def _fetch_one_hop(
    graph,
    seeds: list[str],
    limit: int,
    timeout_ms: int,
) -> list[dict]:
    """One-hop edges anchored at each seed by name.

    Uses ``UNWIND $seeds AS sname / MATCH (a {name: sname})`` so the
    planner does a per-seed indexed lookup instead of a global edge scan
    + post-filter. That's the difference between bounded latency on a
    hub like "AI" and a 30-second timeout.

    Retries once with a smaller ``limit`` on timeout — if the cluster is
    momentarily slow the smaller scan often succeeds.
    """
    if not seeds:
        return []

    query = (
        "UNWIND $seeds AS sname "
        "MATCH (a {name: sname})-[r]-(b) "
        "RETURN a.name, labels(a)[0], type(r), b.name, labels(b)[0], "
        "  coalesce(r.hits, 1) AS hits, coalesce(r.weight, 1.0) AS weight, "
        "  r.last_seen AS last_seen, r.source_chunks AS chunks "
        "ORDER BY hits DESC, weight DESC "
        "LIMIT $limit"
    )
    params = {"seeds": list(seeds), "limit": int(limit)}

    attempts: list[tuple[int, int]] = [(int(timeout_ms), int(limit))]
    # If we have plenty of budget, plan a degraded retry. Otherwise just
    # the one shot — the caller's outer fallback will pick up the slack.
    if timeout_ms >= 4000 and limit > 20:
        attempts.append((max(2000, timeout_ms // 2), max(20, limit // 4)))

    last_exc: Exception | None = None
    for attempt_timeout, attempt_limit in attempts:
        params["limit"] = int(attempt_limit)
        try:
            result = run_query(graph, query, params, timeout_ms=attempt_timeout)
            return _materialise_edges(result)
        except Exception as e:
            last_exc = e
            if not _is_timeout_error(e):
                # Non-timeout error: don't burn the budget retrying.
                break
            _log.warning(
                "_fetch_one_hop timeout  seeds=%d limit=%d timeout_ms=%d — retrying smaller",
                len(seeds), attempt_limit, attempt_timeout,
            )

    _log.warning(
        "_fetch_one_hop gave up  seeds=%s last_error=%s",
        seeds[:3], type(last_exc).__name__ if last_exc else "n/a",
    )
    return []


def _materialise_edges(result) -> list[dict]:
    """Common row → dict mapping for neighbourhood queries. Tolerates the
    hits/weight/last_seen fields being missing on legacy edges."""
    out: list[dict] = []
    if result is None or not getattr(result, "result_set", None):
        return out
    for row in result.result_set:
        if not row or not row[0] or not row[3]:
            continue
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


def get_weighted_neighbourhood(
    org_id: int,
    seed_names: list[str],
    max_hops: int = 2,
    edge_limit: int = 80,
    timeout_ms: int = 8000,
) -> list[dict]:
    """Weighted expansion from one or more seed nodes.

    Returns edges ordered by reinforcement strength (``hits`` then ``weight``).
    Alias-aware: matches on either ``name`` or membership in ``aliases``.

    Resilience strategy (the previous query timed out on hub nodes):
      1. Resolve seeds → canonical names in a cheap, bounded pre-pass so
         the alias-OR predicate doesn't ride along inside the main scan.
      2. Per-seed edge cap so one hub can't drown the others out of the
         result set.
      3. Anchored ``UNWIND $seeds / MATCH (a {name: sname})`` traversal
         — indexed lookup per seed instead of a global edge scan with
         a post-filter.
      4. Hard ``timeout_ms`` on every Cypher call (default 8 s).
      5. On 1-hop timeout: retry with smaller limit before giving up.
      6. 2-hop is opportunistic — only tried if the 1-hop budget didn't
         already exhaust the edge limit, AND it falls back cleanly to
         the 1-hop result on its own timeout. We never escalate a 1-hop
         success into a 2-hop failure.
    """
    if not seed_names:
        return []
    graph = get_graph(org_id)
    raw = [s for s in {str(n).strip() for n in seed_names} if s]
    if not raw:
        return []

    canonical = _resolve_seed_names(graph, raw, timeout_ms=1500) or raw
    # Hard cap on seeds — a fuzzy match against a generic word (e.g. "ai")
    # could otherwise resolve to 50 entities and turn each subsequent
    # query into a fan-out.
    canonical = canonical[:10]

    # Per-seed cap so traversing a single hub doesn't fill the whole
    # result set at the expense of the other seeds.
    per_seed_cap = max(5, edge_limit // max(1, len(canonical)))
    one_hop_limit = min(int(edge_limit), per_seed_cap * len(canonical))

    edges = _fetch_one_hop(graph, canonical, one_hop_limit, timeout_ms=timeout_ms)

    # 2-hop: opportunistic. Skip cleanly if 1-hop already filled, if the
    # caller asked for 1-hop only, or if the 1-hop fetch failed (in which
    # case there's no useful seed pool for the second hop anyway).
    if max_hops > 1 and edges and len(edges) < edge_limit:
        # Build a second-hop seed pool from the strongest 1-hop neighbours
        # — strong edges → likely informative neighbourhood.
        seen_seeds = set(canonical)
        second_seeds: list[str] = []
        for e in edges:
            for nm in (e.get("to"), e.get("from")):
                if nm and nm not in seen_seeds and nm not in second_seeds:
                    second_seeds.append(nm)
                if len(second_seeds) >= 8:
                    break
            if len(second_seeds) >= 8:
                break

        remaining = edge_limit - len(edges)
        if second_seeds and remaining > 0:
            extra = _fetch_one_hop(
                graph,
                second_seeds,
                limit=remaining,
                # Tighter budget for the optional hop. If it can't finish
                # in 4s, drop it; the 1-hop result is already useful.
                timeout_ms=min(int(timeout_ms), 4000),
            )
            edges.extend(extra)

    # Dedupe (a 2-hop expansion can re-emit a 1-hop edge with reversed
    # direction) and order by reinforcement strength + recency.
    out: list[dict] = []
    seen: set[tuple] = set()
    for e in edges:
        # Undirected dedupe key — `(min, rel, max)` collapses both
        # orientations of the same edge.
        a, b = str(e["from"]), str(e["to"])
        key = (min(a, b), str(e["relationship"]), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
        if len(out) >= edge_limit:
            break

    return out


def list_entities_for_resolution(
    org_id: int,
    limit: int = 500,
    min_degree: int = 1,
    timeout_ms: int = 30_000,
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
        result = graph.query(
            "MATCH (n) WHERE n.name IS NOT NULL "
            "RETURN labels(n)[0] AS label, n.name AS name, "
            "coalesce(n.aliases, []) AS aliases, size((n)--()) AS deg "
            "ORDER BY deg DESC LIMIT $limit",
            {"limit": int(limit)},
            timeout=int(timeout_ms),
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
