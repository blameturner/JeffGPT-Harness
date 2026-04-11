import logging
import falkordb

from config import FALKORDB_HOST, FALKORDB_PORT, scoped_graph

_log = logging.getLogger("graph")

client = falkordb.FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)

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
) -> None:
    graph_name = scoped_graph(org_id)
    graph = get_graph(org_id)

    query = (
        f"MERGE (a:{from_type} {{name: $from_name}}) "
        f"MERGE (b:{to_type} {{name: $to_name}}) "
        f"MERGE (a)-[:{relationship}]->(b)"
    )

    _log.info("write  (%s:%s)-[%s]->(%s:%s)  graph=%s", from_name, from_type, relationship, to_name, to_type, graph_name)
    try:
        result = graph.query(query, {"from_name": from_name, "to_name": to_name})
        _log.info("write ok  nodes_created=%d rels_created=%d  graph=%s",
                   result.nodes_created, result.relationships_created, graph_name)
    except Exception:
        _log.error("write failed  (%s)-[%s]->(%s)  graph=%s", from_name, relationship, to_name, graph_name, exc_info=True)
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
        _log.debug("get_sparse_concepts failed org=%s", org_id, exc_info=True)
        return []
    concepts = [row[0] for row in result.result_set if row and row[0]]
    _log.debug("get_sparse_concepts  org=%s found=%d", org_id, len(concepts))
    return concepts


def get_connections(org_id: int, node_name: str) -> list[dict]:
    graph_name = scoped_graph(org_id)
    graph = get_graph(org_id)

    _log.debug("get_connections  node=%s graph=%s", node_name, graph_name)
    try:
        result = graph.query(
            "MATCH (a {name: $name})-[r]->(b) RETURN a.name, type(r), b.name",
            {"name": node_name},
        )
    except Exception:
        _log.error("get_connections failed  node=%s graph=%s", node_name, graph_name, exc_info=True)
        return []

    connections = []
    for row in result.result_set:
        connections.append({
            "from": row[0],
            "relationship": row[1],
            "to": row[2],
        })

    _log.debug("get_connections ok  node=%s found=%d", node_name, len(connections))
    return connections
