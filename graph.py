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
