import logging
import falkordb

from config import FALKORDB_HOST, FALKORDB_PORT, scoped_graph

_log = logging.getLogger("graph")

client = falkordb.FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)

def get_graph(org_id: int):
    name = scoped_graph(org_id)
    _log.debug("select_graph %s", name)
    return client.select_graph(name)

def write_relationship(
        org_id: int,
        from_type: str,
        from_name: str,
        relationship: str,
        to_type: str,
        to_name: str,
) -> None:
    graph = get_graph(org_id)

    query = (
        f"MERGE (a:{from_type} {{name: $from_name}}) "
        f"MERGE (b:{to_type} {{name: $to_name}}) "
        f"MERGE (a)-[:{relationship}]->(b)"
    )

    _log.debug("write  %s -[%s]-> %s  graph=%s", from_name, relationship, to_name, scoped_graph(org_id))
    result = graph.query(query, {"from_name": from_name, "to_name": to_name})
    _log.debug("write ok  nodes_created=%d rels_created=%d",
               result.nodes_created, result.relationships_created)

def get_connections(org_id: int, node_name: str) -> list[dict]:
    graph = get_graph(org_id)

    result = graph.query(
        "MATCH (a {name: $name})-[r]->(b) RETURN a.name, type(r), b.name",
        {"name": node_name},
    )

    connections = []
    for row in result.result_set:
        connections.append({
            "from": row[0],
            "relationship": row[1],
            "to": row[2],
        })

    _log.debug("get_connections %s  found=%d", node_name, len(connections))
    return connections
