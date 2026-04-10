import falkordb

from config import FALKORDB_HOST, FALKORDB_PORT, scoped_graph

client = falkordb.FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)

def get_graph(org_id: int):
    return client.select_graph(scoped_graph(org_id))

def write_relationship(
        org_id: int,
        from_type: str,
        from_name: str,
        relationship: str,
        to_type: str,
        to_name: str,
) -> None:

    graph = get_graph(org_id)

    query = """
    MERGE (a:{from_type} {{name: $from_name}})
    MERGE (b:{to_type} {{name: $to_name}})
    MERGE (a)-[:{relationship}]->(b)
    """.format(from_type=from_type, from_name=from_name, to_type=to_type, relationship=relationship)

    graph.query(query, {"from_name": from_name, "to_name": to_name})

def get_connections(org_id: int, node_name: str) -> list[dict]:  
    graph = get_graph(org_id)

    query = """
    MATCH (a {name: $name})-[r]->(b)
    RETURN a.name, type(r), b.name
    """
    result = graph.query(query, {"name": node_name})

    connections = []
    for row in result.result_set:
        connections.append({
            "from": row[0],
            "relationship": row[1],
            "to": row[2],
        })

    return connections