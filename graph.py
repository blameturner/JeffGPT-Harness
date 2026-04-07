import falkordb

from config import FALKORDB_HOST, FALKORDB_PORT

client = falkordb.FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)

graph = client.select_graph("mst_ag")

def write_relationship(
        from_type: str,
        from_name: str,
        relationship: str,
        to_type: str,
        to_name: str,
) -> None:

    query = """
    MERGE (a:{from_type} {{name: $from_name}})
    MERGE (b:{to_type} {{name: $to_name}})
    MERGE (a)-[:{relationship}]->(b)
    """.format(from_type=from_type, from_name=from_name, to_type=to_type, relationship=relationship)

    graph.query(query, {"from_name": from_name, "to_name": to_name})

def get_connections(node_name: str) -> list[dict]:
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
