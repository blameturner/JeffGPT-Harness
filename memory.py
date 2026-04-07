import chromadb

from config import CHROMA_URL, scoped_collection  # added scoped_collection
from embedder import embed
from urllib import parse

parsed = parse.urlparse(CHROMA_URL)

client = chromadb.HttpClient(
    host=parsed.hostname,
    port=parsed.port
)

def get_collection(org_id: int, name: str):  # added org_id parameter
    return client.get_or_create_collection(scoped_collection(org_id, name))  # scope the name

def remember(text: str, metadata: dict, org_id: int, collection_name: str = "agent_outputs") -> list[str]:  # added org_id
    from chunker import chunk_text
    import uuid

    chunks = chunk_text(text)
    collection = get_collection(org_id, collection_name)  # pass org_id
    ids = []

    for i, chunk in enumerate(chunks):
        vector = embed(chunk)
        chunk_id = str(uuid.uuid4())

        collection.add(
            ids=[chunk_id],
            embeddings=[vector],
            documents=[chunk],
            metadatas=[{**metadata, "chunk_index": i, "org_id": org_id}],  # added org_id to metadata
        )
        ids.append(chunk_id)

    return ids

def recall(query: str, org_id: int, collection_name: str = "agent_outputs", n_results: int = 5) -> list[dict]:  # added org_id
    collection = get_collection(org_id, collection_name)  # pass org_id
    vector = embed(query)

    results = collection.query(
        query_embeddings=[vector],
        n_results=n_results,
    )

    output = []

    for i, doc in enumerate(results["documents"][0]):
        output.append({
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return output