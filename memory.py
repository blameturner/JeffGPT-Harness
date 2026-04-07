import chromadb

from config import CHROMA_URL
from embedder import embed
from urllib import parse

parsed = parse.urlparse(CHROMA_URL)

client = chromadb.HttpClient(
    host=parsed.hostname,
    port=parsed.port
)

def get_collection(name: str):
    return client.get_or_create_collection(name)

def remember(text: str, metadata: dict, collection_name: str = "agent_outputs") -> list[str]:
    from chunker import chunk_text
    import uuid

    chunks = chunk_text(text)
    collection = get_collection(collection_name)
    ids = []

    for i, chunk in enumerate(chunks):
        vector = embed(chunk)
        chunk_id = str(uuid.uuid4())

        collection.add(
            ids=[chunk_id],
            embeddings=[vector],
            documents=[chunk],
            metadatas=[{**metadata, "chunk_index":i}],
        )
        ids.append(chunk_id)

    return ids

def recall (query: str, collection_name: str = "agent_outputs", n_results: int = 5) -> list[dict]:
    collection = get_collection(collection_name)
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
