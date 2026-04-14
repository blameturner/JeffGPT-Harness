import logging
import chromadb

from infra.config import CHROMA_URL, scoped_collection
from infra.embedder import embed
from urllib import parse

_log = logging.getLogger("memory")

parsed = parse.urlparse(CHROMA_URL) if CHROMA_URL else None
if not parsed or not parsed.hostname:
    _log.error("CHROMA_URL is missing or invalid: %s", CHROMA_URL)

client = chromadb.HttpClient(
    host=parsed.hostname if parsed else "localhost",
    port=parsed.port if parsed else 8000,
)
_log.info("chroma client  host=%s port=%s", parsed.hostname if parsed else "localhost", parsed.port if parsed else 8000)

def get_collection(org_id: int, name: str):
    scoped = scoped_collection(org_id, name)
    return client.get_or_create_collection(scoped)

def remember(text: str, metadata: dict, org_id: int, collection_name: str = "agent_outputs") -> list[str]:
    from infra.chunker import chunk_text
    import uuid

    chunks = chunk_text(text)
    scoped = scoped_collection(org_id, collection_name)
    _log.info("remember  collection=%s chunks=%d text_len=%d", scoped, len(chunks), len(text))
    collection = get_collection(org_id, collection_name)
    ids = []

    for i, chunk in enumerate(chunks):
        try:
            vector = embed(chunk)
        except Exception:
            _log.warning("embed failed  collection=%s chunk=%d/%d words=%d, skipping", scoped, i, len(chunks), len(chunk.split()))
            continue
        chunk_id = str(uuid.uuid4())
        try:
            # ChromaDB metadata values must be str, int, float, or bool — filter out None
            clean_meta = {k: v for k, v in {**metadata, "chunk_index": i, "org_id": org_id}.items() if v is not None}
            collection.add(
                ids=[chunk_id],
                embeddings=[vector],
                documents=[chunk],
                metadatas=[clean_meta],
            )
            ids.append(chunk_id)
        except Exception:
            _log.error("chroma add failed  collection=%s chunk=%d/%d", scoped, i, len(chunks), exc_info=True)

    _log.info("remember ok  collection=%s stored=%d/%d", scoped, len(ids), len(chunks))
    return ids

def recall(query: str, org_id: int, collection_name: str = "agent_outputs", n_results: int = 5) -> list[dict]:
    scoped = scoped_collection(org_id, collection_name)
    _log.debug("recall  collection=%s n_results=%d query=%s", scoped, n_results, query[:80])
    collection = get_collection(org_id, collection_name)
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

    _log.info("recall ok  collection=%s returned=%d", scoped, len(output))
    return output
