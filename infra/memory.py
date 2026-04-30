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

def remember(
    text: str,
    metadata: dict,
    org_id: int,
    collection_name: str = "agent_outputs",
    produced_by: str | None = None,
) -> list[str]:
    """Chunk + embed + store text into a Chroma collection.

    `produced_by` is a provenance tag (`chat`, `research_agent`, `digest`,
    `scraper`, `pa_extractor`, ...) so the UI can filter / attribute hits
    later. Falls back to the caller's existing metadata or `"unknown"`.
    """
    from infra.chunker import chunk_text
    from datetime import datetime, timezone
    import uuid

    chunks = chunk_text(text)
    scoped = scoped_collection(org_id, collection_name)
    _log.info("remember  collection=%s chunks=%d text_len=%d", scoped, len(chunks), len(text))
    collection = get_collection(org_id, collection_name)
    ids = []
    now_iso = datetime.now(timezone.utc).isoformat()
    provenance = produced_by or metadata.get("produced_by") or "unknown"

    for i, chunk in enumerate(chunks):
        try:
            vector = embed(chunk)
        except Exception:
            _log.warning("embed failed  collection=%s chunk=%d/%d words=%d, skipping", scoped, i, len(chunks), len(chunk.split()))
            continue
        chunk_id = str(uuid.uuid4())
        try:
            # chroma metadata must be str/int/float/bool — None values crash add()
            clean_meta = {
                k: v for k, v in {
                    **metadata,
                    "chunk_index": i,
                    "org_id": org_id,
                    "produced_by": provenance,
                    "ingested_at": now_iso,
                }.items() if v is not None
            }
            collection.add(
                ids=[chunk_id],
                embeddings=[vector],
                documents=[chunk],
                metadatas=[clean_meta],
            )
            ids.append(chunk_id)
        except Exception:
            _log.error("chroma add failed  collection=%s chunk=%d/%d", scoped, i, len(chunks), exc_info=True)

    _log.info("remember ok  collection=%s stored=%d/%d produced_by=%s", scoped, len(ids), len(chunks), provenance)
    return ids


def forget(chunk_id: str, org_id: int, collection_name: str) -> bool:
    """Delete a single chunk by id from a scoped collection. Returns True
    if the collection acknowledged the delete (which Chroma does even if
    the id wasn't present, so callers should pre-check existence)."""
    try:
        collection = get_collection(org_id, collection_name)
        collection.delete(ids=[chunk_id])
        _log.info("forget  collection=%s id=%s", scoped_collection(org_id, collection_name), chunk_id)
        return True
    except Exception:
        _log.warning("forget failed  id=%s collection=%s", chunk_id, collection_name, exc_info=True)
        return False


def get_chunk(chunk_id: str, org_id: int, collection_name: str) -> dict | None:
    """Look up a single chunk by id. Used by the forget/preview UI flow
    and recall-replay debug endpoint."""
    try:
        collection = get_collection(org_id, collection_name)
        result = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not result or not result.get("ids"):
            return None
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        return {
            "id": chunk_id,
            "text": docs[0] if docs else "",
            "metadata": metas[0] if metas else {},
        }
    except Exception:
        _log.debug("get_chunk failed  id=%s", chunk_id, exc_info=True)
        return None


def collection_health(org_id: int) -> dict:
    """Aggregate health stats for an org's Chroma collections.

    Returns: per-collection counts + a roll-up. The per-collection
    `produced_by` breakdown lets the UI show "this org's memory is 70%
    research, 20% chat, 10% digest". Cheap: peek metadata only via
    Chroma's get(limit=N).
    """
    from datetime import datetime, timezone
    prefix = f"org_{org_id}_"
    out_collections: list[dict] = []
    totals = {"records": 0, "by_produced_by": {}}

    try:
        cols = client.list_collections()
    except Exception:
        _log.warning("collection_health: list_collections failed", exc_info=True)
        return {"org_id": org_id, "collections": [], "totals": totals}

    for c in cols:
        if not c.name.startswith(prefix):
            continue
        try:
            count = c.count()
        except Exception:
            count = 0
        # peek up to 500 metadatas; enough for most orgs to give a stable
        # provenance breakdown without paging the whole collection
        produced_breakdown: dict[str, int] = {}
        ingest_ts: list[str] = []
        try:
            sample = c.get(limit=500, include=["metadatas"])
            for m in (sample.get("metadatas") or []):
                pb = (m or {}).get("produced_by") or "unknown"
                produced_breakdown[pb] = produced_breakdown.get(pb, 0) + 1
                ts = (m or {}).get("ingested_at")
                if ts:
                    ingest_ts.append(str(ts))
        except Exception:
            _log.debug("collection_health: sample failed  name=%s", c.name, exc_info=True)

        oldest = min(ingest_ts) if ingest_ts else None
        newest = max(ingest_ts) if ingest_ts else None
        out_collections.append({
            "name": c.name[len(prefix):],
            "scoped_name": c.name,
            "records": count,
            "by_produced_by": produced_breakdown,
            "oldest_ingested_at": oldest,
            "newest_ingested_at": newest,
            "sampled": min(count, 500),
        })
        totals["records"] += count
        for k, v in produced_breakdown.items():
            totals["by_produced_by"][k] = totals["by_produced_by"].get(k, 0) + v

    out_collections.sort(key=lambda x: x["name"])
    return {
        "org_id": org_id,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "collections": out_collections,
        "totals": totals,
    }

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
    ids_row = (results.get("ids") or [[]])[0]
    for i, doc in enumerate(results["documents"][0]):
        output.append({
            "id": ids_row[i] if i < len(ids_row) else None,
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    _log.info("recall ok  collection=%s returned=%d", scoped, len(output))
    return output
