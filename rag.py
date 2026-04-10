import logging
import requests
from config import RERANKER_URL
from memory import recall
from embedder import EMBED_MAX_WORDS

_log = logging.getLogger("rag")

QUERY_CHUNK_WORDS = EMBED_MAX_WORDS
QUERY_CHUNK_OVERLAP = 50


def _chunk_query(text: str) -> list[str]:
    words = text.split()
    if len(words) <= QUERY_CHUNK_WORDS:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + QUERY_CHUNK_WORDS
        chunks.append(" ".join(words[start:end]))
        start += QUERY_CHUNK_WORDS - QUERY_CHUNK_OVERLAP
    return chunks


def rerank(query: str, texts: list[str]) -> list[str]:
    rerank_query = query if len(query.split()) <= QUERY_CHUNK_WORDS else " ".join(query.split()[:QUERY_CHUNK_WORDS])
    _log.debug("rerank  query=%s docs=%d reranker=%s", rerank_query[:80], len(texts), RERANKER_URL)
    response = requests.post(
        f"{RERANKER_URL}/v1/rerank",
        json={
            "model": "bge-reranker",
            "query": rerank_query,
            "documents": texts
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    sorted_results = sorted(data["results"], key=lambda r: r["relevance_score"], reverse=True)
    _log.debug("rerank ok  top_score=%.3f", sorted_results[0]["relevance_score"] if sorted_results else 0)
    return [texts[r["index"]] for r in sorted_results]


def retrieve(query: str, org_id: int, collection_name: str = "agent_outputs", n_results: int = 10, top_k: int = 3) -> str:
    _log.info("retrieve  org=%d collection=%s n=%d top_k=%d query=%s", org_id, collection_name, n_results, top_k, query[:80])

    query_chunks = _chunk_query(query)
    _log.debug("retrieve  query split into %d chunks", len(query_chunks))

    seen_docs: set[str] = set()
    all_candidates: list[dict] = []
    per_chunk_n = max(3, n_results // len(query_chunks))

    for i, chunk in enumerate(query_chunks):
        try:
            results = recall(chunk, org_id=org_id, collection_name=collection_name, n_results=per_chunk_n)
        except Exception:
            _log.warning("recall failed for query chunk %d/%d", i + 1, len(query_chunks), exc_info=True)
            continue
        for r in results:
            doc_key = r["text"][:200]
            if doc_key not in seen_docs:
                seen_docs.add(doc_key)
                all_candidates.append(r)

    if not all_candidates:
        _log.info("retrieve  no candidates found across %d chunks", len(query_chunks))
        return ""

    _log.debug("retrieve  %d unique candidates from %d chunks", len(all_candidates), len(query_chunks))
    texts = [c["text"][:1500] for c in all_candidates]

    try:
        ranked_texts = rerank(query, texts)
    except Exception:
        _log.error("rerank failed, using unranked candidates", exc_info=True)
        ranked_texts = texts

    top_texts = ranked_texts[:top_k]
    _log.info("retrieve ok  candidates=%d ranked=%d returned=%d chunks_queried=%d", len(all_candidates), len(ranked_texts), len(top_texts), len(query_chunks))

    context_block = "\n\n---\n\n".join(top_texts)
    return f"RELEVANT CONTEXT:\n\n{context_block}"
