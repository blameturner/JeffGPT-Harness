import logging
import requests
from config import RERANKER_URL
from memory import recall

_log = logging.getLogger("rag")


def rerank(query: str, texts: list[str]) -> list[str]:
    _log.debug("rerank  query=%s docs=%d reranker=%s", query[:80], len(texts), RERANKER_URL)
    response = requests.post(
        f"{RERANKER_URL}/v1/rerank",
        json={
            "model": "bge-reranker",
            "query": query,
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
    candidates = recall(query, org_id=org_id, collection_name=collection_name, n_results=n_results)
    if not candidates:
        _log.info("retrieve  no candidates found")
        return ""

    texts = [c["text"][:1500] for c in candidates]

    try:
        ranked_texts = rerank(query, texts)
    except Exception:
        _log.error("rerank failed, using unranked candidates", exc_info=True)
        ranked_texts = texts

    top_texts = ranked_texts[:top_k]
    _log.info("retrieve ok  candidates=%d ranked=%d returned=%d", len(candidates), len(ranked_texts), len(top_texts))

    context_block = "\n\n---\n\n".join(top_texts)
    return f"RELEVANT CONTEXT:\n\n{context_block}"
