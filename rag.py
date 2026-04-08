import requests
from config import RERANKER_URL
from memory import recall


def rerank(query: str, texts: list[str]) -> list[str]:
    pairs = [[query, text] for text in texts]

    response = requests.post(
        f"{RERANKER_URL}/v1/rerank",
        json={
            "model": "bge-reranker",
            "query": query,
            "documents": texts
        }
    )
    response.raise_for_status()
    data = response.json()

    sorted_results = (
        sorted(data["results"],
               key=lambda r: r["relevance_score"],
               reverse=True
               ))

    return [texts[r["index"]] for r in sorted_results]


def retrieve(query: str, org_id: int, collection_name: str = "agent_outputs", n_results: int = 10, top_k: int = 3) -> str:
    candidates = recall(query, org_id=org_id, collection_name=collection_name, n_results=n_results)
    if not candidates:
        return ""

    texts = [c["text"][:1500] for c in candidates] #roughly truncates to 400 tokens - batch size limit on reranker

    ranked_texts = rerank(query, texts)
    top_texts = ranked_texts[:top_k]

    context_block = "\n\n---\n\n".join(top_texts)
    return f"RELEVANT CONTEXT:\n\n{context_block}"

# Step 1 gets candidates from this org's scoped Chroma collection only
# Step 2 extracts text from each
# Step 3 reranks by relevance. only top k
# Step 4 formats into context block