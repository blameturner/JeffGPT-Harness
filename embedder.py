import requests

from config import EMBEDDER_URL

# may need to remove hardcoded name of model later.

def embed(text: str) -> list[float]:
    response = requests.post(
        f"{EMBEDDER_URL}/v1/embeddings",
        json={
            "model": "nomic-embed",
            "input": text
        }
    )
    response.raise_for_status()
    data = response.json()
    return data["data"][0]["embedding"]