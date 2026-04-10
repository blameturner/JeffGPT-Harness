import logging
import requests

from config import EMBEDDER_URL

_log = logging.getLogger("embedder")

def embed(text: str) -> list[float]:
    try:
        response = requests.post(
            f"{EMBEDDER_URL}/v1/embeddings",
            json={
                "model": "nomic-embed",
                "input": text
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]
    except requests.HTTPError as e:
        _log.error("embedder %d from %s: %s", e.response.status_code, EMBEDDER_URL, e.response.text[:300])
        raise
    except requests.ConnectionError:
        _log.error("embedder unreachable at %s", EMBEDDER_URL)
        raise
    except requests.Timeout:
        _log.error("embedder timeout at %s (text_len=%d)", EMBEDDER_URL, len(text))
        raise
    except Exception:
        _log.error("embedder failed", exc_info=True)
        raise