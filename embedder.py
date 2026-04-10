import logging
import time
import requests

from config import EMBEDDER_URL

_log = logging.getLogger("embedder")

EMBED_MAX_WORDS = 180


def embed(text: str) -> list[float]:
    words = text.split()
    text_len = len(words)
    if text_len > EMBED_MAX_WORDS:
        text = " ".join(words[:EMBED_MAX_WORDS])
        _log.debug("embed  truncated %d -> %d words", text_len, EMBED_MAX_WORDS)
    _log.debug("embed  words=%d url=%s", min(text_len, EMBED_MAX_WORDS), EMBEDDER_URL)
    started = time.time()
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
        dims = len(data["data"][0]["embedding"])
        elapsed = round(time.time() - started, 3)
        _log.debug("embed ok  dims=%d %.3fs", dims, elapsed)
        return data["data"][0]["embedding"]
    except requests.HTTPError as e:
        _log.error("embedder %d from %s: %s", e.response.status_code, EMBEDDER_URL, e.response.text[:300])
        raise
    except requests.ConnectionError:
        _log.error("embedder unreachable at %s", EMBEDDER_URL)
        raise
    except requests.Timeout:
        _log.error("embedder timeout at %s (words=%d)", EMBEDDER_URL, text_len)
        raise
    except Exception:
        _log.error("embedder failed", exc_info=True)
        raise
