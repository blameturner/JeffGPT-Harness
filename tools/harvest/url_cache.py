"""Per-URL state cache for conditional-GET re-checks.

File-backed at /app/data/harvest_url_cache.json. Each entry holds the
last-seen ETag, Last-Modified, content_hash, and timestamp for one URL.
Re-fetches send `If-None-Match` / `If-Modified-Since` to short-circuit
on `304 Not Modified` and skip the LLM extraction entirely.

This is a local cache — no DB column required. It survives across
process restarts but not across container rebuilds (acceptable; the
worst case is one wasted re-fetch per URL).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path

_log = logging.getLogger("harvest.url_cache")

_PATH = Path(os.getenv("HARVEST_URL_CACHE_PATH", "/app/data/harvest_url_cache.json"))
_lock = threading.Lock()
_cache: dict | None = None
_dirty = False
_last_save = 0.0
_SAVE_INTERVAL_S = 30.0  # debounce disk writes


def _key(url: str) -> str:
    return hashlib.sha256((url or "").encode("utf-8", errors="ignore")).hexdigest()[:24]


def _load() -> dict:
    if not _PATH.exists():
        return {}
    try:
        with open(_PATH, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        _log.warning("url_cache: failed to read %s", _PATH)
        return {}


def _save(force: bool = False) -> None:
    global _dirty, _last_save
    if not _dirty and not force:
        return
    if not force and (time.time() - _last_save) < _SAVE_INTERVAL_S:
        return
    try:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _PATH.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(_cache or {}, f)
        os.replace(tmp, _PATH)
        _dirty = False
        _last_save = time.time()
    except Exception:
        _log.warning("url_cache: failed to write %s", _PATH, exc_info=True)


def _ensure_loaded() -> dict:
    global _cache
    if _cache is None:
        with _lock:
            if _cache is None:
                _cache = _load()
    return _cache


def get(url: str) -> dict | None:
    """Returns {etag, last_modified, content_hash, fetched_at} or None."""
    if not url:
        return None
    entry = _ensure_loaded().get(_key(url))
    return dict(entry) if isinstance(entry, dict) else None


def set_state(url: str, *, etag: str = "", last_modified: str = "",
              content_hash: str = "") -> None:
    """Record post-fetch state for ``url``."""
    if not url:
        return
    global _dirty
    with _lock:
        cache = _ensure_loaded()
        cache[_key(url)] = {
            "url": url,
            "etag": etag or "",
            "last_modified": last_modified or "",
            "content_hash": content_hash or "",
            "fetched_at": time.time(),
        }
        _dirty = True
    _save()


def flush() -> None:
    """Force write to disk (call from shutdown hook if you have one)."""
    _save(force=True)
