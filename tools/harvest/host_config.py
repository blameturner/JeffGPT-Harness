"""Per-host harvest configuration. File-backed at /app/data/harvest_hosts.json
with an in-memory cache. Falls back to features.harvest.host_policies when
no file entry exists.

Schema:
    {
      "<host>": {
        "rate_limit_per_host_s": float,
        "respect_robots": bool,
        "headless_fallback": bool,
        "connection_id": int,         # use this api_connection for auth
        "notes": str,
        "cool_off_until": float       # unix ts (rare — usually managed by rate_limit)
      }
    }
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from infra.config import get_feature

_log = logging.getLogger("harvest.host_config")

_PATH = Path(os.getenv("HARVEST_HOST_CONFIG_PATH", "/app/data/harvest_hosts.json"))
_lock = threading.Lock()
_cache: dict | None = None


def _load_from_disk() -> dict:
    if not _PATH.exists():
        return {}
    try:
        with open(_PATH, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        _log.warning("host_config: failed to read %s, starting empty", _PATH)
        return {}


def _save_to_disk(data: dict) -> None:
    try:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _PATH.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, _PATH)
    except Exception:
        _log.warning("host_config: failed to write %s", _PATH, exc_info=True)


def _ensure_loaded() -> dict:
    global _cache
    if _cache is None:
        with _lock:
            if _cache is None:
                _cache = _load_from_disk()
    return _cache


def all_hosts() -> dict:
    """Snapshot of file-backed configs (does NOT include features fallback)."""
    return dict(_ensure_loaded())


def get(host: str) -> dict:
    """Resolve config for ``host``. File config wins; falls back to
    ``features.harvest.host_policies[host]``; finally returns {}."""
    if not host:
        return {}
    h = host.lower()
    file_cfg = _ensure_loaded().get(h)
    if file_cfg:
        return dict(file_cfg)
    feature_cfg = get_feature("harvest", "host_policies", {}) or {}
    if isinstance(feature_cfg, dict):
        v = feature_cfg.get(h)
        if isinstance(v, dict):
            return dict(v)
    return {}


def set_host(host: str, fields: dict) -> dict:
    """Upsert config for ``host``. Returns the merged record."""
    if not host:
        raise ValueError("host required")
    h = host.lower()
    with _lock:
        cur = _ensure_loaded()
        existing = dict(cur.get(h) or {})
        existing.update({k: v for k, v in (fields or {}).items() if v is not None})
        cur[h] = existing
        _save_to_disk(cur)
        return dict(existing)


def delete_host(host: str) -> bool:
    if not host:
        return False
    h = host.lower()
    with _lock:
        cur = _ensure_loaded()
        if h in cur:
            cur.pop(h)
            _save_to_disk(cur)
            return True
    return False


def reload() -> None:
    """Force reload from disk (e.g. after manual edit)."""
    global _cache
    with _lock:
        _cache = _load_from_disk()
