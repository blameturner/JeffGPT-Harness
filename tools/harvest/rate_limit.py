"""Per-host token-bucket rate limiter for the harvest fetcher.

Default 1 request/second per host. Configurable per-host via the
`harvest_host_policies` JSON in features config. Thread-safe.

The limiter never blocks for more than `max_wait_s` (default 30s) —
beyond that, the caller is told to skip and try later. This keeps Huey
workers from being held hostage by a slow host.
"""
from __future__ import annotations

import logging
import threading
import time
from urllib.parse import urlparse

from infra.config import get_feature

_log = logging.getLogger("harvest.rate_limit")

_DEFAULT_RATE_S = 1.0
_DEFAULT_MAX_WAIT_S = 30.0

_locks: dict[str, threading.Lock] = {}
_locks_master = threading.Lock()
_next_allowed: dict[str, float] = {}
_failure_count: dict[str, int] = {}
_cool_off_until: dict[str, float] = {}


def _host_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _lock_for(host: str) -> threading.Lock:
    with _locks_master:
        lk = _locks.get(host)
        if lk is None:
            lk = threading.Lock()
            _locks[host] = lk
        return lk


def _per_host_rate_s(host: str, policy_default: float | None = None) -> float:
    """Resolve rate: file-backed host_config → features fallback → policy default → global default."""
    # File-backed host_config wins (mutable at runtime via the API).
    try:
        from tools.harvest.host_config import get as _hc_get
        cfg = _hc_get(host) or {}
        v = cfg.get("rate_limit_per_host_s")
        if v is not None:
            return max(0.0, float(v))
    except Exception:
        pass
    overrides = get_feature("harvest", "host_policies", {}) or {}
    if isinstance(overrides, dict):
        h = overrides.get(host)
        if isinstance(h, dict):
            v = h.get("rate_limit_per_host_s")
            if v is not None:
                try:
                    return max(0.0, float(v))
                except Exception:
                    pass
    if policy_default is not None:
        return max(0.0, float(policy_default))
    return _DEFAULT_RATE_S


def acquire(url: str, *, policy_default_rate_s: float | None = None,
            max_wait_s: float = _DEFAULT_MAX_WAIT_S) -> bool:
    """Block until it's safe to fetch ``url``. Returns False if we'd have to
    wait longer than ``max_wait_s`` (caller should skip / retry later)."""
    host = _host_of(url)
    if not host:
        return True
    now = time.time()

    # Cool-off check (per-host failure circuit breaker)
    cool_until = _cool_off_until.get(host, 0.0)
    if cool_until > now:
        _log.debug("rate_limit: host %s in cool-off for %.1fs more", host, cool_until - now)
        return False

    rate_s = _per_host_rate_s(host, policy_default_rate_s)
    if rate_s <= 0:
        return True

    lk = _lock_for(host)
    with lk:
        next_allowed = _next_allowed.get(host, 0.0)
        wait = max(0.0, next_allowed - time.time())
        if wait > max_wait_s:
            return False
        if wait > 0:
            time.sleep(wait)
        _next_allowed[host] = time.time() + rate_s
    return True


def record_failure(url: str, *, threshold: int = 5, cool_off_s: float = 3600.0) -> None:
    """Increment per-host failure counter; trip cool-off if threshold reached."""
    host = _host_of(url)
    if not host:
        return
    n = _failure_count.get(host, 0) + 1
    _failure_count[host] = n
    if n >= threshold:
        _cool_off_until[host] = time.time() + cool_off_s
        _failure_count[host] = 0
        _log.warning(
            "rate_limit: host %s entered cool-off for %.0fs after %d failures",
            host, cool_off_s, threshold,
        )


def record_success(url: str) -> None:
    """Reset per-host failure counter on success."""
    host = _host_of(url)
    if host:
        _failure_count[host] = 0


def status(host: str | None = None) -> dict:
    """Snapshot for the status endpoint."""
    if host:
        return {
            "host": host,
            "next_allowed_s": _next_allowed.get(host, 0.0),
            "failure_count": _failure_count.get(host, 0),
            "cool_off_until": _cool_off_until.get(host, 0.0),
        }
    return {
        "tracked_hosts": len(_next_allowed),
        "in_cool_off": [
            h for h, t in _cool_off_until.items() if t > time.time()
        ],
    }
