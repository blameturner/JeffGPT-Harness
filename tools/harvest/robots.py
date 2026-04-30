"""robots.txt cache + check.

Fetches /robots.txt at most once per host per 24h. Disabled per-host via
features.harvest.host_policies[host].respect_robots = false.
"""
from __future__ import annotations

import logging
import threading
import time
from urllib.parse import urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests

from infra.config import get_feature

_log = logging.getLogger("harvest.robots")

_CACHE_TTL_S = 24 * 3600
_USER_AGENT = "mst-harness/1.0 (+contact)"

_lock = threading.Lock()
_cache: dict[str, tuple[float, RobotFileParser | None]] = {}


def _robots_url(target_url: str) -> str:
    p = urlparse(target_url)
    return urlunparse((p.scheme or "https", p.netloc, "/robots.txt", "", "", ""))


def _host_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _per_host_respects(host: str, policy_default: bool) -> bool:
    # File-backed host_config wins (mutable at runtime via the API).
    try:
        from tools.harvest.host_config import get as _hc_get
        cfg = _hc_get(host) or {}
        if "respect_robots" in cfg:
            return bool(cfg["respect_robots"])
    except Exception:
        pass
    overrides = get_feature("harvest", "host_policies", {}) or {}
    if isinstance(overrides, dict):
        h = overrides.get(host)
        if isinstance(h, dict):
            v = h.get("respect_robots")
            if v is not None:
                return bool(v)
    return bool(policy_default)


def _load_robots(target_url: str) -> RobotFileParser | None:
    """Fetch and parse robots.txt for the host of ``target_url``. Returns
    None on error (treated as 'no robots' = allowed). Cached for 24h."""
    host = _host_of(target_url)
    if not host:
        return None
    now = time.time()
    with _lock:
        entry = _cache.get(host)
        if entry and entry[0] > now:
            return entry[1]

    rp = RobotFileParser()
    rp.set_url(_robots_url(target_url))
    try:
        resp = requests.get(
            _robots_url(target_url),
            timeout=5,
            headers={"User-Agent": _USER_AGENT},
        )
        if resp.status_code == 200 and resp.text:
            rp.parse(resp.text.splitlines())
        elif resp.status_code in (401, 403):
            # Treat as fully restricted
            rp.parse(["User-agent: *", "Disallow: /"])
        else:
            # 404 / 5xx → no robots → allow all
            rp = None
    except Exception as e:
        _log.debug("robots.txt fetch failed for %s: %s", host, e)
        rp = None

    with _lock:
        _cache[host] = (now + _CACHE_TTL_S, rp)
    return rp


def can_fetch(url: str, *, policy_respects: bool = True,
              user_agent: str = _USER_AGENT) -> bool:
    """Returns True if the URL is allowed by robots (or robots is disabled
    or unreachable)."""
    if not policy_respects:
        return True
    host = _host_of(url)
    if not _per_host_respects(host, policy_respects):
        return True
    rp = _load_robots(url)
    if rp is None:
        return True
    try:
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True


def clear_cache(host: str | None = None) -> None:
    with _lock:
        if host is None:
            _cache.clear()
        else:
            _cache.pop(host.lower(), None)
