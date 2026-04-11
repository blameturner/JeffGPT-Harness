from __future__ import annotations

import logging
import threading
from contextlib import contextmanager

from config import MODEL_PARALLEL_SLOTS, MODELS

_log = logging.getLogger("web_search.models")


# Enrichment / crawl helpers must NEVER resolve to the reasoner model.
_ENRICHMENT_SAFE_ROLES = ("tool", "fast")

# slot semaphores bound harness-side concurrency to the llama.cpp --parallel N
# value so the backend queue never builds up; MODEL_PARALLEL_SLOTS must match
# the llama-server --parallel flag for each role
_FAST_SLOT_SEM = threading.Semaphore(MODEL_PARALLEL_SLOTS)
_TOOL_SLOT_SEM = threading.Semaphore(MODEL_PARALLEL_SLOTS)


@contextmanager
def fast_slot():
    _FAST_SLOT_SEM.acquire()
    try:
        yield
    finally:
        _FAST_SLOT_SEM.release()


@contextmanager
def tool_slot():
    _TOOL_SLOT_SEM.acquire()
    try:
        yield
    finally:
        _TOOL_SLOT_SEM.release()


def _resolve_safe_model(preferred_role: str) -> tuple[str | None, str | None]:
    assert preferred_role in _ENRICHMENT_SAFE_ROLES, f"unsafe role {preferred_role!r}"

    chain = [preferred_role] + [r for r in _ENRICHMENT_SAFE_ROLES if r != preferred_role]
    for role in chain:
        entry = MODELS.get(role)
        if isinstance(entry, dict) and entry.get("url"):
            if role != preferred_role:
                _log.debug("no '%s' role in catalog, falling back to %s",
                           preferred_role, role)
            return entry.get("url"), entry.get("model_id") or role

    reasoner_entry = MODELS.get("reasoner")
    reasoner_url = reasoner_entry.get("url") if isinstance(reasoner_entry, dict) else None
    seen_urls: set[str] = set()
    for v in MODELS.values():
        if not isinstance(v, dict):
            continue
        url = v.get("url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        if v.get("role") == "reasoner":
            continue
        if reasoner_url and url == reasoner_url:
            continue
        _log.debug(
            "no named tool/fast role for %s, falling back to %s",
            preferred_role, v.get("role"),
        )
        return url, v.get("model_id") or v.get("role")

    _log.error(
        "no safe enrichment model available  preferred=%s catalog_roles=%s",
        preferred_role,
        sorted({v.get("role") for v in MODELS.values() if isinstance(v, dict)}),
    )
    return None, None


def _fast_model() -> tuple[str | None, str | None]:
    return _resolve_safe_model("fast")


def _tool_model() -> tuple[str | None, str | None]:
    return _resolve_safe_model("tool")
