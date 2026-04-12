"""
Load-aware model resolver for the enrichment + tool paths.

Design:
- Every candidate role has its own semaphore bounded to MODEL_PARALLEL_SLOTS,
  matching llama.cpp --parallel N for that container.
- `acquire_model(pool)` atomically picks the least-loaded healthy role in a
  named pool and acquires its slot, yielding (url, model_id). Callers must
  use the yielded url — never read the URL separately and then acquire a
  slot, because the resolver may select a different role on each call.
- The "fast" pool is t1_primary → t1_secondary → t2_coder. Enrichment
  summariser / relationship extractor picks whichever has the most free
  slots at call time, so if t1_primary is hot, work drifts to t1_secondary
  or t2_coder.
- The "tool" pool is t3_tool → t1_primary → t1_secondary → t2_coder. t3_tool
  is preferred for short classifier/JSON calls; if it's saturated or down,
  fall back load-aware across the rest.
- Neither pool includes the reasoner (no model registers under that role in
  the current deployment, but _assert_not_reasoner() is still used by
  callers as belt-and-braces).

Backward-compat:
- `_fast_model()` / `_tool_model()` still exist for callers that only need
  a URL without holding a slot (e.g. history summariser, query generator).
  They return the least-loaded healthy role in the pool at snapshot time.
- `fast_slot()` / `tool_slot()` are thin wrappers over `acquire_model` that
  discard the yielded url/model_id, for callers that only want concurrency
  bounding. New code should prefer `acquire_model` because the shim can't
  tell the caller which role was picked.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Iterator

from config import MODEL_PARALLEL_SLOTS, MODELS, REASONER_ROLE

_log = logging.getLogger("web_search.models")


# ---------- Pools ----------

# Role names match the MODEL_<NAME>_URL env var convention, lowercased and
# with the leading MODEL_ stripped (see config._collect_role_env_vars).
# Fast pool: used for enrichment summarisation and relationship extraction.
# Only non-thinking models — Gemma 4 thinking can't be disabled per-request.
_FAST_POOL: tuple[str, ...] = ("t3_tool", "t2_coder")
# Tool pool: used for planner, classifiers, web search summariser.
# t3_tool preferred (fast 4B), t2_coder fallback (no thinking).
_TOOL_POOL: tuple[str, ...] = ("t3_tool", "t2_coder")

_POOLS: dict[str, tuple[str, ...]] = {
    "fast": _FAST_POOL,
    "tool": _TOOL_POOL,
}

# Safe enrichment roles — used for a startup-time assertion and for the
# _assert_not_reasoner guard in enrichment.models.
_ENRICHMENT_SAFE_ROLES: tuple[str, ...] = tuple(sorted(set(_FAST_POOL + _TOOL_POOL)))


# ---------- Per-role slots ----------

_role_semaphores: dict[str, threading.Semaphore] = {}
_role_sem_lock = threading.Lock()


def _sem_for(role: str) -> threading.Semaphore:
    with _role_sem_lock:
        sem = _role_semaphores.get(role)
        if sem is None:
            sem = threading.Semaphore(MODEL_PARALLEL_SLOTS)
            _role_semaphores[role] = sem
        return sem


def _free_slots(sem: threading.Semaphore) -> int:
    # threading.Semaphore._value is private but stable across CPython
    # versions; it represents the remaining permits before blocking.
    return int(getattr(sem, "_value", 0))


def _present_candidates(pool: tuple[str, ...]) -> list[tuple[str, dict, threading.Semaphore]]:
    out: list[tuple[str, dict, threading.Semaphore]] = []
    for role in pool:
        entry = MODELS.get(role)
        if isinstance(entry, dict) and entry.get("url"):
            out.append((role, entry, _sem_for(role)))
    return out


# ---------- Atomic pick + acquire ----------

@contextmanager
def acquire_model(pool_name: str) -> Iterator[tuple[str | None, str | None]]:
    """
    Atomically pick the least-loaded healthy model in the named pool and
    acquire its slot. Yields (url, model_id). Callers MUST post to the
    yielded url; never mix with a separately-resolved URL.

    If no model in the pool is in the catalog, yields (None, None) and
    does NOT acquire a slot — callers should short-circuit.
    """
    pool = _POOLS.get(pool_name, ())
    if not pool:
        _log.error("acquire_model unknown pool=%s", pool_name)
        yield None, None
        return

    present = _present_candidates(pool)
    if not present:
        _log.error(
            "no model available in pool=%s catalog_roles=%s",
            pool_name,
            sorted({v.get("role") for v in MODELS.values() if isinstance(v, dict)}),
        )
        yield None, None
        return

    # Rank by most free slots first; ties broken by pool order (earlier = preferred).
    ranked = sorted(
        enumerate(present),
        key=lambda it: (-_free_slots(it[1][2]), it[0]),
    )

    acquired_role: str | None = None
    acquired_sem: threading.Semaphore | None = None
    acquired_entry: dict = {}

    for _, (role, entry, sem) in ranked:
        if sem.acquire(blocking=False):
            acquired_role = role
            acquired_sem = sem
            acquired_entry = entry
            break

    if acquired_sem is None:
        # All ranked slots busy — block on the best-ranked role. This is
        # still the fairest choice even if its free count changed between
        # rank and block.
        _, (role, entry, sem) = ranked[0]
        _log.debug("pool=%s all slots busy — blocking on %s", pool_name, role)
        sem.acquire(blocking=True)
        acquired_role = role
        acquired_sem = sem
        acquired_entry = entry

    url = acquired_entry.get("url")
    model_id = acquired_entry.get("model_id") or acquired_role
    free_now = _free_slots(acquired_sem)
    _log.debug(
        "pool=%s acquired role=%s free_after=%d url=%s",
        pool_name, acquired_role, free_now, url,
    )
    try:
        yield url, model_id
    finally:
        acquired_sem.release()


# ---------- Read-only discovery (for URL-only callers) ----------

def _best_in_pool(pool: tuple[str, ...]) -> tuple[str | None, str | None]:
    present = _present_candidates(pool)
    if not present:
        return None, None
    present.sort(key=lambda rns: -_free_slots(rns[2]))
    role, entry, _sem = present[0]
    return entry.get("url"), entry.get("model_id") or role


def _fast_model() -> tuple[str | None, str | None]:
    """
    Return (url, model_id) for the least-loaded healthy fast-pool model.
    Best-effort snapshot — callers that ALSO need concurrency bounding
    should use `acquire_model("fast")` instead.
    """
    return _best_in_pool(_FAST_POOL)


def _tool_model() -> tuple[str | None, str | None]:
    """Read-only discovery for the tool pool. See `_fast_model`."""
    return _best_in_pool(_TOOL_POOL)


# ---------- Back-compat slot shims ----------

@contextmanager
def fast_slot() -> Iterator[None]:
    """
    Deprecated — prefer `acquire_model("fast")` so pick and slot are atomic.
    This shim acquires a slot in the least-loaded fast-pool role but throws
    away which role it picked, so the caller's separate `_fast_model()` URL
    may belong to a different role. That's acceptable only for callers that
    treat slot acquisition as pure concurrency bounding.
    """
    with acquire_model("fast"):
        yield


@contextmanager
def tool_slot() -> Iterator[None]:
    """Deprecated — prefer `acquire_model("tool")`."""
    with acquire_model("tool"):
        yield
