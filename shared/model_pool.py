from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Iterator

from infra.config import MODEL_PARALLEL_SLOTS, MODELS, REASONER_ROLE

_log = logging.getLogger("web_search.models")


# pools exclude gemma 4 — thinking can't be disabled per-request
_FAST_POOL: tuple[str, ...] = ("t3_tool", "t2_coder")
_TOOL_POOL: tuple[str, ...] = ("t3_tool", "t2_coder")

_POOLS: dict[str, tuple[str, ...]] = {
    "fast": _FAST_POOL,
    "tool": _TOOL_POOL,
}

_SAFE_ROLES: tuple[str, ...] = tuple(sorted(set(_FAST_POOL + _TOOL_POOL)))


_role_semaphores: dict[str, threading.Semaphore] = {}
_role_sem_lock = threading.Lock()

# background tasks watch this and back off when user requests are waiting
_user_requests_waiting = threading.Event()


def _sem_for(role: str) -> threading.Semaphore:
    with _role_sem_lock:
        sem = _role_semaphores.get(role)
        if sem is None:
            from infra.config import ROLE_PARALLEL_SLOTS
            slots = ROLE_PARALLEL_SLOTS.get(role, MODEL_PARALLEL_SLOTS)
            sem = threading.Semaphore(slots)
            _role_semaphores[role] = sem
        return sem


def _free_slots(sem: threading.Semaphore) -> int:
    # _value is private but stable across CPython versions
    return int(getattr(sem, "_value", 0))


def _present_candidates(pool: tuple[str, ...]) -> list[tuple[str, dict, threading.Semaphore]]:
    out: list[tuple[str, dict, threading.Semaphore]] = []
    for role in pool:
        entry = MODELS.get(role)
        if isinstance(entry, dict) and entry.get("url"):
            out.append((role, entry, _sem_for(role)))
    return out


@contextmanager
def acquire_model(pool_name: str, priority: bool = False) -> Iterator[tuple[str | None, str | None]]:
    # caller MUST post to yielded url — resolver may pick a different role than a separate URL lookup
    if priority:
        _user_requests_waiting.set()

    pool = _POOLS.get(pool_name, ())
    if not pool:
        _log.error("acquire_model unknown pool=%s", pool_name)
        if priority:
            _user_requests_waiting.clear()
        yield None, None
        return

    present = _present_candidates(pool)
    if not present:
        _log.error(
            "no model available in pool=%s catalog_roles=%s",
            pool_name,
            sorted({v.get("role") for v in MODELS.values() if isinstance(v, dict)}),
        )
        if priority:
            _user_requests_waiting.clear()
        yield None, None
        return

    # ties broken by pool order — earlier entries are preferred
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
        _, (role, entry, sem) = ranked[0]
        _log.info("pool=%s all slots busy — blocking on %s (priority=%s)", pool_name, role, priority)
        sem.acquire(blocking=True)
        acquired_role = role
        acquired_sem = sem
        acquired_entry = entry

    if priority:
        _user_requests_waiting.clear()

    url = acquired_entry.get("url")
    model_id = acquired_entry.get("model_id") or acquired_role
    free_now = _free_slots(acquired_sem)
    _log.info(
        "pool=%s acquired role=%s free_after=%d (priority=%s)",
        pool_name, acquired_role, free_now, priority,
    )
    try:
        yield url, model_id
    finally:
        acquired_sem.release()


def _best_in_pool(pool: tuple[str, ...]) -> tuple[str | None, str | None]:
    present = _present_candidates(pool)
    if not present:
        return None, None
    present.sort(key=lambda rns: -_free_slots(rns[2]))
    role, entry, _sem = present[0]
    return entry.get("url"), entry.get("model_id") or role


def _fast_model() -> tuple[str | None, str | None]:
    # best-effort snapshot — doesn't bound concurrency, prefer acquire_model("fast") for that
    return _best_in_pool(_FAST_POOL)


def _tool_model() -> tuple[str | None, str | None]:
    return _best_in_pool(_TOOL_POOL)


@contextmanager
def fast_slot() -> Iterator[None]:
    # deprecated shim — pick and slot aren't atomic, caller's separate url may be for a different role
    with acquire_model("fast"):
        yield


@contextmanager
def tool_slot() -> Iterator[None]:
    with acquire_model("tool"):
        yield


@contextmanager
def acquire_role(role: str, priority: bool = False) -> Iterator[tuple[str | None, str | None]]:
    entry = MODELS.get(role)
    if not isinstance(entry, dict) or not entry.get("url"):
        _log.error(
            "acquire_role: role=%s not in catalog. available=%s",
            role,
            sorted({v.get("role") for v in MODELS.values() if isinstance(v, dict)}),
        )
        yield None, None
        return

    if priority:
        _user_requests_waiting.set()

    sem = _sem_for(role)
    if not sem.acquire(blocking=False):
        _log.info("role=%s slot busy — blocking (priority=%s)", role, priority)
        sem.acquire(blocking=True)

    if priority:
        _user_requests_waiting.clear()

    url = entry.get("url")
    model_id = entry.get("model_id") or role
    free_now = _free_slots(sem)
    _log.info("role=%s acquired free_after=%d (priority=%s)", role, free_now, priority)
    try:
        yield url, model_id
    finally:
        sem.release()
