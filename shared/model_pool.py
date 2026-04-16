from __future__ import annotations

import contextvars
import logging
import threading
import time
from contextlib import contextmanager
from typing import Iterator

from infra.config import MODEL_PARALLEL_SLOTS, MODELS

_log = logging.getLogger("web_search.models")


# pools exclude gemma 4 — thinking can't be disabled per-request
_FAST_POOL: tuple[str, ...] = ("t3_tool", "t2_coder")
_TOOL_POOL: tuple[str, ...] = ("t3_tool", "t2_coder")

_POOLS: dict[str, tuple[str, ...]] = {
    "fast": _FAST_POOL,
    "tool": _TOOL_POOL,
}

_SAFE_ROLES: tuple[str, ...] = tuple(sorted(set(_FAST_POOL + _TOOL_POOL)))


class _PrioritySemaphore:
    """Semaphore that wakes priority waiters before normal waiters on each release.

    Slots already in flight (currently-running LLM calls) are never interrupted —
    they complete normally.  But when a slot is released, any priority (user-facing
    chat/code) waiters are woken *before* queued background (scraper/summariser)
    waiters regardless of arrival order.  This means a chat request that arrives
    while scrapers are saturating the pool will get the very next free slot,
    rather than queuing behind a backlog of background work — without either
    side silently failing or dropping its request.
    """

    __slots__ = ("_lock", "_value", "_priority_q", "_normal_q")

    def __init__(self, value: int) -> None:
        self._lock = threading.Lock()
        self._value = value
        self._priority_q: list[threading.Event] = []
        self._normal_q: list[threading.Event] = []

    def acquire(self, blocking: bool = True, priority: bool = False) -> bool:
        with self._lock:
            if self._value > 0:
                self._value -= 1
                return True
            if not blocking:
                return False
            ev = threading.Event()
            if priority:
                self._priority_q.append(ev)
            else:
                self._normal_q.append(ev)
        ev.wait()
        return True

    def release(self) -> None:
        with self._lock:
            if self._priority_q:
                # Priority waiter gets the slot; don't increment _value.
                self._priority_q.pop(0).set()
            elif self._normal_q:
                self._normal_q.pop(0).set()
            else:
                self._value += 1

    def free_slots(self) -> int:
        with self._lock:
            return self._value

    def queued_priority(self) -> int:
        with self._lock:
            return len(self._priority_q)

    def queued_normal(self) -> int:
        with self._lock:
            return len(self._normal_q)


_role_semaphores: dict[str, _PrioritySemaphore] = {}
_role_sem_lock = threading.Lock()

_user_priority_ctx: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_user_priority_ctx", default=False
)


@contextmanager
def user_priority_scope() -> Iterator[None]:
    token = _user_priority_ctx.set(True)
    try:
        yield
    finally:
        _user_priority_ctx.reset(token)


def user_priority_active() -> bool:
    return _user_priority_ctx.get()

_user_requests_waiting = threading.Event()

_USER_YIELD_TIMEOUT_S = 120.0
_USER_YIELD_POLL_S = 0.25


def _yield_to_user_if_waiting(
    label: str,
    candidates: "list[tuple[str, dict, _PrioritySemaphore]] | None" = None,
) -> None:
    if not _user_requests_waiting.is_set():
        return
    if candidates and any(_free_slots(sem) > 0 for _, _, sem in candidates):
        return
    waited = 0.0
    while _user_requests_waiting.is_set() and waited < _USER_YIELD_TIMEOUT_S:
        time.sleep(_USER_YIELD_POLL_S)
        if not _user_requests_waiting.is_set():
            break
        waited += _USER_YIELD_POLL_S
    if waited > 0:
        _log.info("background acquirer (%s) yielded %.1fs to user request", label, waited)


def _sem_for(role: str) -> _PrioritySemaphore:
    with _role_sem_lock:
        sem = _role_semaphores.get(role)
        if sem is None:
            from infra.config import ROLE_PARALLEL_SLOTS
            slots = ROLE_PARALLEL_SLOTS.get(role, MODEL_PARALLEL_SLOTS)
            sem = _PrioritySemaphore(int(slots) if slots is not None else 1)
            _role_semaphores[role] = sem
        return sem


def _free_slots(sem: _PrioritySemaphore) -> int:
    return sem.free_slots()


def _present_candidates(pool: tuple[str, ...]) -> list[tuple[str, dict, _PrioritySemaphore]]:
    out: list[tuple[str, dict, _PrioritySemaphore]] = []
    for role in pool:
        entry = MODELS.get(role)
        if isinstance(entry, dict) and entry.get("url"):
            out.append((role, entry, _sem_for(role)))
    return out


def _catalog_roles() -> list[str]:
    return sorted(
        role for role in (v.get("role") for v in MODELS.values() if isinstance(v, dict))
        if isinstance(role, str) and role
    )


@contextmanager
def acquire_model(pool_name: str, priority: bool = False) -> Iterator[tuple[str | None, str | None]]:
    # caller MUST post to yielded url — resolver may pick a different role than a separate URL lookup
    # Auto-promote to priority when the call stack is inside user_priority_scope().
    if not priority and user_priority_active():
        priority = True

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
            _catalog_roles(),
        )
        yield None, None
        return

    if priority:
        _user_requests_waiting.set()
    else:
        # Background callers only yield when every slot is occupied — if a free
        # slot exists the priority caller can take it directly and there's no
        # contention to resolve.
        _yield_to_user_if_waiting(f"pool={pool_name}", present)

    # ties broken by pool order — earlier entries are preferred
    ranked = sorted(
        enumerate(present),
        key=lambda it: (-_free_slots(it[1][2]), it[0]),
    )

    acquired_role: str | None = None
    acquired_sem: _PrioritySemaphore | None = None
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
        sem.acquire(blocking=True, priority=priority)
        acquired_role = role
        acquired_sem = sem
        acquired_entry = entry

    assert acquired_sem is not None

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
            _catalog_roles(),
        )
        yield None, None
        return

    # Auto-promote to priority when the call stack is inside user_priority_scope().
    if not priority and user_priority_active():
        priority = True

    # Compute semaphore before the yield check so we can pass free-slot info.
    sem = _sem_for(role)

    if priority:
        _user_requests_waiting.set()
    else:
        # Background callers only yield when the role's slot is fully occupied —
        # if a slot is free the priority caller can take it directly.
        _yield_to_user_if_waiting(f"role={role}", [(role, entry, sem)])
    if not sem.acquire(blocking=False):
        _log.info("role=%s slot busy — blocking (priority=%s)", role, priority)
        sem.acquire(blocking=True, priority=priority)

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
