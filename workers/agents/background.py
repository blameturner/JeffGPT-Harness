"""Shared post-turn background work for chat and code agents.

Phases:
  1. RAG embedding (per-conversation collection)
  2. Knowledge embedding (shared knowledge collection)
  3. Graph extraction queued to tool_queue (inside knowledge block)
  4. Conversation summarisation (always runs)
  5. Optional extra phase (e.g. code checklist extraction)
"""

from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

_log = logging.getLogger(__name__)


@dataclass
class PostTurnConfig:
    conversation_id: int
    org_id: int
    user_message: str
    output: str
    model: str
    history: list[dict]
    collection_name: str           # per-conversation RAG collection
    knowledge_collection: str      # shared knowledge collection name
    rag_enabled: bool
    knowledge_enabled: bool
    source: str = "chat"           # "chat" or "code" — for graph extraction source tag
    db: Any = None                 # NocodbClient instance
    extra_phase: Callable | None = None  # optional phase 5 callback

    # --- summary persistence callbacks ---
    # list_messages(conversation_id) -> list[dict]
    list_messages_fn: Callable | None = None
    # patch_summary(existing_id, content) -> None
    patch_summary_fn: Callable | None = None
    # create_summary(conversation_id, org_id, content) -> None
    create_summary_fn: Callable | None = None

    # Extra metadata merged into the embedding metadata dict
    extra_metadata: dict = field(default_factory=dict)


def _phase_count(config: PostTurnConfig) -> int:
    return 5 if config.extra_phase else 4


def run_post_turn_work(config: PostTurnConfig) -> None:
    """Execute all post-turn background phases sequentially.

    Designed to be called from a daemon thread.
    """
    from memory import remember
    from workers.chat.history import maybe_summarise

    # Lazy import to avoid circular dependency
    from workers.agents.base import _get_summary_event

    cid = config.conversation_id
    n = _phase_count(config)
    _t_bg = time.perf_counter()
    _log.info("%s conv=%s  post-turn background starting", config.source, cid)

    base_metadata = {
        "conversation_id": cid,
        "model": config.model,
        "turn_time": time.time(),
    }
    base_metadata.update(config.extra_metadata)

    turn_text = f"USER: {config.user_message}\n\nASSISTANT: {config.output}"

    # ------------------------------------------------------------------
    # Phase 1: RAG embedding
    # ------------------------------------------------------------------
    if config.rag_enabled and config.output:
        _t = time.perf_counter()
        try:
            remember(
                text=turn_text,
                metadata=base_metadata,
                org_id=config.org_id,
                collection_name=config.collection_name,
            )
            _log.info(
                "%s conv=%s  [1/%d] RAG embedded to %s  %.2fs",
                config.source, cid, n, config.collection_name, time.perf_counter() - _t,
            )
        except Exception:
            _log.error("%s conv=%s  [1/%d] RAG embed FAILED", config.source, cid, n, exc_info=True)

    # ------------------------------------------------------------------
    # Phase 2: Knowledge embedding
    # ------------------------------------------------------------------
    if config.knowledge_enabled and config.output:
        _t = time.perf_counter()
        try:
            remember(
                text=turn_text,
                metadata=base_metadata,
                org_id=config.org_id,
                collection_name=config.knowledge_collection,
            )
            _log.info(
                "%s conv=%s  [2/%d] knowledge embedded  %.2fs",
                config.source, cid, n, time.perf_counter() - _t,
            )
        except Exception:
            _log.error("%s conv=%s  [2/%d] knowledge embed FAILED", config.source, cid, n, exc_info=True)

        # --------------------------------------------------------------
        # Phase 3: Graph extraction (queued, not inline)
        # --------------------------------------------------------------
        try:
            from workers.tool_queue import get_tool_queue
            tq = get_tool_queue()
            if tq:
                job_id = tq.submit(
                    job_type="graph_extract",
                    payload={
                        "user_text": config.user_message,
                        "assistant_text": config.output,
                        "conversation_id": cid,
                        "org_id": config.org_id,
                    },
                    source=config.source,
                    org_id=config.org_id,
                    priority=5,
                )
                _log.info(
                    "%s conv=%s  [3/%d] graph extraction queued  job=%s",
                    config.source, cid, n, job_id,
                )
        except Exception:
            _log.error("%s conv=%s  [3/%d] graph extraction queue FAILED", config.source, cid, n, exc_info=True)

    # ------------------------------------------------------------------
    # Phase 4: Background summarisation
    # ------------------------------------------------------------------
    summary_ev = _get_summary_event(cid)
    summary_ev.clear()  # mark as running
    _t = time.perf_counter()
    try:
        full_history = config.history + [
            {"role": "user", "content": config.user_message},
            {"role": "assistant", "content": config.output},
        ]
        summarised_history, bg_summary_event = maybe_summarise(full_history, truncate_only=False)
        if bg_summary_event and not bg_summary_event.get("fallback"):
            topics = bg_summary_event.get("topics", [])
            summary_content = ""
            for m in summarised_history:
                if m.get("role") == "system" and "[Conversation summary]" in (m.get("content") or ""):
                    summary_content = m["content"]
                    break
            if summary_content:
                try:
                    _persist_summary(config, summary_content, cid, topics, _t, n)
                except Exception:
                    _log.error("%s conv=%s  [4/%d] summary persist FAILED", config.source, cid, n, exc_info=True)
            else:
                _log.info(
                    "%s conv=%s  [4/%d] summary produced but empty — skipped persist",
                    config.source, cid, n,
                )
        elif bg_summary_event and bg_summary_event.get("fallback"):
            _log.info(
                "%s conv=%s  [4/%d] summary skipped — model unavailable, truncation only",
                config.source, cid, n,
            )
        else:
            _log.info(
                "%s conv=%s  [4/%d] summary skipped — under threshold (%d messages)",
                config.source, cid, n, len(full_history),
            )
    except Exception:
        _log.error("%s conv=%s  [4/%d] summary FAILED", config.source, cid, n, exc_info=True)
    finally:
        summary_ev.set()  # mark as done, unblock any waiting turn

    # ------------------------------------------------------------------
    # Phase 5: extra phase callback (e.g. code checklist extraction)
    # ------------------------------------------------------------------
    if config.extra_phase:
        try:
            config.extra_phase()
        except Exception:
            _log.error("%s conv=%s  [5/%d] extra phase FAILED", config.source, cid, n, exc_info=True)

    _log.info(
        "%s conv=%s  post-turn complete  total=%.2fs",
        config.source, cid, time.perf_counter() - _t_bg,
    )


def _persist_summary(
    config: PostTurnConfig,
    summary_content: str,
    cid: int,
    topics: list,
    phase_start: float,
    n: int,
) -> None:
    """Persist or update the conversation summary via config callbacks."""
    if config.list_messages_fn is None:
        _log.warning("%s conv=%s  [4/%d] no list_messages_fn — cannot persist summary", config.source, cid, n)
        return

    existing_msgs = config.list_messages_fn(cid)
    existing_id = None
    for msg in existing_msgs:
        if msg.get("role") == "system" and "[Conversation summary]" in (msg.get("content") or ""):
            existing_id = msg.get("Id")
            break

    if existing_id and config.patch_summary_fn:
        config.patch_summary_fn(existing_id, summary_content)
        _log.info(
            "%s conv=%s  [4/%d] summary updated  topics=%s  %.2fs",
            config.source, cid, n, topics, time.perf_counter() - phase_start,
        )
    elif config.create_summary_fn:
        config.create_summary_fn(cid, config.org_id, summary_content)
        _log.info(
            "%s conv=%s  [4/%d] summary created  topics=%s  %.2fs",
            config.source, cid, n, topics, time.perf_counter() - phase_start,
        )
    else:
        _log.warning(
            "%s conv=%s  [4/%d] missing persistence callback — summary not saved",
            config.source, cid, n,
        )
