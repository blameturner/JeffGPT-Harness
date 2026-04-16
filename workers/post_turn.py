from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

_log = logging.getLogger(__name__)


def ingest_output(
    *,
    output: str,
    user_text: str = "",
    org_id: int,
    conversation_id: int = 0,
    model: str = "",
    rag_collection: str = "",
    knowledge_collection: str = "",
    source: str = "",
    extra_metadata: dict | None = None,
    queue_graph_extract: bool = True,
) -> None:
    """Embed an output and queue graph extraction. Safe to call from any background context.

    Mirrors the embed/graph phases of `run_post_turn_work` without the summary machinery,
    so one-shot producers (planned_search, research) can feed RAG + FalkorDB the same way
    chat/code conversations do.
    """
    if not output:
        return

    from infra.memory import remember

    metadata: dict = {
        "conversation_id": conversation_id,
        "model": model,
        "turn_time": time.time(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    turn_text = f"USER: {user_text}\n\nASSISTANT: {output}" if user_text else output

    if rag_collection:
        try:
            remember(
                text=turn_text,
                metadata=metadata,
                org_id=org_id,
                collection_name=rag_collection,
            )
            _log.info("%s conv=%s  ingest RAG embedded to %s", source, conversation_id, rag_collection)
        except Exception:
            _log.error("%s conv=%s  ingest RAG FAILED coll=%s", source, conversation_id, rag_collection, exc_info=True)

    if knowledge_collection:
        try:
            remember(
                text=turn_text,
                metadata=metadata,
                org_id=org_id,
                collection_name=knowledge_collection,
            )
            _log.info("%s conv=%s  ingest knowledge embedded to %s", source, conversation_id, knowledge_collection)
        except Exception:
            _log.error("%s conv=%s  ingest knowledge FAILED coll=%s", source, conversation_id, knowledge_collection, exc_info=True)

    if queue_graph_extract and (user_text or output):
        try:
            from workers.tool_queue import get_tool_queue
            tq = get_tool_queue()
            if tq:
                job_id = tq.submit(
                    job_type="graph_extract",
                    payload={
                        "user_text": user_text,
                        "assistant_text": output,
                        "conversation_id": conversation_id,
                        "org_id": org_id,
                    },
                    source=source,
                    org_id=org_id,
                    priority=4,
                )
                _log.info("%s conv=%s  ingest graph_extract queued  job=%s", source, conversation_id, job_id)
        except Exception:
            _log.error("%s conv=%s  ingest graph_extract queue FAILED", source, conversation_id, exc_info=True)


@dataclass
class PostTurnConfig:
    conversation_id: int
    org_id: int
    user_message: str
    output: str
    model: str
    history: list[dict]
    collection_name: str
    knowledge_collection: str
    rag_enabled: bool
    knowledge_enabled: bool
    source: str = "chat"
    db: Any = None
    extra_phase: Callable | None = None

    list_messages_fn: Callable | None = None
    patch_summary_fn: Callable | None = None
    create_summary_fn: Callable | None = None

    extra_metadata: dict = field(default_factory=dict)


def _phase_count(config: PostTurnConfig) -> int:
    return 5 if config.extra_phase else 4


def run_post_turn_work(config: PostTurnConfig) -> None:
    from infra.memory import remember
    from workers.chat.history import maybe_summarise

    # lazy import — circular dep with workers.base
    from workers.base import _get_summary_event

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
                    priority=4,
                )
                _log.info(
                    "%s conv=%s  [3/%d] graph extraction queued  job=%s",
                    config.source, cid, n, job_id,
                )
        except Exception:
            _log.error("%s conv=%s  [3/%d] graph extraction queue FAILED", config.source, cid, n, exc_info=True)

    summary_ev = _get_summary_event(cid)
    summary_ev.clear()  # mark running — next turn will wait on this
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
        summary_ev.set()  # must set even on failure — otherwise next turn blocks forever

    if config.extra_phase:
        try:
            config.extra_phase()
        except Exception:
            _log.error("%s conv=%s  [5/%d] extra phase FAILED", config.source, cid, n, exc_info=True)

    # Reset the idle clock now that the full turn window (including summarising) is
    # closed.  The backoff timer only starts from this point — not from when the LLM
    # finished streaming.  planned_search jobs are submitted to the tool_queue and run
    # outside this window, so they are correctly unaffected.
    try:
        from workers.tool_queue import touch_chat_activity
        touch_chat_activity()
    except Exception:
        pass

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
