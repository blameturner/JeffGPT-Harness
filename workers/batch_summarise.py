"""
Batch chat summarisation — scheduled job that compresses old conversation
history using the tool model. Runs during idle time (4am AEST).

Processes both chat conversations and code conversations that have enough
history to benefit from compression but haven't been summarised yet.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from config import get_function_config
from nocodb_client import NocodbClient
from workers.enrichment.db import EnrichmentDB
from workers.enrichment.models import model_call

_log = logging.getLogger("batch.summarise")

SUMMARY_MIN_MESSAGES = 10
SUMMARY_MIN_CHARS = 8000
MAX_CONVERSATIONS_PER_RUN = 20


def _summarise_conversation(
    db: NocodbClient,
    conv_id: int,
    org_id: int,
    messages: list[dict],
    table_type: str,
) -> bool:
    """Summarise a single conversation. Returns True on success."""
    cfg = get_function_config("batch_summarise")
    max_transcript = cfg.get("max_input_chars", 24000)

    keep_tail = 4
    old = messages[:-keep_tail] if len(messages) > keep_tail else []
    if not old:
        return False

    transcript_parts = []
    chars = 0
    for m in old:
        line = f"{m.get('role', 'user').upper()}: {m.get('content') or ''}\n"
        if chars + len(line) > max_transcript:
            break
        transcript_parts.append(line)
        chars += len(line)
    transcript = "".join(transcript_parts)

    prompt = (
        "Compress this chat history into a concise factual summary "
        "(<= 400 words). Preserve names, decisions, open questions, "
        "and instructions. No preamble.\n\n"
        f"{transcript}"
    )

    try:
        summary, _tokens = model_call("batch_summarise", prompt)
    except Exception:
        _log.error("batch summarise model call failed  %s conv=%s", table_type, conv_id, exc_info=True)
        return False

    if not summary:
        return False

    try:
        if table_type == "chat":
            db.add_message(
                conversation_id=conv_id,
                org_id=org_id,
                role="system",
                content=f"[Conversation summary]\n{summary}",
                model="batch_summarise",
            )
            db.update_conversation(conv_id, {"summary_applied": True})
        else:
            db.add_code_message(
                conversation_id=conv_id,
                org_id=org_id,
                role="system",
                content=f"[Conversation summary]\n{summary}",
                model="batch_summarise",
            )
            db.update_code_conversation(conv_id, {"summary_applied": True})
        _log.info("batch summarise ok  %s conv=%s summary_chars=%d", table_type, conv_id, len(summary))
        return True
    except Exception:
        _log.error("batch summarise persist failed  %s conv=%s", table_type, conv_id, exc_info=True)
        return False


def _process_table(
    db: NocodbClient,
    table: str,
    list_fn,
    messages_fn,
    table_type: str,
    stats: dict,
):
    """Process one conversation table (chat or code)."""
    try:
        conversations = list_fn()
    except Exception:
        _log.error("batch summarise: failed to list %s", table_type, exc_info=True)
        return

    for convo in conversations[:MAX_CONVERSATIONS_PER_RUN]:
        conv_id = convo.get("Id")
        if not conv_id:
            continue

        if convo.get("summary_applied"):
            stats["skipped"] += 1
            continue

        try:
            messages = messages_fn(conv_id)
        except Exception:
            stats["failed"] += 1
            continue

        if len(messages) < SUMMARY_MIN_MESSAGES:
            stats["skipped"] += 1
            continue

        total_chars = sum(len(m.get("content") or "") for m in messages)
        if total_chars < SUMMARY_MIN_CHARS:
            stats["skipped"] += 1
            continue

        org_id = int(convo.get("org_id") or 1)
        ok = _summarise_conversation(db, conv_id, org_id, messages, table_type)
        if ok:
            stats["summarised"] += 1
        else:
            stats["failed"] += 1
        stats["processed"] += 1


def run_batch_summarise() -> dict:
    """Summarise old conversations that haven't been compressed yet."""
    _log.info("batch summarise starting")
    started = time.time()

    from workers.jobs import STORE
    active_jobs = sum(1 for j in STORE._jobs.values() if not j.done)
    if active_jobs > 0:
        _log.info("batch summarise deferred — %d active chat jobs", active_jobs)
        return {"status": "deferred", "reason": f"{active_jobs} active jobs"}

    db = NocodbClient()
    stats = {"processed": 0, "summarised": 0, "skipped": 0, "failed": 0}

    # Chat conversations
    _process_table(
        db,
        table="conversations",
        list_fn=lambda: db._get("conversations", params={"sort": "-CreatedAt", "limit": 200}).get("list", []),
        messages_fn=lambda cid: db.list_messages(cid),
        table_type="chat",
        stats=stats,
    )

    # Code conversations
    _process_table(
        db,
        table="code_conversations",
        list_fn=lambda: db._get("code_conversations", params={"sort": "-CreatedAt", "limit": 200}).get("list", []),
        messages_fn=lambda cid: db.list_code_messages(cid),
        table_type="code",
        stats=stats,
    )

    elapsed = round(time.time() - started, 1)
    _log.info("batch summarise done  %s elapsed=%.1fs", stats, elapsed)

    try:
        edb = EnrichmentDB()
        cycle_id = f"batch_summarise_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        edb.log_event(
            cycle_id, "batch_summarise",
            message=f"processed={stats['processed']} summarised={stats['summarised']} skipped={stats['skipped']} failed={stats['failed']} elapsed={elapsed}s",
            duration_seconds=elapsed,
        )
    except Exception:
        pass

    return stats
