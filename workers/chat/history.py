from __future__ import annotations

import logging

_log = logging.getLogger("chat.history")


# Budget: keep prompt-eval fast on CPU.
# ~4 chars ≈ 1 token.  8K chars ≈ 2K tokens of history — comfortable headroom
# for system prompt + search context without blowing past model context or
# causing multi-minute prompt eval.
MAX_HISTORY_CHARS = 8_000          # hard cap on total history chars sent to model
KEEP_RECENT_EXCHANGES = 3          # minimum full exchanges (user+assistant pairs) to preserve
MAX_SINGLE_MESSAGE_CHARS = 2_000   # truncate any single message longer than this


def _truncate_message(msg: dict) -> dict:
    """Trim a single message to MAX_SINGLE_MESSAGE_CHARS, keeping head + tail."""
    content = msg.get("content") or ""
    if len(content) <= MAX_SINGLE_MESSAGE_CHARS:
        return msg
    half = MAX_SINGLE_MESSAGE_CHARS // 2
    trimmed = content[:half] + "\n\n[...earlier content trimmed...]\n\n" + content[-half:]
    return {**msg, "content": trimmed}


def maybe_summarise(history: list[dict]) -> tuple[list[dict], dict | None]:
    if not history:
        return history, None

    total_chars = sum(len(m.get("content") or "") for m in history)

    # If history is already small enough, just truncate individual long messages.
    if total_chars <= MAX_HISTORY_CHARS and len(history) <= KEEP_RECENT_EXCHANGES * 2 + 2:
        return [_truncate_message(m) for m in history], None

    _log.info(
        "history trimming  messages=%d chars=%d cap=%d",
        len(history), total_chars, MAX_HISTORY_CHARS,
    )

    # Keep the last N exchanges (user+assistant pairs).
    # Walk backwards to find pair boundaries.
    keep_count = min(len(history), KEEP_RECENT_EXCHANGES * 2)
    recent = history[-keep_count:]

    # Truncate each kept message individually.
    recent = [_truncate_message(m) for m in recent]

    # If still over budget after keeping minimum exchanges, drop oldest kept pairs.
    while len(recent) > 2:
        chars = sum(len(m.get("content") or "") for m in recent)
        if chars <= MAX_HISTORY_CHARS:
            break
        # Drop oldest pair (user + assistant)
        recent = recent[2:] if len(recent) >= 4 else recent[-2:]

    dropped = len(history) - len(recent)
    if dropped > 0:
        _log.info("history trimmed  kept=%d dropped=%d", len(recent), dropped)

    event = None
    if dropped > 0:
        event = {
            "type": "summarised",
            "removed": dropped,
            "summary_chars": 0,
            "fallback": True,
        }

    return recent, event
