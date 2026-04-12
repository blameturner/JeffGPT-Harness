from __future__ import annotations

import logging

_log = logging.getLogger("chat.history")


# ~4 chars per token. 4000 tokens of history = ~16K chars = ~136s prompt eval on CPU.
# Trigger summarisation before history gets expensive.
MAX_SUMMARY_INPUT_CHARS = 16_000
SUMMARISE_TRIGGER_CHARS = 16_000
FALLBACK_RECENT_MESSAGES = 8


def maybe_summarise(history: list[dict]) -> tuple[list[dict], dict | None]:
    total_chars = sum(len(m.get("content") or "") for m in history)
    if total_chars <= SUMMARISE_TRIGGER_CHARS:
        return history, None

    _log.info("summarisation triggered  messages=%d chars=%d threshold=%d", len(history), total_chars, SUMMARISE_TRIGGER_CHARS)

    keep_tail = 4
    if len(history) <= keep_tail:
        return history, None

    old = history[:-keep_tail]
    recent = history[-keep_tail:]

    buf: list[str] = []
    used = 0
    for m in old:
        line = f"{m.get('role', 'user').upper()}: {m.get('content') or ''}\n"
        if used + len(line) > MAX_SUMMARY_INPUT_CHARS:
            break
        buf.append(line)
        used += len(line)
    transcript = "".join(buf)

    # Never call a model during the interactive chat path — just truncate.
    # Model-based summarisation is too slow on CPU and blocks the chat flow.
    _log.info("summarisation: truncating to last %d messages (dropped %d)", FALLBACK_RECENT_MESSAGES, max(0, len(history) - FALLBACK_RECENT_MESSAGES))
    trimmed = history[-FALLBACK_RECENT_MESSAGES:]
    return trimmed, {
        "type": "summarised",
        "removed": max(0, len(history) - len(trimmed)),
        "summary_chars": 0,
        "fallback": True,
    }
