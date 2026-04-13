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
SUMMARISE_THRESHOLD_CHARS = 6_000  # trigger RWKV summarisation above this
SUMMARISE_THRESHOLD_MESSAGES = 8   # or this many messages


def _truncate_message(msg: dict) -> dict:
    """Trim a single message to MAX_SINGLE_MESSAGE_CHARS, keeping head + tail."""
    content = msg.get("content") or ""
    if len(content) <= MAX_SINGLE_MESSAGE_CHARS:
        return msg
    half = MAX_SINGLE_MESSAGE_CHARS // 2
    trimmed = content[:half] + "\n\n[...earlier content trimmed...]\n\n" + content[-half:]
    return {**msg, "content": trimmed}


def _total_chars(history: list[dict]) -> int:
    return sum(len(m.get("content") or "") for m in history)


def maybe_summarise(history: list[dict]) -> tuple[list[dict], dict | None]:
    """Summarise older history via RWKV when it exceeds thresholds.

    Keeps the most recent KEEP_RECENT_EXCHANGES exchanges verbatim.
    Everything older gets compressed into a single [Conversation summary]
    system message via the ``chat_summarise`` model config.

    Falls back to truncation if the model call fails or is unavailable.
    """
    if not history:
        return history, None

    total = _total_chars(history)

    # Small enough — just truncate individual long messages, no summary needed.
    if total <= SUMMARISE_THRESHOLD_CHARS and len(history) <= SUMMARISE_THRESHOLD_MESSAGES:
        return [_truncate_message(m) for m in history], None

    # Split: recent messages to keep verbatim, older messages to summarise.
    keep_count = min(len(history), KEEP_RECENT_EXCHANGES * 2)
    older = history[:-keep_count] if keep_count < len(history) else []
    recent = history[-keep_count:]

    if not older:
        # Not enough older messages to warrant summarisation — just truncate.
        return [_truncate_message(m) for m in recent], None

    # Build text block from older messages for summarisation.
    # Check if there's already a summary message we should preserve/extend.
    existing_summary = ""
    msgs_to_summarise = older
    if older and older[0].get("role") == "system" and "[Conversation summary]" in (older[0].get("content") or ""):
        existing_summary = older[0]["content"]
        msgs_to_summarise = older[1:]

    # Feed full message content to the summariser so it can capture
    # important context.  The RWKV call itself is bounded by max_input_chars.
    older_text = ""
    for m in msgs_to_summarise:
        role = m.get("role", "user")
        content = m.get("content") or ""
        older_text += f"{role}: {content}\n\n"

    if not older_text.strip() and existing_summary:
        # Only the previous summary exists, no new messages to compress.
        result = [{"role": "system", "content": existing_summary}] + [_truncate_message(m) for m in recent]
        return result, None

    # Call RWKV to summarise the older messages.
    summary = _call_rwkv_summarise(older_text.strip(), existing_summary)

    if summary:
        summary_msg = {"role": "system", "content": f"[Conversation summary]\n{summary}"}
        result = [summary_msg] + [_truncate_message(m) for m in recent]
        event = {
            "type": "summarised",
            "removed": len(older),
            "summary_chars": len(summary),
            "fallback": False,
        }
        _log.info("history summarised  older=%d chars=%d summary=%d",
                   len(older), _total_chars(older), len(summary))
        return result, event

    # Fallback: truncation only (RWKV unavailable or failed).
    _log.warning("RWKV summarise failed — falling back to truncation")
    recent = [_truncate_message(m) for m in recent]

    # Drop oldest kept pairs if still over budget.
    while len(recent) > 2:
        if _total_chars(recent) <= MAX_HISTORY_CHARS:
            break
        recent = recent[2:] if len(recent) >= 4 else recent[-2:]

    dropped = len(history) - len(recent)
    event = None
    if dropped > 0:
        event = {
            "type": "summarised",
            "removed": dropped,
            "summary_chars": 0,
            "fallback": True,
        }
    return recent, event


def _call_rwkv_summarise(older_text: str, existing_summary: str) -> str | None:
    """Call the RWKV model to compress conversation history."""
    try:
        from config import get_function_config
        from workers.enrichment.models import model_call

        cfg = get_function_config("chat_summarise")
        max_input = cfg.get("max_input_chars", 16000)

        parts = []
        if existing_summary:
            parts.append(f"Previous summary:\n{existing_summary}\n\n")
        parts.append(f"New messages to incorporate:\n{older_text[:max_input]}")

        prompt = (
            "Compress the following conversation history into a concise factual summary. "
            "Preserve: names, decisions, open questions, key facts, instructions, and "
            "any context the user would need to continue the conversation.\n"
            "Keep under 400 words. Output only the summary.\n\n"
            + "".join(parts)
        )

        summary, _tokens = model_call("chat_summarise", prompt)
        if summary and len(summary) > 20:
            return summary.strip()
        return None
    except Exception:
        _log.error("RWKV chat summarise failed", exc_info=True)
        return None
