from __future__ import annotations

import logging

_log = logging.getLogger("chat.history")


# 8K chars ≈ 2K tokens — keeps CPU prompt-eval under a minute with headroom for system+search
MAX_HISTORY_CHARS = 8_000
KEEP_RECENT_EXCHANGES = 3
MAX_SINGLE_MESSAGE_CHARS = 2_000
SUMMARISE_THRESHOLD_CHARS = 5_000
SUMMARISE_THRESHOLD_MESSAGES = 6


def _truncate_message(msg: dict) -> dict:
    content = msg.get("content") or ""
    if len(content) <= MAX_SINGLE_MESSAGE_CHARS:
        return msg
    half = MAX_SINGLE_MESSAGE_CHARS // 2
    trimmed = content[:half] + "\n\n[...earlier content trimmed...]\n\n" + content[-half:]
    return {**msg, "content": trimmed}


def _total_chars(history: list[dict]) -> int:
    return sum(len(m.get("content") or "") for m in history)


def extract_conversation_topics(history: list[dict]) -> list[str]:
    import re
    from tools.search.queries import _extract_keywords

    summary_topics: list[str] = []
    for m in history:
        if m.get("role") == "system" and "[Conversation summary]" in (m.get("content") or ""):
            content = m["content"]
            for line in content.split("\n"):
                if line.strip().upper().startswith("TOPICS:"):
                    topic_str = line.split(":", 1)[1].strip()
                    summary_topics = [t.strip().lower() for t in re.split(r"[,;]", topic_str) if t.strip()]
                    break
            break

    scores: dict[str, float] = {}

    # Summary topics are useful long-tail memory, but lower weight than recent turns.
    for t in summary_topics:
        scores[t] = scores.get(t, 0.0) + 0.6

    # Recency-weighted terms from the last N non-system messages.
    recent_msgs = [m for m in history if m.get("role") in ("user", "assistant")][-12:]
    n = len(recent_msgs)
    for i, m in enumerate(recent_msgs):
        content = str(m.get("content") or "")[:1200]
        kws = _extract_keywords(content)
        if not kws:
            continue
        # newer messages score higher (linear recency weighting)
        weight = 0.6 + ((i + 1) / max(1, n))
        for kw in kws:
            k = kw.strip().lower()
            if len(k) < 3 or k.isdigit():
                continue
            scores[k] = scores.get(k, 0.0) + weight

    if not scores:
        _log.debug("extract_topics: no summary/topics extracted from %d messages", len(history))
        return []

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    topics = [k for k, _ in ranked[:10]]
    _log.debug("extract_topics: summary=%d recent_msgs=%d resolved=%s", len(summary_topics), n, topics)
    return topics


def maybe_summarise(history: list[dict], truncate_only: bool = False) -> tuple[list[dict], dict | None]:
    # truncate_only=True keeps this off the hot path — caller runs the model call in bg
    if not history:
        return history, None

    total = _total_chars(history)
    threshold_chars, threshold_messages, keep_recent_exchanges = _summary_thresholds()

    if total <= threshold_chars and len(history) <= threshold_messages:
        return [_truncate_message(m) for m in history], None

    keep_count = min(len(history), keep_recent_exchanges * 2)
    older = history[:-keep_count] if keep_count < len(history) else []
    recent = history[-keep_count:]

    if not older:
        return [_truncate_message(m) for m in recent], None

    existing_summary = ""
    msgs_to_summarise = older
    if older and older[0].get("role") == "system" and "[Conversation summary]" in (older[0].get("content") or ""):
        existing_summary = older[0]["content"]
        msgs_to_summarise = older[1:]

    older_text = ""
    for m in msgs_to_summarise:
        role = m.get("role", "user")
        content = m.get("content") or ""
        older_text += f"{role}: {content}\n\n"

    if not older_text.strip() and existing_summary:
        result = [{"role": "system", "content": existing_summary}] + [_truncate_message(m) for m in recent]
        return result, None

    if truncate_only:
        # preserve existing summary from a prior bg run — truncation must not drop it
        if existing_summary:
            result = [{"role": "system", "content": existing_summary}] + [_truncate_message(m) for m in recent]
            return result, None
        summary, topics = None, []
    else:
        summary, topics = _call_summarise(older_text.strip(), existing_summary)

    if summary:
        topics_line = f"\nTOPICS: {', '.join(topics)}" if topics else ""
        summary_msg = {"role": "system", "content": f"[Conversation summary]\n{summary}{topics_line}"}
        result = [summary_msg] + [_truncate_message(m) for m in recent]
        event = {
            "type": "summarised",
            "removed": len(older),
            "summary_chars": len(summary),
            "topics": topics,
            "fallback": False,
        }
        _log.info("summariser: compressed %d older messages (%d chars) into %d char summary  topics=%s",
                   len(older), _total_chars(older), len(summary), topics)
        return result, event

    if not truncate_only:
        _log.warning("summariser: model call failed — falling back to truncation")
    recent = [_truncate_message(m) for m in recent]

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


def _parse_summary_and_topics(raw: str) -> tuple[str, list[str]]:
    import re
    lines = raw.strip().split("\n")
    topics: list[str] = []
    summary_lines: list[str] = []
    for line in lines:
        if line.strip().upper().startswith("TOPICS:"):
            topic_str = line.split(":", 1)[1].strip()
            topics = [t.strip().lower() for t in re.split(r"[,;]", topic_str) if t.strip()]
        else:
            summary_lines.append(line)
    summary = "\n".join(summary_lines).strip()
    return summary, topics[:10]


def _call_summarise(older_text: str, existing_summary: str) -> tuple[str, list[str]] | tuple[None, list[str]]:
    try:
        from infra.config import get_function_config
        from shared.models import model_call

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
            "Keep under 400 words.\n\n"
            "After the summary, on a new line output:\n"
            "TOPICS: keyword1, keyword2, keyword3, ...\n"
            "List the 3-8 most important technical terms, product names, "
            "languages, or domain topics discussed (e.g. \"javascript, arrays, "
            "map function, data transformation\"). These will be used to "
            "improve web search queries.\n\n"
            + "".join(parts)
        )

        raw, _tokens = model_call("chat_summarise", prompt)
        if raw and len(raw) > 20:
            return _parse_summary_and_topics(raw.strip())
        return None, []
    except Exception:
        _log.error("chat summarise failed", exc_info=True)
        return None, []


def _summary_thresholds() -> tuple[int, int, int]:
    """Runtime-configurable summary thresholds.

    Lower defaults make post-turn summaries happen more often, improving topic
    extraction used by search query generation on subsequent turns.
    """
    try:
        from infra.config import get_feature

        cfg = get_feature("chat", "chat_summarise", {}) or {}
        if not isinstance(cfg, dict):
            cfg = {}

        chars = int(cfg.get("threshold_chars", SUMMARISE_THRESHOLD_CHARS) or SUMMARISE_THRESHOLD_CHARS)
        messages = int(cfg.get("threshold_messages", SUMMARISE_THRESHOLD_MESSAGES) or SUMMARISE_THRESHOLD_MESSAGES)
        keep_recent = int(cfg.get("keep_recent_exchanges", KEEP_RECENT_EXCHANGES) or KEEP_RECENT_EXCHANGES)
    except Exception:
        chars = SUMMARISE_THRESHOLD_CHARS
        messages = SUMMARISE_THRESHOLD_MESSAGES
        keep_recent = KEEP_RECENT_EXCHANGES

    return max(1000, chars), max(4, messages), max(1, keep_recent)
