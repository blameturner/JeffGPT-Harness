from __future__ import annotations

import logging

import requests

from workers.web_search import _tool_model

_log = logging.getLogger("chat.history")


MAX_SUMMARY_INPUT_CHARS = 48_000
SUMMARISE_TRIGGER_CHARS = int(128_000 * 0.8)
FALLBACK_RECENT_MESSAGES = 12


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

    fast_url, fast_model = _tool_model()
    summary_text: str | None = None

    if fast_url:
        try:
            resp = requests.post(
                f"{fast_url}/v1/chat/completions",
                json={
                    "model": fast_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You compress chat history. Given a transcript, "
                                "produce a concise factual summary (<= 400 words) "
                                "preserving names, decisions, open questions, and "
                                "any instructions the user gave. No preamble."
                            ),
                        },
                        {"role": "user", "content": transcript},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 800,
                },
                timeout=120,
            )
            resp.raise_for_status()
            summary_text = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            _log.warning("summarisation failed, falling back: %s", e)
            summary_text = None

    if summary_text:
        new_history = [
            {
                "role": "system",
                "content": f"Earlier conversation summary:\n{summary_text}",
            }
        ] + recent
        return new_history, {
            "type": "summarised",
            "removed": len(old),
            "summary_chars": len(summary_text),
        }

    trimmed = history[-FALLBACK_RECENT_MESSAGES:]
    return trimmed, {
        "type": "summarised",
        "removed": max(0, len(history) - len(trimmed)),
        "summary_chars": 0,
        "fallback": True,
    }
