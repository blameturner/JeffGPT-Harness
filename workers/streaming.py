"""Standalone HTTP streaming logic for OpenAI-compatible chat completions."""

from __future__ import annotations

import json
import logging
from typing import Callable

import requests

_log = logging.getLogger(__name__)


def stream_model_response(
    url: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    emit: Callable[[dict], None],
) -> tuple[list[str], dict, str]:
    """Stream a chat completion and return collected chunks, usage, and resolved model name.

    Args:
        url: Base URL of the OpenAI-compatible API (e.g. ``http://localhost:8080``).
            ``/v1/chat/completions`` is appended automatically.
        model: Model identifier to send in the request payload.
        messages: Conversation messages in OpenAI chat format.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        emit: Callback invoked for every streaming event.  Called with dicts of
            the form ``{"type": "chunk", "text": ...}`` for content tokens and
            ``{"type": "thinking", "text": ...}`` for reasoning/thinking tokens.

    Returns:
        A tuple of ``(chunks, usage, final_model)`` where *chunks* is a list of
        content strings received, *usage* is the usage dict from the final SSE
        event (or empty), and *final_model* is the model name reported by the
        server.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    chunks: list[str] = []
    final_usage: dict = {}
    final_model: str = model
    in_think_block = False
    think_tokens = 0
    first_content_emitted = False
    reasoning_chunks: list[str] = []

    with requests.post(
        f"{url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=(30, 3600),
    ) as response:
        response.raise_for_status()
        response.encoding = "utf-8"
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if not raw_line.startswith("data:"):
                continue
            data = raw_line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            if event.get("model"):
                final_model = event["model"]
            usage = event.get("usage")
            if usage:
                final_usage = usage

            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}

            # Some models put thinking in reasoning_content, actual answer in content.
            # Others put everything in content with <think> tags.
            reasoning = delta.get("reasoning_content")
            text = delta.get("content")

            if reasoning and not text:
                # Model is still thinking -- stream to UI as thinking event
                think_tokens += 1
                reasoning_chunks.append(reasoning)
                emit({"type": "thinking", "text": reasoning})
                continue

            if not text:
                continue

            # Filter <think>...</think> tags in content (Qwen/RWKV style)
            if not first_content_emitted and "<think>" in text:
                in_think_block = True
                think_tokens += 1
                # Strip the <think> tag, emit remaining text as thinking
                after_tag = text.split("<think>", 1)[1] if "<think>" in text else ""
                if after_tag:
                    reasoning_chunks.append(after_tag)
                    emit({"type": "thinking", "text": after_tag})
                continue
            if in_think_block:
                think_tokens += 1
                if "</think>" in text:
                    in_think_block = False
                    # Emit any text before the closing tag as thinking
                    before_tag = text.split("</think>", 1)[0]
                    if before_tag:
                        reasoning_chunks.append(before_tag)
                        emit({"type": "thinking", "text": before_tag})
                    # Emit any text after the closing tag as content
                    after_tag = text.split("</think>", 1)[1].strip()
                    if after_tag:
                        first_content_emitted = True
                        chunks.append(after_tag)
                        emit({"type": "chunk", "text": after_tag})
                    if think_tokens > 1:
                        _log.info("model thinking done  tokens=%d", think_tokens)
                else:
                    reasoning_chunks.append(text)
                    emit({"type": "thinking", "text": text})
                continue

            first_content_emitted = True
            chunks.append(text)
            emit({"type": "chunk", "text": text})

    # Fallback: if model put everything in thinking (reasoning_content field
    # OR <think> tags) and never produced actual content, use the reasoning
    # as the output.  Better than losing the response entirely.
    if not chunks and reasoning_chunks:
        _log.warning(
            "model returned no content, using thinking as fallback  think_tokens=%d reasoning_chars=%d",
            think_tokens,
            sum(len(r) for r in reasoning_chunks),
        )
        full_reasoning = "".join(reasoning_chunks)
        chunks.append(full_reasoning)
        emit({"type": "chunk", "text": full_reasoning})

    if think_tokens > 0:
        _log.info(
            "model thinking summary  think_tokens=%d content_tokens=%d",
            think_tokens,
            len(chunks),
        )

    return chunks, final_usage, final_model
