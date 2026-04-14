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

            # two model conventions: reasoning_content field vs <think> tags inline in content
            reasoning = delta.get("reasoning_content")
            text = delta.get("content")

            if reasoning and not text:
                think_tokens += 1
                reasoning_chunks.append(reasoning)
                emit({"type": "thinking", "text": reasoning})
                continue

            if not text:
                continue

            # qwen/rwkv style: strip <think>...</think> from content and re-emit as thinking
            if not first_content_emitted and "<think>" in text:
                in_think_block = True
                think_tokens += 1
                after_tag = text.split("<think>", 1)[1] if "<think>" in text else ""
                if after_tag:
                    reasoning_chunks.append(after_tag)
                    emit({"type": "thinking", "text": after_tag})
                continue
            if in_think_block:
                think_tokens += 1
                if "</think>" in text:
                    in_think_block = False
                    before_tag = text.split("</think>", 1)[0]
                    if before_tag:
                        reasoning_chunks.append(before_tag)
                        emit({"type": "thinking", "text": before_tag})
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

    # fallback: if model emitted only thinking and no content, promote thinking to output
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
