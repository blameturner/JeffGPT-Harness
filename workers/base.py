from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator

from infra.config import MODELS, get_model_url, refresh_models
from infra.nocodb_client import NocodbClient

_log = logging.getLogger("agent.base")

SUMMARY_WAIT_TIMEOUT = 15

# per-conv summary event — blocks new turn while previous turn's bg summary still running
_summary_events: dict[int, threading.Event] = {}
_summary_lock = threading.Lock()


def _get_summary_event(conversation_id: int) -> threading.Event:
    with _summary_lock:
        if conversation_id not in _summary_events:
            ev = threading.Event()
            ev.set()  # initial state: no summary in progress
            _summary_events[conversation_id] = ev
        return _summary_events[conversation_id]


@dataclass
class ChatResult:
    output: str
    model: str
    conversation_id: int
    tokens_input: int = 0
    tokens_output: int = 0
    duration_seconds: float = 0.0
    rag_enabled: bool = False
    context_chars: int = 0


class BaseAgent:
    def __init__(self, model: str, org_id: int, search_enabled: bool = False):
        url = get_model_url(model)
        if not url:
            refresh_models()
            url = get_model_url(model)
        if not url:
            options = sorted({
                v["role"] for v in MODELS.values() if isinstance(v, dict)
            })
            raise ValueError(
                f"Model '{model}' not available. Options: {options}"
            )
        self.model = model
        self.org_id = org_id
        self.url = url
        self.search_enabled = search_enabled
        self._search_mode = "standard"
        self.db = NocodbClient()

    @staticmethod
    def _default_collection(conversation_id: int) -> str:
        return f"chat_{conversation_id}"

    @staticmethod
    def _truthy(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return False

    @staticmethod
    def _tool_model_url() -> tuple[str | None, str | None]:
        from shared.model_pool import _tool_model
        return _tool_model()

    def _call_model(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        emit: Callable[[dict], None],
    ) -> tuple[list[str], dict, str]:
        from workers.streaming import stream_model_response
        return stream_model_response(
            url=self.url,
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            emit=emit,
        )

    def send(
        self,
        user_message: str,
        conversation_id: int | None = None,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        rag_enabled: bool | None = None,
        rag_collection: str | None = None,
        knowledge_enabled: bool | None = None,
    ) -> ChatResult:
        parts: list[str] = []
        final: dict = {}
        conv_id = conversation_id or 0
        for event in self.send_streaming(
            user_message=user_message,
            conversation_id=conversation_id,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            rag_enabled=rag_enabled,
            rag_collection=rag_collection,
            knowledge_enabled=knowledge_enabled,
        ):
            etype = event.get("type")
            if etype == "meta":
                conv_id = event.get("conversation_id") or conv_id
            elif etype == "chunk":
                parts.append(event.get("text", ""))
            elif etype == "done":
                final = event
            elif etype == "error":
                raise RuntimeError(event.get("message") or "chat stream error")

        return ChatResult(
            output="".join(parts),
            model=final.get("model", self.model),
            conversation_id=final.get("conversation_id", conv_id),
            tokens_input=int(final.get("tokens_input") or 0),
            tokens_output=int(final.get("tokens_output") or 0),
            duration_seconds=float(final.get("duration_seconds") or 0.0),
            rag_enabled=bool(final.get("rag_enabled")),
            context_chars=int(final.get("context_chars") or 0),
        )

    def send_streaming(self, **kwargs) -> Iterator[dict]:
        raise NotImplementedError("subclass must implement send_streaming")
