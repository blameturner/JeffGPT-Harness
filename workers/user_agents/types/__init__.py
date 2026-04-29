"""Agent type registry. Each type maps to a handler module with run(ctx)."""
from __future__ import annotations

from . import document, queue, producer, responder, supervisor

REGISTRY = {
    "document":   document.run,
    "queue":      queue.run,
    "producer":   producer.run,
    "responder":  responder.run,
    "supervisor": supervisor.run,
}


def dispatch(type_name: str, ctx):
    handler = REGISTRY.get((type_name or "").lower())
    if not handler:
        raise ValueError(f"unknown agent type: {type_name}")
    return handler(ctx)
