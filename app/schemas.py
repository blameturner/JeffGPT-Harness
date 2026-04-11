from __future__ import annotations

from pydantic import BaseModel


class ConversationUpdate(BaseModel):
    title: str | None = None
    code_checklist: list | None = None
    # §7 opt-out: False downgrades contextual_enrichment turns to chitchat
    contextual_grounding_enabled: bool | None = None
