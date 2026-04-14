from __future__ import annotations

from pydantic import BaseModel


class ConversationUpdate(BaseModel):
    title: str | None = None
    code_checklist: list | None = None
    # false here forces contextual_enrichment turns to be treated as chitchat (search phase §7 opt-out)
    contextual_grounding_enabled: bool | None = None
