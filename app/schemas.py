from __future__ import annotations

from pydantic import BaseModel


class ConversationUpdate(BaseModel):
    title: str | None = None
    code_checklist: list | None = None
    # false here forces contextual_enrichment turns to be treated as chitchat (search phase §7 opt-out)
    contextual_grounding_enabled: bool | None = None
    # Properties-tab fields (per-conversation defaults)
    system_note: str | None = None
    default_response_style: str | None = None
    polish_pass_default: bool | None = None
    strict_grounding_default: bool | None = None
    ask_back_default: bool | None = None
    memory_extract_every_n_turns: int | None = None
    memory_token_budget: int | None = None
    saved_fragments_json: list | None = None


class ChatMemoryItemCreate(BaseModel):
    category: str  # fact | decision | thread
    text: str
    pinned: bool = False
    status: str = "active"
    confidence: int = 0
    source_message_id: int | None = None


class ChatMemoryItemUpdate(BaseModel):
    text: str | None = None
    category: str | None = None
    pinned: bool | None = None
    status: str | None = None  # active | proposed | rejected
    confidence: int | None = None
