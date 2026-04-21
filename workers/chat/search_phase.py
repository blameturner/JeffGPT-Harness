from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchPhaseResult:
    search_context: str = ""
    search_sources: list[dict] = field(default_factory=list)
    search_confidence: str = "none"
    search_status: str = "not_used"
    search_note: str = ""
    search_errored: bool = False
    intent_dict: dict | None = None
