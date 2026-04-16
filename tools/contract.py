from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ToolName(str, Enum):
    WEB_SEARCH = "web_search"
    RAG_LOOKUP = "rag_lookup"
    PLANNED_SEARCH = "planned_search"
    URL_SCRAPER = "url_scraper"


class ToolAction(BaseModel):
    tool: ToolName
    params: dict
    reason: str = ""


class ToolPlan(BaseModel):
    actions: list[ToolAction] = Field(default_factory=list, max_length=4)
    summary: str = ""


class ToolResult(BaseModel):
    tool: ToolName
    action_index: int
    ok: bool
    data: str
    elapsed_s: float = 0.0


class ToolContext(BaseModel):
    plan_summary: str = ""
    results: list[ToolResult] = Field(default_factory=list)

    def to_system_block(self) -> str:
        if not self.results:
            return ""
        parts = [f"[Tool results — {self.plan_summary}]"]
        for r in self.results:
            status = "OK" if r.ok else "FAILED"
            parts.append(f"\n--- {r.tool.value} ({status}, {r.elapsed_s:.1f}s) ---")
            parts.append(r.data)
        parts.append("\n[End tool results]")
        return "\n".join(parts)
