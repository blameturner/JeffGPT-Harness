from __future__ import annotations

import base64
import time
from typing import Iterator, Literal

from workers.chat_agent import ChatAgent


PLAN_SYSTEM = (
    "You are a senior software engineer in PLAN mode. "
    "Analyse the user's request and any attached files, then produce a "
    "structured implementation plan. DO NOT write code. Output sections: "
    "Context, Approach, Files to change (with paths), Step-by-step plan, "
    "Risks, Verification. Be specific and actionable."
)

EXECUTE_SYSTEM = (
    "You are a senior software engineer in EXECUTE mode. "
    "Write complete, working code that implements the approved plan (if one "
    "is provided) or the user's request. Use fenced code blocks with language "
    "tags. Include full file contents where practical, not snippets. After "
    "the code, add a short 'Notes' section only if there is something the "
    "user must do manually."
)

DEBUG_SYSTEM = (
    "You are a senior software engineer in DEBUG mode. "
    "Diagnose the root cause of the reported issue using the attached files "
    "and the user's description. Output sections: Symptom, Root cause, Fix "
    "(with code), Verification. Do not speculate — if evidence is missing, "
    "ask for it."
)

_SYSTEMS: dict[str, str] = {
    "plan": PLAN_SYSTEM,
    "execute": EXECUTE_SYSTEM,
    "debug": DEBUG_SYSTEM,
}

Mode = Literal["plan", "execute", "debug"]


def _decode_files(files: list[dict] | None) -> str:
    if not files:
        return ""
    parts: list[str] = ["<attached_files>"]
    for f in files:
        name = (f.get("name") or "unnamed").strip()
        b64 = f.get("content_b64") or ""
        try:
            content = base64.b64decode(b64).decode("utf-8", errors="replace")
        except Exception as e:
            content = f"[decode failed: {e}]"
        parts.append(f"### {name}\n```\n{content}\n```")
    parts.append("</attached_files>")
    return "\n\n".join(parts)


class CodeAgent(ChatAgent):

    def __init__(
        self,
        model: str,
        org_id: int,
        mode: Mode = "plan",
        approved_plan: str | None = None,
        files: list[dict] | None = None,
    ):
        super().__init__(model=model, org_id=org_id, search_enabled=False)
        if mode not in _SYSTEMS:
            raise ValueError(f"Invalid mode '{mode}'. Must be plan|execute|debug.")
        self.mode: Mode = mode
        self.approved_plan = approved_plan
        self.files = files or []

    def run_streaming(
        self,
        user_message: str,
        conversation_id: int | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> Iterator[dict]:
        system_prompt = _SYSTEMS[self.mode]
        file_context = _decode_files(self.files)
        pieces: list[str] = []
        if file_context:
            pieces.append(file_context)
        if self.mode == "execute" and self.approved_plan:
            pieces.append(f"<approved_plan>\n{self.approved_plan}\n</approved_plan>")
        pieces.append(user_message)
        composed_message = "\n\n".join(pieces)

        yield {"type": "meta", "mode": self.mode, "conversation_id": conversation_id}

        payload: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": composed_message},
        ]

        start = time.time()
        accumulated: list[str] = []
        final_usage: dict = {}
        final_model = self.model
        errored = False

        for event in self._call_model_streaming(payload, temperature, max_tokens):
            etype = event.get("type")
            if etype == "chunk":
                accumulated.append(event["text"])
                yield event
            elif etype == "done":
                final_usage = event.get("usage") or {}
                final_model = event.get("model") or final_model
                break
            elif etype == "error":
                errored = True
                yield event
                return

        if errored:
            return

        duration = round(time.time() - start, 2)
        yield {
            "type": "done",
            "mode": self.mode,
            "model": str(final_model),
            "tokens_input": int(final_usage.get("prompt_tokens") or 0),
            "tokens_output": int(final_usage.get("completion_tokens") or 0),
            "duration_seconds": duration,
            "output": "".join(accumulated),
        }
