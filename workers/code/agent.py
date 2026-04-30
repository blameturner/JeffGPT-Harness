from __future__ import annotations

import base64
import json
import logging
import re
import time
import uuid
from typing import Iterator, Literal

from infra.config import BASE_SYSTEM_PROMPT
from infra.memory import remember
from infra.rag import retrieve
from shared.temporal import build_temporal_context
from workers.base import BaseAgent, SUMMARY_WAIT_TIMEOUT, _get_summary_event
from workers.chat.history import maybe_summarise
from workers.code.config import code_max_tokens, code_style_prompt, code_temperature

_log = logging.getLogger("code")

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
    "tags. Include full file contents where practical, not snippets."
)

REVIEW_SYSTEM = (
    "You are a senior software engineer in REVIEW mode. "
    "Provide detailed code review feedback on the attached files. "
    "Output sections: Summary, Issues (with severity), Recommendations, "
    "Code snippets where helpful."
)

EXPLAIN_SYSTEM = (
    "You are a senior software engineer in EXPLAIN mode. "
    "Explain how the code works in clear, simple terms. "
    "Output sections: Overview, Key concepts, Flow, Examples."
)

_SYSTEMS: dict[str, str] = {
    "plan": PLAN_SYSTEM,
    "execute": EXECUTE_SYSTEM,
    "review": REVIEW_SYSTEM,
    "explain": EXPLAIN_SYSTEM,
}

Mode = Literal["plan", "execute", "review", "explain"]


def _decode_file_content(b64: str) -> str:
    try:
        return base64.b64decode(b64 or "").decode("utf-8", errors="replace")
    except Exception as e:
        return f"[decode failed: {e}]"


def _render_files_block(files: list[dict] | None) -> str:
    if not files:
        return ""
    parts: list[str] = ["<attached_files>"]
    for f in files:
        name = (f.get("name") or "unnamed").strip()
        content = f.get("content") or _decode_file_content(f.get("content_b64") or "")
        parts.append(f"### {name}\n```\n{content}\n```")
    parts.append("</attached_files>")
    return "\n\n".join(parts)


def _files_to_storage(files: list[dict] | None) -> list[dict]:
    if not files:
        return []
    out: list[dict] = []
    for f in files:
        name = (f.get("name") or "unnamed").strip()
        content = f.get("content")
        if content is None:
            content = _decode_file_content(f.get("content_b64") or "")
        out.append({"name": name, "content": content})
    return out


def _parse_plan_checklist(plan_text: str) -> list[str]:
    if not plan_text.strip():
        return []
    steps: list[str] = []
    for line in plan_text.splitlines():
        m = re.match(r"^\s*(?:[-*]|\d+[.)])\s+(.+)$", line)
        if not m:
            continue
        step = m.group(1).strip()
        if step:
            steps.append(step[:160])
        if len(steps) >= 30:
            break
    return steps


class CodeAgent(BaseAgent):
    def __init__(
        self,
        model: str,
        org_id: int,
        mode: Mode = "plan",
        approved_plan: str | None = None,
        files: list[dict] | None = None,
        search_enabled: bool = False,
    ):
        super().__init__(model=model, org_id=org_id, search_enabled=search_enabled)
        if mode not in _SYSTEMS:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {sorted(_SYSTEMS)}.")
        self.mode: Mode = mode
        self.approved_plan = approved_plan
        self.files = files or []

    def _load_workspace(self, conversation_id: int) -> list[dict]:
        try:
            msgs = self.db.list_code_messages(conversation_id, org_id=self.org_id)
        except Exception:
            _log.error("workspace load failed", exc_info=True)
            return []
        for m in reversed(msgs):
            if m.get("role") != "user":
                continue
            raw = m.get("files_json")
            if not raw:
                continue
            if isinstance(raw, list):
                return raw
            if isinstance(raw, str):
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                if isinstance(data, list):
                    return data
        return []

    def run_job(
        self,
        job,
        user_message: str,
        conversation_id: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        title: str | None = None,
        codebase_collection: str | None = None,
        response_style: str | None = None,
        knowledge_enabled: bool | None = None,
    ) -> None:
        from shared.jobs import STORE

        if temperature is None:
            temperature = code_temperature(response_style)
        if max_tokens is None:
            max_tokens = code_max_tokens(response_style)

        def emit(event: dict):
            STORE.append(job, event)

        system_prompt = _SYSTEMS[self.mode]
        style_key, style_prompt = code_style_prompt(response_style)

        if conversation_id is None:
            convo = self.db.create_code_conversation(
                org_id=self.org_id,
                model=self.model,
                title=(title or user_message)[:80],
                mode=self.mode,
                knowledge_enabled=bool(knowledge_enabled),
            )
            conversation_id = int(convo["Id"])
            history: list[dict] = []
        else:
            convo = self.db.get_code_conversation(conversation_id, org_id=self.org_id)
            if not convo:
                emit({"type": "error", "message": f"Code conversation {conversation_id} not found"})
                return
            summary_ev = _get_summary_event(conversation_id)
            if not summary_ev.is_set():
                emit({"type": "status", "phase": "summarising_previous", "message": "Updating conversation context..."})
                summary_ev.wait(timeout=SUMMARY_WAIT_TIMEOUT)
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in self.db.list_code_messages(conversation_id, org_id=self.org_id)
                if m.get("content")
            ]
            if not self.files:
                self.files = self._load_workspace(conversation_id)

        try:
            self.db.update_code_conversation(conversation_id, {"status": "processing", "rag_collection": self.mode})
        except Exception:
            _log.warning("status update to processing failed  conv=%s", conversation_id)

        emit({"type": "meta", "mode": self.mode, "conversation_id": conversation_id})

        pieces: list[str] = []
        files_block = _render_files_block(self.files)
        if files_block:
            pieces.append(files_block)
        if self.mode == "execute" and self.approved_plan:
            pieces.append(f"<approved_plan>\n{self.approved_plan}\n</approved_plan>")
        pieces.append(user_message)
        composed_message = "\n\n".join(pieces)

        storage_files = _files_to_storage(self.files)
        try:
            self.db.add_code_message(
                conversation_id=conversation_id,
                org_id=self.org_id,
                role="user",
                content=user_message,
                model=self.model,
                mode=self.mode,
                files_json=storage_files or None,
                response_style=style_key,
            )
        except Exception:
            _log.error("user message persist failed", exc_info=True)

        history, summary_event = maybe_summarise(history, truncate_only=True)
        if summary_event:
            emit(summary_event)

        payload: list[dict] = [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "system", "content": build_temporal_context()},
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": style_prompt},
        ]

        if codebase_collection:
            try:
                codebase_context = retrieve(
                    query=user_message,
                    org_id=self.org_id,
                    collection_name=codebase_collection,
                    n_results=10,
                    top_k=5,
                )
                if codebase_context:
                    payload.append(
                        {
                            "role": "system",
                            "content": "Relevant codebase context:\n\n" + codebase_context,
                        }
                    )
            except Exception:
                _log.error("codebase RAG retrieve failed", exc_info=True)

        payload.extend(history)
        payload.append({"role": "user", "content": composed_message})

        start = time.time()
        try:
            chunks, final_usage, final_model = self._call_model(payload, float(temperature), int(max_tokens), emit)
        except Exception:
            _log.error("model call failed  conv=%s", conversation_id, exc_info=True)
            try:
                self.db.update_code_conversation(conversation_id, {"status": "error"})
            except Exception:
                pass
            emit({"type": "error", "message": "model call failed"})
            return

        output = "".join(chunks)
        duration = round(time.time() - start, 2)
        tokens_input = int(final_usage.get("prompt_tokens") or 0)
        tokens_output = int(final_usage.get("completion_tokens") or 0)

        if output:
            try:
                self.db.add_code_message(
                    conversation_id=conversation_id,
                    org_id=self.org_id,
                    role="assistant",
                    content=output,
                    model=str(final_model),
                    tokens_input=tokens_input,
                    tokens_output=tokens_output,
                    mode=self.mode,
                    response_style=style_key,
                )
            except Exception:
                _log.error("assistant message persist failed  conv=%s", conversation_id, exc_info=True)

        try:
            self.db.update_code_conversation(conversation_id, {"status": "complete"})
        except Exception:
            _log.warning("status update to complete failed  conv=%s", conversation_id)

        emit(
            {
                "type": "done",
                "mode": self.mode,
                "conversation_id": conversation_id,
                "model": str(final_model),
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
                "duration_seconds": duration,
                "output": output,
            }
        )

        try:
            remember(
                text=f"USER ({self.mode}): {user_message}\n\nASSISTANT: {output}",
                metadata={"conversation_id": conversation_id, "model": str(final_model), "mode": self.mode, "turn_time": time.time()},
                org_id=self.org_id,
                collection_name=f"code_{conversation_id}",
            )
        except Exception:
            _log.debug("code memory write skipped", exc_info=True)

        if self.mode == "plan" and output:
            try:
                checklist_steps = _parse_plan_checklist(output)
                if checklist_steps:
                    self.db.update_code_conversation(conversation_id, {"code_checklist": checklist_steps})
            except Exception:
                _log.debug("plan checklist persist skipped", exc_info=True)

    def run_streaming(
        self,
        user_message: str,
        conversation_id: int | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        title: str | None = None,
        codebase_collection: str | None = None,
        response_style: str | None = None,
    ) -> Iterator[dict]:
        from shared.jobs import Job

        job = Job(uuid.uuid4().hex)
        self.run_job(
            job,
            user_message=user_message,
            conversation_id=conversation_id,
            temperature=temperature,
            max_tokens=max_tokens,
            title=title,
            codebase_collection=codebase_collection,
            response_style=response_style,
        )
        yield from job.events
