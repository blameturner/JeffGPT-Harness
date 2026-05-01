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
from infra.project_context import build_project_context_pack, coerce_retrieval_scope
from infra.prompts import assemble_code_system_prompt
from infra.rag import retrieve
from shared.temporal import build_temporal_context
from workers.base import BaseAgent, SUMMARY_WAIT_TIMEOUT, _get_summary_event
from workers.chat.history import maybe_summarise
from workers.code.config import CODE_MODES, code_max_tokens, code_style_prompt, code_temperature, resolve_code_mode
from workers.code.fs_parser import apply_file_fences
from workers.code.fs_tools import apply_tool_directives, parse_tool_directives

INTERACTIVE_FS_MAX_HOPS = 3

_log = logging.getLogger("code")

Mode = Literal["chat", "plan", "execute", "apply", "review", "explain", "decide", "scaffold", "refine"]

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
        project_id: int | None = None,
        interactive_fs: bool = False,
    ):
        super().__init__(model=model, org_id=org_id, search_enabled=search_enabled)
        resolved = resolve_code_mode(mode)
        if resolved not in CODE_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {sorted(CODE_MODES)}.")
        self.mode: Mode = resolved  # type: ignore[assignment]
        self.approved_plan = approved_plan
        self.files = files or []
        self.project_id = project_id
        self.interactive_fs = interactive_fs

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
        from shared.model_pool import _user_priority_ctx
        from shared.models import set_model_usage_context
        from workers.tool_queue import begin_chat_turn

        # Code turns are interactive work; mark priority + active-turn gate so
        # background queue/model callers cannot compete until this turn ends.
        _priority_token = _user_priority_ctx.set(True)
        begin_chat_turn()
        _turn_finalised = {"done": False}

        def _finalise_turn() -> None:
            if _turn_finalised["done"]:
                return
            _turn_finalised["done"] = True
            try:
                from workers.tool_queue import end_chat_turn
                end_chat_turn()
            except Exception:
                _log.debug("code end_chat_turn failed", exc_info=True)

        set_model_usage_context(org_id=self.org_id, source="code", conversation_id=conversation_id)

        if temperature is None:
            temperature = code_temperature(response_style)
        if max_tokens is None:
            max_tokens = code_max_tokens(response_style)

        stream_buffer = ""
        seen_fence_keys: set[str] = set()
        streamed_changes: list[dict] = []

        def emit_raw(event: dict):
            STORE.append(job, event)

        def emit(event: dict):
            nonlocal stream_buffer
            emit_raw(event)
            if not self.project_id or self.interactive_fs:
                return
            if event.get("type") != "chunk":
                return
            text = event.get("text") or ""
            if not text:
                return
            stream_buffer += text
            try:
                changes = apply_file_fences(
                    db=self.db,
                    project_id=self.project_id,
                    response_text=stream_buffer,
                    conversation_id=conversation_id,
                    assistant_message_id=None,
                    seen_keys=seen_fence_keys,
                )
            except Exception:
                # Chunk boundaries can bisect logical outputs; full-turn fallback still runs.
                return
            for ch in changes:
                streamed_changes.append(ch)
                if ch.get("permission_required"):
                    try:
                        self.db.add_project_audit_event(
                            project_id=int(self.project_id),
                            actor=f"agent:{conversation_id}" if conversation_id else "agent",
                            kind="permission_request",
                            payload={"path": ch.get("path"), "reason": ch.get("reason") or "locked file"},
                        )
                    except Exception:
                        _log.debug("permission_request audit write skipped", exc_info=True)
                    emit_raw(
                        {
                            "type": "permission_request",
                            "project_id": self.project_id,
                            "conversation_id": conversation_id,
                            "path": ch.get("path"),
                            "reason": ch.get("reason") or "locked file",
                        }
                    )
                    continue
                emit_raw(
                    {
                        "type": "file_changed",
                        "project_id": self.project_id,
                        "conversation_id": conversation_id,
                        **ch,
                    }
                )

        style_key, style_prompt = code_style_prompt(response_style)
        project = None
        pinned_context = ""
        path_manifest = ""
        context_notice = ""

        if self.project_id and not self.db.get_project(self.project_id, org_id=self.org_id):
            emit({"type": "error", "message": f"Project {self.project_id} not found"})
            _finalise_turn()
            _user_priority_ctx.reset(_priority_token)
            return

        if conversation_id is None:
            convo = self.db.create_code_conversation(
                org_id=self.org_id,
                model=self.model,
                title=(title or user_message)[:80],
                mode=self.mode,
                knowledge_enabled=bool(knowledge_enabled),
                project_id=self.project_id,
                interactive_fs=self.interactive_fs,
            )
            conversation_id = int(convo["Id"])
            history: list[dict] = []
        else:
            convo = self.db.get_code_conversation(conversation_id, org_id=self.org_id)
            if not convo:
                emit({"type": "error", "message": f"Code conversation {conversation_id} not found"})
                _finalise_turn()
                _user_priority_ctx.reset(_priority_token)
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
            convo_project_id = convo.get("project_id")
            if convo_project_id and self.project_id and int(convo_project_id) != int(self.project_id):
                emit(
                    {
                        "type": "error",
                        "message": f"Conversation {conversation_id} is scoped to project {convo_project_id}",
                    }
                )
                _finalise_turn()
                _user_priority_ctx.reset(_priority_token)
                return
            if convo_project_id and not self.project_id:
                self.project_id = int(convo_project_id)
            if not self.files:
                self.files = self._load_workspace(conversation_id)

        glossary_terms: list[str] = []
        if self.project_id:
            project = self.db.get_project(self.project_id, org_id=self.org_id)
            if project:
                pack = build_project_context_pack(self.db, self.project_id)
                pinned_context = str(pack.get("pinned_context") or "")
                path_manifest = str(pack.get("path_manifest") or "")
                context_notice = str(pack.get("context_notice") or "")
                glossary_terms = list(pack.get("glossary_terms") or [])

        system_prompt = assemble_code_system_prompt(
            mode=self.mode,
            style_prompt=style_prompt,
            project_name=(project or {}).get("name") or "",
            project_slug=(project or {}).get("slug") or "",
            system_note=(project or {}).get("system_note") or "",
            pinned_context=pinned_context,
            path_manifest=path_manifest,
            context_notice=context_notice,
            interactive_fs=self.interactive_fs,
            glossary_terms=glossary_terms,
        )

        try:
            self.db.update_code_conversation(conversation_id, {"status": "processing", "rag_collection": self.mode})
        except Exception:
            _log.warning("status update to processing failed  conv=%s", conversation_id)

        emit({"type": "meta", "mode": self.mode, "conversation_id": conversation_id})

        pieces: list[str] = []
        files_block = _render_files_block(self.files)
        if files_block:
            pieces.append(files_block)
        if self.mode == "apply" and self.approved_plan:
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
        ]

        retrieval_collections: list[str] = []
        if codebase_collection:
            retrieval_collections = [codebase_collection]
        elif project:
            retrieval_collections = coerce_retrieval_scope(project.get("retrieval_scope"))

        if retrieval_collections:
            rag_blocks: list[str] = []
            for col in retrieval_collections[:3]:
                try:
                    codebase_context = retrieve(
                        query=user_message,
                        org_id=self.org_id,
                        collection_name=col,
                        n_results=8,
                        top_k=3,
                    )
                    if codebase_context:
                        rag_blocks.append(f"Collection: {col}\n{codebase_context}")
                except Exception:
                    _log.error("codebase RAG retrieve failed  collection=%s", col, exc_info=True)
            if rag_blocks:
                payload.append({"role": "system", "content": "Relevant retrieval context:\n\n" + "\n\n".join(rag_blocks)})

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
            _finalise_turn()
            _user_priority_ctx.reset(_priority_token)
            return

        output = "".join(chunks)
        duration = round(time.time() - start, 2)
        tokens_input = int(final_usage.get("prompt_tokens") or 0)
        tokens_output = int(final_usage.get("completion_tokens") or 0)

        if output:
            assistant_row = None
            try:
                assistant_row = self.db.add_code_message(
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

            if self.project_id and self.interactive_fs:
                # Multi-turn tool loop: parse directives, apply, feed results back, re-invoke up to N hops.
                cumulative_changes: list[dict] = []
                hop_output = output
                hop_assistant_id = int(assistant_row["Id"]) if assistant_row and assistant_row.get("Id") else None
                for hop in range(INTERACTIVE_FS_MAX_HOPS):
                    try:
                        changes, tool_events = apply_tool_directives(
                            db=self.db,
                            project_id=self.project_id,
                            response_text=hop_output,
                            conversation_id=conversation_id,
                            assistant_message_id=hop_assistant_id,
                        )
                    except Exception:
                        _log.error("interactive fs apply failed  hop=%d", hop, exc_info=True)
                        break
                    for tev in tool_events:
                        emit_raw({"type": "tool_result", "project_id": self.project_id, "conversation_id": conversation_id, "hop": hop, **tev})
                        if tev.get("permission_required"):
                            emit_raw({
                                "type": "permission_request",
                                "project_id": self.project_id,
                                "conversation_id": conversation_id,
                                "path": tev.get("path"),
                                "reason": tev.get("data") or "locked file",
                            })
                    cumulative_changes.extend(changes)
                    # If no directives at all, we're done.
                    directives = parse_tool_directives(hop_output)
                    if not directives:
                        break
                    # Build a follow-up message with tool results so the model can decide what next.
                    results_block = "\n\n".join(
                        f"```tool_result name={tev.get('tool')} ok={tev.get('ok')}\n{tev.get('data')}\n```"
                        for tev in tool_events
                    )
                    payload.append({"role": "assistant", "content": hop_output})
                    payload.append({"role": "user", "content": f"Tool results:\n\n{results_block}\n\nContinue, or end with a final summary."})
                    try:
                        chunks2, _u2, _m2 = self._call_model(payload, float(temperature), int(max_tokens), emit_raw)
                    except Exception:
                        _log.error("interactive fs follow-up call failed  hop=%d", hop, exc_info=True)
                        break
                    hop_output = "".join(chunks2)
                    if not hop_output.strip():
                        break
                    try:
                        followup_row = self.db.add_code_message(
                            conversation_id=conversation_id,
                            org_id=self.org_id,
                            role="assistant",
                            content=hop_output,
                            model=str(final_model),
                            mode=self.mode,
                            response_style=style_key,
                        )
                        hop_assistant_id = int(followup_row["Id"]) if followup_row and followup_row.get("Id") else hop_assistant_id
                    except Exception:
                        _log.debug("follow-up message persist skipped", exc_info=True)
                if cumulative_changes:
                    emit_raw({"type": "workspace_changed", "project_id": self.project_id, "changes": cumulative_changes})
            elif self.project_id and not self.interactive_fs:
                try:
                    changes = apply_file_fences(
                        db=self.db,
                        project_id=self.project_id,
                        response_text=output,
                        conversation_id=conversation_id,
                        assistant_message_id=int(assistant_row["Id"]) if assistant_row and assistant_row.get("Id") else None,
                        seen_keys=seen_fence_keys,
                    )
                    combined = [*streamed_changes, *changes]
                    for ch in changes:
                        if ch.get("permission_required"):
                            try:
                                self.db.add_project_audit_event(
                                    project_id=int(self.project_id),
                                    actor=f"agent:{conversation_id}" if conversation_id else "agent",
                                    kind="permission_request",
                                    payload={"path": ch.get("path"), "reason": ch.get("reason") or "locked file"},
                                )
                            except Exception:
                                _log.debug("permission_request audit write skipped", exc_info=True)
                            emit_raw(
                                {
                                    "type": "permission_request",
                                    "project_id": self.project_id,
                                    "conversation_id": conversation_id,
                                    "path": ch.get("path"),
                                    "reason": ch.get("reason") or "locked file",
                                }
                            )
                    if combined:
                        emit_raw({"type": "workspace_changed", "project_id": self.project_id, "changes": combined})
                except Exception:
                    _log.error("workspace fence apply failed  project=%s conv=%s", self.project_id, conversation_id, exc_info=True)

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

        _finalise_turn()
        _user_priority_ctx.reset(_priority_token)

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
