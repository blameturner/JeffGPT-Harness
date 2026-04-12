from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
import uuid
from typing import Iterator, Literal

import requests

from config import BASE_SYSTEM_PROMPT, TOOLS_FRAMEWORK_ENABLED, no_think_params
from workers.search.temporal import build_temporal_context
from memory import remember
from rag import retrieve
from tools.framework.contract import ToolContext
from tools.framework.dispatcher import execute_plan
from tools.framework.gate import gate_check
from tools.framework.planner import generate_plan
from workers.chat_agent import ChatAgent
from workers.styles import code_style_prompt

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


def _parse_plan_checklist(plan_text: str, fast_url: str, fast_model: str) -> list[str]:
    if not plan_text.strip():
        return []
    prompt = (
        "Extract the concrete step-by-step actions from this engineering "
        "plan as a JSON array of short strings (one per step, ≤ 15 words "
        "each). Return ONLY the JSON array, no prose.\n\n"
        f"PLAN:\n{plan_text[:6000]}"
    )
    try:
        resp = requests.post(
            f"{fast_url}/v1/chat/completions",
            json={
                "model": fast_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 400,
                **no_think_params(),
            },
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        match = re.search(r"\[.*\]", raw, re.S)
        if not match:
            return []
        data = json.loads(match.group(0))
        if not isinstance(data, list):
            return []
        return [str(x).strip() for x in data if str(x).strip()][:30]
    except Exception as e:
        _log.warning("plan checklist parse failed: %s", e)
        return []


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

    def _load_workspace(self, conversation_id: int) -> list[dict]:
        try:
            msgs = self.db.list_code_messages(conversation_id)
        except Exception as e:
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
                    if isinstance(data, list):
                        return data
                except Exception:
                    continue
        return []

    def run_job(
        self,
        job,
        user_message: str,
        conversation_id: int | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        title: str | None = None,
        codebase_collection: str | None = None,
        response_style: str | None = None,
        knowledge_enabled: bool | None = None,
    ) -> None:
        from workers.jobs import STORE

        def emit(event: dict):
            STORE.append(job, event)

        system_prompt = _SYSTEMS[self.mode]
        style_key, style_prompt = code_style_prompt(response_style)

        history: list[dict] = []
        is_new = False
        if conversation_id is None:
            try:
                convo = self.db.create_code_conversation(
                    org_id=self.org_id,
                    model=self.model,
                    title=(title or user_message)[:80],
                    mode=self.mode,
                    knowledge_enabled=bool(knowledge_enabled),
                )
                conversation_id = convo["Id"]
                is_new = True
            except Exception:
                _log.error("create_code_conversation failed", exc_info=True)
                emit({"type": "error", "message": "failed to create conversation"})
                return
        else:
            try:
                convo = self.db.get_code_conversation(conversation_id)
                if not convo:
                    emit({"type": "error", "message": f"Code conversation {conversation_id} not found"})
                    return
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in self.db.list_code_messages(conversation_id)
                ]
                if not self.files:
                    self.files = self._load_workspace(conversation_id)
            except Exception:
                _log.error("load code conversation failed", exc_info=True)
                emit({"type": "error", "message": "failed to load conversation"})
                return

        convo_knowledge = self._truthy(convo.get("knowledge_enabled")) or bool(knowledge_enabled)
        _log.info("code conversation flags  conv=%s knowledge=%s mode=%s", conversation_id, convo_knowledge, self.mode)

        if conversation_id is not None and not is_new:
            try:
                self.db.update_code_conversation(conversation_id, {"rag_collection": self.mode})
            except Exception:
                _log.warning("mode patch failed", exc_info=True)

        if conversation_id is not None:
            try:
                self.db.update_code_conversation(conversation_id, {"status": "processing"})
            except Exception:
                _log.warning("status update to processing failed  conv=%s", conversation_id)

        _log.debug("turn start  conv=%s mode=%s model=%s org=%d", conversation_id, self.mode, self.model, self.org_id)
        emit({"type": "meta", "mode": self.mode, "conversation_id": conversation_id})

        files_block = _render_files_block(self.files)
        pieces: list[str] = []
        if files_block:
            pieces.append(files_block)
        if self.mode == "execute" and self.approved_plan:
            pieces.append(f"<approved_plan>\n{self.approved_plan}\n</approved_plan>")
        pieces.append(user_message)
        composed_message = "\n\n".join(pieces)

        codebase_context = ""
        if codebase_collection:
            try:
                codebase_context = retrieve(
                    query=user_message,
                    org_id=self.org_id,
                    collection_name=codebase_collection,
                    n_results=10,
                    top_k=5,
                )
            except Exception:
                _log.error("codebase RAG retrieve failed", exc_info=True)

        # Tools framework — gate → plan → dispatch. Shares the same flag as chat
        # so ops can disable the whole pipeline in one place. Emits tool_status
        # events through the same STORE channel the frontend already handles.
        tool_context: ToolContext = ToolContext()
        if TOOLS_FRAMEWORK_ENABLED:
            last_assistant = ""
            for turn in reversed(history):
                if turn.get("role") == "assistant":
                    last_assistant = (turn.get("content") or "")[:800]
                    break
            hints = gate_check(
                user_message,
                conversation_context=last_assistant,
                mode="code",
            )
            _log.info("tools gate  conv=%s mode=%s hints=%s",
                      conversation_id, self.mode, sorted(hints) or "[]")

            if hints:
                emit({
                    "type": "tool_status",
                    "phase": "thinking",
                    "summary": "Preparing tools...",
                    "tools": sorted(hints),
                })

                async def _plan_and_run() -> ToolContext:
                    convo_summary = ""
                    if history:
                        last = history[-1].get("content") or ""
                        convo_summary = last[:200]
                    plan = await generate_plan(
                        user_message=user_message,
                        hints=hints,
                        conversation_summary=convo_summary,
                    )
                    if plan is None:
                        return ToolContext()
                    # Inject scope into each action's params. Code sessions have
                    # their own memory collection plus the attached codebase (if
                    # any); rag_lookup uses whichever it receives in params.
                    code_collection = f"code_{conversation_id}" if conversation_id else "code_knowledge"
                    for a in plan.actions:
                        a.params["_org_id"] = self.org_id
                        a.params["_collection"] = code_collection
                        # Widen rag_lookup to the cross-session code collections
                        # plus the attached codebase_collection if present.
                        if a.tool.value == "rag_lookup" and "collections" not in a.params:
                            cols = [code_collection, "code_knowledge"]
                            if codebase_collection:
                                cols.append(codebase_collection)
                            a.params["collections"] = cols
                    emit({
                        "type": "tool_status",
                        "phase": "planning",
                        "summary": plan.summary,
                        "tools": [a.tool.value for a in plan.actions],
                    })
                    return await execute_plan(plan, emit)

                try:
                    tool_context = asyncio.run(_plan_and_run())
                except Exception:
                    _log.error("tools framework failed  conv=%s",
                               conversation_id, exc_info=True)
                    tool_context = ToolContext()

        storage_files = _files_to_storage(self.files)
        if conversation_id is not None:
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

        payload: list[dict] = [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "system", "content": build_temporal_context()},
            {"role": "system", "content": system_prompt},
        ]
        if codebase_context:
            payload.append({
                "role": "system",
                "content": (
                    "The following snippets were retrieved from the project "
                    "codebase. Use them where relevant and prefer them over "
                    "guessing.\n\n" + codebase_context
                ),
            })
        tool_block = tool_context.to_system_block()
        if tool_block:
            payload.append({"role": "system", "content": tool_block})
        # style placed last so it sits closest to the user turn without
        # breaking history/turn alternation.
        payload.append({"role": "system", "content": style_prompt})
        payload.extend(history)
        payload.append({"role": "user", "content": composed_message})

        _log.debug("model call   conv=%s messages=%d temp=%.1f max_tokens=%d", conversation_id, len(payload), temperature, max_tokens)
        start = time.time()

        try:
            chunks, final_usage, final_model = self._call_model(payload, temperature, max_tokens, emit)
        except Exception:
            _log.error("model call failed  conv=%s", conversation_id, exc_info=True)
            if conversation_id is not None:
                try:
                    self.db.update_code_conversation(conversation_id, {"status": "error"})
                except Exception:
                    _log.warning("status update to error failed  conv=%s", conversation_id)
            emit({"type": "error", "message": "model call failed"})
            return

        duration = round(time.time() - start, 2)
        output = "".join(chunks)
        tokens_input = int(final_usage.get("prompt_tokens") or 0)
        tokens_output = int(final_usage.get("completion_tokens") or 0)
        _log.info("turn done    conv=%s mode=%s model=%s in=%d out=%d %.1fs", conversation_id, self.mode, final_model, tokens_input, tokens_output, duration)

        # Persist assistant message immediately — this is the critical write
        # that survives page navigation. Everything after this is background work.
        if output and conversation_id is not None:
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
                _log.info("persisted assistant message  conv=%s chars=%d", conversation_id, len(output))
            except Exception:
                _log.error("assistant message persist failed  conv=%s", conversation_id, exc_info=True)

        # Plan checklist must emit BEFORE done so the frontend receives it.
        checklist_steps = None
        if self.mode == "plan" and output:
            tool_url, tool_model = self._tool_model_url()
            if tool_url:
                checklist_steps = _parse_plan_checklist(output, tool_url, tool_model)
                if checklist_steps:
                    emit({"type": "plan_checklist", "steps": checklist_steps})

        if conversation_id is not None:
            try:
                self.db.update_code_conversation(conversation_id, {"status": "complete"})
            except Exception:
                _log.warning("status update to complete failed  conv=%s", conversation_id)

        emit({
            "type": "done",
            "mode": self.mode,
            "conversation_id": conversation_id,
            "model": str(final_model),
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "duration_seconds": duration,
            "output": output,
        })

        # Background: memory writes and graph extraction involve CPU-bound
        # embedding calls. Run after the user already has their response.
        import threading

        def _post_turn_work():
            if output and conversation_id is not None:
                try:
                    remember(
                        text=f"USER ({self.mode}): {user_message}\n\nASSISTANT: {output}",
                        metadata={"conversation_id": conversation_id, "model": str(final_model), "mode": self.mode, "turn_time": time.time()},
                        org_id=self.org_id,
                        collection_name=f"code_{conversation_id}",
                    )
                except Exception:
                    _log.error("memory write failed  conv=%s", conversation_id, exc_info=True)

            if convo_knowledge and output:
                try:
                    remember(
                        text=f"USER ({self.mode}): {user_message}\n\nASSISTANT: {output}",
                        metadata={"conversation_id": conversation_id, "model": str(final_model), "mode": self.mode, "turn_time": time.time()},
                        org_id=self.org_id,
                        collection_name="code_knowledge",
                    )
                except Exception:
                    _log.error("code_knowledge write failed  conv=%s", conversation_id, exc_info=True)
                try:
                    from workers.chat.graph import extract_and_write_graph
                    extract_and_write_graph(user_message, output, conversation_id, self.org_id)
                except Exception:
                    _log.error("graph extraction failed  conv=%s", conversation_id, exc_info=True)

            if checklist_steps and conversation_id is not None:
                try:
                    self.db.update_code_conversation(conversation_id, {"code_checklist": checklist_steps})
                except Exception:
                    _log.error("checklist persist failed  conv=%s", conversation_id, exc_info=True)

        threading.Thread(target=_post_turn_work, daemon=True).start()

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
        from workers.jobs import Job
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
