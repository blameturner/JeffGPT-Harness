from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
import uuid
from typing import Iterator, Literal

from infra.config import BASE_SYSTEM_PROMPT, TOOLS_FRAMEWORK_ENABLED, is_feature_enabled
from shared.temporal import build_temporal_context
from infra.memory import remember
from infra.rag import retrieve
from tools.contract import ToolContext
from tools.dispatcher import execute_plan
from tools.gate import gate_check
from tools.planner import generate_plan
from workers.base import BaseAgent, _get_summary_event, SUMMARY_WAIT_TIMEOUT
from workers.chat.history import maybe_summarise, extract_conversation_topics
from workers.code.config import code_style_prompt, code_max_tokens, code_temperature

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

REVIEW_SYSTEM = (
    "You are a senior software engineer in REVIEW mode. "
    "Provide detailed code review feedback on the attached files. "
    "Output sections: Summary, Issues (with severity), Recommendations, "
    "Code snippets where helpful."
)

EXPLAIN_SYSTEM = (
    "You are a senior software engineer in EXPLAIN mode. "
    "Explain how the code works in clear, simple terms. "
    "Output sections: Overview, Key concepts, Flow, Examples. "
    "Use analogies where helpful."
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
    from shared.models import model_call
    prompt = (
        "Extract the concrete step-by-step actions from this engineering "
        "plan as a JSON array of short strings (one per step, ≤ 15 words "
        "each). Return ONLY the JSON array, no prose.\n\n"
        f"PLAN:\n{plan_text[:6000]}"
    )
    try:
        raw, _tokens = model_call("tool_planner", prompt, max_tokens=400, temperature=0.0)
        if not raw:
            return []
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

        # signal active session so queue workers back off
        from workers.tool_queue import touch_chat_activity
        touch_chat_activity()

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
                # must wait on prev turn's bg summary — else we'd read stale summary from DB
                summary_ev = _get_summary_event(conversation_id)
                if not summary_ev.is_set():
                    emit({"type": "status", "phase": "summarising_previous", "message": "Updating conversation context..."})
                    waited = summary_ev.wait(timeout=SUMMARY_WAIT_TIMEOUT)
                    if waited:
                        _log.info("code conv=%s  waited for background summary — ready", conversation_id)
                    else:
                        _log.warning("code conv=%s  background summary wait timed out after %ds", conversation_id, SUMMARY_WAIT_TIMEOUT)
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

        tool_context: ToolContext = ToolContext()
        web_search_enabled = is_feature_enabled("web_search")
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
            if not web_search_enabled or not self.search_enabled:
                hints.discard("web_search")
            _log.info("tools gate  conv=%s mode=%s hints=%s web_search=%s",
                      conversation_id, self.mode, sorted(hints) or "[]", web_search_enabled)

            if hints:
                tool_labels = {
                    "web_search": "web search",
                    "rag_lookup": "conversation history lookup",
                }
                hint_names = [tool_labels.get(h, h) for h in sorted(hints)]
                instant_summary = f"Running {', '.join(hint_names)} for: {user_message[:80]}"
                emit({
                    "type": "tool_status",
                    "phase": "planning",
                    "summary": instant_summary,
                    "tools": sorted(hints),
                })

                code_collection = f"code_{conversation_id}" if conversation_id else "code_knowledge"

                if hints == {"web_search"}:
                    _log.info("web_search fast-path  conv=%s", conversation_id)
                    from tools.search.queries import generate_broad_queries
                    from workers.chat.history import extract_conversation_topics
                    convo_topics = extract_conversation_topics(history)
                    queries = generate_broad_queries(user_message, max_queries=5, conversation_topics=convo_topics)
                    _log.info("web_search fast-path queries  conv=%s queries=%s", conversation_id, queries)

                    if queries:
                        from tools.contract import ToolAction, ToolName, ToolPlan
                        plan = ToolPlan(
                            actions=[ToolAction(
                                tool=ToolName.WEB_SEARCH,
                                params={
                                    "queries": queries,
                                    "_org_id": self.org_id,
                                    "_collection": code_collection,
                                    "_mode": "normal",
                                },
                                reason="web search",
                            )],
                            summary=instant_summary,
                        )
                        try:
                            tool_context = asyncio.run(execute_plan(plan, emit))
                        except Exception:
                            _log.error("web_search fast-path failed  conv=%s", conversation_id, exc_info=True)
                            tool_context = ToolContext()
                else:
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
                        for a in plan.actions:
                            a.params["_org_id"] = self.org_id
                            a.params["_collection"] = code_collection
                            a.params["_conversation_id"] = conversation_id
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

        history, summary_event = maybe_summarise(history, truncate_only=True)
        if summary_event:
            emit(summary_event)

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
        # style last — sits closest to user turn without breaking role alternation
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

        # must persist BEFORE emitting done — drop here loses the turn even though chunks streamed
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

        import threading

        def _post_turn_work():
            _t_bg = time.perf_counter()
            _log.info("code conv=%s  post-turn background starting  mode=%s", conversation_id, self.mode)

            if output and conversation_id is not None:
                _t = time.perf_counter()
                try:
                    remember(
                        text=f"USER ({self.mode}): {user_message}\n\nASSISTANT: {output}",
                        metadata={"conversation_id": conversation_id, "model": str(final_model), "mode": self.mode, "turn_time": time.time()},
                        org_id=self.org_id,
                        collection_name=f"code_{conversation_id}",
                    )
                    _log.info("code conv=%s  [1/5] RAG embedded to code_%s  %.2fs", conversation_id, conversation_id, time.perf_counter() - _t)
                except Exception:
                    _log.error("code conv=%s  [1/5] RAG embed FAILED", conversation_id, exc_info=True)

            if convo_knowledge and output:
                _t = time.perf_counter()
                try:
                    remember(
                        text=f"USER ({self.mode}): {user_message}\n\nASSISTANT: {output}",
                        metadata={"conversation_id": conversation_id, "model": str(final_model), "mode": self.mode, "turn_time": time.time()},
                        org_id=self.org_id,
                        collection_name="code_knowledge",
                    )
                    _log.info("code conv=%s  [2/5] knowledge embedded  %.2fs", conversation_id, time.perf_counter() - _t)
                except Exception:
                    _log.error("code conv=%s  [2/5] knowledge embed FAILED", conversation_id, exc_info=True)

                try:
                    from workers.tool_queue import get_tool_queue
                    tq = get_tool_queue()
                    if tq:
                        job_id = tq.submit(
                            job_type="graph_extract",
                            payload={
                                "user_text": user_message,
                                "assistant_text": output,
                                "conversation_id": conversation_id,
                                "org_id": self.org_id,
                            },
                            source="code",
                            org_id=self.org_id,
                            priority=5,
                        )
                        _log.info("code conv=%s  [3/5] graph extraction queued  job=%s", conversation_id, job_id)
                except Exception:
                    _log.error("code conv=%s  [3/5] graph extraction queue FAILED", conversation_id, exc_info=True)

            # bg summary produces summary+topics consumed by the NEXT turn's gate
            summary_ev = _get_summary_event(conversation_id)
            summary_ev.clear()  # blocks next turn until set()
            _t = time.perf_counter()
            try:
                full_history = history + [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": output},
                ]
                summarised_history, bg_summary_event = maybe_summarise(full_history, truncate_only=False)
                if bg_summary_event and not bg_summary_event.get("fallback"):
                    topics = bg_summary_event.get("topics", [])
                    summary_content = ""
                    for m in summarised_history:
                        if m.get("role") == "system" and "[Conversation summary]" in (m.get("content") or ""):
                            summary_content = m["content"]
                            break
                    if summary_content:
                        try:
                            existing_msgs = self.db.list_code_messages(conversation_id)
                            existing_id = None
                            for msg in existing_msgs:
                                if msg.get("role") == "system" and "[Conversation summary]" in (msg.get("content") or ""):
                                    existing_id = msg.get("Id")
                                    break
                            if existing_id:
                                self.db._patch("code_messages", existing_id, {
                                    "Id": existing_id,
                                    "content": summary_content,
                                })
                                _log.info("code conv=%s  [4/5] summary updated  topics=%s  %.2fs", conversation_id, topics, time.perf_counter() - _t)
                            else:
                                self.db.add_code_message(
                                    conversation_id=conversation_id,
                                    org_id=self.org_id,
                                    role="system",
                                    content=summary_content,
                                    model="summariser",
                                    mode=self.mode,
                                )
                                _log.info("code conv=%s  [4/5] summary created  topics=%s  %.2fs", conversation_id, topics, time.perf_counter() - _t)
                        except Exception:
                            _log.error("code conv=%s  [4/5] summary persist FAILED", conversation_id, exc_info=True)
                    else:
                        _log.info("code conv=%s  [4/5] summary produced but empty — skipped persist", conversation_id)
                elif bg_summary_event and bg_summary_event.get("fallback"):
                    _log.info("code conv=%s  [4/5] summary skipped — model unavailable, truncation only", conversation_id)
                else:
                    _log.info("code conv=%s  [4/5] summary skipped — under threshold (%d messages)", conversation_id, len(full_history))
            except Exception:
                _log.error("code conv=%s  [4/5] summary FAILED", conversation_id, exc_info=True)
            finally:
                summary_ev.set()  # must set even on failure or next turn blocks forever

            if self.mode == "plan" and output and conversation_id is not None:
                try:
                    checklist_steps = _parse_plan_checklist(output)
                    if checklist_steps:
                        self.db.update_code_conversation(conversation_id, {"code_checklist": checklist_steps})
                        _log.info("code conv=%s  [5/5] plan checklist persisted  steps=%d", conversation_id, len(checklist_steps))
                    else:
                        _log.info("code conv=%s  [5/5] plan checklist — no steps extracted", conversation_id)
                except Exception:
                    _log.error("code conv=%s  [5/5] plan checklist FAILED", conversation_id, exc_info=True)

            _log.info("code conv=%s  post-turn complete  total=%.2fs", conversation_id, time.perf_counter() - _t_bg)

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
