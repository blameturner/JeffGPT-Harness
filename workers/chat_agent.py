import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Iterator

import requests

from config import MODELS, TOOLS_FRAMEWORK_ENABLED, get_model_url, refresh_models
from nocodb_client import NocodbClient
from memory import remember
from tools.framework.contract import ToolAction, ToolContext, ToolName, ToolPlan
from tools.framework.dispatcher import execute_plan
from tools.framework.gate import gate_check
from tools.framework.planner import generate_plan
from workers.search.intent import classify_message_intent
from workers.search.models import _tool_model
from workers.search.queries import generate_search_queries
from workers.styles import chat_style_prompt
from workers.chat.history import maybe_summarise
from workers.chat.graph import extract_and_write_graph
from workers.chat.payload import build_chat_payload
from workers.chat.search_phase import SearchPhaseResult, run_search_phase
from workers.chat.rag_phase import submit_rag_future, collect_rag, cancel_rag
from workers.chat.persistence import (
    schedule_status_processing_write,
    schedule_user_message_write,
    persist_assistant_message,
)

_log = logging.getLogger("chat")


@dataclass
class ChatResult:
    output: str
    model: str
    conversation_id: int
    tokens_input: int
    tokens_output: int
    duration_seconds: float
    rag_enabled: bool
    context_chars: int


class ChatAgent:
    def __init__(self, model: str, org_id: int, search_enabled: bool = False):
        url = get_model_url(model)
        if not url:
            refresh_models()
            url = get_model_url(model)
        if not url:
            options = sorted({
                v["role"] for v in MODELS.values() if isinstance(v, dict)
            })
            raise ValueError(
                f"Model '{model}' not available. Options: {options}"
            )
        self.model = model
        self.org_id = org_id
        self.url = url
        self.search_enabled = search_enabled
        self.db = NocodbClient()

    @staticmethod
    def _default_collection(conversation_id: int) -> str:
        return f"chat_{conversation_id}"

    @staticmethod
    def _truthy(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return False

    @staticmethod
    def _tool_model_url() -> tuple[str | None, str | None]:
        return _tool_model()

    def run_job(
        self,
        job,
        user_message: str,
        conversation_id: int | None = None,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        rag_enabled: bool | None = None,
        rag_collection: str | None = None,
        knowledge_enabled: bool | None = None,
        search_consent_declined: bool = False,
        response_style: str | None = None,
    ) -> None:
        from workers.jobs import STORE

        def emit(event: dict):
            etype = event.get("type", "")
            if etype != "chunk":
                _log.info("emit  type=%s %s", etype, event.get("phase") or event.get("summary", "")[:60] or "")
            STORE.append(job, event)

        _turn_start = time.perf_counter()
        spans: dict[str, int] = {}

        def _span(name: str, t_start: float) -> None:
            spans[name] = int((time.perf_counter() - t_start) * 1000)

        _t = time.perf_counter()
        if conversation_id is None:
            convo = self.db.create_conversation(
                org_id=self.org_id,
                model=self.model,
                title=user_message[:80],
                rag_enabled=bool(rag_enabled),
                rag_collection=rag_collection or "",
                knowledge_enabled=bool(knowledge_enabled),
            )
            conversation_id = convo["Id"]
            history: list[dict] = []
        else:
            convo = self.db.get_conversation(conversation_id)
            if not convo:
                emit({"type": "error", "message": f"Conversation {conversation_id} not found"})
                return
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in self.db.list_messages(conversation_id)
            ]
        _span("load_convo_ms", _t)

        convo_rag_enabled = self._truthy(convo.get("rag_enabled"))
        collection_name = (
            (convo.get("rag_collection") or "").strip()
            or self._default_collection(conversation_id)
        )
        convo_knowledge = self._truthy(convo.get("knowledge_enabled")) or bool(knowledge_enabled)
        _log.info("conversation flags  conv=%s rag=%s knowledge=%s (raw=%s, param=%s)", conversation_id, convo_rag_enabled, convo_knowledge, convo.get("knowledge_enabled"), knowledge_enabled)

        _t = time.perf_counter()
        schedule_status_processing_write(self.db, conversation_id)
        _span("status_processing_ms", _t)

        _log.debug("turn start  conv=%s model=%s org=%d", conversation_id, self.model, self.org_id)
        emit({"type": "meta", "conversation_id": conversation_id})

        rag_executor, rag_future = submit_rag_future(
            user_message=user_message,
            org_id=self.org_id,
            collection_name=collection_name,
            enabled=convo_rag_enabled,
        )

        _t_search = time.perf_counter()
        tool_context: ToolContext = ToolContext()

        if TOOLS_FRAMEWORK_ENABLED:
            # New path: heuristic gate → planner → parallel tool dispatch.
            # search_result stays empty-default; legacy consent flow is skipped.
            search_result = SearchPhaseResult()

            # Use the last assistant turn (if any) as context so follow-up
            # questions like "and what about Melbourne?" reopen web_search.
            last_assistant = ""
            for turn in reversed(history):
                if turn.get("role") == "assistant":
                    last_assistant = (turn.get("content") or "")[:800]
                    break
            hints = gate_check(user_message, conversation_context=last_assistant)
            _log.info("tools gate  conv=%s hints=%s", conversation_id, sorted(hints) or "[]")

            if hints:
                tool_labels = {
                    "web_search": "web search",
                    "rag_lookup": "conversation history lookup",
                    "code_exec": "code execution",
                }
                hint_names = [tool_labels.get(h, h) for h in sorted(hints)]
                instant_summary = f"Running {', '.join(hint_names)} for: {user_message[:80]}"
                emit({
                    "type": "tool_status",
                    "phase": "planning",
                    "summary": instant_summary,
                    "tools": sorted(hints),
                })

                _t_tools = time.perf_counter()

                if hints == {"web_search"}:
                    # Fast path: skip planner AND intent classifier entirely.
                    # Build queries directly from the user message — zero model calls.
                    _log.info("web_search fast-path  conv=%s", conversation_id)
                    # Simple heuristic intent for query generation — no model call
                    words = user_message.lower().split()
                    entities = [w for w in user_message.split() if len(w) > 3 and w[0].isupper()]
                    intent_dict = {
                        "intent": "factual_lookup",
                        "entities": entities or [user_message[:60]],
                        "time_sensitive": any(w in words for w in ("latest", "recent", "current", "today", "now", "2025", "2026")),
                        "confidence": "medium",
                        "search_policy": "focused",
                    }
                    queries = generate_search_queries(intent_dict, message=user_message)
                    _log.info("web_search fast-path queries  conv=%s queries=%s", conversation_id, queries)

                    if queries:
                        search_mode = getattr(self, '_search_mode', 'normal')
                        plan = ToolPlan(
                            actions=[ToolAction(
                                tool=ToolName.WEB_SEARCH,
                                params={
                                    "queries": queries,
                                    "_org_id": self.org_id,
                                    "_collection": collection_name,
                                    "_mode": search_mode,
                                },
                                reason="web search",
                            )],
                            summary=instant_summary,
                        )
                        try:
                            tool_context = asyncio.run(execute_plan(plan, emit))
                            _log.info("web_search fast-path done  conv=%s results=%d elapsed=%.2fs", conversation_id, len(tool_context.results), time.perf_counter() - _t_tools)
                        except Exception:
                            _log.error("web_search fast-path failed  conv=%s", conversation_id, exc_info=True)
                            tool_context = ToolContext()
                else:
                    # Full planner path for multi-tool or non-search hints.
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
                            a.params["_collection"] = collection_name
                        emit({
                            "type": "tool_status",
                            "phase": "planning",
                            "summary": plan.summary,
                            "tools": [a.tool.value for a in plan.actions],
                        })
                        return await execute_plan(plan, emit)

                    try:
                        _log.info("tools framework starting  conv=%s hints=%s", conversation_id, sorted(hints))
                        tool_context = asyncio.run(_plan_and_run())
                        _log.info("tools framework done  conv=%s results=%d elapsed=%.2fs", conversation_id, len(tool_context.results), time.perf_counter() - _t_tools)
                    except Exception:
                        _log.error("tools framework failed  conv=%s", conversation_id, exc_info=True)
                        tool_context = ToolContext()

            # Map web_search results back onto search_result so the downstream
            # payload build and persistence code paths stay unchanged.
            for r in tool_context.results:
                if r.tool.value == "web_search" and r.ok:
                    search_result.search_context = r.data
                    search_result.search_status = "used"
                    search_result.search_confidence = "high"
                    break
        else:
            search_result = run_search_phase(
                user_message=user_message,
                history=history,
                convo=convo,
                conversation_id=conversation_id,
                org_id=self.org_id,
                search_enabled=self.search_enabled,
                search_consent_declined=search_consent_declined,
                emit=emit,
                span=_span,
            )
        _span("search_total_ms", _t_search)

        if search_result.consent_required:
            try:
                self.db.update_conversation(conversation_id, {"status": "awaiting_consent"})
            except Exception:
                pass
            emit({
                "type": "search_consent_required",
                "query": user_message,
                "reason": search_result.consent_reason,
            })
            emit({
                "type": "done",
                "conversation_id": conversation_id,
                "awaiting": "search_consent",
                "model": self.model,
                "tokens_input": 0,
                "tokens_output": 0,
                "duration_seconds": 0.0,
                "rag_enabled": False,
                "context_chars": 0,
            })
            cancel_rag(rag_future, rag_executor)
            return

        search_context = search_result.search_context
        search_sources = search_result.search_sources
        search_confidence = search_result.search_confidence
        search_status = search_result.search_status
        search_note = search_result.search_note
        intent_dict = search_result.intent_dict

        # Prepend tool-framework results (rag_lookup, code_exec, etc.) onto
        # search_context so they ride the same system-prompt injection path.
        # web_search results were already mapped onto search_result.search_context.
        if tool_context.results:
            non_web = [r for r in tool_context.results if r.tool.value != "web_search"]
            if non_web:
                block = ToolContext(
                    plan_summary=tool_context.plan_summary, results=non_web,
                ).to_system_block()
                search_context = (block + "\n\n" + (search_context or "")).strip()

        _t = time.perf_counter()
        style_key, style_prompt = chat_style_prompt(response_style)
        _span("style_resolve_ms", _t)

        _t = time.perf_counter()
        _user_msg_written = schedule_user_message_write(
            db=self.db,
            conversation_id=conversation_id,
            org_id=self.org_id,
            user_message=user_message,
            model=self.model,
            style_key=style_key,
        )
        _span("user_msg_persist_ms", _t)

        _t = time.perf_counter()
        rag_context = collect_rag(rag_future, rag_executor)
        _span("rag_retrieve_ms", _t)

        _t = time.perf_counter()
        history, summary_event = maybe_summarise(history)
        if summary_event:
            emit(summary_event)
        _span("summarise_ms", _t)

        _t = time.perf_counter()
        payload = build_chat_payload(
            history=history,
            user_message=user_message,
            style_prompt=style_prompt,
            system=system,
            search_context=search_context,
            search_note=search_note,
            rag_context=rag_context,
        )
        _span("payload_build_ms", _t)

        _span("pre_model_total_ms", _turn_start)
        _log.info(
            "turn pre-model  conv=%s " + " ".join(f"{k}=%d" for k in spans),
            conversation_id, *spans.values(),
        )

        _log.debug("model call   conv=%s messages=%d temp=%.1f max_tokens=%d", conversation_id, len(payload), temperature, max_tokens)
        start = time.time()

        try:
            chunks, final_usage, final_model = self._call_model(payload, temperature, max_tokens, emit)
        except Exception:
            _log.error("model call failed  conv=%s", conversation_id, exc_info=True)
            try:
                self.db.update_conversation(conversation_id, {"status": "error"})
            except Exception:
                _log.warning("status update to error failed  conv=%s", conversation_id)
            emit({"type": "error", "message": "model call failed"})
            return

        duration = round(time.time() - start, 2)
        output = "".join(chunks)
        tokens_input = int(final_usage.get("prompt_tokens") or 0)
        tokens_output = int(final_usage.get("completion_tokens") or 0)
        _log.info("turn done    conv=%s model=%s in=%d out=%d %.1fs chars=%d", conversation_id, final_model, tokens_input, tokens_output, duration, len(output))

        # Persist to NocoDB BEFORE emitting done — this is the critical write
        # that must not be lost. If this fails, the user still has streamed chunks.
        if output:
            if not _user_msg_written.wait(timeout=10.0):
                _log.warning("user message write still pending after 10s  conv=%s", conversation_id)
            _t = time.perf_counter()
            persist_ok = persist_assistant_message(
                db=self.db,
                conversation_id=conversation_id,
                org_id=self.org_id,
                output=output,
                final_model=final_model,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                style_key=style_key,
                search_sources=search_sources,
                search_status=search_status,
                search_confidence=search_confidence,
                search_context=search_context,
                intent_dict=intent_dict,
            )
            _log.info("persist done  conv=%s ok=%s %.2fs", conversation_id, persist_ok, time.perf_counter() - _t)
            if not persist_ok:
                emit({"type": "error", "message": "assistant message persist failed"})

        try:
            self.db.update_conversation(conversation_id, {"status": "complete"})
        except Exception:
            _log.warning("status update to complete failed  conv=%s", conversation_id)

        emit({
            "type": "done",
            "conversation_id": conversation_id,
            "model": str(final_model),
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "duration_seconds": duration,
            "rag_enabled": convo_rag_enabled,
            "context_chars": len(rag_context),
            "response_style": style_key,
            "search_used": bool(search_sources) or search_status in ("used", "no_results", "error"),
            "search_status": search_status,
            "search_confidence": search_confidence,
            "search_source_count": len(search_sources),
        })

        # Only memory/graph work in background — these are non-critical.
        import threading

        def _post_turn_work():
            _t_bg = time.perf_counter()

            # Memory and graph
            if convo_rag_enabled and output:
                _t = time.perf_counter()
                try:
                    remember(
                        text=f"USER: {user_message}\n\nASSISTANT: {output}",
                        metadata={"conversation_id": conversation_id, "model": self.model, "turn_time": time.time()},
                        org_id=self.org_id,
                        collection_name=collection_name,
                    )
                    _log.info("bg: rag remember done  conv=%s %.2fs", conversation_id, time.perf_counter() - _t)
                except Exception:
                    _log.error("bg: rag remember failed  conv=%s", conversation_id, exc_info=True)

            if convo_knowledge and output:
                _t = time.perf_counter()
                try:
                    remember(
                        text=f"USER: {user_message}\n\nASSISTANT: {output}",
                        metadata={"conversation_id": conversation_id, "model": self.model, "turn_time": time.time()},
                        org_id=self.org_id,
                        collection_name="chat_knowledge",
                    )
                    _log.info("bg: knowledge remember done  conv=%s %.2fs", conversation_id, time.perf_counter() - _t)
                except Exception:
                    _log.error("bg: knowledge remember failed  conv=%s", conversation_id, exc_info=True)
                _t = time.perf_counter()
                try:
                    extract_and_write_graph(user_message, output, conversation_id, self.org_id)
                    _log.info("bg: graph extraction done  conv=%s %.2fs", conversation_id, time.perf_counter() - _t)
                except Exception:
                    _log.error("bg: graph extraction failed  conv=%s", conversation_id, exc_info=True)

            _log.info("bg: post-turn complete  conv=%s total=%.2fs", conversation_id, time.perf_counter() - _t_bg)

        threading.Thread(target=_post_turn_work, daemon=True).start()

    def send_streaming(
        self,
        user_message: str,
        conversation_id: int | None = None,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        rag_enabled: bool | None = None,
        rag_collection: str | None = None,
        knowledge_enabled: bool | None = None,
        search_consent_declined: bool = False,
        response_style: str | None = None,
    ) -> Iterator[dict]:
        from workers.jobs import Job
        job = Job(uuid.uuid4().hex)
        self.run_job(
            job,
            user_message=user_message,
            conversation_id=conversation_id,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            rag_enabled=rag_enabled,
            rag_collection=rag_collection,
            knowledge_enabled=knowledge_enabled,
            search_consent_declined=search_consent_declined,
            response_style=response_style,
        )
        yield from job.events

    def _call_model(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        emit: Callable[[dict], None],
    ) -> tuple[list[str], dict, str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        chunks: list[str] = []
        final_usage: dict = {}
        final_model: str = self.model
        in_think_block = False
        think_tokens = 0
        first_content_emitted = False
        reasoning_chunks: list[str] = []

        with requests.post(
            f"{self.url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=(30, 3600),
        ) as response:
            response.raise_for_status()
            response.encoding = "utf-8"
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                if not raw_line.startswith("data:"):
                    continue
                data = raw_line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if event.get("model"):
                    final_model = event["model"]
                usage = event.get("usage")
                if usage:
                    final_usage = usage

                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}

                # Some models put thinking in reasoning_content, actual answer in content.
                # Others put everything in content with <think> tags.
                reasoning = delta.get("reasoning_content")
                text = delta.get("content")

                if reasoning and not text:
                    # Model is still thinking — collect but don't emit yet
                    think_tokens += 1
                    reasoning_chunks.append(reasoning)
                    continue

                if not text:
                    continue

                # Filter <think>...</think> tags in content (Qwen3 style)
                if not first_content_emitted and "<think>" in text:
                    in_think_block = True
                    think_tokens += 1
                    continue
                if in_think_block:
                    think_tokens += 1
                    if "</think>" in text:
                        in_think_block = False
                        if think_tokens > 1:
                            _log.info("model thinking done  tokens=%d", think_tokens)
                    continue

                first_content_emitted = True
                chunks.append(text)
                emit({"type": "chunk", "text": text})

        # Fallback: if model put everything in reasoning_content and content was
        # always empty, use the reasoning as the output. Better than losing the response.
        if not chunks and reasoning_chunks:
            _log.warning("model returned no content, using reasoning_content as fallback  tokens=%d", len(reasoning_chunks))
            full_reasoning = "".join(reasoning_chunks)
            chunks.append(full_reasoning)
            emit({"type": "chunk", "text": full_reasoning})

        if think_tokens > 0:
            _log.info("model thinking summary  think_tokens=%d content_tokens=%d", think_tokens, len(chunks))

        return chunks, final_usage, final_model

    def send(
        self,
        user_message: str,
        conversation_id: int | None = None,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        rag_enabled: bool | None = None,
        rag_collection: str | None = None,
        knowledge_enabled: bool | None = None,
    ) -> ChatResult:
        parts: list[str] = []
        final: dict = {}
        conv_id = conversation_id or 0
        for event in self.send_streaming(
            user_message=user_message,
            conversation_id=conversation_id,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            rag_enabled=rag_enabled,
            rag_collection=rag_collection,
            knowledge_enabled=knowledge_enabled,
        ):
            etype = event.get("type")
            if etype == "meta":
                conv_id = event.get("conversation_id") or conv_id
            elif etype == "chunk":
                parts.append(event.get("text", ""))
            elif etype == "done":
                final = event
            elif etype == "error":
                raise RuntimeError(event.get("message") or "chat stream error")

        return ChatResult(
            output="".join(parts),
            model=final.get("model", self.model),
            conversation_id=final.get("conversation_id", conv_id),
            tokens_input=int(final.get("tokens_input") or 0),
            tokens_output=int(final.get("tokens_output") or 0),
            duration_seconds=float(final.get("duration_seconds") or 0.0),
            rag_enabled=bool(final.get("rag_enabled")),
            context_chars=int(final.get("context_chars") or 0),
        )
