import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Iterator

import requests

from config import MODELS, TOOLS_FRAMEWORK_ENABLED, get_model_url, is_feature_enabled, refresh_models
from nocodb_client import NocodbClient
from memory import remember
from tools.framework.contract import ToolAction, ToolContext, ToolName, ToolPlan
from tools.framework.dispatcher import execute_plan
from tools.framework.gate import gate_check
from tools.framework.planner import generate_plan
from workers.search.queries import generate_broad_queries, generate_search_queries
from workers.styles import chat_style_prompt
from workers.chat.history import maybe_summarise, extract_conversation_topics
# graph extraction is now queued via tool_queue, not called directly
from workers.chat.payload import build_chat_payload
from workers.chat.search_phase import SearchPhaseResult, run_search_phase
from workers.chat.rag_phase import submit_rag_future, collect_rag, cancel_rag
from workers.chat.persistence import (
    schedule_status_processing_write,
    schedule_user_message_write,
    persist_assistant_message,
)

import threading

_log = logging.getLogger("chat")

# Per-conversation summarisation locks. When a background summary is running,
# the event is cleared.  Next turn waits on it (with timeout) so the summary
# is available for topic extraction and context.
_summary_locks: dict[int, threading.Event] = {}
_summary_locks_mu = threading.Lock()

SUMMARY_WAIT_TIMEOUT = 30  # max seconds to wait for a running summary


def _get_summary_event(conversation_id: int) -> threading.Event:
    with _summary_locks_mu:
        ev = _summary_locks.get(conversation_id)
        if ev is None:
            ev = threading.Event()
            ev.set()  # not running by default
            _summary_locks[conversation_id] = ev
        return ev


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
        from workers.search.models import _tool_model
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

        # Signal active session so queue workers back off.
        from workers.tool_queue import touch_chat_activity
        touch_chat_activity()

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
            # Wait for any in-flight background summary to finish so we
            # pick up the latest summary + topics from the DB.
            summary_ev = _get_summary_event(conversation_id)
            if not summary_ev.is_set():
                emit({"type": "status", "phase": "summarising_previous", "message": "Updating conversation context..."})
                waited = summary_ev.wait(timeout=SUMMARY_WAIT_TIMEOUT)
                if not waited:
                    _log.warning("summary wait timed out  conv=%s", conversation_id)
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
        web_search_enabled = is_feature_enabled("web_search")

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
            if not web_search_enabled or not self.search_enabled:
                hints.discard("web_search")
            _log.info("tools gate  conv=%s hints=%s web_search=%s search_enabled=%s", conversation_id, sorted(hints) or "[]", web_search_enabled, self.search_enabled)

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

                # Deep search is a UI toggle (search_mode="deep"), not
                # auto-detected.  When active, queue deep_search jobs AND
                # still run a normal web_search inline so the main model
                # has immediate context while the deep results arrive later.
                search_mode = getattr(self, '_search_mode', 'normal')

                # Tools that can be dispatched directly without the planner.
                # web_search / deep_search: heuristic queries (zero model calls).
                # rag_lookup: just a ChromaDB similarity search.
                # code_exec: needs planner for code generation — only tool
                #            that truly requires the planner.
                _DIRECT_TOOLS = {"web_search", "rag_lookup"}

                if hints <= _DIRECT_TOOLS:
                    # Fast path: build plan directly without invoking
                    # the planner model.  Zero model calls for dispatch.
                    _log.info("fast-path  conv=%s hints=%s mode=%s", conversation_id, sorted(hints), search_mode)
                    actions: list[ToolAction] = []

                    if "web_search" in hints:
                        convo_topics = extract_conversation_topics(history)
                        # If RAG is running, collect it early to extract
                        # context keywords for search query enrichment.
                        if rag_future and not convo_topics:
                            try:
                                early_rag = rag_future.result(timeout=2)
                                if early_rag:
                                    from workers.search.queries import _extract_keywords
                                    rag_kw = _extract_keywords(early_rag[:2000])
                                    # Deduplicate against message keywords
                                    msg_kw_lower = {k.lower() for k in _extract_keywords(user_message)}
                                    convo_topics = [k.lower() for k in rag_kw if k.lower() not in msg_kw_lower][:8]
                                    _log.info("rag-enriched topics  conv=%s topics=%s", conversation_id, convo_topics)
                            except Exception:
                                pass  # timeout or error — proceed without
                        queries = generate_broad_queries(user_message, max_queries=5, conversation_topics=convo_topics)
                        _log.info("fast-path queries  conv=%s queries=%s", conversation_id, queries)
                        if queries:
                            actions.append(ToolAction(
                                tool=ToolName.WEB_SEARCH,
                                params={
                                    "queries": queries,
                                    "_org_id": self.org_id,
                                    "_collection": collection_name,
                                },
                                reason="web search",
                            ))

                    # When UI has deep search toggled, also queue deep_search
                    # jobs for thorough background analysis.
                    if search_mode == "deep" and "web_search" in hints:
                        deep_queries = generate_broad_queries(user_message, max_queries=10)
                        _log.info("deep search queuing  conv=%s queries=%s", conversation_id, deep_queries)
                        if deep_queries:
                            actions.append(ToolAction(
                                tool=ToolName.DEEP_SEARCH,
                                params={
                                    "queries": deep_queries,
                                    "_org_id": self.org_id,
                                    "_conversation_id": conversation_id,
                                },
                                reason="deep search (queued)",
                            ))

                    if "rag_lookup" in hints:
                        actions.append(ToolAction(
                            tool=ToolName.RAG_LOOKUP,
                            params={
                                "query": user_message[:500],
                                "_org_id": self.org_id,
                                "_collection": collection_name,
                            },
                            reason="conversation history",
                        ))

                    if actions:
                        plan = ToolPlan(actions=actions, summary=instant_summary)
                        try:
                            tool_context = asyncio.run(execute_plan(plan, emit))
                            _log.info("fast-path done  conv=%s results=%d elapsed=%.2fs", conversation_id, len(tool_context.results), time.perf_counter() - _t_tools)
                        except Exception:
                            _log.error("fast-path failed  conv=%s", conversation_id, exc_info=True)
                            tool_context = ToolContext()
                else:
                    # Full planner path — only for tools that need model-generated
                    # params (deep_search queries, code_exec code).
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

            # Map tool results back onto search_result so the downstream
            # payload build and persistence code paths stay unchanged.
            has_queued_jobs = False
            for r in tool_context.results:
                if r.tool.value == "web_search" and r.ok:
                    search_result.search_context = r.data
                    search_result.search_status = "used"
                    search_result.search_confidence = "high"
                elif r.tool.value == "web_search" and not r.ok:
                    search_result.search_status = "no_results"
                    search_result.search_confidence = "none"
                    search_result.search_note = (
                        "Web search was attempted but found no relevant results. "
                        "Answer from your own knowledge, and suggest 1-2 specific "
                        "search terms the user could try to find what they need."
                    )
                elif r.tool.value == "deep_search" and r.ok:
                    # Deep search is queued — tell the UI.
                    has_queued_jobs = True
                    emit({
                        "type": "jobs_queued",
                        "tool": "deep_search",
                        "message": r.data,
                        "status": "waiting",
                    })
                    # Include acknowledgment in main model context so it
                    # can tell the user results are being researched.
                    if not search_result.search_context:
                        search_result.search_context = r.data
                        search_result.search_status = "queued"
                        search_result.search_confidence = "pending"
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
        history, summary_event = maybe_summarise(history, truncate_only=True)
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
                # Queue graph extraction instead of running inline — it uses
                # RWKV and would block the summariser and other work.
                try:
                    from workers.tool_queue import get_tool_queue
                    tq = get_tool_queue()
                    if tq:
                        tq.submit(
                            job_type="graph_extract",
                            payload={
                                "user_text": user_message,
                                "assistant_text": output,
                                "conversation_id": conversation_id,
                            },
                            source="chat",
                            org_id=self.org_id,
                            priority=5,
                        )
                        _log.info("bg: graph extraction queued  conv=%s", conversation_id)
                except Exception:
                    _log.error("bg: graph extraction queue failed  conv=%s", conversation_id, exc_info=True)

            # Background summarisation — run summary so it's ready for
            # the next turn.  Signal via event so next turn can wait if needed.
            summary_ev = _get_summary_event(conversation_id)
            summary_ev.clear()  # mark as running
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
                            existing_msgs = self.db.list_messages(conversation_id)
                            existing_id = None
                            for msg in existing_msgs:
                                if msg.get("role") == "system" and "[Conversation summary]" in (msg.get("content") or ""):
                                    existing_id = msg.get("Id")
                                    break
                            if existing_id:
                                self.db._patch("messages", existing_id, {
                                    "Id": existing_id,
                                    "content": summary_content,
                                })
                            else:
                                self.db.add_message(
                                    conversation_id=conversation_id,
                                    org_id=self.org_id,
                                    role="system",
                                    content=summary_content,
                                    model="summariser",
                                )
                        except Exception:
                            _log.debug("bg: summary persist failed  conv=%s", conversation_id, exc_info=True)
                    _log.info("bg: summarise done  conv=%s topics=%s %.2fs",
                              conversation_id, topics, time.perf_counter() - _t)
            except Exception:
                _log.error("bg: summarise failed  conv=%s", conversation_id, exc_info=True)
            finally:
                summary_ev.set()  # mark as done, unblock any waiting turn

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
                    # Model is still thinking — stream to UI as thinking event
                    think_tokens += 1
                    reasoning_chunks.append(reasoning)
                    emit({"type": "thinking", "text": reasoning})
                    continue

                if not text:
                    continue

                # Filter <think>...</think> tags in content (Qwen/RWKV style)
                if not first_content_emitted and "<think>" in text:
                    in_think_block = True
                    think_tokens += 1
                    # Strip the <think> tag, emit remaining text as thinking
                    after_tag = text.split("<think>", 1)[1] if "<think>" in text else ""
                    if after_tag:
                        reasoning_chunks.append(after_tag)
                        emit({"type": "thinking", "text": after_tag})
                    continue
                if in_think_block:
                    think_tokens += 1
                    if "</think>" in text:
                        in_think_block = False
                        # Emit any text before the closing tag as thinking
                        before_tag = text.split("</think>", 1)[0]
                        if before_tag:
                            reasoning_chunks.append(before_tag)
                            emit({"type": "thinking", "text": before_tag})
                        # Emit any text after the closing tag as content
                        after_tag = text.split("</think>", 1)[1].strip()
                        if after_tag:
                            first_content_emitted = True
                            chunks.append(after_tag)
                            emit({"type": "chunk", "text": after_tag})
                        if think_tokens > 1:
                            _log.info("model thinking done  tokens=%d", think_tokens)
                    else:
                        reasoning_chunks.append(text)
                        emit({"type": "thinking", "text": text})
                    continue

                first_content_emitted = True
                chunks.append(text)
                emit({"type": "chunk", "text": text})

        # Fallback: if model put everything in thinking (reasoning_content field
        # OR <think> tags) and never produced actual content, use the reasoning
        # as the output.  Better than losing the response entirely.
        if not chunks and reasoning_chunks:
            _log.warning("model returned no content, using thinking as fallback  think_tokens=%d reasoning_chars=%d",
                         think_tokens, sum(len(r) for r in reasoning_chunks))
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
