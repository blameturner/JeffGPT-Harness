"""ChatAgent — orchestrates chat conversation turns.

Inherits from BaseAgent for model calling, streaming, and utilities.
Uses extracted modules for approval handling, background work, and streaming.
"""

import asyncio
import logging
import time
import threading
import uuid
from typing import Callable, Iterator

from infra.config import TOOLS_FRAMEWORK_ENABLED, is_feature_enabled
from infra.memory import remember
from tools.contract import ToolAction, ToolContext, ToolName, ToolPlan
from tools.dispatcher import execute_plan
from tools.gate import gate_check
from tools.planner import generate_plan
from workers.base import BaseAgent, ChatResult, _get_summary_event, SUMMARY_WAIT_TIMEOUT
from tools.search.queries import generate_broad_queries
from shared.styles import chat_style_prompt
from workers.chat.history import maybe_summarise, extract_conversation_topics
from workers.chat.payload import build_chat_payload
from workers.chat.search_phase import SearchPhaseResult, run_search_phase
from workers.chat.rag_phase import submit_rag_future, collect_rag, cancel_rag
from workers.chat.persistence import (
    schedule_status_processing_write,
    schedule_user_message_write,
    persist_assistant_message,
)

_log = logging.getLogger("chat")


class ChatAgent(BaseAgent):
    # __init__, _truthy, _default_collection, _tool_model_url, _call_model,
    # send, _search_mode all inherited from BaseAgent.

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
        from shared.jobs import STORE

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
                if waited:
                    _log.info("chat conv=%s  waited for background summary — ready", conversation_id)
                else:
                    _log.warning("chat conv=%s  background summary wait timed out after %ds — proceeding without", conversation_id, SUMMARY_WAIT_TIMEOUT)
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
        _log.info("chat conv=%s  flags: rag=%s knowledge=%s search=%s", conversation_id, convo_rag_enabled, convo_knowledge, self.search_enabled)

        _t = time.perf_counter()
        schedule_status_processing_write(self.db, conversation_id)
        _span("status_processing_ms", _t)

        _log.info("chat conv=%s  turn start  model=%s org=%d messages=%d", conversation_id, self.model, self.org_id, len(history))
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
            _t_tools = time.perf_counter()

            hints = gate_check(user_message, conversation_context=last_assistant)
            if not web_search_enabled or not self.search_enabled:
                hints.discard("web_search")
            _log.info("chat conv=%s  gate: hints=%s", conversation_id, sorted(hints) or "none")

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

                # Tools that can be dispatched directly without the planner.
                # web_search: heuristic queries (zero model calls).
                # rag_lookup: just a ChromaDB similarity search.
                _DIRECT_TOOLS = {"web_search", "rag_lookup"}

                if hints <= _DIRECT_TOOLS and hints:
                    _log.info("chat conv=%s  search fast-path  tools=%s", conversation_id, sorted(hints))
                    actions: list[ToolAction] = []

                    if "web_search" in hints:
                        convo_topics = extract_conversation_topics(history)
                        if convo_topics:
                            _log.info("chat conv=%s  topics from summary: %s", conversation_id, convo_topics)
                        # If RAG is running, collect it early to extract
                        # context keywords for search query enrichment.
                        if rag_future and not convo_topics:
                            try:
                                early_rag = rag_future.result(timeout=2)
                                if early_rag:
                                    from tools.search.queries import _extract_keywords
                                    rag_kw = _extract_keywords(early_rag[:2000])
                                    msg_kw_lower = {k.lower() for k in _extract_keywords(user_message)}
                                    convo_topics = [k.lower() for k in rag_kw if k.lower() not in msg_kw_lower][:8]
                                    _log.info("chat conv=%s  topics from RAG: %s", conversation_id, convo_topics)
                                else:
                                    _log.info("chat conv=%s  early RAG returned empty — no topic enrichment", conversation_id)
                            except Exception as e:
                                _log.info("chat conv=%s  early RAG unavailable (%s) — proceeding without topics", conversation_id, type(e).__name__)
                        queries = generate_broad_queries(user_message, max_queries=5, conversation_topics=convo_topics)
                        _log.info("chat conv=%s  search queries (%d): %s", conversation_id, len(queries), [q[:60] for q in queries])
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
                            ok = sum(1 for r in tool_context.results if r.ok)
                            _log.info("chat conv=%s  search complete  results=%d/%d elapsed=%.2fs", conversation_id, ok, len(tool_context.results), time.perf_counter() - _t_tools)
                        except Exception:
                            _log.error("chat conv=%s  search failed", conversation_id, exc_info=True)
                            tool_context = ToolContext()
                else:
                    # Full planner path — for tools that need model-generated params.
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

        # Prepend tool-framework results (rag_lookup, etc.) onto
        # search_context so they ride the same system-prompt injection path.
        # web_search results are already mapped onto
        # search_result.search_context — don't duplicate them.
        _ALREADY_MAPPED = {"web_search"}
        if tool_context.results:
            non_web = [r for r in tool_context.results if r.tool.value not in _ALREADY_MAPPED]
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
            search_status=search_status,
        )
        _span("payload_build_ms", _t)

        _span("pre_model_total_ms", _turn_start)
        _log.info(
            "chat conv=%s  pre-model ready  " + " ".join(f"{k}=%dms" for k in spans),
            conversation_id, *spans.values(),
        )

        _log.info("chat conv=%s  sending to model  messages=%d temp=%.1f max_tokens=%d rag_chars=%d search_chars=%d", conversation_id, len(payload), temperature, max_tokens, len(rag_context), len(search_context))
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
        _log.info("chat conv=%s  model response complete  model=%s tokens_in=%d tokens_out=%d duration=%.1fs chars=%d", conversation_id, final_model, tokens_input, tokens_output, duration, len(output))

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
            _log.info("chat conv=%s  post-turn background starting", conversation_id)

            # 1. RAG embedding
            if convo_rag_enabled and output:
                _t = time.perf_counter()
                try:
                    remember(
                        text=f"USER: {user_message}\n\nASSISTANT: {output}",
                        metadata={"conversation_id": conversation_id, "model": self.model, "turn_time": time.time()},
                        org_id=self.org_id,
                        collection_name=collection_name,
                    )
                    _log.info("chat conv=%s  [1/4] RAG embedded to %s  %.2fs", conversation_id, collection_name, time.perf_counter() - _t)
                except Exception:
                    _log.error("chat conv=%s  [1/4] RAG embed FAILED", conversation_id, exc_info=True)

            # 2. Knowledge embedding
            if convo_knowledge and output:
                _t = time.perf_counter()
                try:
                    remember(
                        text=f"USER: {user_message}\n\nASSISTANT: {output}",
                        metadata={"conversation_id": conversation_id, "model": self.model, "turn_time": time.time()},
                        org_id=self.org_id,
                        collection_name="chat_knowledge",
                    )
                    _log.info("chat conv=%s  [2/4] knowledge embedded  %.2fs", conversation_id, time.perf_counter() - _t)
                except Exception:
                    _log.error("chat conv=%s  [2/4] knowledge embed FAILED", conversation_id, exc_info=True)

                # 3. Graph extraction (queued, not inline)
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
                            source="chat",
                            org_id=self.org_id,
                            priority=5,
                        )
                        _log.info("chat conv=%s  [3/4] graph extraction queued  job=%s", conversation_id, job_id)
                except Exception:
                    _log.error("chat conv=%s  [3/4] graph extraction queue FAILED", conversation_id, exc_info=True)

            # 4. Background summarisation — produces summary + topics for next turn
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
                                _log.info("chat conv=%s  [4/4] summary updated  topics=%s  %.2fs", conversation_id, topics, time.perf_counter() - _t)
                            else:
                                self.db.add_message(
                                    conversation_id=conversation_id,
                                    org_id=self.org_id,
                                    role="system",
                                    content=summary_content,
                                    model="summariser",
                                )
                                _log.info("chat conv=%s  [4/4] summary created  topics=%s  %.2fs", conversation_id, topics, time.perf_counter() - _t)
                        except Exception:
                            _log.error("chat conv=%s  [4/4] summary persist FAILED", conversation_id, exc_info=True)
                    else:
                        _log.info("chat conv=%s  [4/4] summary produced but empty — skipped persist", conversation_id)
                elif bg_summary_event and bg_summary_event.get("fallback"):
                    _log.info("chat conv=%s  [4/4] summary skipped — model unavailable, truncation only", conversation_id)
                else:
                    _log.info("chat conv=%s  [4/4] summary skipped — under threshold (%d messages)", conversation_id, len(full_history))
            except Exception:
                _log.error("chat conv=%s  [4/4] summary FAILED", conversation_id, exc_info=True)
            finally:
                summary_ev.set()  # mark as done, unblock any waiting turn

            _log.info("chat conv=%s  post-turn complete  total=%.2fs", conversation_id, time.perf_counter() - _t_bg)

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
        from shared.jobs import Job
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

    # _call_model, send inherited from BaseAgent
