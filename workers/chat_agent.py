import concurrent.futures
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterator
from zoneinfo import ZoneInfo

import requests

from config import BASE_SYSTEM_PROMPT, CHAT_TIMEZONE, MODELS, get_model_url, refresh_models
from nocodb_client import NocodbClient
from rag import retrieve
from memory import remember
from workers.web_search import (
    run_web_search,
    classify_message_intent,
    _tool_model,
    build_temporal_context,
    CHAT_INTENT_CHITCHAT,
    CHAT_INTENT_CONTEXTUAL,
    SEARCH_POLICY_NONE,
)
from workers.styles import chat_style_prompt

_log = logging.getLogger("chat")


MAX_SUMMARY_INPUT_CHARS = 48_000
SUMMARISE_TRIGGER_CHARS = int(128_000 * 0.8)
FALLBACK_RECENT_MESSAGES = 12


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
    """Generic chat agent — ChatGPT/Claude style, streaming.

    Conversation history is persisted to NocoDB (conversations + messages).
    If the conversation row has rag_enabled=1, each turn retrieves from and
    writes back to a per-conversation Chroma collection. If knowledge_enabled
    is set at creation time, each turn also fires a fast-model entity
    extraction call and writes concept edges to FalkorDB.
    """

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

    def _maybe_summarise(self, history: list[dict]) -> tuple[list[dict], dict | None]:
        """If history is too large, replace the oldest portion with a summary.

        Returns (new_history, event_dict_or_None). The event should be yielded
        to the client as a `summarised` SSE event when not None.
        """
        total_chars = sum(len(m.get("content") or "") for m in history)
        if total_chars <= SUMMARISE_TRIGGER_CHARS:
            return history, None

        _log.info("summarisation triggered  messages=%d chars=%d threshold=%d", len(history), total_chars, SUMMARISE_TRIGGER_CHARS)

        keep_tail = 4
        if len(history) <= keep_tail:
            return history, None

        old = history[:-keep_tail]
        recent = history[-keep_tail:]

        buf: list[str] = []
        used = 0
        for m in old:
            line = f"{m.get('role', 'user').upper()}: {m.get('content') or ''}\n"
            if used + len(line) > MAX_SUMMARY_INPUT_CHARS:
                break
            buf.append(line)
            used += len(line)
        transcript = "".join(buf)

        fast_url, fast_model = self._tool_model_url()
        summary_text: str | None = None

        if fast_url:
            try:
                resp = requests.post(
                    f"{fast_url}/v1/chat/completions",
                    json={
                        "model": fast_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You compress chat history. Given a transcript, "
                                    "produce a concise factual summary (<= 400 words) "
                                    "preserving names, decisions, open questions, and "
                                    "any instructions the user gave. No preamble."
                                ),
                            },
                            {"role": "user", "content": transcript},
                        ],
                        "temperature": 0.2,
                        "max_tokens": 800,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                summary_text = resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                _log.warning("summarisation failed, falling back: %s", e)
                summary_text = None

        if summary_text:
            new_history = [
                {
                    "role": "system",
                    "content": f"Earlier conversation summary:\n{summary_text}",
                }
            ] + recent
            return new_history, {
                "type": "summarised",
                "removed": len(old),
                "summary_chars": len(summary_text),
            }

        trimmed = history[-FALLBACK_RECENT_MESSAGES:]
        return trimmed, {
            "type": "summarised",
            "removed": max(0, len(history) - len(trimmed)),
            "summary_chars": 0,
            "fallback": True,
        }

    def _extract_and_write_graph(
        self,
        user_text: str,
        assistant_text: str,
        conversation_id: int,
    ) -> None:
        """Delegate to the ironclad enrichment extractor for this chat turn.

        Fire-and-forget (caller runs us in a daemon thread). Uses the
        shared ``_extract_relationships`` helper from
        :mod:`workers.enrichment_agent` so the chat path and the
        enrichment path both emit the same high-quality, richly-typed
        relationships into FalkorDB — rather than the chat path's old
        duplicate minimal-prompt + tool-model path which was producing
        shallow "IS_A" triples.

        Deferred import to avoid pulling enrichment_agent at module load.
        """
        _log.info("graph extraction starting  conv=%d", conversation_id)
        try:
            # Deferred import — enrichment_agent is heavy (FalkorDB, NocoDB).
            from workers.enrichment_agent import _extract_relationships
        except Exception as e:
            _log.warning("graph extraction skipped — import failed: %s", e)
            return

        # Combine the user turn and the assistant reply into a single
        # content blob the extractor can reason over. The ironclad
        # prompt handles the rest (specific entities, causal verbs,
        # deep relationships only).
        combined = (
            f"USER TURN:\n{user_text.strip()}\n\n"
            f"ASSISTANT REPLY:\n{assistant_text.strip()}"
        )

        try:
            written, tokens = _extract_relationships(combined, self.org_id)
            _log.info(
                "graph extraction done  conv=%d written=%d tokens=%d",
                conversation_id, written, tokens,
            )
        except Exception:
            _log.warning("graph extraction failed", exc_info=True)

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
            STORE.append(job, event)

        # Phase 0: per-turn timing spans. `_span(name, t)` records elapsed ms
        # from `t` into `spans`, which is dumped as a single log line just
        # before the model call so we can see where pre-model latency went.
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

        # Phase 3: fire-and-forget status=processing. Nothing downstream
        # in this turn reads conversation status, so it's safe to background.
        _t = time.perf_counter()

        def _bg_status_processing():
            try:
                self.db.update_conversation(conversation_id, {"status": "processing"})
            except Exception:
                _log.warning("status update to processing failed  conv=%s", conversation_id)

        threading.Thread(target=_bg_status_processing, daemon=True).start()
        _span("status_processing_ms", _t)

        _log.debug("turn start  conv=%s model=%s org=%d", conversation_id, self.model, self.org_id)
        emit({"type": "meta", "conversation_id": conversation_id})

        # Phase 1: submit RAG retrieval to a per-turn background executor so
        # it overlaps with the search decision/execution that runs next on
        # the foreground. Search stays on the foreground so its SSE events
        # (`searching`, `search_complete`) land in-order relative to `meta`
        # and `chunk`. RAG emits no events, so backgrounding is safe.
        rag_executor: concurrent.futures.ThreadPoolExecutor | None = None
        rag_future: concurrent.futures.Future | None = None
        if convo_rag_enabled:
            rag_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="chat-rag"
            )
            rag_future = rag_executor.submit(
                retrieve,
                query=user_message,
                org_id=self.org_id,
                collection_name=collection_name,
                n_results=10,
                top_k=3,
            )

        _t_search = time.perf_counter()
        search_context = ""
        search_sources: list[dict] = []
        search_confidence = "none"
        search_status = "not_used"
        search_note = ""
        search_errored = False
        intent_dict: dict | None = None

        # --- Intent classification (§2) ---
        # One classifier call drives routing, search budget, response
        # template, and style. Conversation-aware via history. Safe
        # fallback on any failure — the classifier never blocks the
        # turn on its own.
        if not search_consent_declined:
            _t_classifier = time.perf_counter()
            try:
                intent_dict = classify_message_intent(user_message, history=history)
            except Exception:
                _log.warning("classify_message_intent failed", exc_info=True)
                intent_dict = None
            _span("intent_classify_ms", _t_classifier)

            # Respect per-conversation contextual grounding toggle.
            # `contextual_grounding_enabled` defaults to True (opt-in
            # default on). When off, contextual_enrichment intent is
            # downgraded to chitchat and no search fires.
            grounding_ok = self._truthy(
                convo.get("contextual_grounding_enabled", True)
                if convo.get("contextual_grounding_enabled") is not None
                else True
            )
            if (
                intent_dict
                and intent_dict.get("intent") == CHAT_INTENT_CONTEXTUAL
                and not grounding_ok
            ):
                _log.info(
                    "contextual grounding disabled  conv=%s — downgrading to chitchat",
                    conversation_id,
                )
                intent_dict["intent"] = CHAT_INTENT_CHITCHAT
                intent_dict["search_policy"] = SEARCH_POLICY_NONE
                intent_dict["response_template"] = "chitchat_casual"

            if intent_dict:
                emit({
                    "type": "intent_classified",
                    "route": intent_dict.get("route"),
                    "intent": intent_dict.get("intent"),
                    "secondary_intent": intent_dict.get("secondary_intent"),
                    "entities": intent_dict.get("entities") or [],
                    "confidence": intent_dict.get("confidence"),
                })

        # --- Search dispatch ---
        # Unified path: the intent's search_policy decides everything.
        # Unless the user explicitly toggled search on, we still require
        # `not search_consent_declined` for the flow to run at all (same
        # user-facing behaviour as before).
        policy = (intent_dict or {}).get("search_policy", SEARCH_POLICY_NONE)
        confidence = (intent_dict or {}).get("confidence", "low")
        intent_label = (intent_dict or {}).get("intent") or CHAT_INTENT_CHITCHAT

        # Auto-search: user explicitly enabled search, OR classifier is
        # high-confidence that the turn needs search. Either path bypasses
        # the consent prompt and dispatches directly.
        should_auto_search = (
            intent_dict is not None
            and policy != SEARCH_POLICY_NONE
            and (self.search_enabled or confidence == "high")
        )

        # Consent required: search disabled, classifier not high-confidence,
        # user hasn't already declined. Prompts the UI for permission.
        needs_consent = (
            intent_dict is not None
            and policy != SEARCH_POLICY_NONE
            and not self.search_enabled
            and confidence in ("medium", "low")
            and not search_consent_declined
        )

        if should_auto_search:
            _log.info(
                "dispatching search  conv=%s intent=%s policy=%s",
                conversation_id,
                intent_dict.get("intent"),
                intent_dict.get("search_policy"),
            )
            emit({"type": "searching"})
            _t_exec = time.perf_counter()
            try:
                search_context, search_sources, search_confidence = run_web_search(
                    user_message,
                    self.org_id,
                    intent_dict=intent_dict,
                    history=history,
                )
            except Exception:
                _log.error("web search failed", exc_info=True)
                search_context, search_sources, search_confidence = "", [], "failed"
                search_errored = True
            _span("search_execute_ms", _t_exec)
            emit({
                "type": "search_complete",
                "source_count": len(search_sources),
                "sources": search_sources,
                "ok": bool(search_sources),
                "confidence": search_confidence,
            })
            if search_errored:
                search_status = "error"
            elif search_sources:
                search_status = "used"
            elif search_confidence == "deferred":
                search_status = "deferred"
                # contextual enrichment timed out — emit explicit event
                # so the UI can show "still looking..." and potentially
                # pull the data via a follow-up query later.
                emit({
                    "type": "search_deferred",
                    "entities": (intent_dict or {}).get("entities") or [],
                })
            elif search_confidence == "failed":
                search_status = "failed"
                # run_web_search already put the failure-context block
                # into `search_context` with explicit "do not invent"
                # instructions — no additional search_note needed.
            else:
                search_status = "no_results"
            if search_status in ("no_results",):
                search_note = (
                    "SEARCH STATUS: A live web search was performed but the "
                    "search engine returned no results at all"
                    + (" (the search backend errored)." if search_errored
                       else " for this query.")
                    + " Tell the user you searched but found nothing, then "
                    "answer from your own knowledge with a recency caveat. "
                    "Do NOT say you cannot search — you can and did, "
                    "it just returned empty this time."
                )
        elif needs_consent:
            # Classifier thinks search might help but the user hasn't
            # explicitly enabled it — prompt the UI for consent and
            # pause the turn. Same pattern as the pre-refactor flow.
            search_status = "consent_required"
            reason = (
                f"this looks like a {intent_label.replace('_', ' ')} — "
                "may benefit from a web search"
            )
            _log.info(
                "search consent required  conv=%s intent=%s confidence=%s",
                conversation_id, intent_label, confidence,
            )
            try:
                self.db.update_conversation(conversation_id, {"status": "awaiting_consent"})
            except Exception:
                pass
            emit({
                "type": "search_consent_required",
                "query": user_message,
                "reason": reason,
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
            # If we abort here for consent, cancel any backgrounded RAG
            # retrieval so it doesn't keep running after return.
            if rag_future is not None:
                rag_future.cancel()
            if rag_executor is not None:
                rag_executor.shutdown(wait=False)
            return
        elif search_consent_declined:
            search_status = "declined"
            search_note = (
                "SEARCH STATUS: The user declined a live web search for this "
                "question. Answer from general knowledge and explicitly flag "
                "that anything time-sensitive may be out of date. Do NOT "
                "claim you lack the ability to search — the user chose not "
                "to allow it this turn."
            )
        # Anything else (pure chitchat, no intent classified, policy=none):
        # no search, no consent prompt, proceed to synthesis as normal.
        _span("search_total_ms", _t_search)

        _t = time.perf_counter()
        style_key, style_prompt = chat_style_prompt(response_style)
        _span("style_resolve_ms", _t)

        # Phase 3: background the user-message persist. The assistant-message
        # write later on must happen after this lands (NocoDB server-stamps
        # created_at per POST arrival, so reversed POST order would invert
        # `list_messages` on the next turn). We gate the assistant write on
        # `_user_msg_written` — in practice the model call between here and
        # the gate runs for seconds, so the event is always already set.
        _t = time.perf_counter()
        _user_msg_written = threading.Event()

        def _bg_user_msg():
            try:
                self.db.add_message(
                    conversation_id=conversation_id,
                    org_id=self.org_id,
                    role="user",
                    content=user_message,
                    model=self.model,
                    response_style=style_key,
                )
            except Exception:
                _log.error("user message persist failed", exc_info=True)
            finally:
                _user_msg_written.set()

        threading.Thread(target=_bg_user_msg, daemon=True).start()
        _span("user_msg_persist_ms", _t)

        # Phase 1: collect RAG. If the future finished during the search
        # block, this is essentially free. 45s is a safety valve, not a
        # bound — we're overlapping, not capping.
        _t = time.perf_counter()
        rag_context = ""
        if rag_future is not None:
            try:
                rag_context = rag_future.result(timeout=45)
            except Exception:
                _log.error("RAG retrieval failed", exc_info=True)
                rag_context = ""
            finally:
                if rag_executor is not None:
                    rag_executor.shutdown(wait=False)
        _span("rag_retrieve_ms", _t)

        _t = time.perf_counter()
        history, summary_event = self._maybe_summarise(history)
        if summary_event:
            emit(summary_event)
        _span("summarise_ms", _t)

        _t = time.perf_counter()
        payload: list[dict] = []
        payload.append({"role": "system", "content": BASE_SYSTEM_PROMPT})
        # Current date/time — injected fresh every turn so the model reasons
        # from wall-clock time, not its training cutoff. See
        # workers.web_search.build_temporal_context.
        payload.append({"role": "system", "content": build_temporal_context()})
        if system:
            payload.append({"role": "system", "content": system})
        if search_context:
            payload.append({"role": "system", "content": search_context})
        if search_note:
            payload.append({"role": "system", "content": search_note})
        if rag_context:
            payload.append({
                "role": "system",
                "content": (
                    "The following context was retrieved from this "
                    "conversation's memory. Use it where relevant.\n\n"
                    f"{rag_context}"
                ),
            })
        # style last: closest to history without breaking turn alternation.
        # If the model's template supports interleaved system messages, move
        # this append to immediately before the user turn instead.
        payload.append({"role": "system", "content": style_prompt})
        payload.extend(history)
        payload.append({"role": "user", "content": user_message})
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

        if output:
            # Phase 3: ensure the backgrounded user-message POST landed at
            # NocoDB before we POST the assistant message, so server-stamped
            # created_at preserves user→assistant ordering in list_messages.
            if not _user_msg_written.wait(timeout=10.0):
                _log.warning("user message write still pending after 10s  conv=%s", conversation_id)
            # Classification metadata for the message row. The
            # intent_entities and search_queries columns are JSON-typed
            # in NocoDB — pass native Python lists, not json.dumps'd
            # strings. `requests.post(json=payload, ...)` in NocodbClient
            # will serialise them to real JSON arrays on the wire.
            intent_meta: dict = {}
            if intent_dict:
                intent_meta = {
                    "intent": intent_dict.get("intent"),
                    "intent_entities": intent_dict.get("entities") or [],
                    "search_queries": (
                        [s.get("url") for s in search_sources if s.get("url")]
                        if search_sources
                        else []
                    ),
                }
                if search_status == "failed":
                    intent_meta["search_status_reason"] = "reformulation_exhausted"
                elif search_status == "deferred":
                    intent_meta["search_status_reason"] = "contextual_latency_cap"

            try:
                msg_row = self.db.add_message(
                    conversation_id=conversation_id,
                    org_id=self.org_id,
                    role="assistant",
                    content=output,
                    model=str(final_model),
                    tokens_input=tokens_input,
                    tokens_output=tokens_output,
                    response_style=style_key,
                    search_used=bool(search_sources) or search_status in ("used", "no_results", "error"),
                    search_status=search_status,
                    search_confidence=search_confidence,
                    search_source_count=len(search_sources),
                    search_context_text=search_context,
                    **intent_meta,
                )
                _log.info("persisted assistant message  conv=%s chars=%d", conversation_id, len(output))
                if search_sources and msg_row.get("Id"):
                    try:
                        self.db.add_message_search_sources(
                            message_id=msg_row["Id"],
                            conversation_id=conversation_id,
                            org_id=self.org_id,
                            sources=search_sources,
                        )
                    except Exception:
                        _log.error("search sources persist failed  conv=%s", conversation_id, exc_info=True)
            except Exception:
                _log.error("assistant message persist failed  conv=%s", conversation_id, exc_info=True)

        if convo_rag_enabled and output:
            try:
                remember(
                    text=f"USER: {user_message}\n\nASSISTANT: {output}",
                    metadata={"conversation_id": conversation_id, "model": self.model, "turn_time": time.time()},
                    org_id=self.org_id,
                    collection_name=collection_name,
                )
            except Exception:
                _log.error("memory write failed", exc_info=True)

        if convo_knowledge and output:
            try:
                remember(
                    text=f"USER: {user_message}\n\nASSISTANT: {output}",
                    metadata={"conversation_id": conversation_id, "model": self.model, "turn_time": time.time()},
                    org_id=self.org_id,
                    collection_name="chat_knowledge",
                )
            except Exception:
                _log.error("chat_knowledge write failed", exc_info=True)
            self._extract_and_write_graph(user_message, output, conversation_id)

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
        """Sync wrapper — collects events from run_job for callers that need a generator."""
        from workers.jobs import Job, STORE
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
        # Tight loop — reads the full model SSE stream without suspending.
        # Each chunk is pushed to emit() immediately for real-time delivery.
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

        with requests.post(
            f"{self.url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=(10, 600),
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
                if choices:
                    delta = choices[0].get("delta") or {}
                    text = delta.get("content")
                    if text:
                        chunks.append(text)
                        emit({"type": "chunk", "text": text})

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
        """Drain send_streaming into a single ChatResult — sync callers only."""
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
