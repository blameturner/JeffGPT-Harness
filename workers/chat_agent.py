import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Iterator

import requests

from config import MODELS, get_model_url, refresh_models
from nocodb_client import NocodbClient
from rag import retrieve
from memory import remember
from graph import write_relationship
from workers.web_search import run_web_search, needs_web_search
from workers.styles import chat_style_prompt


# --- Summarisation constants -----------------------------------------------
# Cap for how much history we feed into the fast model for summarisation.
# The fast model runs with a 16384-token context; ~48k chars is a safe cap.
MAX_SUMMARY_INPUT_CHARS = 48_000

# Rough char-budget before we trigger summarisation. Assume a chat model
# window of ~32k tokens ≈ 128k chars; trigger at 80%.
SUMMARISE_TRIGGER_CHARS = int(128_000 * 0.8)

# Fallback when summarisation fails: keep this many most-recent messages.
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

    # --- helpers -----------------------------------------------------------

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

    def _fast_model_url(self) -> tuple[str | None, str | None]:
        """Return (url, model_id) for the configured 'fast' role, or (None, None)."""
        entry = MODELS.get("fast")
        if not isinstance(entry, dict):
            return None, None
        return entry.get("url"), entry.get("model_id") or "fast"

    # --- summarisation -----------------------------------------------------

    def _maybe_summarise(self, history: list[dict]) -> tuple[list[dict], dict | None]:
        """If history is too large, replace the oldest portion with a summary.

        Returns (new_history, event_dict_or_None). The event should be yielded
        to the client as a `summarised` SSE event when not None.
        """
        total_chars = sum(len(m.get("content") or "") for m in history)
        if total_chars <= SUMMARISE_TRIGGER_CHARS:
            return history, None

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

        fast_url, fast_model = self._fast_model_url()
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
                print(f"[chat] summarisation failed, falling back: {e}")
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

        # Fallback: just trim to most recent messages, never error.
        trimmed = history[-FALLBACK_RECENT_MESSAGES:]
        return trimmed, {
            "type": "summarised",
            "removed": max(0, len(history) - len(trimmed)),
            "summary_chars": 0,
            "fallback": True,
        }

    # --- knowledge graph ---------------------------------------------------

    def _extract_and_write_graph(
        self,
        user_text: str,
        assistant_text: str,
        conversation_id: int,
    ) -> None:
        """Fast-model entity extraction + FalkorDB write. Fire-and-forget.

        Safe when the chat model IS the fast model — this runs in a daemon
        thread and does not stream, so there is no recursion risk.
        """
        fast_url, fast_model = self._fast_model_url()
        if not fast_url:
            return

        prompt = (
            "Extract concepts and relations from this chat turn. "
            "Return ONLY JSON of the form:\n"
            '{"entities": ["name", ...], '
            '"relations": [["a", "REL", "b"], ...]}\n'
            "Use short UPPER_SNAKE relation names. No prose.\n\n"
            f"USER: {user_text}\n\nASSISTANT: {assistant_text}"
        )

        try:
            resp = requests.post(
                f"{fast_url}/v1/chat/completions",
                json={
                    "model": fast_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 400,
                },
                timeout=60,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?", "", raw).rstrip("`").strip()
            data = json.loads(raw)
        except Exception as e:
            print(f"[chat] graph extraction failed: {e}")
            return

        relations = data.get("relations") or []
        for row in relations:
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                continue
            a, rel, b = row
            if not (isinstance(a, str) and isinstance(rel, str) and isinstance(b, str)):
                continue
            rel_safe = re.sub(r"[^A-Z0-9_]", "_", rel.upper()) or "RELATED_TO"
            try:
                write_relationship(
                    org_id=self.org_id,
                    from_type="Concept",
                    from_name=a[:200],
                    relationship=rel_safe,
                    to_type="Concept",
                    to_name=b[:200],
                )
            except Exception as e:
                print(f"[chat] graph write failed ({a}-{rel}->{b}): {e}")

    # --- streaming send ----------------------------------------------------

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
        """Yield SSE-ready event dicts for a single chat turn.

        Events:
          {"type": "meta", "conversation_id": int}
          {"type": "summarised", ...}
          {"type": "chunk", "text": str}
          {"type": "done", "tokens_input": int, "tokens_output": int, ...}
          {"type": "error", "message": str}
        """
        if conversation_id is None:
            convo = self.db.create_conversation(
                org_id=self.org_id,
                model=self.model,
                title=user_message[:80],
                rag_enabled=bool(rag_enabled),
                rag_collection=rag_collection or "",
            )
            conversation_id = convo["Id"]
            history: list[dict] = []
        else:
            convo = self.db.get_conversation(conversation_id)
            if not convo:
                yield {"type": "error", "message": f"Conversation {conversation_id} not found"}
                return
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in self.db.list_messages(conversation_id)
            ]

        convo_rag_enabled = self._truthy(convo.get("rag_enabled"))
        collection_name = (
            (convo.get("rag_collection") or "").strip()
            or self._default_collection(conversation_id)
        )
        convo_knowledge = self._truthy(convo.get("knowledge_enabled")) or bool(knowledge_enabled)

        yield {"type": "meta", "conversation_id": conversation_id}

        search_context = ""
        search_sources: list[str] = []
        search_note = ""
        search_errored = False
        if self.search_enabled:
            yield {"type": "searching"}
            try:
                search_context, search_sources = run_web_search(user_message, self.org_id)
            except Exception as e:
                print(f"[chat] web search failed: {e}")
                search_context, search_sources = "", []
                search_errored = True
            yield {
                "type": "search_complete",
                "source_count": len(search_sources),
                "sources": search_sources,
                "ok": bool(search_sources),
            }
            if not search_sources:
                search_note = (
                    "SEARCH STATUS: You attempted a live web search for this "
                    "question but it returned no usable results "
                    + ("(the search backend errored)." if search_errored
                       else "(all candidate pages failed to fetch or were irrelevant).")
                    + " In your reply, clearly tell the user that you tried "
                    "to search and couldn't retrieve live results, then "
                    "answer from general knowledge with an explicit caveat "
                    "about recency. Do NOT claim you lack the ability to "
                    "search — you have it, it just failed this time."
                )
        elif not search_consent_declined:
            try:
                needs, reason = needs_web_search(user_message)
            except Exception as e:
                print(f"[chat] needs_web_search failed: {e}")
                needs, reason = False, ""
            if needs:
                yield {
                    "type": "search_consent_required",
                    "query": user_message,
                    "reason": reason or "question appears to need live information",
                }
                yield {
                    "type": "done",
                    "conversation_id": conversation_id,
                    "awaiting": "search_consent",
                    "model": self.model,
                    "tokens_input": 0,
                    "tokens_output": 0,
                    "duration_seconds": 0.0,
                    "rag_enabled": False,
                    "context_chars": 0,
                }
                return
        else:
            search_note = (
                "SEARCH STATUS: The user declined a live web search for this "
                "question. Answer from general knowledge and explicitly flag "
                "that anything time-sensitive may be out of date. Do NOT "
                "claim you lack the ability to search — the user chose not "
                "to allow it this turn."
            )

        style_key, style_prompt = chat_style_prompt(response_style)

        try:
            self.db.add_message(
                conversation_id=conversation_id,
                org_id=self.org_id,
                role="user",
                content=user_message,
                model=self.model,
                response_style=style_key,
            )
        except Exception as e:
            print(f"[chat] user message persist failed: {e}")

        rag_context = ""
        if convo_rag_enabled:
            try:
                rag_context = retrieve(
                    query=user_message,
                    org_id=self.org_id,
                    collection_name=collection_name,
                    n_results=10,
                    top_k=3,
                )
            except Exception as e:
                print(f"[chat] RAG retrieval failed: {e}")
                rag_context = ""

        history, summary_event = self._maybe_summarise(history)
        if summary_event:
            yield summary_event

        payload: list[dict] = []
        payload.append({"role": "system", "content": style_prompt})
        if system:
            payload.append({"role": "system", "content": system})
        if search_context:
            payload.append({
                "role": "system",
                "content": search_context,
            })
        if search_note:
            payload.append({
                "role": "system",
                "content": search_note,
            })
        if rag_context:
            payload.append({
                "role": "system",
                "content": (
                    "The following context was retrieved from this "
                    "conversation's memory. Use it where relevant.\n\n"
                    f"{rag_context}"
                ),
            })
        payload.extend(history)
        payload.append({"role": "user", "content": user_message})

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
        output = "".join(accumulated)
        tokens_input = int(final_usage.get("prompt_tokens") or 0)
        tokens_output = int(final_usage.get("completion_tokens") or 0)

        try:
            self.db.add_message(
                conversation_id=conversation_id,
                org_id=self.org_id,
                role="assistant",
                content=output,
                model=str(final_model),
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                response_style=style_key,
            )
        except Exception as e:
            print(f"[chat] assistant message persist failed: {e}")

        if convo_rag_enabled:
            try:
                remember(
                    text=f"USER: {user_message}\n\nASSISTANT: {output}",
                    metadata={
                        "conversation_id": conversation_id,
                        "model": self.model,
                        "turn_time": time.time(),
                    },
                    org_id=self.org_id,
                    collection_name=collection_name,
                )
            except Exception as e:
                print(f"[chat] memory write failed: {e}")

        if convo_knowledge:
            try:
                remember(
                    text=f"USER: {user_message}\n\nASSISTANT: {output}",
                    metadata={
                        "conversation_id": conversation_id,
                        "model": self.model,
                        "turn_time": time.time(),
                    },
                    org_id=self.org_id,
                    collection_name="chat_knowledge",
                )
            except Exception as e:
                print(f"[chat] chat_knowledge write failed: {e}")

            threading.Thread(
                target=self._extract_and_write_graph,
                args=(user_message, output, conversation_id),
                daemon=True,
            ).start()

        yield {
            "type": "done",
            "conversation_id": conversation_id,
            "model": str(final_model),
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "duration_seconds": duration,
            "rag_enabled": convo_rag_enabled,
            "context_chars": len(rag_context),
        }

    # --- model call (streaming) --------------------------------------------

    def _call_model_streaming(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> Iterator[dict]:
        """Stream from llama.cpp-compatible chat-completions endpoint."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        final_usage: dict = {}
        final_model: str = self.model
        try:
            with requests.post(
                f"{self.url}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=600,
            ) as response:
                response.raise_for_status()
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
                            yield {"type": "chunk", "text": text}
        except Exception as e:
            yield {"type": "error", "message": str(e)}
            return

        yield {"type": "done", "usage": final_usage, "model": final_model}

    # --- blocking wrapper (kept for any internal sync callers) -------------

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
