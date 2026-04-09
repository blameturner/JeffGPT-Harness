from __future__ import annotations

import base64
import json
import re
import time
from typing import Iterator, Literal

import requests

from memory import remember
from rag import retrieve
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
    """Normalise an incoming files payload to plain {name, content} for
    persistence. Accepts either base64 (`content_b64`) or plain (`content`)."""
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
    """Fast-model parse of a plan into an ordered checklist of step strings.
    Returns [] on failure — caller should tolerate."""
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
        print(f"[code] plan checklist parse failed: {e}")
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
        """Pull the most recent non-empty files_json from this conversation's
        user messages. Used to re-hydrate the file workspace on follow-up
        turns so the client doesn't need to resend attachments."""
        try:
            msgs = self.db.list_code_messages(conversation_id)
        except Exception as e:
            print(f"[code] workspace load failed: {e}")
            return []
        for m in reversed(msgs):
            if m.get("role") != "user":
                continue
            raw = m.get("files_json")
            if not raw:
                continue
            # NocoDB JSON columns return a list/dict directly; older rows or
            # string-encoded values need parsing.
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

    def run_streaming(
        self,
        user_message: str,
        conversation_id: int | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        title: str | None = None,
        codebase_collection: str | None = None,
    ) -> Iterator[dict]:
        system_prompt = _SYSTEMS[self.mode]

        # --- load or create code conversation ------------------------------
        history: list[dict] = []
        is_new = False
        if conversation_id is None:
            try:
                convo = self.db.create_code_conversation(
                    org_id=self.org_id,
                    model=self.model,
                    title=(title or user_message)[:80],
                    mode=self.mode,
                )
                conversation_id = convo["Id"]
                is_new = True
            except Exception as e:
                print(f"[code] create_code_conversation failed: {e}")
        else:
            try:
                convo = self.db.get_code_conversation(conversation_id)
                if not convo:
                    yield {"type": "error", "message": f"Code conversation {conversation_id} not found"}
                    return
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in self.db.list_code_messages(conversation_id)
                ]
                # If caller sent no files, re-hydrate the prior workspace.
                if not self.files:
                    self.files = self._load_workspace(conversation_id)
            except Exception as e:
                print(f"[code] load history failed: {e}")

        # Patch the conversation's stored mode on every turn so the sidebar
        # badge reflects the latest mode in a switched session.
        if conversation_id is not None and not is_new:
            try:
                self.db.update_code_conversation(
                    conversation_id, {"rag_collection": self.mode}
                )
            except Exception as e:
                print(f"[code] mode patch failed: {e}")

        yield {"type": "meta", "mode": self.mode, "conversation_id": conversation_id}

        # --- build the composed user message -------------------------------
        files_block = _render_files_block(self.files)
        pieces: list[str] = []
        if files_block:
            pieces.append(files_block)
        if self.mode == "execute" and self.approved_plan:
            pieces.append(f"<approved_plan>\n{self.approved_plan}\n</approved_plan>")
        pieces.append(user_message)
        composed_message = "\n\n".join(pieces)

        # --- optional: retrieve from a codebase RAG collection -------------
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
            except Exception as e:
                print(f"[code] codebase retrieve failed: {e}")
                codebase_context = ""

        # --- persist user turn ---------------------------------------------
        storage_files = _files_to_storage(self.files)
        if conversation_id is not None:
            try:
                self.db.add_code_message(
                    conversation_id=conversation_id,
                    org_id=self.org_id,
                    role="user",
                    content=user_message,  # store the raw user text, not the composed blob
                    model=self.model,
                    mode=self.mode,
                    files_json=storage_files or None,
                )
            except Exception as e:
                print(f"[code] user message persist failed: {e}")

        payload: list[dict] = [{"role": "system", "content": system_prompt}]
        if codebase_context:
            payload.append({
                "role": "system",
                "content": (
                    "The following snippets were retrieved from the project "
                    "codebase. Use them where relevant and prefer them over "
                    "guessing.\n\n" + codebase_context
                ),
            })
        payload.extend(history)
        payload.append({"role": "user", "content": composed_message})

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

        # --- persist assistant turn ----------------------------------------
        if conversation_id is not None:
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
                )
            except Exception as e:
                print(f"[code] assistant message persist failed: {e}")

        # --- save to semantic memory (Chroma), mirroring chat RAG ---------
        if conversation_id is not None:
            try:
                remember(
                    text=f"USER ({self.mode}): {user_message}\n\nASSISTANT: {output}",
                    metadata={
                        "conversation_id": conversation_id,
                        "model": str(final_model),
                        "mode": self.mode,
                        "turn_time": time.time(),
                    },
                    org_id=self.org_id,
                    collection_name=f"code_{conversation_id}",
                )
            except Exception as e:
                print(f"[code] memory write failed: {e}")

        # --- plan → checklist (plan mode only) -----------------------------
        if self.mode == "plan":
            fast_url, fast_model = self._fast_model_url()
            if fast_url:
                steps = _parse_plan_checklist(output, fast_url, fast_model)
                if steps:
                    if conversation_id is not None:
                        try:
                            self.db.update_code_conversation(
                                conversation_id, {"code_checklist": steps}
                            )
                        except Exception as e:
                            print(f"[code] checklist persist failed: {e}")
                    yield {"type": "plan_checklist", "steps": steps}

        yield {
            "type": "done",
            "mode": self.mode,
            "conversation_id": conversation_id,
            "model": str(final_model),
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "duration_seconds": duration,
            "output": output,
        }
