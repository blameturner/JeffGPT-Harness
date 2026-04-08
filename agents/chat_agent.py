import time
import requests
from dataclasses import dataclass
from config import MODELS, get_model_url, refresh_models
from nocodb_client import NocodbClient
from rag import retrieve
from memory import remember


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
    """Generic chat agent — ChatGPT/Claude style.

    Conversation history is persisted to NocoDB (conversations + messages).
    If the conversation row has rag_enabled=1, each turn also retrieves from
    a per-conversation Chroma collection and writes user+assistant turns
    back to memory — giving the chat a growing, searchable memory.
    """

    def __init__(self, model: str, org_id: int):
        url = get_model_url(model)
        if not url:
            # Lazy re-discover in case model containers came up after the harness.
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
        self.db = NocodbClient()

    @staticmethod
    def _default_collection(conversation_id: int) -> str:
        return f"chat_{conversation_id}"

    @staticmethod
    def _truthy(value) -> bool:
        # NocoDB checkbox columns can come back as bool, 0/1, or "true"/"false".
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return False

    def _call_model(self, messages: list[dict], temperature: float, max_tokens: int) -> dict:
        response = requests.post(
            f"{self.url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    def send(
        self,
        user_message: str,
        conversation_id: int | None = None,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        rag_enabled: bool | None = None,
        rag_collection: str | None = None,
    ) -> ChatResult:
        # --- Load or create the conversation ---
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
                raise ValueError(f"Conversation {conversation_id} not found")
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in self.db.list_messages(conversation_id)
            ]

        convo_rag_enabled = self._truthy(convo.get("rag_enabled"))
        collection_name = (
            (convo.get("rag_collection") or "").strip()
            or self._default_collection(conversation_id)
        )

        # --- Persist the incoming user message up-front ---
        self.db.add_message(
            conversation_id=conversation_id,
            org_id=self.org_id,
            role="user",
            content=user_message,
            model=self.model,
        )

        # --- Optional RAG retrieval ---
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
                # Don't let RAG failures break the chat — log and continue.
                print(f"[chat] RAG retrieval failed: {e}")
                rag_context = ""

        # --- Build the model payload ---
        payload: list[dict] = []
        if system:
            payload.append({"role": "system", "content": system})
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
        data = self._call_model(payload, temperature, max_tokens)
        duration = round(time.time() - start, 2)

        output = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        tokens_input = usage.get("prompt_tokens", 0)
        tokens_output = usage.get("completion_tokens", 0)

        # --- Persist the assistant reply ---
        self.db.add_message(
            conversation_id=conversation_id,
            org_id=self.org_id,
            role="assistant",
            content=output,
            model=str(data.get("model", self.model)),
            tokens_input=tokens_input,
            tokens_output=tokens_output,
        )

        # --- Write this turn to memory for future RAG ---
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

        return ChatResult(
            output=output,
            model=str(data.get("model", self.model)),
            conversation_id=conversation_id,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            duration_seconds=duration,
            rag_enabled=convo_rag_enabled,
            context_chars=len(rag_context),
        )
