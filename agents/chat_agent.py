import time
import requests
from dataclasses import dataclass
from config import MODELS, get_model_url, refresh_models
from nocodb_client import NocodbClient


@dataclass
class ChatResult:
    output: str
    model: str
    conversation_id: int
    tokens_input: int
    tokens_output: int
    duration_seconds: float


class ChatAgent:
    """Generic chat agent — no RAG, no persona config. ChatGPT/Claude style.

    Model is selected per-request by the caller (frontend). Conversation
    history is persisted to NocoDB (conversations + messages tables).
    """

    def __init__(self, model: str, org_id: int):
        url = get_model_url(model)
        if not url:
            # Lazy re-discover in case the model containers came up after the harness did.
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
    ) -> ChatResult:
        # Load or create conversation
        if conversation_id is None:
            convo = self.db.create_conversation(
                org_id=self.org_id,
                model=self.model,
                title=user_message[:80],
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

        # Persist the incoming user message
        self.db.add_message(
            conversation_id=conversation_id,
            org_id=self.org_id,
            role="user",
            content=user_message,
            model=self.model,
        )

        # Build payload
        payload: list[dict] = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend(history)
        payload.append({"role": "user", "content": user_message})

        start = time.time()
        data = self._call_model(payload, temperature, max_tokens)
        duration = round(time.time() - start, 2)

        output = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        tokens_input = usage.get("prompt_tokens", 0)
        tokens_output = usage.get("completion_tokens", 0)

        # Persist assistant reply
        self.db.add_message(
            conversation_id=conversation_id,
            org_id=self.org_id,
            role="assistant",
            content=output,
            model=str(data.get("model", self.model)),
            tokens_input=tokens_input,
            tokens_output=tokens_output,
        )

        return ChatResult(
            output=output,
            model=str(data.get("model", self.model)),
            conversation_id=conversation_id,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            duration_seconds=duration,
        )
