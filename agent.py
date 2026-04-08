import json
import time
import requests
from dataclasses import dataclass
from config import MODELS
from rag import retrieve
from memory import remember
from nocodb_client import NocodbClient


@dataclass
class RunResult:
    output: str
    tokens_input: int
    tokens_output: int
    context_tokens: int
    duration_seconds: float
    model_name: str


class Agent:
    def __init__(self, agent_name: str, org_id: int):
        self.agent_name = agent_name
        self.org_id = org_id
        self.db = NocodbClient()

        self.config = self.db.get_agent(agent_name, org_id)
        if not self.config:
            raise ValueError(f"Agent {agent_name} not found on org {org_id}")
        assert self.config is not None

    def _get_model_url(self) -> str:
        model_key = self.config["model"].lower()
        url = MODELS.get(model_key)
        if not url:
            raise ValueError(f"Model {model_key} not found in variables - add MODEL_{model_key.upper()}=url")
        return url

    def _build_prompt(self, task: str, context: str) -> list[dict]:
        persona = self.config.get("persona", "")
        template = self.config.get("system_prompt_template", "")

        import datetime
        system_prompt = persona
        if template:
            filled_template = template.format(
                task=task,
                date=datetime.date.today().isoformat(),
                products=self.config.get("products") or "",
            )
            system_prompt = f"{system_prompt}\n\n{filled_template}"

        user_message = task
        if context:
            user_message = f"{context}\n\n---\n\nTASK:\n{task}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

    def _call_model(self, messages: list[dict]) -> dict:
        url = self._get_model_url()

        response = requests.post(
            f"{url}/v1/chat/completions",
            json={
                "model": self.config["model"],
                "messages": messages,
                "temperature": self.config.get("temperature", 0.7),
                "max_tokens": self.config.get("max_tokens", 1000)
            },
            timeout=300
        )
        response.raise_for_status()
        return response.json()

    def run(self, task: str, product: str = "") -> RunResult:
        context = ""
        context_tokens = 0

        # RAG retrieval — only if enabled for this agent
        if self.config.get("rag_enabled"):
            context = retrieve(
                query=task,
                org_id=self.org_id,
                collection_name=self.config.get("rag_collection", "agent_outputs"),
                n_results=self.config.get("rag_n_candidates", 10),
                top_k=self.config.get("rag_top_k", 3),
            )

        # Everything below runs regardless of whether RAG is enabled
        messages = self._build_prompt(task, context)

        start_time = time.time()
        response_data = self._call_model(messages)
        duration_seconds = round(time.time() - start_time, 2)

        output = response_data["choices"][0]["message"]["content"]
        usage = response_data.get("usage", {})
        tokens_input = usage.get("prompt_tokens", 0)
        tokens_output = usage.get("completion_tokens", 0)
        model_name = response_data.get("model", self.config["model"])

        chroma_ids = remember(
            text=output,
            metadata={
                "agent": self.agent_name,
                "product": product,
                "task": task[:200],
            },
            org_id=self.org_id,
            collection_name=self.config.get("rag_collection", "agent_outputs")
        )

        run = self.db.create_run(
            agent=self.config,
            org_id=self.org_id,
            task_description=task,
            product=product
        )

        self.db.complete_run(
            run_id=run["Id"],
            summary=output[:500],
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            context_tokens=context_tokens,
            duration_seconds=duration_seconds,
            quality_score=0,
            model_name= str(response_data.get("model", self.config["model"]))
        )

        self.db.save_output(
            run=run,
            full_text=output,
            chroma_ids=json.dumps(chroma_ids)
        )

        return RunResult(
            output=output,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            context_tokens=context_tokens,
            duration_seconds=duration_seconds,
            model_name= str(response_data.get("model", self.config["model"]))
        )