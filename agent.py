import json
import logging
import time
from typing import Iterator
import requests
from dataclasses import dataclass
from config import get_model_url
from rag import retrieve
from memory import remember
from nocodb_client import NocodbClient

_log = logging.getLogger("agent")


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
            _log.error("agent not found  name=%s org=%d", agent_name, org_id)
            raise ValueError(f"Agent {agent_name} not found on org {org_id}")
        _log.info("agent loaded  name=%s org=%d model=%s", agent_name, org_id, self.config.get("model"))

    def _get_model_url(self) -> str:
        model_key = self.config["model"].lower()
        url = get_model_url(model_key)
        if not url:
            raise ValueError(
                f"Model '{model_key}' not found in catalog — "
                f"set MODEL_{model_key.upper()}_URL in the environment"
            )
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

    def _call_model_streaming(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        url: str | None = None,
        model: str | None = None,
    ) -> Iterator[dict]:
        """Stream chat completions from a llama.cpp-compatible server.

        Yields:
          {"type": "chunk", "text": str}
          {"type": "done", "usage": {...}, "model": str}
          {"type": "error", "message": str}  (terminal, on failure)

        Never raises — callers consume until the generator is exhausted.
        """
        url = url or self._get_model_url()
        model = model or self.config["model"]
        temperature = temperature if temperature is not None else self.config.get("temperature", 0.7)
        max_tokens = max_tokens if max_tokens is not None else self.config.get("max_tokens", 1000)

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        final_usage: dict = {}
        final_model: str = model
        try:
            with requests.post(
                f"{url}/v1/chat/completions",
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
                            yield {"type": "chunk", "text": text}
        except Exception as e:
            yield {"type": "error", "message": str(e)}
            return

        yield {"type": "done", "usage": final_usage, "model": final_model}

    def run(self, task: str, product: str = "") -> RunResult:
        _log.info("run  agent=%s task=%s", self.agent_name, task[:100])
        context = ""
        context_tokens = 0

        if self.config.get("rag_enabled"):
            context = retrieve(
                query=task,
                org_id=self.org_id,
                collection_name=self.config.get("rag_collection", "agent_outputs"),
                n_results=self.config.get("rag_n_candidates", 10),
                top_k=self.config.get("rag_top_k", 3),
            )

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
            chroma_ids=chroma_ids,
        )

        return RunResult(
            output=output,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            context_tokens=context_tokens,
            duration_seconds=duration_seconds,
            model_name= str(response_data.get("model", self.config["model"]))
        )

    def run_streaming(self, task: str, product: str = "") -> Iterator[dict]:
        """Streaming variant of run().

        Yields the same event shapes as `_call_model_streaming`, plus
        persists the run to NocoDB + Chroma after the `done` event arrives.
        Token counts come from the final streaming chunk — never zero.
        """
        _log.debug("run_streaming start  agent=%s org=%d", self.config.get("name"), self.org_id)
        context = ""
        context_tokens = 0

        if self.config.get("rag_enabled"):
            try:
                context = retrieve(
                    query=task,
                    org_id=self.org_id,
                    collection_name=self.config.get("rag_collection", "agent_outputs"),
                    n_results=self.config.get("rag_n_candidates", 10),
                    top_k=self.config.get("rag_top_k", 3),
                )
            except Exception as e:
                _log.error("RAG retrieval failed", exc_info=True)
                context = ""

        messages = self._build_prompt(task, context)

        start_time = time.time()
        accumulated: list[str] = []
        final_usage: dict = {}
        final_model: str = self.config["model"]
        errored = False

        for event in self._call_model_streaming(messages):
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

        duration_seconds = round(time.time() - start_time, 2)
        output = "".join(accumulated)
        tokens_input = int(final_usage.get("prompt_tokens") or 0)
        tokens_output = int(final_usage.get("completion_tokens") or 0)
        _log.info("run done     agent=%s model=%s in=%d out=%d %.1fs", self.config.get("name"), final_model, tokens_input, tokens_output, duration_seconds)

        try:
            chroma_ids = remember(
                text=output,
                metadata={
                    "agent": self.agent_name,
                    "product": product,
                    "task": task[:200],
                },
                org_id=self.org_id,
                collection_name=self.config.get("rag_collection", "agent_outputs"),
            )
        except Exception as e:
            _log.error("memory write failed", exc_info=True)
            chroma_ids = []

        try:
            run = self.db.create_run(
                agent=self.config,
                org_id=self.org_id,
                task_description=task,
                product=product,
            )
            self.db.complete_run(
                run_id=run["Id"],
                summary=output[:500],
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                context_tokens=context_tokens,
                duration_seconds=duration_seconds,
                quality_score=0,
                model_name=str(final_model),
            )
            self.db.save_output(
                run=run,
                full_text=output,
                chroma_ids=chroma_ids,
            )
        except Exception as e:
            _log.error("run persistence failed", exc_info=True)

        yield {
            "type": "done",
            "output": output,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "duration_seconds": duration_seconds,
            "model": str(final_model),
        }