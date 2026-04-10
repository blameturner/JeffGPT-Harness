import json
import logging
from typing import Iterator
from pydantic import ValidationError
from agent import Agent, RunResult
from schemas.base import AgentOutput

_log = logging.getLogger("generator")

class GeneratorAgent(Agent):

    RESPONSE_FORMAT = AgentOutput.prompt_template()

    def __init__(self, agent_name: str, org_id: int):
        super().__init__(agent_name, org_id)

    def _build_prompt(self, task: str, context: str) -> list[dict]:
        messages = super()._build_prompt(task, context)

        messages[-1]["content"] = (
            messages[-1]["content"] + "\n\n" + self.RESPONSE_FORMAT
        )

        return messages

    def _parse_response(self, raw: str) -> AgentOutput | None:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("JSON"):
                    cleaned = cleaned[4:]

            data = json.loads(cleaned)
            return AgentOutput(**data)

        except json.JSONDecodeError as e:
            _log.warning("JSON parse error: %s", e)
            return None

        except ValidationError as e:
            _log.warning("validation error: %s", e)
            return None

    def run(self, task: str, product: str = "") -> AgentOutput | None:
        result = super().run(task, product)

        if result is None:
            return None

        parsed = self._parse_response(result.output)

        if parsed is None:
            _log.warning("failed to parse structured output for task: %s", task[:100])
            return None

        return parsed

    def run_streaming(self, task: str, product: str = "") -> Iterator[dict]:
        accumulated: list[str] = []
        for event in super().run_streaming(task, product):
            etype = event.get("type")
            if etype == "chunk":
                accumulated.append(event["text"])
                yield event
            elif etype == "error":
                yield event
                return
            elif etype == "done":
                raw = event.get("output") or "".join(accumulated)
                parsed = self._parse_response(raw)
                yield event
                yield {
                    "type": "parsed",
                    "output": parsed.model_dump() if parsed else None,
                    "parse_ok": parsed is not None,
                }
