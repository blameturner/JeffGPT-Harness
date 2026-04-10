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
        _log.info("init  agent=%s org=%d", agent_name, org_id)
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
            _log.warning("JSON parse error: %s  raw=%s", e, raw[:200])
            return None

        except ValidationError as e:
            _log.warning("validation error: %s", e)
            return None

    def run(self, task: str, product: str = "") -> AgentOutput | None:
        _log.info("run  task=%s product=%s", task[:100], product[:50])
        result = super().run(task, product)

        if result is None:
            _log.warning("run returned None for task: %s", task[:100])
            return None

        parsed = self._parse_response(result.output)

        if parsed is None:
            _log.warning("failed to parse structured output for task: %s", task[:100])
            return None

        _log.info("run ok  task=%s", task[:100])
        return parsed

    def run_streaming(self, task: str, product: str = "") -> Iterator[dict]:
        _log.info("run_streaming  task=%s product=%s", task[:100], product[:50])
        accumulated: list[str] = []
        for event in super().run_streaming(task, product):
            etype = event.get("type")
            if etype == "chunk":
                accumulated.append(event["text"])
                yield event
            elif etype == "error":
                _log.error("run_streaming error: %s", event.get("message"))
                yield event
                return
            elif etype == "done":
                raw = event.get("output") or "".join(accumulated)
                parsed = self._parse_response(raw)
                _log.info("run_streaming done  parse_ok=%s chars=%d", parsed is not None, len(raw))
                yield event
                yield {
                    "type": "parsed",
                    "output": parsed.model_dump() if parsed else None,
                    "parse_ok": parsed is not None,
                }
