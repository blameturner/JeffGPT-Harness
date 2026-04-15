import json
import logging
from typing import Any
from infra.config import get_feature
from shared.models import model_call

_log = logging.getLogger("research.critic")

DEFAULT_CONFIDENCE_THRESHOLD = 80


def analyze_gaps(topic: str, content: str, schema: dict, context: str = "") -> dict:
    prompt = f"""You are a Research Critic. Analyze the draft research content for gaps and sufficiency.

TOPIC: {topic}
SCHEMA (data to extract): {json.dumps(schema)}

DRAFT CONTENT:
{content[:20000]}

CONTEXT (additional):
{context[:5000] if context else "No additional context."}

Respond with ONLY valid JSON (no explanation):
{{
  "gaps_found": [
    {{"field": "field_name", "status": "missing|insufficient|superficial", "needed": "specific detail needed"}}
  ],
  "new_search_requirements": ["specific query to fill gap 1", "specific query to fill gap 2"],
  "confidence_score": 0-100,
  "ready_for_completion": true|false,
  "notes": "overall assessment"
}}"""

    try:
        result, _ = model_call("research_agent", prompt, max_tokens=1000, temperature=0.2)
        return json.loads(result)
    except json.JSONDecodeError as e:
        _log.warning("critic parse failed  topic=%s  error=%s", topic[:40], e)
        return {
            "gaps_found": [],
            "new_search_requirements": [],
            "confidence_score": 50,
            "ready_for_completion": True,
            "notes": f"Parse error: {e}"
        }
    except Exception as e:
        _log.warning("critic failed  topic=%s  error=%s", topic[:40], e)
        return {
            "gaps_found": [],
            "new_search_requirements": [],
            "confidence_score": 0,
            "ready_for_completion": False,
            "notes": f"Error: {e}"
        }


def get_confidence_threshold() -> int:
    return get_feature("research", "confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)