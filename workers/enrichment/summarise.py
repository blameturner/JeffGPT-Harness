from __future__ import annotations

import json
import logging

from config import get_function_config
from workers.enrichment.models import model_call

_log = logging.getLogger("enrichment_agent.summarise")


def _summarise(text: str) -> tuple[str, int]:
    cfg = get_function_config("enrichment_summarise")
    max_input = cfg.get("max_input_chars", 12000)
    _log.debug("summarise  text_len=%d max_input=%d", len(text), max_input)
    prompt = (
        "Summarise the following page for a knowledge base. Be factual, "
        "≤ 250 words, preserve key names, numbers, dates, and verbatim "
        "quotes where notable.\n\n"
        f"PAGE:\n{text[:max_input]}"
    )
    summary, tokens = model_call("enrichment_summarise", prompt)
    if summary:
        _log.info("summarise ok  summary_len=%d tokens=%d", len(summary), tokens)
    else:
        _log.warning("summarise returned empty  tokens=%d", tokens)
    return summary, tokens


def _salvage_json_array(text: str) -> list | None:
    # truncated JSON from token limits — find last complete object, close the array
    last_close = text.rfind("}")
    if last_close == -1:
        return None
    truncated = text[:last_close + 1].rstrip(", \n\t") + "]"
    if not truncated.startswith("["):
        truncated = "[" + truncated
    try:
        result = json.loads(truncated)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    return None
