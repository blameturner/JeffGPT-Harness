from __future__ import annotations

import json
import logging

from config import MAX_SUMMARY_INPUT_CHARS
from workers.enrichment.models import _fast_call

_log = logging.getLogger("enrichment_agent.summarise")


def _summarise(text: str) -> tuple[str, int]:
    _log.debug("summarise  text_len=%d", len(text))
    prompt = (
        "Summarise the following page for a knowledge base. Be factual, "
        "≤ 250 words, preserve key names, numbers, dates, and verbatim "
        "quotes where notable.\n\n"
        f"PAGE:\n{text[:MAX_SUMMARY_INPUT_CHARS]}"
    )
    summary, tokens = _fast_call(prompt, max_tokens=300)
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
