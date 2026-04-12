from __future__ import annotations

import logging
import time

import httpx

from config import MODELS, REASONER_ROLE, get_function_config, no_think_params
from workers.search.models import acquire_model, acquire_role

_log = logging.getLogger("enrichment_agent.models")

FAST_TIMEOUT = 3600

# Roles that must never receive enrichment/tool traffic.
_CHAT_ONLY_FUNCTIONS = frozenset({"chat", "code"})


def _assert_not_reasoner(url: str | None, function_name: str) -> None:
    if function_name in _CHAT_ONLY_FUNCTIONS:
        return
    if not url:
        return
    reasoner_entry = MODELS.get(REASONER_ROLE)
    if not isinstance(reasoner_entry, dict):
        return
    reasoner_url = reasoner_entry.get("url")
    if reasoner_url and url == reasoner_url:
        raise RuntimeError(
            f"refusing to dispatch {function_name} call to reasoner "
            f"(url={url}). Non-chat functions must not use the reasoner."
        )


def _raw_model_call(
    label: str,
    url: str,
    model_id: str | None,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, int]:
    """Low-level model call. Slot acquisition is the caller's responsibility."""
    started = time.time()
    _log.info(
        "%s start  url=%s model=%s prompt_len=%d max_tokens=%d",
        label, url, model_id, len(prompt), max_tokens,
    )
    try:
        r = httpx.post(
            f"{url}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                **no_think_params(),
            },
            timeout=FAST_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        text = (msg.get("content") or "").strip()
        if not text and msg.get("reasoning_content"):
            text = msg["reasoning_content"].strip()
            _log.warning("%s content empty, using reasoning_content as fallback", label)
        usage = data.get("usage") or {}
        tokens = int(usage.get("total_tokens") or (len(prompt) // 4 + max_tokens))
        elapsed = round(time.time() - started, 2)
        _log.info("%s ok  tokens=%d %.2fs", label, tokens, elapsed)
        return text, tokens
    except httpx.HTTPStatusError as e:
        _log.error(
            "%s %d from %s: %s",
            label, e.response.status_code, url, e.response.text[:300],
        )
        return "", 0
    except httpx.TimeoutException:
        _log.error(
            "%s timeout after %ds from %s (prompt_len=%d)",
            label, FAST_TIMEOUT, url, len(prompt),
        )
        return "", 0
    except Exception:
        _log.error("%s call failed from %s", label, url, exc_info=True)
        return "", 0


def model_call(
    function_name: str,
    prompt: str,
    priority: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, int]:
    """Config-driven model call.

    Reads role, temperature, max_tokens from config.json for the given
    function_name. Acquires a slot for that role, calls the model, returns
    (response_text, tokens_used).

    Caller can override temperature/max_tokens for one-off needs; config
    values are used when not provided.
    """
    cfg = get_function_config(function_name)
    role = cfg["role"]
    temp = temperature if temperature is not None else cfg.get("temperature", 0.2)
    mt = max_tokens if max_tokens is not None else cfg.get("max_tokens", 200)

    with acquire_role(role, priority=priority) as (url, model_id):
        if not url:
            _log.error("%s: no model for role=%s", function_name, role)
            return "", 0
        _assert_not_reasoner(url, function_name)
        return _raw_model_call(function_name, url, model_id, prompt, mt, temp)


# ---------------------------------------------------------------------------
# Legacy wrappers — kept during migration, will be removed.
# ---------------------------------------------------------------------------

def _tool_call(prompt: str, max_tokens: int, temperature: float = 0.2) -> tuple[str, int]:
    with acquire_model("tool") as (url, model_id):
        if not url:
            return "", 0
        return _raw_model_call("tool_call", url, model_id, prompt, max_tokens, temperature)


def _fast_call(prompt: str, max_tokens: int, temperature: float = 0.2) -> tuple[str, int]:
    with acquire_model("fast") as (url, model_id):
        if not url:
            return "", 0
        return _raw_model_call("fast_call", url, model_id, prompt, max_tokens, temperature)
