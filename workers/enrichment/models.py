from __future__ import annotations

import logging
import time

import httpx

from config import MODELS, REASONER_ROLE
from workers.search.models import _fast_model, _tool_model, fast_slot, tool_slot

_log = logging.getLogger("enrichment_agent.models")

FAST_TIMEOUT = 180


def _assert_not_reasoner(url: str | None) -> None:
    # belt-and-braces: the resolver already restricts enrichment to tool/fast roles
    if not url:
        return
    reasoner_entry = MODELS.get(REASONER_ROLE)
    if not isinstance(reasoner_entry, dict):
        return
    reasoner_url = reasoner_entry.get("url")
    if reasoner_url and url == reasoner_url:
        raise RuntimeError(
            f"refusing to dispatch enrichment call to reasoner "
            f"(url={url}). Enrichment must use tool/fast roles only."
        )


def _model_call(
    role_label: str,
    url: str | None,
    model_id: str | None,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, int]:
    if not url:
        _log.error(
            "%s model not available — no safe role resolved in catalog",
            role_label,
        )
        return "", 0
    _assert_not_reasoner(url)
    started = time.time()
    _log.debug(
        "%s    url=%s model=%s prompt_len=%d max_tokens=%d",
        role_label, url, model_id, len(prompt), max_tokens,
    )
    try:
        r = httpx.post(
            f"{url}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=FAST_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage") or {}
        tokens = int(usage.get("total_tokens") or (len(prompt) // 4 + max_tokens))
        elapsed = round(time.time() - started, 2)
        _log.debug("%s ok  tokens=%d %.2fs", role_label, tokens, elapsed)
        return text, tokens
    except httpx.HTTPStatusError as e:
        _log.error(
            "%s %d from %s: %s",
            role_label, e.response.status_code, url, e.response.text[:300],
        )
        return "", 0
    except httpx.TimeoutException:
        _log.error(
            "%s timeout after %ds from %s (prompt_len=%d)",
            role_label, FAST_TIMEOUT, url, len(prompt),
        )
        return "", 0
    except Exception:
        _log.error("%s call failed from %s", role_label, url, exc_info=True)
        return "", 0


def _tool_call(prompt: str, max_tokens: int, temperature: float = 0.2) -> tuple[str, int]:
    url, model_id = _tool_model()
    with tool_slot():
        return _model_call("tool_call", url, model_id, prompt, max_tokens, temperature)


def _fast_call(prompt: str, max_tokens: int, temperature: float = 0.2) -> tuple[str, int]:
    url, model_id = _fast_model()
    with fast_slot():
        return _model_call("fast_call", url, model_id, prompt, max_tokens, temperature)
