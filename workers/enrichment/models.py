from __future__ import annotations

import logging
import time

import httpx

from config import MODELS, REASONER_ROLE, no_think_params
from workers.search.models import acquire_model

_log = logging.getLogger("enrichment_agent.models")

FAST_TIMEOUT = 600


def _assert_not_reasoner(url: str | None) -> None:
    # Belt-and-braces: the resolver already restricts enrichment to the fast
    # and tool pools. This guards against catalog misconfiguration that ever
    # registers a reasoner model under one of those roles.
    if not url:
        return
    reasoner_entry = MODELS.get(REASONER_ROLE)
    if not isinstance(reasoner_entry, dict):
        return
    reasoner_url = reasoner_entry.get("url")
    if reasoner_url and url == reasoner_url:
        raise RuntimeError(
            f"refusing to dispatch enrichment call to reasoner "
            f"(url={url}). Enrichment must use fast/tool pools only."
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
    _log.info(
        "%s start  url=%s model=%s prompt_len=%d max_tokens=%d",
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
                **no_think_params(),
            },
            timeout=FAST_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage") or {}
        tokens = int(usage.get("total_tokens") or (len(prompt) // 4 + max_tokens))
        elapsed = round(time.time() - started, 2)
        _log.info("%s ok  tokens=%d %.2fs", role_label, tokens, elapsed)
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
    with acquire_model("tool") as (url, model_id):
        return _model_call("tool_call", url, model_id, prompt, max_tokens, temperature)


def _fast_call(prompt: str, max_tokens: int, temperature: float = 0.2) -> tuple[str, int]:
    with acquire_model("fast") as (url, model_id):
        return _model_call("fast_call", url, model_id, prompt, max_tokens, temperature)
