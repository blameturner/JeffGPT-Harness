"""Harvest extractor — turns fetched text into a summary + optional structured fields.

The "summary" path reuses the existing `scrape_summariser` model role so
behaviour matches what the cron-driven scraper has been doing.
The "structured" path uses a generic JSON-extracting LLM call against
the policy's schema.
"""
from __future__ import annotations

import concurrent.futures as _futures
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from infra.config import get_feature
from shared.models import model_call

_log = logging.getLogger("harvest.extractor")

# Hard per-call cap. Local models can wedge on big prompts; we never want a
# single URL to block the whole harvest. Configurable via
# features.harvest.extractor_timeout_s (default 600s).
_EXTRACTOR_TIMEOUT_DEFAULT_S = 600


def _timeout_s() -> int:
    raw = get_feature("harvest", "extractor_timeout_s", _EXTRACTOR_TIMEOUT_DEFAULT_S)
    try:
        v = int(raw)
        return v if v > 0 else _EXTRACTOR_TIMEOUT_DEFAULT_S
    except Exception:
        return _EXTRACTOR_TIMEOUT_DEFAULT_S


def _bounded_model_call(function_name: str, prompt: str, **kwargs) -> tuple[str, int] | None:
    """Run model_call in a worker thread with a hard timeout. Returns None on
    timeout / error so the caller can degrade gracefully."""
    timeout_s = _timeout_s()
    ex = _futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="harvest-extract")
    try:
        fut = ex.submit(model_call, function_name, prompt, **kwargs)
        try:
            return fut.result(timeout=timeout_s)
        except _futures.TimeoutError:
            _log.warning("extractor: %s timed out after %ds", function_name, timeout_s)
            return None
        except Exception as e:
            _log.warning("extractor: %s failed: %s", function_name, str(e)[:200])
            return None
    finally:
        ex.shutdown(wait=False)


@dataclass
class ExtractResult:
    summary: str = ""
    fields: dict = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.summary or self.fields)


def _strip_html_tags(html: str) -> str:
    """Cheap HTML-to-text. Existing scraper uses a similar approach.

    Doesn't try to be perfect — extractor LLM is robust to noisy input
    and we cap upstream so excessive markup is fine."""
    if not html:
        return ""
    # Drop <script> / <style> blocks
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Replace tags with whitespace
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    # Decode common entities
    cleaned = (
        cleaned.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _summarise(text: str, *, model_role: str = "scrape_summariser",
               max_input_chars: int = 12000) -> str:
    if not text:
        return ""
    # Redact obvious secrets / PII before any LLM call. Defence in depth —
    # auth_connections-fetched bodies could contain bearer tokens etc.
    try:
        from tools.harvest.connectors import redact_secrets
        text = redact_secrets(text)
    except Exception:
        pass
    excerpt = text[:max_input_chars]
    prompt = (
        "Summarise the following web page in 4-6 sentences. Capture the topic, "
        "the most important factual claims (with dates/numbers if present), "
        "and the takeaway. Output prose only, no bullets, no headings, no preamble.\n\n"
        f"PAGE:\n{excerpt}"
    )
    res = _bounded_model_call(model_role, prompt)
    if not res:
        return ""
    text_out, _ = res
    return (text_out or "").strip()


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE)


def _strip_fence(s: str) -> str:
    s = s.strip()
    s = _FENCE_RE.sub("", s)
    return s.strip()


def _clean_json_text(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def _extract_json(raw: str) -> dict | None:
    if not raw:
        return None
    s = _clean_json_text(_strip_fence(raw))
    # Find first {...}
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(s[start:end + 1])
    except (json.JSONDecodeError, TypeError):
        return None


def _extract_fields(text: str, schema: dict, *, max_input_chars: int = 12000,
                    model_role: str = "research_planner") -> dict:
    # Redact before structured-extract too.
    try:
        from tools.harvest.connectors import redact_secrets
        text = redact_secrets(text)
    except Exception:
        pass
    """Generic structured extraction against a schema.

    Schema is `{field: type}` where type ∈ {text, numeric, date, percent}.
    Returns whatever fields the model could populate; missing fields are
    omitted, not stuffed with "unknown".
    """
    if not text or not schema:
        return {}
    field_lines = "\n".join(f'  "{k}": "{v}"' for k, v in schema.items())
    prompt = f"""Extract the following fields from the page below. Return ONLY a JSON object with the field names as keys. If a field is not present in the page, omit it (do NOT write null or "unknown").

FIELD SCHEMA (key: expected type):
{field_lines}

Rules:
- Output raw JSON only — no markdown fences, no preamble, no trailing prose.
- Numeric / percent fields should be numbers (not strings) when present.
- Date fields should be ISO-format strings (YYYY-MM-DD) when possible.
- Text fields should be concise (max 200 chars each).

PAGE:
{text[:max_input_chars]}"""
    res = _bounded_model_call(model_role, prompt, temperature=0.1)
    if not res:
        return {}
    raw, _ = res
    parsed = _extract_json(raw or "")
    if not isinstance(parsed, dict):
        return {}
    # Filter to known schema keys only
    return {k: v for k, v in parsed.items() if k in schema}


def extract(content_text: str, *, content_type: str = "text/html",
            schema: Optional[dict] = None,
            summarise: bool = True,
            summary_model: str = "scrape_summariser",
            extract_model: str = "research_planner") -> ExtractResult:
    """Run summary + (optional) structured extraction on fetched content.

    If ``content_type`` is HTML, strips tags first; otherwise uses the
    text as-is (PDFs/RSS already plain text by the time they reach here).
    """
    out = ExtractResult()
    if not content_text:
        out.error = "empty_input"
        return out

    if "html" in (content_type or "").lower():
        text = _strip_html_tags(content_text)
    else:
        text = content_text

    if not text.strip():
        out.error = "empty_after_strip"
        return out

    if summarise:
        out.summary = _summarise(text, model_role=summary_model)

    if schema:
        out.fields = _extract_fields(text, schema, model_role=extract_model)

    return out
