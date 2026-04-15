"""Relevance classifier — the second pathfinder-side agent.

Runs after a scrape_target has been scraped. Reads the extracted text,
asks an LLM to rate the page's relevance for the enrichment corpus, and
patches the scrape_targets row accordingly:

- relevant            → status="ok"           (kept active, will keep getting re-scraped)
- borderline          → status="ok"           (kept, flagged low in relevance_label)
- not relevant / spam → status="rejected", active=0  (scraper stops picking it)

Triggered by scrape_target_job on successful scrape. Optional — the scraper still
works if this handler isn't registered.
"""
from __future__ import annotations

import json
import logging
import re

from infra.config import get_feature, get_function_config
from infra.nocodb_client import NocodbClient
from shared.models import model_call

_log = logging.getLogger("scraper.classifier")

DEFAULT_CLASSIFIER_FUNCTION = "scrape_classifier"

_PROMPT = """You are reviewing a web page that was auto-discovered for an
enrichment corpus. Classify whether the page is worth keeping for
retrieval-augmented search.

URL: {url}

PAGE CONTENT:
{text}

Respond with ONLY a single JSON object, no prose:
{{
  "relevance": "high" | "medium" | "low" | "rejected",
  "score": <0-100 integer>,
  "reason": "<one short sentence>"
}}

Guidelines:
- "rejected" = spam, parked domain, login wall, empty shell, near-duplicate boilerplate,
  unrelated off-topic content, or pages that are essentially navigation/footer only.
- "low" = thin content, tangential relevance, mostly links.
- "medium" = useful context but not canonical.
- "high" = substantive primary content on a recognisable topic.
"""


def _extract_json(raw: str) -> dict | None:
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s).rstrip("`").strip()
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def classify_relevance_job(payload: dict) -> dict:
    """Tool-queue handler: classify a scraped page and update scrape_targets.status.

    Payload: {target_id, url, text, org_id}
    """
    target_id = payload.get("target_id")
    url = (payload.get("url") or "").strip()
    text = (payload.get("text") or "").strip()
    org_id = int(payload.get("org_id") or 0)

    if not target_id or not url or not text:
        return {"status": "error", "reason": "missing_fields"}

    function_name = get_feature("scraper", "classifier_model", DEFAULT_CLASSIFIER_FUNCTION)
    try:
        cfg = get_function_config(function_name)
    except KeyError:
        _log.warning("classifier model %r not configured — skipping", function_name)
        return {"status": "skipped", "reason": "model_not_configured"}

    max_input = int(cfg.get("max_input_chars", 8000))
    prompt = _PROMPT.format(url=url, text=text[:max_input])

    try:
        raw, _tokens = model_call(function_name, prompt)
    except Exception as e:
        _log.warning("classifier call failed  target_id=%s  url=%s  error=%s",
                     target_id, url[:80], e)
        return {"status": "error", "reason": str(e)[:200]}

    parsed = _extract_json(raw)
    if not parsed:
        _log.warning("classifier parse failed  target_id=%s  raw=%s", target_id, (raw or "")[:200])
        return {"status": "error", "reason": "unparseable_llm_output"}

    label = str(parsed.get("relevance") or "").strip().lower()
    if label not in ("high", "medium", "low", "rejected"):
        label = "low"  # default to conservative-keep
    try:
        score = max(0, min(100, int(parsed.get("score") or 0)))
    except Exception:
        score = 0
    reason = str(parsed.get("reason") or "")[:500]

    # Decide the NocoDB status + active flag.
    # Schema enum for scrape_targets.status: "ok" | "error" | "rejected" | null.
    if label == "rejected":
        patch: dict = {"status": "rejected", "active": 0, "last_scrape_error": reason[:500]}
    else:
        patch = {"status": "ok", "last_scrape_error": ""}
    # Optional columns — NocoDB drops these silently if the column doesn't exist.
    patch["relevance_score"] = score
    patch["relevance_label"] = label

    client = NocodbClient()
    try:
        client._patch("scrape_targets", int(target_id), patch)
    except Exception:
        _log.warning("classifier patch failed  target_id=%s", target_id, exc_info=True)
        return {"status": "error", "reason": "patch_failed"}

    _log.info(
        "classifier ok  target_id=%s url=%s label=%s score=%d",
        target_id, url[:100], label, score,
    )
    return {
        "status": "ok",
        "target_id": target_id,
        "url": url,
        "relevance": label,
        "score": score,
        "org_id": org_id,
    }
