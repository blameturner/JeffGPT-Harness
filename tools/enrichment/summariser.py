from __future__ import annotations

import logging

from infra.config import get_feature, get_function_config
from infra.memory import remember
from infra.nocodb_client import NocodbClient
from shared.models import model_call

_log = logging.getLogger("scraper.summariser")

DEFAULT_SUMMARY_FUNCTION = "scrape_summariser"


def summarise_page_job(payload: dict) -> dict:
    """Tool-queue handler: takes scraped page text, runs the summariser model, persists the summary."""
    url = payload.get("url") or ""
    text = payload.get("text") or ""
    org_id = int(payload.get("org_id") or 0)
    source = payload.get("source") or "scrape"
    scrape_target_id = payload.get("scrape_target_id")
    discovery_id = payload.get("discovery_id")

    if not url or not text:
        return {"status": "error", "reason": "missing_url_or_text"}

    function_name = get_feature("scraper", "summariser_model", DEFAULT_SUMMARY_FUNCTION)
    try:
        cfg = get_function_config(function_name)
    except KeyError:
        _log.warning("summariser model %r not configured", function_name)
        return {"status": "error", "reason": "model_not_configured"}

    max_input = int(cfg.get("max_input_chars", 12000))
    prompt = (
        "Summarise the following web page in 4-8 sentences. "
        "Capture: what the page is, the key facts/claims, and any concrete "
        "data (numbers, dates, names). Do not invent details.\n\n"
        f"URL: {url}\n\n"
        f"PAGE:\n{text[:max_input]}"
    )

    try:
        summary, _tokens = model_call(function_name, prompt)
        summary = (summary or "").strip()
    except Exception as e:
        _log.warning("summarise failed  url=%s  error=%s", url[:80], e)
        return {"status": "error", "reason": str(e)[:200]}

    if not summary:
        return {"status": "error", "reason": "empty_summary"}

    metadata = {
        "url": url,
        "source": source,
        "kind": "summary",
    }
    if scrape_target_id:
        metadata["scrape_target_id"] = scrape_target_id
    if discovery_id:
        metadata["discovery_id"] = discovery_id

    chunks = 0
    try:
        chunk_ids = remember(summary, metadata, org_id, collection_name="discovery_summaries")
        chunks = len(chunk_ids or [])
    except Exception:
        _log.warning("summary embed failed  url=%s", url[:80], exc_info=True)

    # surface the summary back onto the originating row (if a column exists; nocodb
    # silently drops unknown fields, so this is best-effort).
    if scrape_target_id:
        try:
            NocodbClient()._patch("scrape_targets", int(scrape_target_id), {"summary": summary[:2000]})
        except Exception:
            pass
    if discovery_id:
        try:
            NocodbClient()._patch("discovery", int(discovery_id), {"summary": summary[:2000]})
        except Exception:
            pass

    _log.info("summarise ok  url=%s chars=%d chunks=%d", url[:100], len(summary), chunks)
    return {
        "status": "ok",
        "url": url,
        "summary_chars": len(summary),
        "chunks": chunks,
    }
