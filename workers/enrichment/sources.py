from __future__ import annotations

import json
import logging
import re

import httpx

from config import CATEGORY_COLLECTIONS, MAX_SUMMARY_INPUT_CHARS
from workers.enrichment.db import EnrichmentDB
from workers.enrichment.models import _tool_call
from workers.enrichment.summarise import _salvage_json_array

_log = logging.getLogger("enrichment_agent.sources")


def _verify_url_reachable(url: str) -> bool:
    try:
        r = httpx.head(
            url, timeout=10, follow_redirects=True,
            headers={"User-Agent": "mst-harness/1.0"},
        )
        return r.status_code < 400
    except Exception:
        return False


def _discover_sources(
    text: str,
    source_url: str,
    org_id: int,
    cycle_id: str,
    db: EnrichmentDB,
) -> int:
    _log.debug("discovering sources from %s  org=%d", source_url[:80], org_id)

    try:
        already_known = db.list_tracked_urls(org_id)
    except Exception:
        already_known = set()

    prompt = (
        "You are evaluating page content to find authoritative external sources "
        "worth ONGOING monitoring in a knowledge base. Apply strict quality criteria:\n\n"
        "REQUIRED — a source MUST have ALL of:\n"
        "1. INSTITUTIONAL AUTHORITY: official organisation site, established publication, "
        "   government/regulatory body, or recognised industry group. NOT personal blogs, "
        "   social media profiles, or anonymous authors.\n"
        "2. REGULAR UPDATES: publishes new content on a recurring basis (daily, weekly, "
        "   monthly). NOT one-off articles, static pages, or archived content.\n"
        "3. ORIGINAL CONTENT: primary source with original reporting, research, data, or "
        "   documentation. NOT aggregators, scrapers, or sites that just repost others.\n"
        "4. EDITORIAL STANDARDS: has editorial review or institutional accountability. "
        "   NOT unmoderated user-generated content.\n\n"
        "ALWAYS REJECT: social media (Twitter/X, Reddit, LinkedIn, Facebook, Instagram), "
        "forums, Medium/Substack (unless from a known institution), paywalled sites, "
        "SEO spam, content farms, aggregator/scraper sites, personal hobby blogs, "
        "YouTube channels, podcast pages, GitHub repos (unless official project docs).\n\n"
        "From the content below, identify up to 3 external sources that meet ALL "
        "criteria. For each, return a JSON object with:\n"
        "- url: the source's main page or feed URL (not a deep link to one article)\n"
        "- name: the organisation or publication name\n"
        "- category: one of documentation, news, competitive, regulatory, research, "
        "  security, model_releases\n"
        "- authority: WHO maintains this source and WHY they are authoritative "
        "  (e.g. 'Official Apache Foundation docs, maintained by core committers')\n"
        "- update_frequency: estimated publication cadence (e.g. 'weekly', 'daily')\n"
        "- confidence_score: 1-10 where 8+ = clearly meets all criteria, "
        "  7 = probably meets criteria, below 7 = uncertain or missing a criterion\n\n"
        "If NO sources meet ALL criteria, return []. Be selective — 0 suggestions "
        "is better than a weak suggestion.\n\n"
        f"CONTENT:\n{text[:MAX_SUMMARY_INPUT_CHARS]}"
    )
    raw, tokens = _tool_call(prompt, max_tokens=600)
    if not raw:
        _log.debug("discover_sources returned empty from %s", source_url[:80])
        return tokens
    try:
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip()).strip()
        items = json.loads(cleaned)
    except Exception:
        items = _salvage_json_array(cleaned) if cleaned else None
        if items is None:
            _log.warning("discover_sources unparseable from %s: %s", source_url[:80], raw[:200])
            return tokens
    if not isinstance(items, list):
        _log.warning("discover_sources non-list from %s", source_url[:80])
        return tokens

    recorded = 0
    for item in items[:3]:
        try:
            score = int(item.get("confidence_score") or 0)
            if score < 7:
                _log.debug("discover_sources skip low-score=%d url=%s", score, str(item.get("url", ""))[:80])
                continue
            category = str(item.get("category") or "").lower()
            if category not in CATEGORY_COLLECTIONS:
                continue
            url = str(item.get("url") or "").strip()
            if not url or not url.startswith("http"):
                continue

            if url in already_known:
                _log.debug("discover_sources skip already-tracked url=%s", url[:80])
                continue

            if not _verify_url_reachable(url):
                _log.info("discover_sources skip unreachable url=%s", url[:80])
                continue

            authority = str(item.get("authority") or "")
            freq = str(item.get("update_frequency") or "")
            reason = f"{authority}. Updates: {freq}" if authority else str(item.get("reason") or "")

            confidence = "high" if score >= 8 else "medium"
            _log.debug("suggesting source  url=%s category=%s score=%d from=%s", url[:80], category, score, source_url[:60])
            db.record_suggestion(
                org_id=org_id,
                url=url,
                name=str(item.get("name") or url),
                category=category,
                reason=reason[:500],
                confidence=confidence,
                confidence_score=score,
                suggested_by_url=source_url,
                suggested_by_cycle=cycle_id,
            )
            already_known.add(url)
            recorded += 1
        except Exception:
            _log.error("suggestion record failed", exc_info=True)
    _log.info("discover_sources  from=%s candidates=%d recorded=%d", source_url[:80], len(items), recorded)
    return tokens
