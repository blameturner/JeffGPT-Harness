from __future__ import annotations

import json
import logging
import re

from config import CATEGORY_COLLECTIONS
from graph import get_sparse_concepts
from workers.enrichment.db import EnrichmentDB
from workers.enrichment.models import model_call
from workers.enrichment.sources import _verify_url_reachable
from workers.enrichment.summarise import _salvage_json_array
from workers.search.engine import searxng_search

_log = logging.getLogger("enrichment_agent.proactive")


def _proactive_search(
    org_id: int,
    cycle_id: str,
    db: EnrichmentDB,
    budget_remaining: int,
    sparse_concepts: list[str] | None = None,
) -> int:
    # reuse caller's sparse_concepts if supplied — saves a FalkorDB round-trip
    if sparse_concepts is None:
        sparse_concepts = get_sparse_concepts(org_id, limit=5)
    concepts = sparse_concepts[:5]

    if not concepts:
        return 0

    try:
        already_known = db.list_tracked_urls(org_id)
    except Exception:
        already_known = set()

    total_tokens = 0
    for concept in concepts:
        if total_tokens >= budget_remaining:
            break
        queries = [
            f'"{concept}" official documentation site',
            f'"{concept}" research publications regulatory',
        ]
        candidates: list[dict] = []
        for q in queries:
            candidates.extend(searxng_search(q, max_results=3))
            if len(candidates) >= 6:
                break

        seen: set[str] = set()
        filtered: list[dict] = []
        for r in candidates:
            url = r.get("url", "")
            if url in seen or url in already_known:
                continue
            seen.add(url)
            filtered.append(r)

        if not filtered:
            continue

        candidates_text = "\n".join(
            f"{i+1}. TITLE: {r.get('title', '')}\n   URL: {r.get('url', '')}\n   SNIPPET: {r.get('snippet', '')}"
            for i, r in enumerate(filtered[:5])
        )
        prompt = (
            f"You are evaluating search results about '{concept}' to find sources "
            "worth ONGOING monitoring in a knowledge base.\n\n"
            "A good monitoring target MUST have ALL of:\n"
            "- INSTITUTIONAL AUTHORITY: maintained by a recognised organisation, "
            "  not a personal blog or social media\n"
            "- REGULAR UPDATES: publishes new content on a recurring basis\n"
            "- ORIGINAL CONTENT: primary source, not an aggregator or reposter\n"
            "- RELEVANCE: directly covers this topic area with depth\n\n"
            "REJECT: social media, forums, Medium/Substack, paywalled sites, "
            "personal blogs, content farms, aggregators, YouTube, GitHub issues.\n\n"
            f"CANDIDATES:\n{candidates_text}\n\n"
            "For each candidate worth monitoring, return a JSON object with:\n"
            "- index: the candidate number\n"
            "- category: one of documentation, news, competitive, regulatory, "
            "  research, security, model_releases\n"
            "- authority: WHO maintains this and WHY they are credible\n"
            "- score: 1-10 (8+ = clearly authoritative, 7 = probably good)\n\n"
            "Return a JSON array. If NONE are worth monitoring, return []. "
            "Be very selective — most search results are NOT good monitoring targets."
        )
        raw, tokens = model_call("enrichment_source_discovery", prompt)
        total_tokens += tokens
        if not raw:
            continue
        try:
            cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip()).strip()
            evaluations = json.loads(cleaned)
        except Exception:
            evaluations = _salvage_json_array(cleaned) if cleaned else None
            if evaluations is None:
                continue
        if not isinstance(evaluations, list):
            continue

        for ev in evaluations:
            try:
                score = int(ev.get("score") or 0)
                if score < 7:
                    continue
                idx = int(ev.get("index", 0)) - 1
                if idx < 0 or idx >= len(filtered):
                    continue
                r = filtered[idx]
                url = r.get("url", "")
                category = str(ev.get("category") or "").lower()
                if category not in CATEGORY_COLLECTIONS:
                    continue

                if not _verify_url_reachable(url):
                    _log.debug("proactive skip unreachable url=%s", url[:80])
                    continue

                authority = str(ev.get("authority") or "")
                reason = f"Proactive: {authority}" if authority else f"Sparse coverage of {concept}"
                db.record_suggestion(
                    org_id=org_id,
                    url=url,
                    name=r.get("title") or url,
                    category=category,
                    reason=reason[:500],
                    confidence="high" if score >= 8 else "medium",
                    confidence_score=score,
                    suggested_by_url=None,
                    suggested_by_cycle=cycle_id,
                )
                already_known.add(url)
            except Exception:
                continue

    db.log_event(
        cycle_id, "proactive_search", org_id=org_id,
        message=f"concepts={len(concepts)}", tokens_used=total_tokens,
    )
    return total_tokens
