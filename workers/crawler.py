"""Knowledge crawler: frontier management, link discovery, staleness, fan-out.

This module owns everything about *which URLs to fetch next* and *when*. It
is deliberately separate from the content pipeline (validate/summarise/graph/
chroma) which lives in :mod:`workers.enrichment_agent`.

Rules this module lives by:

1. **MUST NOT call the reasoner model.** The reasoner is reserved for
   interactive chat and future synthesis agents. LLM-calling helpers
   (:func:`select_crawl_paths`, :func:`expand_frontier`) take a ``tool_call``
   callable as a parameter so this module imports *nothing* model-related at
   runtime. The call-site in :mod:`workers.enrichment_agent` passes its own
   ``_tool_call`` which enforces the reasoner guard.
2. **Thread-safe.** Polite-delay state and the robots cache are protected by
   locks because :func:`fan_out` can run crawler calls concurrently.
3. **No circular imports.** :class:`workers.enrichment_agent.EnrichmentDB` is
   only referenced under ``TYPE_CHECKING``; at runtime the ``db`` parameter
   is duck-typed.
"""
from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
import urllib.robotparser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, TYPE_CHECKING
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from config import MODEL_PARALLEL_SLOTS

if TYPE_CHECKING:
    from workers.enrichment_agent import EnrichmentDB

_log = logging.getLogger("crawler")


# --- Crawl policy constants -------------------------------------------------

MAX_CRAWL_DEPTH = 5
CRAWL_LINK_CAP = 50

POLITE_DELAY_MIN = 2
POLITE_DELAY_MAX = 5
POLITE_SAME_DOMAIN_MIN = 4
POLITE_SAME_DOMAIN_MAX = 8

ROBOTS_CACHE_TTL = 3600  # re-check robots.txt after 1 hour


# --- Module state (all mutations protected by the relevant lock) -----------

_last_domain_hit: dict[str, float] = {}
_polite_lock = threading.Lock()

ROBOTS_CACHE: dict[str, tuple[urllib.robotparser.RobotFileParser, float]] = {}
_robots_lock = threading.Lock()


# --- Polite-delay helper ----------------------------------------------------

def apply_polite_delay(domain: str) -> float:
    """Sleep for a random polite interval scaled by same-domain recency.

    Returns the delay actually applied (seconds). Thread-safe: the
    ``_last_domain_hit`` timestamp is updated to the *projected wake time*
    BEFORE sleeping, so concurrent callers for the same domain see the
    slot as already taken and compute their own delay on top of it. This
    matters once §6-style fan-out is extended to run multiple sources
    concurrently — without this, two threads could pick the same domain
    at the same moment and both treat it as idle.
    """
    if not domain:
        delay = random.uniform(POLITE_DELAY_MIN, POLITE_DELAY_MAX)
        time.sleep(delay)
        return delay
    with _polite_lock:
        now = time.time()
        last_hit = _last_domain_hit.get(domain, 0)
        if now - last_hit < POLITE_SAME_DOMAIN_MAX:
            delay = random.uniform(POLITE_SAME_DOMAIN_MIN, POLITE_SAME_DOMAIN_MAX)
        else:
            delay = random.uniform(POLITE_DELAY_MIN, POLITE_DELAY_MAX)
        # Publish the projected wake time *under the lock* so any other
        # thread that observes _last_domain_hit next sees this slot as
        # in-progress, not idle.
        _last_domain_hit[domain] = now + delay
    time.sleep(delay)
    return delay


# --- robots.txt -------------------------------------------------------------

def check_robots(url: str) -> bool:
    """Return True if our crawler is allowed to fetch this URL.

    Caches the parsed ``RobotFileParser`` per host for ``ROBOTS_CACHE_TTL``
    seconds. Fails **open** (returns True) when robots.txt is unreachable or
    malformed — we don't want a flaky server to halt crawling.
    """
    parsed = urlparse(url)
    host = f"{parsed.scheme}://{parsed.netloc}"
    now = time.time()
    with _robots_lock:
        cached = ROBOTS_CACHE.get(host)
    if cached is not None:
        rp, fetched_at = cached
        if now - fetched_at < ROBOTS_CACHE_TTL:
            try:
                return rp.can_fetch("mst-harness", url)
            except Exception:
                return True
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(f"{host}/robots.txt")
    try:
        rp.read()
    except Exception:
        return True
    with _robots_lock:
        ROBOTS_CACHE[host] = (rp, now)
        if len(ROBOTS_CACHE) > 500:
            cutoff = now - ROBOTS_CACHE_TTL
            stale = [k for k, (_, t) in ROBOTS_CACHE.items() if t < cutoff]
            for k in stale:
                del ROBOTS_CACHE[k]
    try:
        return rp.can_fetch("mst-harness", url)
    except Exception:
        return True


# --- Link discovery ---------------------------------------------------------

def extract_internal_links(url: str, source: dict | None = None) -> list[str]:
    """Fetch a page and return de-duped internal (same-domain) links.

    Capped at :data:`CRAWL_LINK_CAP` to keep prompts bounded. The
    ``source`` argument is reserved for future per-target customisation and
    is currently unused.
    """
    # Deferred import to avoid a module-load-time dependency on web_search.
    from workers.web_search import (
        BROWSER_HEADERS, SCRAPE_TIMEOUT, _is_safe_url, _is_blocklisted,
    )
    parsed_base = urlparse(url)
    base_domain = parsed_base.netloc.lower()

    try:
        resp = httpx.get(
            url, timeout=SCRAPE_TIMEOUT, follow_redirects=True,
            headers=BROWSER_HEADERS,
        )
        resp.raise_for_status()
    except Exception:
        _log.debug("link extraction failed for %s", url[:120])
        return []

    try:
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception:
        return []

    seen: set[str] = set()
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("#", "javascript:", "mailto:")):
            continue
        full = urljoin(url, href).split("#")[0].split("?")[0]
        parsed = urlparse(full)
        if parsed.netloc.lower() != base_domain:
            continue
        if full in seen or full == url:
            continue
        if not _is_safe_url(full) or _is_blocklisted(full):
            continue
        seen.add(full)
        links.append(full)
        if len(links) >= CRAWL_LINK_CAP:
            break

    _log.debug("extracted %d internal links from %s", len(links), url[:80])
    return links


# --- Link ranking (tool LLM, dependency-injected) ---------------------------

def select_crawl_paths(
    links: list[str],
    source_url: str,
    budget_remaining: int,
    tool_call: Callable[[str, int, float], tuple[str, int]],
    sparse_concepts: list[str] | None = None,
) -> tuple[list[str], int]:
    """Rank internal links by content value using the tool model.

    ``tool_call`` is injected so this module doesn't import the model
    helpers directly — keeps the circular-import surface at zero.

    If ``sparse_concepts`` is provided, the prompt is extended with a
    hint biasing selection toward links that might cover those topics.
    This implements §5 (graph-aware link ranking).

    Returns ``(selected_urls, tokens_used)``.
    """
    if not links:
        return [], 0

    max_pages = min(MAX_CRAWL_DEPTH, max(1, budget_remaining // 2000))
    link_list = "\n".join(f"- {u}" for u in links[:CRAWL_LINK_CAP])
    sparse_hint = ""
    if sparse_concepts:
        joined = ", ".join(sparse_concepts[:8])
        sparse_hint = (
            "\nWe currently have SPARSE coverage of these topics in our "
            f"knowledge base: {joined}. Prioritise links that look like "
            "they would cover any of those topics.\n"
        )
    prompt = (
        f"You are selecting which pages to crawl from the site {source_url}.\n"
        f"Below are internal links found on that page. Select up to {max_pages} "
        "links that are most likely to contain substantive, unique content worth "
        "indexing in a knowledge base.\n\n"
        "Prefer:\n"
        "- Article/content pages over navigation/index pages\n"
        "- Pages with specific topics over generic landing pages\n"
        "- Documentation, guides, research, or news over login/signup/about/contact\n\n"
        "SKIP: login pages, user profiles, search pages, privacy policies, "
        "terms of service, duplicate/paginated versions of the same content."
        f"{sparse_hint}\n\n"
        f"LINKS:\n{link_list}\n\n"
        "Return ONLY a JSON array of the selected URLs, nothing else. "
        "If none are worth crawling, return []."
    )
    raw, tokens = tool_call(prompt, 400, 0.2)
    if not raw:
        return [], tokens
    try:
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip()).strip()
        selected = json.loads(cleaned)
    except Exception:
        _log.warning("crawl path selection unparseable: %s", raw[:200])
        return [], tokens
    if not isinstance(selected, list):
        return [], tokens
    link_set = set(links)
    filtered = [u for u in selected if isinstance(u, str) and u in link_set][:max_pages]
    return filtered, tokens


# --- Staleness --------------------------------------------------------------

def compute_next_crawl_at(
    last_scraped_at: datetime,
    base_hours: float,
    consecutive_unchanged: int,
) -> datetime:
    """Exponential backoff scheduler.

    ``next = last + base_hours * 2^min(n, 4)``. Caps at 16× so a 1-hour
    source never stretches past 16 hours; a 24-hour source never past
    ~16 days. Reset to zero on content change — caller's responsibility.
    """
    multiplier = 2 ** min(max(consecutive_unchanged, 0), 4)
    return last_scraped_at + timedelta(hours=base_hours * multiplier)


def should_recrawl(row: dict, now: datetime | None = None) -> bool:
    """Return True if a ``scrape_targets`` row is due for re-crawl.

    Prefers ``next_crawl_at`` when set (the new adaptive-staleness path);
    falls back to ``last_scraped_at + frequency_hours`` for legacy rows
    where the new column is NULL.
    """
    if not row.get("active"):
        return False
    now = now or datetime.now(timezone.utc)

    next_at = row.get("next_crawl_at")
    if next_at:
        try:
            target = datetime.fromisoformat(str(next_at).replace("Z", "+00:00"))
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            return target <= now
        except Exception:
            pass  # fall through to legacy path

    last = row.get("last_scraped_at")
    if not last:
        return True  # never scraped
    try:
        last_dt = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
    except Exception:
        return True
    # NocoDB's scrape_targets schema uses `frequency` as the column name in
    # some deployments and `frequency_hours` in others. Tolerate both.
    freq = float(row.get("frequency_hours") or row.get("frequency") or 24)
    return last_dt + timedelta(hours=freq) <= now


# --- Frontier expansion (§3: auto-crawl same-domain children) --------------

def expand_frontier(
    parent: dict,
    db: "EnrichmentDB",
    org_id: int,
    category: str,
    budget_remaining: int,
    tool_call: Callable[[str, int, float], tuple[str, int]],
    sparse_concepts: list[str] | None = None,
) -> tuple[int, int]:
    """Discover same-domain sub-pages from a trusted parent and auto-create
    them as ``scrape_targets`` rows (inside the trust envelope).

    External-domain discoveries continue to go through the suggestions
    queue — see ``_discover_sources`` in ``workers.enrichment_agent``.

    Returns ``(children_created, tokens_used)``.
    """
    parent_id = parent.get("Id")
    parent_url = parent.get("url") or ""
    parent_depth = int(parent.get("depth") or 0)

    if parent_depth + 1 > MAX_CRAWL_DEPTH:
        _log.debug("frontier skipped  parent=%s at depth cap %d", parent_id, parent_depth)
        return 0, 0

    internal_links = extract_internal_links(parent_url, source=parent)
    if not internal_links:
        return 0, 0

    try:
        already_tracked = db.list_tracked_urls(org_id)
    except Exception:
        already_tracked = set()
    new_links = [u for u in internal_links if u not in already_tracked]
    if not new_links:
        return 0, 0

    selected, tokens = select_crawl_paths(
        new_links,
        parent_url,
        budget_remaining,
        tool_call=tool_call,
        sparse_concepts=sparse_concepts,
    )
    if not selected:
        return 0, tokens

    now_utc = datetime.now(timezone.utc)
    first_probe = now_utc + timedelta(minutes=5)
    freq_hours = parent.get("frequency_hours") or parent.get("frequency") or 24
    use_playwright = bool(parent.get("use_playwright")) or False
    enrichment_agent_id = parent.get("enrichment_agent_id")
    parent_name = parent.get("name") or parent_url

    created = 0
    for child_url in selected:
        if not check_robots(child_url):
            _log.debug("frontier skip robots  %s", child_url[:120])
            continue
        try:
            # Reuse the existing EnrichmentDB.create_source path that the UI
            # and suggestion-approval flow also go through. This keeps all
            # scrape_targets inserts on one code path.
            db.create_source({
                "org_id": org_id,
                "url": child_url,
                "name": parent_name,
                "category": category,
                "enrichment_agent_id": enrichment_agent_id,
                "parent_target": parent_id,
                "depth": parent_depth + 1,
                "discovered_from": parent_url,
                "auto_crawled": True,
                "use_playwright": use_playwright,
                "frequency_hours": freq_hours,
                "next_crawl_at": first_probe.isoformat(),
                "active": True,
            })
            created += 1
        except Exception:
            _log.error(
                "failed to create frontier child %s",
                child_url[:120], exc_info=True,
            )

    _log.info(
        "frontier expanded  parent=%s depth=%d selected=%d created=%d",
        parent_id, parent_depth + 1, len(selected), created,
    )
    return created, tokens


# --- Concurrent fan-out (§6) -----------------------------------------------

def fan_out(
    calls: list[Callable[[], Any]],
    *,
    max_workers: int | None = None,
    label: str = "fan_out",
) -> list[Any]:
    """Run zero-arg callables concurrently, preserving input order.

    Bounded by :data:`config.MODEL_PARALLEL_SLOTS` by default so
    llama.cpp's parallel slot pool isn't over-scheduled. Exceptions from
    individual calls are logged and replaced with ``None`` in the results
    list — the caller decides how to handle partial failure.

    Example::

        [summary_tuple, rels_tuple, disc_tokens] = fan_out(
            [
                lambda: _summarise(text),
                lambda: _extract_relationships(text, org_id),
                lambda: _discover_sources(text, url, org_id, cycle_id, db),
            ],
            label="process_source",
        )
    """
    if not calls:
        return []
    workers = max_workers or MODEL_PARALLEL_SLOTS
    workers = max(1, min(workers, len(calls)))
    results: list[Any] = [None] * len(calls)
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=label) as pool:
        future_to_index = {pool.submit(fn): i for i, fn in enumerate(calls)}
        for fut in as_completed(future_to_index):
            i = future_to_index[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                _log.warning("%s item %d failed: %s", label, i, e)
                results[i] = None
    elapsed = round(time.time() - started, 2)
    _log.debug(
        "%s complete  items=%d workers=%d %.2fs",
        label, len(calls), workers, elapsed,
    )
    return results
