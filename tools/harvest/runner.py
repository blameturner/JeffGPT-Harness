"""Harvest runner — single tool-queue handler for the harvest pipeline.

Synchronous, single-pass design. The runner walks seeds, fetches each
URL, extracts, and persists — all within one tool-queue job. We rely on
the queue's per-handler max_workers and the heartbeat watchdog to keep
the system honest under long runs.

A future iteration can split this into per-URL sub-jobs with a finaliser
(see scraper-pathfinder-refactor.md). For v1 we keep it simple and
linear so the path of one harvest is easy to read in logs.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import deque
from datetime import datetime, timezone
from urllib.parse import urlparse

from infra.nocodb_client import NocodbClient
from tools._org import resolve_org_id
from tools.harvest import extractor as ext_mod
from tools.harvest import fetcher as fetch_mod
from tools.harvest import url_cache
from tools.harvest import walker
from tools.harvest import persister
from tools.harvest.policy import HarvestPolicy, get_policy

_log = logging.getLogger("harvest.runner")

_TABLE = "harvest_runs"

# Cap kept events so artifacts_json doesn't unbound-grow on a 2000-page run.
# The Live drawer only needs the recent tail.
_EVENT_LOG_MAX = 200

# Hard cap on the in-memory walker-overflow buffer. The seeding step trims to
# `features.harvest.suggestions_max_per_run` (default 50) anyway, so anything
# beyond a few hundred is wasted memory. Belt-and-braces against a runaway
# walker on a link-heavy site.
_WALKED_OVERFLOW_MAX = 500


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _patch(client: NocodbClient, run_id: int, fields: dict) -> None:
    try:
        client._patch(_TABLE, run_id, fields)
    except Exception:
        _log.warning("harvest_runs patch failed run=%d fields=%s", run_id, list(fields), exc_info=True)


def _patch_progress(client: NocodbClient, run_id: int, counters: dict,
                    events_tail: list[dict] | None = None) -> None:
    """Progress patch that also writes the recent event tail into
    artifacts_json["events"]. Reads the row first so we don't clobber
    any per-policy artifacts the persister has accumulated."""
    fields = dict(counters)
    if events_tail is not None:
        try:
            rows = client._get(_TABLE, params={"where": f"(Id,eq,{run_id})", "limit": 1}).get("list", [])
            row = rows[0] if rows else {}
            raw = row.get("artifacts_json") or "{}"
            try:
                arts = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except (json.JSONDecodeError, TypeError):
                arts = {}
            if not isinstance(arts, dict):
                arts = {}
            arts["events"] = list(events_tail)[-_EVENT_LOG_MAX:]
            fields["artifacts_json"] = json.dumps(arts)
        except Exception:
            _log.warning("event-log merge failed run=%d", run_id, exc_info=True)
    _patch(client, run_id, fields)


def _load_run(run_id: int) -> tuple[NocodbClient, dict | None]:
    client = NocodbClient()
    try:
        rows = client._get(_TABLE, params={"where": f"(Id,eq,{run_id})", "limit": 1}).get("list", [])
    except Exception:
        _log.warning("harvest_runs load failed run=%d", run_id, exc_info=True)
        return client, None
    return client, (rows[0] if rows else None)


def _params(run: dict) -> dict:
    raw = run.get("params_json") or "{}"
    try:
        val = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except (json.JSONDecodeError, TypeError):
        val = {}
    return val if isinstance(val, dict) else {}


# ── seed strategies ────────────────────────────────────────────────────────

def _seeds_literal_url(seed: str) -> list[str]:
    return [seed] if seed else []


def _seeds_url_list(seed: str, params: dict) -> list[str]:
    # Netscape-style bookmarks export — extract every <A HREF="..."> URL.
    if (params or {}).get("source") == "bookmarks_html" and isinstance(seed, str) and "<a" in seed.lower():
        hrefs = re.findall(r'(?i)<a[^>]+href=["\']([^"\']+)["\']', seed)
        return [h for h in hrefs if h.startswith("http")]
    if isinstance(seed, list):
        return [s for s in seed if isinstance(s, str) and s.startswith("http")]
    # Try JSON-encoded list (router serialises a list seed as JSON when storing)
    if isinstance(seed, str) and seed.startswith("["):
        try:
            parsed = json.loads(seed)
            if isinstance(parsed, list):
                return [s for s in parsed if isinstance(s, str) and s.startswith("http")]
        except (json.JSONDecodeError, TypeError):
            pass
    if isinstance(seed, str) and seed:
        parts = re.split(r"[\s,]+", seed.strip())
        return [p for p in parts if p.startswith("http")]
    urls = (params or {}).get("urls") or []
    return [u for u in urls if isinstance(u, str) and u.startswith("http")]


def _seeds_topic_search(seed: str, params: dict, org_id: int, max_seeds: int) -> list[str]:
    """Resolve seed URLs for a topic — searxng-only.

    This used to call ``run_web_search`` from the orchestrator, which chains:
    LLM query-generation → searxng → LLM rerank → parallel scrape → LLM
    extract. Five steps, any of which returning empty (rerank dropping all,
    scrape failing, extract returning nothing) makes harvest get zero seeds.
    The runner is going to fetch + extract anyway — seeding only needs URLs.

    We hit searxng directly with the user's query, plus a couple of
    light reformulations if the first pass returns nothing. No LLM, no
    scrape, no extract.
    """
    if not seed:
        return []
    try:
        from tools.search.engine import searxng_search, _dedupe
    except Exception:
        _log.warning("searxng client unavailable for topic_search")
        return []

    query = seed.strip()
    raw: list[dict] = []
    try:
        raw = searxng_search(query, max_results=max(max_seeds, 10))
    except Exception as e:
        _log.warning("topic_search searxng failed for %r: %s", query[:120], e)
        raw = []

    # Cheap fallbacks if the literal query returned nothing — cover the
    # 'too narrow' case without invoking the LLM.
    if not raw and " " in query:
        # Try the first 3-5 keywords only.
        head = " ".join(query.split()[:5])
        if head and head != query:
            try:
                raw = searxng_search(head, max_results=max(max_seeds, 10))
            except Exception:
                pass

    if not raw:
        _log.info("topic_search returned 0 URLs query=%r", query[:120])
        return []

    out: list[str] = []
    seen: set[str] = set()
    for r in _dedupe(raw):
        u = (r.get("url") or "").strip()
        if u and u not in seen and u.startswith("http"):
            seen.add(u)
            out.append(u)
        if len(out) >= max_seeds:
            break
    _log.info("topic_search resolved %d URLs query=%r", len(out), query[:120])
    return out


def _seeds_sitemap_expand(seed: str) -> list[str]:
    if not seed:
        return []
    if not seed.endswith(".xml") and "/sitemap" not in seed:
        # Best-effort: try /sitemap.xml on the host
        p = urlparse(seed)
        seed = f"{p.scheme or 'https'}://{p.netloc}/sitemap.xml"
    fr = fetch_mod.fetch(seed, timeout_s=20, policy_default_rate_s=1.0)
    if not fr.ok:
        return []
    return walker.extract_sitemap_urls(fr.text, max_urls=500)


def _seeds_rss_feed(seed: str) -> list[str]:
    if not seed:
        return []
    fr = fetch_mod.fetch(seed, timeout_s=20, policy_default_rate_s=1.0)
    if not fr.ok:
        return []
    entries = walker.extract_rss_entries(fr.text, max_entries=200)
    return [e["url"] for e in entries if e.get("url")]


def _seeds_table_column(seed: str, params: dict, client: NocodbClient,
                        url_to_row_out: dict[str, int] | None = None) -> list[str]:
    """Pull URLs from a NocoDB table column. Populates ``url_to_row_out`` (if
    provided) with URL → source row Id, avoiding a second full-table scan
    later in the runner.

    seed = ignored. params = {table, column, filter?, missing_only?, limit?}
    """
    table = (params.get("table") or "").strip()
    column = (params.get("column") or "url").strip()
    if not table:
        return []
    where_parts = []
    f = params.get("filter")
    if f:
        where_parts.append(str(f))
    if params.get("missing_only"):
        # NocoDB v2 uses (col,is,null) — `isblank` is silently ignored,
        # which would mean we scan the entire table. Explicit null check.
        miss_col = params.get("missing_field") or "summary"
        where_parts.append(f"({miss_col},is,null)")
    where = "~and".join(where_parts) if where_parts else ""
    limit = int(params.get("limit") or 500)
    try:
        rows = client._get_paginated(table, params={
            **({"where": where} if where else {}),
            "limit": limit,
        })
    except Exception:
        _log.warning("table_column fetch failed table=%s", table, exc_info=True)
        return []
    out: list[str] = []
    seen: set[str] = set()
    for r in rows:
        u = str(r.get(column) or "").strip()
        if u and u.startswith("http") and u not in seen:
            seen.add(u)
            out.append(u)
            if url_to_row_out is not None:
                rid = r.get("Id")
                if rid is not None:
                    url_to_row_out[u] = int(rid)
    return out


def _seeds_criteria_search(seed: str, params: dict, org_id: int, max_seeds: int) -> list[str]:
    """For job_board_collector / funding_tracker / etc. — like topic_search
    but lets the policy stuff site filters in via params."""
    sites = params.get("sites") or []
    base_q = seed or params.get("query") or ""
    if not base_q:
        return []
    queries = []
    if sites:
        queries.extend(f"{base_q} site:{s}" for s in sites)
    else:
        queries.append(base_q)
    out: list[str] = []
    seen: set[str] = set()
    for q in queries:
        urls = _seeds_topic_search(q, params, org_id, max_seeds)
        for u in urls:
            if u not in seen:
                seen.add(u)
                out.append(u)
            if len(out) >= max_seeds:
                return out
    return out


def _resolve_seeds(policy: HarvestPolicy, run: dict, client: NocodbClient,
                   org_id: int,
                   url_to_row_out: dict[str, int] | None = None) -> list[str]:
    """Resolve seeds. For table_column strategy, populates ``url_to_row_out``
    with URL → source row Id so the runner does not have to perform a second
    full-table scan to map URLs back to rows for knowledge_update / graph_node
    persistence.
    """
    seed = run.get("seed") or ""
    params = _params(run)
    sp = policy.seed_strategy
    sp_params = {**policy.seed_strategy_params, **params}
    max_seeds = int(sp_params.get("max_seeds") or policy.max_pages)

    if sp == "literal_url":
        return _seeds_literal_url(seed)
    if sp == "url_list":
        return _seeds_url_list(seed, sp_params)
    if sp == "topic_search":
        return _seeds_topic_search(seed, sp_params, org_id, max_seeds)
    if sp == "sitemap_expand":
        return _seeds_sitemap_expand(seed)
    if sp == "rss_feed":
        return _seeds_rss_feed(seed)
    if sp == "table_column":
        return _seeds_table_column(seed, sp_params, client, url_to_row_out=url_to_row_out)
    if sp == "criteria_search":
        return _seeds_criteria_search(seed, sp_params, org_id, max_seeds)
    _log.warning("unknown seed_strategy: %s", sp)
    return []


# ── per-URL processing ────────────────────────────────────────────────────

# Per-URL outcome tags used by run_harvest to track distinct phases.
_OUT_PERSISTED = "persisted"
_OUT_UNCHANGED = "unchanged"     # 304 Not Modified — no extract, no persist
_OUT_SKIPPED = "skipped"         # robots / rate-limited / cool-off
_OUT_FETCH_FAIL = "fetch_fail"
_OUT_EXTRACT_FAIL = "extract_fail"
_OUT_PERSIST_FAIL = "persist_fail"


def _process_url(url: str, *, policy: HarvestPolicy, run_id: int, run: dict,
                 client: NocodbClient, org_id: int,
                 artifacts_buffer: list,
                 source_row_id: int | None = None,
                 source_table: str | None = None) -> tuple[str, list[str]]:
    """Fetch + extract + persist one URL.

    Returns (outcome, child_urls). ``outcome`` is one of the _OUT_* constants
    so the caller tracks distinct phases (persisted vs unchanged vs failed).
    Artifact-mode persistence appends to ``artifacts_buffer`` for a single
    flush at end-of-run rather than read-modify-writing harvest_runs once
    per URL (avoids quadratic cost + write race).
    """
    # broken_link_sweep — HEAD only, no LLM, flag dead/moved URLs.
    if policy.head_only:
        head_fr = fetch_mod.head(
            url,
            timeout_s=15,
            policy_default_rate_s=policy.rate_limit_per_host_s,
            policy_respects_robots=policy.respect_robots,
            org_id=org_id,
        )
        if head_fr.skipped_reason:
            return _OUT_SKIPPED, []
        is_dead = not (200 <= head_fr.status_code < 400)
        if source_row_id is not None and source_table:
            persister.persist_knowledge_update(
                table=source_table, row_id=source_row_id,
                fields={"dead_url": is_dead, "last_check_status": head_fr.status_code or 0},
                client=client,
            )
            return _OUT_PERSISTED, []
        return _OUT_PERSIST_FAIL, []

    # Conditional GET — feed prior etag / last-modified.
    prior = url_cache.get(url) or {}
    fr = fetch_mod.fetch(
        url,
        timeout_s=30,
        policy_default_rate_s=policy.rate_limit_per_host_s,
        policy_respects_robots=policy.respect_robots,
        headless_fallback=False,  # opt-in via host_config.headless_fallback
        if_none_match=prior.get("etag") or None,
        if_modified_since=prior.get("last_modified") or None,
        org_id=org_id,
    )

    # 304 — distinct phase, do NOT count as persisted.
    if fr.skipped_reason == "not_modified":
        _log.info("harvest 304 url=%s (unchanged since last fetch)", url[:120])
        return _OUT_UNCHANGED, []

    if fr.skipped_reason:
        _log.info("harvest skip url=%s reason=%s", url[:120], fr.skipped_reason)
        return _OUT_SKIPPED, []

    if not fr.ok:
        _log.info("harvest fetch_failed url=%s status=%d err=%s",
                  url[:120], fr.status_code, fr.error)
        if policy.persist_target == "knowledge_update" and source_row_id is not None and source_table:
            # Mark fetched-but-failed so a future broken-link sweep picks it up.
            persister.persist_knowledge_update(
                table=source_table, row_id=source_row_id,
                fields={"last_check_status": fr.status_code or 0},
                client=client,
            )
        return _OUT_FETCH_FAIL, []

    # Always cache etag/last-modified after a successful 200, regardless of
    # downstream extract/persist outcome — saves bandwidth on the next run.
    if fr.etag or fr.last_modified:
        try:
            url_cache.set_state(
                fr.final_url or url,
                etag=fr.etag, last_modified=fr.last_modified,
                content_hash=fr.content_hash,
            )
        except Exception:
            pass

    # Extraction — only run summary if the persist target wants it AND policy
    # allows LLM cost (max_cost_usd > 0). knowledge_update with no schema
    # still gets a summary so we can populate the row's summary column;
    # broken_link_sweep already returned above.
    do_summary = policy.persist_target in ("knowledge", "knowledge_update")
    if policy.persist_target == "graph_node":
        do_summary = False  # graph nodes are property bags, not prose

    er = ext_mod.extract(
        fr.text,
        content_type=fr.content_type,
        schema=policy.extract_schema,
        summarise=do_summary,
    )
    if not er.ok:
        return _OUT_EXTRACT_FAIL, []

    # Persistence per policy
    persisted = False
    if policy.persist_target == "knowledge":
        persisted = persister.persist_knowledge(
            url=fr.final_url or url,
            summary=er.summary,
            fields=er.fields or None,
            org_id=org_id,
            run_id=run_id,
            policy_name=policy.name,
            topic=run.get("seed") or "",
        )
    elif policy.persist_target == "knowledge_update":
        if source_row_id is not None and source_table:
            update_fields: dict = {}
            if er.summary:
                update_fields["summary"] = er.summary[:5000]
            if er.fields:
                update_fields.update({k: v for k, v in er.fields.items() if v is not None})
            persisted = persister.persist_knowledge_update(
                table=source_table, row_id=source_row_id,
                fields=update_fields, client=client,
            )
    elif policy.persist_target == "graph_node":
        if source_row_id is not None and source_table and er.fields:
            persisted = persister.persist_knowledge_update(
                table=source_table, row_id=source_row_id,
                fields=er.fields, client=client,
            )
    elif policy.persist_target == "artifacts":
        item = {"url": fr.final_url or url, "summary": er.summary, **(er.fields or {})}
        artifacts_buffer.append(item)
        persisted = True

    if not persisted:
        return _OUT_PERSIST_FAIL, []

    # Walk children if enabled
    children: list[str] = []
    if policy.walk_enabled and "html" in (fr.content_type or "").lower():
        children = walker.extract_links(
            fr.text, base_url=fr.final_url or url,
            same_host_only=policy.walk_same_host_only,
            link_class=policy.walk_link_class,
            url_pattern=policy.walk_url_pattern,
            max_links=policy.walk_max_pages,
        )

    return _OUT_PERSISTED, children


# ── public handler ────────────────────────────────────────────────────────

def run_harvest(run_id: int) -> dict:
    """Tool-queue handler. Synchronously runs a harvest from the
    `harvest_runs` row at ``run_id``."""
    t0 = time.time()
    client, run = _load_run(run_id)
    if not run:
        return {"status": "not_found", "run_id": run_id}

    policy_name = run.get("policy") or ""
    policy = get_policy(policy_name)
    if policy is None:
        _patch(client, run_id, {
            "status": "failed",
            "error_message": f"unknown policy: {policy_name}",
            "finished_at": _now_iso(),
        })
        return {"status": "failed", "run_id": run_id, "error": f"unknown policy: {policy_name}"}

    org_id = resolve_org_id(run.get("org_id"))
    if org_id <= 0:
        org_id = 1

    _patch(client, run_id, {"status": "planning", "started_at": _now_iso()})
    _log.info("harvest:%s:%d START seed=%r", policy.name, run_id, (run.get("seed") or "")[:120])

    # 1. Seeds — table_column populates url_to_row directly so we never need
    #    a second full-table scan.
    url_to_row: dict[str, int] = {}
    try:
        seeds = _resolve_seeds(policy, run, client, org_id, url_to_row_out=url_to_row)
    except Exception as e:
        _log.error("harvest seed resolution failed run=%d", run_id, exc_info=True)
        _patch(client, run_id, {
            "status": "failed",
            "error_message": f"seed_resolution_error: {str(e)[:300]}",
            "finished_at": _now_iso(),
        })
        return {"status": "failed", "run_id": run_id, "error": str(e)[:200]}

    seeds = seeds[: policy.max_pages]
    _patch(client, run_id, {"status": "fetching", "urls_planned": len(seeds)})
    if not seeds:
        _patch(client, run_id, {
            "status": "completed",
            "error_message": "no seed URLs resolved",
            "finished_at": _now_iso(),
        })
        return {"status": "completed", "run_id": run_id, "note": "no_seeds"}

    # 2. Walk
    visited: set[str] = set()
    queue: deque = deque((u, 0) for u in seeds)
    fetched = 0
    persisted_count = 0
    unchanged_count = 0
    skipped_count = 0
    failed = 0
    artifacts_buffer: list[dict] = []
    deadline = time.time() + policy.timeout_total_s

    source_table = _params(run).get("table") or ""
    source_table = source_table.strip() or None

    # Tail buffer of per-URL events for the Live drawer. Flushed into
    # artifacts_json["events"] on every progress patch so the API can
    # serve a recent log without re-reading per-row state.
    event_log: list[dict] = []
    walked_overflow: list[str] = []  # URLs the walker discovered but couldn't enqueue

    while queue and time.time() < deadline:
        url, depth = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        if len(visited) > policy.max_pages:
            break

        fetched += 1
        outcome, children = _process_url(
            url, policy=policy, run_id=run_id, run=run,
            client=client, org_id=org_id,
            artifacts_buffer=artifacts_buffer,
            source_row_id=url_to_row.get(url),
            source_table=source_table,
        )
        if outcome == _OUT_PERSISTED:
            persisted_count += 1
        elif outcome == _OUT_UNCHANGED:
            unchanged_count += 1
        elif outcome == _OUT_SKIPPED:
            skipped_count += 1
        else:
            failed += 1

        event_log.append({
            "ts": _now_iso(),
            "url": url[:300],
            "outcome": outcome,
            "depth": depth,
        })
        if len(event_log) > _EVENT_LOG_MAX:
            event_log = event_log[-_EVENT_LOG_MAX:]

        # Periodic progress patch + yield slot. Default cadence is every 3
        # URLs (was 10) so the Live UI updates promptly. Tunable via
        # features.harvest.yield_every_n_urls / yield_pause_s in config.json.
        try:
            from infra.config import get_feature
            yield_every = int(get_feature("harvest", "yield_every_n_urls", 3) or 3)
            yield_pause = float(get_feature("harvest", "yield_pause_s", 1) or 1)
        except Exception:
            yield_every, yield_pause = 3, 1.0
        if yield_every > 0 and fetched % yield_every == 0:
            _patch_progress(client, run_id, {
                "urls_fetched": fetched,
                "urls_persisted": persisted_count,
                "urls_unchanged": unchanged_count,
                "urls_skipped": skipped_count,
                "urls_failed": failed,
            }, events_tail=event_log)
            if yield_pause > 0:
                time.sleep(yield_pause)

        # Enqueue children only if we still have headroom for them.
        # Anything past headroom becomes a candidate for the suggestions
        # bridge — pathfinder will pick those up.
        if children and policy.walk_enabled and depth + 1 < policy.walk_max_depth:
            headroom = policy.max_pages - (len(visited) + len(queue))
            for i, c in enumerate(children):
                if c in visited:
                    continue
                if i < headroom:
                    queue.append((c, depth + 1))
                elif len(walked_overflow) < _WALKED_OVERFLOW_MAX:
                    walked_overflow.append(c)

    # 3. Single flush for artifacts mode — avoids quadratic
    #    read-modify-write on harvest_runs.artifacts_json per item.
    if artifacts_buffer:
        try:
            persister.persist_artifact_batch(
                run_id=run_id, key=policy.name, items=artifacts_buffer,
                mode=policy.persist_mode, client=client,
            )
        except Exception:
            _log.warning("harvest artifact flush failed run=%d", run_id, exc_info=True)

    # 4. Bridge: feed candidate URLs into pathfinder via suggested_scrape_targets.
    #    Walker overflow is the cheapest signal — URLs the harvest found but
    #    couldn't process this run because of max_pages / max_depth. Pathfinder
    #    then promotes them through the existing approve flow.
    seeded_suggestions = 0
    if walked_overflow:
        try:
            seeded_suggestions = _seed_suggestions_from_overflow(
                client=client, run_id=run_id, policy_name=policy.name,
                org_id=org_id, urls=walked_overflow,
            )
        except Exception:
            _log.warning("suggestion seeding failed run=%d", run_id, exc_info=True)

    elapsed = round(time.time() - t0, 1)
    # Status logic:
    #   completed = at least some URLs landed in persistence (or stayed
    #               unchanged via 304); harvest did its job.
    #   failed    = we fetched things but nothing made it through extract/persist.
    if persisted_count > 0 or unchanged_count > 0:
        final_status = "completed"
    elif fetched == 0:
        final_status = "completed"
    else:
        final_status = "failed"

    # Final write also pushes the residual event tail into artifacts_json so
    # the Live drawer has a complete record after status flips terminal.
    _patch_progress(client, run_id, {
        "status": final_status,
        "urls_fetched": fetched,
        "urls_persisted": persisted_count,
        "urls_unchanged": unchanged_count,
        "urls_skipped": skipped_count,
        "urls_failed": failed,
        "finished_at": _now_iso(),
    }, events_tail=event_log)
    _log.info(
        "harvest:%s:%d DONE %.1fs  fetched=%d persisted=%d unchanged=%d skipped=%d failed=%d seeded=%d",
        policy.name, run_id, elapsed,
        fetched, persisted_count, unchanged_count, skipped_count, failed, seeded_suggestions,
    )
    return {
        "status": final_status, "run_id": run_id,
        "fetched": fetched, "persisted": persisted_count,
        "unchanged": unchanged_count, "skipped": skipped_count, "failed": failed,
        "seeded_suggestions": seeded_suggestions,
    }


def _seed_suggestions_from_overflow(*, client: NocodbClient, run_id: int,
                                    policy_name: str, org_id: int,
                                    urls: list[str]) -> int:
    """Write walked-but-unfetched URLs into ``suggested_scrape_targets`` so
    pathfinder picks them up. Capped (default 50/run) and dedup'd against
    rows that already exist for this org.

    Disable per-run by setting ``features.harvest.seed_suggestions=false``.
    """
    if not urls:
        return 0
    try:
        from infra.config import get_feature, NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS
        if not bool(get_feature("harvest", "seed_suggestions", True)):
            return 0
        cap = int(get_feature("harvest", "suggestions_max_per_run", 50) or 50)
    except Exception:
        from infra.config import NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS
        cap = 50

    # Dedup within the candidate list first (preserve walker order).
    seen: set[str] = set()
    candidates: list[str] = []
    for u in urls:
        if not isinstance(u, str) or not u.startswith("http"):
            continue
        if u in seen:
            continue
        seen.add(u)
        candidates.append(u)
        if len(candidates) >= cap:
            break

    if not candidates:
        return 0

    # Skip URLs already known to suggested_scrape_targets for this org
    # (cheap check: one batched lookup, not a per-URL round trip).
    table = NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS
    existing: set[str] = set()
    try:
        rows = client._get_paginated(
            table,
            params={"where": f"(org_id,eq,{org_id})", "limit": 1000},
        )
        existing = {str(r.get("url") or "").strip() for r in rows if r.get("url")}
    except Exception:
        _log.warning("suggestion dedup lookup failed", exc_info=True)

    written = 0
    reason = f"discovered by harvest run #{run_id} ({policy_name})"
    for u in candidates:
        if u in existing:
            continue
        try:
            client._post(table, {
                "org_id": org_id,
                "url": u,
                "title": u[:200],
                "query": f"harvest:{policy_name}",
                "relevance": "medium",
                "score": 60,
                "reason": reason,
                "status": "pending",
            })
            written += 1
        except Exception:
            _log.warning("suggestion insert failed url=%s", u[:120], exc_info=True)
    if written:
        _log.info("harvest:%s:%d seeded %d suggestion(s) into pathfinder", policy_name, run_id, written)
    return written


def finalise_harvest(run_id: int) -> dict:
    """Reserved for future split-into-subjobs design. Currently a no-op
    that just confirms the run state — the synchronous runner already
    finalises in-line."""
    client, run = _load_run(run_id)
    if not run:
        return {"status": "not_found", "run_id": run_id}
    return {"status": run.get("status") or "unknown", "run_id": run_id}
