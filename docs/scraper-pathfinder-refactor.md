# Harvest System — Scraper / Pathfinder Refactor & Extension

A complete design for layering a generic **harvest** system on top of the
existing scraper / pathfinder. Adds breadth (20 harvest policies) and
quality of life (rate limiting, conditional GET, headless fallback,
format loaders, manual triggers) **without** breaking the current
pipeline and **without** materially changing the database.

## Constraints

1. **No breakage.** Existing cron pipeline (`discover_agent` →
   `suggested_scrape_targets` → manual approve → `pathfinder` →
   `scrape_targets` → `scraper` → `knowledge`) keeps running unchanged.
2. **Minimal DB changes.** One new table, zero column additions to
   existing tables.
3. **Reuse existing queue.** Harvest jobs run on the existing tool
   queue with new handler types — no new queue infrastructure.
4. **Bounded.** Every harvest has hard caps on pages, depth, time, and
   LLM cost. No open-ended walks.
5. **Composable.** New harvest patterns are mostly configuration, not
   code. Adding the 21st is one small file.

---

## Database changes (only critical ones)

### New: `harvest_runs` (one table)

| field | type | purpose |
|---|---|---|
| `Id` | int | NocoDB primary |
| `policy` | SingleLineText | policy name (`topic_seeder`, `domain_crawler`, …) |
| `seed` | SingleLineText | the input that started this run (URL / topic / table+column / criteria JSON) |
| `params_json` | LongText | full input parameters as JSON, for retry / audit |
| `status` | SingleLineText | `queued / planning / fetching / extracting / persisting / completed / failed / cancelled` |
| `urls_planned` | Number | count of seed URLs after planning |
| `urls_fetched` | Number | count successfully fetched |
| `urls_extracted` | Number | count with successful extraction |
| `urls_persisted` | Number | count written to a target table |
| `urls_failed` | Number | count that errored |
| `artifacts_json` | LongText | per-policy structured outputs (diffs, structured rows, summary stats) — same pattern as `research_plans.schema._artifacts` |
| `cost_usd` | Decimal | accumulated LLM cost |
| `started_at` / `finished_at` | DateTime | bounds |
| `error_message` | LongText | last error if failed |
| `org_id` | Number | tenant scope |
| `parent_run_id` | Number \| null | for retries / re-runs (chains) |

**That's the only new table.** Everything else fits into existing
tables:

- The fetch queue uses the existing `scrape_targets` row pattern (rows
  are inserted with a `harvest_run_id` carried in the existing
  metadata field — see "Provenance" below).
- Content output goes to the existing `knowledge` table unchanged.
- Graph extraction uses the existing graph tables unchanged.
- Per-policy structured outputs (competitor diffs, job postings,
  funding rounds, etc.) live in `harvest_runs.artifacts_json` until /
  unless they earn their own table.

### Provenance (no new columns)

A harvest needs to know "which knowledge rows came from this run".
Three options, in order of preference:

1. **Stash run-id in the existing metadata column.** `knowledge` rows
   in this codebase typically have a free-form `metadata` or `extra`
   JSON column (used by research's `extra_metadata`). Add
   `{"harvest_run_id": N, "harvest_policy": "..."}` to that. Zero
   schema change.
2. **If no metadata column exists**, ride the existing `source` /
   `tags` / `notes` text column with a parseable prefix like
   `harvest:42:topic_seeder` — ugly but workable as a stopgap.
3. **Last resort**: add one column. Postpone until #1 and #2 prove
   insufficient.

The existing scraper's `extra_metadata` (passed to `ingest_output`) is
the natural place; `harvest_run_id` joins cleanly.

---

## Module layout

```
tools/harvest/
├── __init__.py
├── fetcher.py         # 1 module: HTTP fetch + content-type loader plugins
├── walker.py          # 1 module: link extraction strategies
├── extractor.py       # 1 module: summariser + schema-driven JSON extractor
├── persister.py       # 1 module: write to knowledge / update existing rows / append to artifacts
├── policy.py          # HarvestPolicy dataclass + registry
├── runner.py          # 1 module: orchestrator (single tool-queue handler)
├── rate_limit.py      # per-host token bucket + cool-off
├── robots.py          # robots.txt cache + check
└── policies/
    ├── __init__.py
    ├── topic_seeder.py
    ├── domain_crawler.py
    ├── sitemap_harvester.py
    ├── citation_crawler.py
    ├── ...            # one file per policy, ~20-40 lines each
    └── feed_watcher.py
```

Existing files (`scraper.py`, `pathfinder.py`, `discover_agent.py`,
`dispatcher.py`, `summariser.py`) are unchanged in v1. They get small
internal updates in Phase 1 to delegate fetch/walk to the new core, but
their **public functions and queue handlers stay identical**.

---

## Data flow

```
        ┌────────────────────┐
        │  POST /harvest/X   │  (or scheduled trigger)
        └─────────┬──────────┘
                  │
                  ▼
        ┌────────────────────┐
        │  insert harvest_run│  status=queued
        │  enqueue 'harvest' │  payload={run_id}
        │   tool_queue job   │
        └─────────┬──────────┘
                  │
                  ▼   (Huey worker)
        ┌────────────────────┐
        │  runner.run(run_id)│
        └─────────┬──────────┘
                  │
                  ├── 1. plan() → seed URLs (LLM if needed)
                  │
                  ├── 2. for each seed: fetch + walk
                  │   (each fetch = own scrape_page subjob,
                  │    runs through existing scraper queue)
                  │
                  ├── 3. for each fetched page: extract
                  │   (LLM call; bounded by per-job cost cap)
                  │
                  ├── 4. persist
                  │   (knowledge rows + run.artifacts_json updates)
                  │
                  └── 5. mark run completed/failed
                       update urls_* counts
```

The runner does NOT loop synchronously through every URL. It enqueues
sub-jobs (existing `scrape_page` handler) and an aggregator that
finalises the run when all sub-jobs are done. This keeps any single
Huey worker free, lets long crawls run in parallel, and naturally
inherits the retry / backoff / heartbeat behaviour we already built.

---

## The `HarvestPolicy` contract

```python
# tools/harvest/policy.py
from dataclasses import dataclass, field
from typing import Callable, Literal

@dataclass(frozen=True)
class HarvestPolicy:
    name: str

    # ── Seed strategy ────────────────────────────────────────────
    # How the seed input becomes a list of URLs to start fetching.
    # Strategies (registered functions): "literal_url", "url_list",
    # "topic_search", "sitemap_expand", "rss_feed", "table_column",
    # "criteria_search".
    seed_strategy: str
    seed_strategy_params: dict = field(default_factory=dict)

    # ── Walk strategy ────────────────────────────────────────────
    # How discovered URLs are filtered for further fetching.
    walk_enabled: bool = False
    walk_max_depth: int = 1
    walk_max_pages: int = 50
    walk_same_host_only: bool = True
    walk_link_class: Literal["all", "article", "doc", "sitemap"] = "all"
    walk_url_pattern: str | None = None  # optional regex filter

    # ── Extraction ──────────────────────────────────────────────
    # None → just summary. Otherwise: structured field schema.
    extract_schema: dict | None = None
    extract_entities: bool = True

    # ── Persistence ─────────────────────────────────────────────
    # Where the output lands.
    persist_target: Literal[
        "knowledge",          # standard RAG ingestion
        "knowledge_update",   # update existing knowledge rows
        "graph_node",         # write/update graph node properties
        "artifacts",          # stash in harvest_runs.artifacts_json only
    ] = "knowledge"
    persist_mode: Literal["insert", "upsert", "diff"] = "insert"

    # ── Bounds ──────────────────────────────────────────────────
    rate_limit_per_host_s: float = 1.0
    max_pages: int = 100
    max_depth: int = 2
    max_cost_usd: float = 5.0
    timeout_total_s: int = 3600
    respect_robots: bool = True
```

**Adding a new policy** = create one `tools/harvest/policies/<name>.py`
that exports a single `POLICY = HarvestPolicy(...)` constant, plus
optionally one helper function for any custom extract logic. The
runner discovers policies at startup via the package `__init__.py`.

---

## The 20 policies as concrete configs

Showing each as the dataclass instantiation. Same runner runs all of
them; only the policy differs.

### Knowledge seeders

```python
# topic_seeder.py
POLICY = HarvestPolicy(
    name="topic_seeder",
    seed_strategy="topic_search",
    seed_strategy_params={"max_seeds": 20, "include_wikipedia": True},
    walk_enabled=False,
    persist_target="knowledge",
    max_pages=20, max_cost_usd=2.0,
)

# domain_crawler.py
POLICY = HarvestPolicy(
    name="domain_crawler",
    seed_strategy="literal_url",
    walk_enabled=True,
    walk_max_depth=2, walk_max_pages=80,
    walk_same_host_only=True, walk_link_class="article",
    persist_target="knowledge",
    max_pages=80, max_cost_usd=4.0,
    rate_limit_per_host_s=2.0,
)

# sitemap_harvester.py
POLICY = HarvestPolicy(
    name="sitemap_harvester",
    seed_strategy="sitemap_expand",
    walk_enabled=False,
    persist_target="knowledge",
    max_pages=200, max_cost_usd=8.0,
    rate_limit_per_host_s=1.5,
)

# citation_crawler.py
POLICY = HarvestPolicy(
    name="citation_crawler",
    seed_strategy="literal_url",
    walk_enabled=True,
    walk_max_depth=2, walk_max_pages=50,
    walk_same_host_only=False, walk_link_class="all",
    walk_url_pattern=r"(arxiv|doi|pubmed|jstor)",
    extract_entities=True,
    persist_target="knowledge",   # citation edges land in graph via extract_entities
    max_pages=50, max_cost_usd=3.0,
)

# newsletter_archive.py
POLICY = HarvestPolicy(
    name="newsletter_archive",
    seed_strategy="literal_url",
    walk_enabled=True,
    walk_max_depth=2, walk_max_pages=300,
    walk_same_host_only=True, walk_link_class="article",
    persist_target="knowledge",
    max_pages=300, max_cost_usd=10.0,
)

# forum_harvester.py
POLICY = HarvestPolicy(
    name="forum_harvester",
    seed_strategy="topic_search",
    seed_strategy_params={"sites": ["news.ycombinator.com", "reddit.com"]},
    walk_enabled=True,
    walk_max_depth=1, walk_max_pages=40,
    extract_schema={"question": "text", "top_answer": "text", "score": "numeric"},
    persist_target="knowledge",
    max_pages=40,
)
```

### Targeted intelligence

```python
# competitor_watcher.py
POLICY = HarvestPolicy(
    name="competitor_watcher",
    seed_strategy="url_list",
    walk_enabled=True,
    walk_max_depth=1, walk_max_pages=10,
    walk_url_pattern=r"(pricing|product|features|blog)",
    extract_schema={"section": "text", "headline": "text", "summary": "text"},
    persist_target="artifacts",   # diff records stay in harvest_runs.artifacts_json
    persist_mode="diff",
    max_pages=20,
)

# tooling_landscape.py
POLICY = HarvestPolicy(
    name="tooling_landscape",
    seed_strategy="criteria_search",
    seed_strategy_params={"max_seeds": 30},
    walk_enabled=True, walk_max_depth=1, walk_max_pages=3,
    walk_url_pattern=r"(pricing|features|about)",
    extract_schema={
        "name": "text", "vendor": "text",
        "primary_use_case": "text", "pricing_model": "text",
        "license": "text", "key_strength": "text", "key_weakness": "text",
    },
    persist_target="artifacts",
    persist_mode="upsert",
    max_pages=90,
)

# job_board_collector.py
POLICY = HarvestPolicy(
    name="job_board_collector",
    seed_strategy="criteria_search",
    seed_strategy_params={"sites": "<configured>"},
    walk_enabled=True, walk_max_depth=2, walk_max_pages=200,
    extract_schema={
        "title": "text", "company": "text", "location": "text",
        "remote": "text", "salary": "text", "posted_date": "date",
        "requirements": "text", "url": "text",
    },
    persist_target="artifacts",
    persist_mode="upsert",
    max_pages=200,
)

# funding_tracker.py / press_monitor.py / patent_watcher.py
# Same shape: criteria_search + url_pattern + extract_schema +
# persist_target="artifacts", persist_mode="upsert".
```

### Backfill / sweep

```python
# url_column_backfill.py
POLICY = HarvestPolicy(
    name="url_column_backfill",
    seed_strategy="table_column",
    seed_strategy_params={"missing_only": True},
    walk_enabled=False,
    persist_target="knowledge_update",   # writes back to source rows
    persist_mode="upsert",
    max_pages=500,
)

# stale_refresher.py
POLICY = HarvestPolicy(
    name="stale_refresher",
    seed_strategy="table_column",
    seed_strategy_params={"order_by": "updated_at", "asc": True},
    walk_enabled=False,
    persist_target="knowledge_update",
    persist_mode="upsert",
    max_pages=200,
)

# broken_link_sweep.py
POLICY = HarvestPolicy(
    name="broken_link_sweep",
    seed_strategy="table_column",
    walk_enabled=False,
    extract_schema=None,   # no LLM, just HEAD requests
    persist_target="knowledge_update",
    persist_mode="upsert",
    max_pages=2000,
    max_cost_usd=0.0,      # no LLM
)

# entity_profile_backfill.py
POLICY = HarvestPolicy(
    name="entity_profile_backfill",
    seed_strategy="table_column",     # graph_nodes table, name column
    walk_enabled=True, walk_max_depth=1, walk_max_pages=3,
    extract_schema={
        "role": "text", "employer": "text", "summary": "text",
        "homepage": "text", "social": "text",
    },
    persist_target="graph_node",
    persist_mode="upsert",
    max_pages=300,
)

# gap_closer.py — consumes a coverage_findings row
POLICY = HarvestPolicy(
    name="gap_closer",
    seed_strategy="topic_search",
    seed_strategy_params={"queries_from": "coverage_finding"},
    walk_enabled=False,
    persist_target="knowledge",
    max_pages=15, max_cost_usd=1.5,
)
```

### Personal / inbox

```python
# reading_list_ingest.py
POLICY = HarvestPolicy(
    name="reading_list_ingest",
    seed_strategy="url_list",
    walk_enabled=False,
    extract_schema={"reading_time_min": "numeric", "key_takeaway": "text"},
    persist_target="knowledge",   # tagged in metadata as reading_queue
    max_pages=200,
)

# bookmark_importer.py — bookmarks export, parse + scrape
POLICY = HarvestPolicy(
    name="bookmark_importer",
    seed_strategy="url_list",
    seed_strategy_params={"source": "bookmarks_html"},
    walk_enabled=False,
    extract_schema={"category": "text"},
    persist_target="knowledge",
    max_pages=2000,
)

# feed_watcher.py — polls RSS, ingests new items
POLICY = HarvestPolicy(
    name="feed_watcher",
    seed_strategy="rss_feed",
    walk_enabled=False,
    persist_target="knowledge",
    persist_mode="upsert",       # by guid
    max_pages=50,
)
```

---

## Endpoints

### Triggering

```
POST   /harvest/{policy}                  body: {seed, params}
                                          → 202 {run_id}
POST   /harvest/scrape-now                body: {url}
                                          → 202 {run_id}  (special-case literal_url)
POST   /harvest/bulk-upload               body: {urls: [...]}
                                          → 202 {run_id}  (special-case url_list)
```

### Inspection

```
GET    /harvest/runs                      list, with filters
GET    /harvest/runs/{id}                 full state of one run
GET    /harvest/runs/{id}/urls            per-URL outcomes
GET    /harvest/runs/{id}/artifacts       structured outputs (parsed from artifacts_json)
GET    /harvest/policies                  list registered policies + their schemas
```

### Lifecycle

```
POST   /harvest/runs/{id}/cancel          stop in-flight; PR-style soft cancel
POST   /harvest/runs/{id}/retry           re-run with same params; chain via parent_run_id
```

### Per-host config

```
GET    /harvest/hosts                     list per-host policies
PATCH  /harvest/hosts/{host}              {rate_limit_per_host_s, respect_robots,
                                          headless, cool_off_until, notes}
```

(Per-host config lives in a tiny KV section of an existing settings table or
in a `host_policies` JSON column — no new table required.)

---

## Queue handler integration

One new tool-queue handler type: `harvest_run`.

```python
# In app/lifespan.py:
from tools.harvest.runner import run_harvest

tool_queue.register("harvest_run", HandlerConfig(
    handler=lambda p: run_harvest(p["run_id"]),
    max_workers=2,
    priority_default=4,
    source="harvest",
))
```

Sub-jobs (per-URL fetch + extract) reuse the existing `scrape_page`
handler. The runner enqueues those, then enqueues a final
`harvest_finalise` job that depends on all the sub-jobs (using the
existing `depends_on` queue feature) — so when all per-URL work is
done, the finaliser updates `harvest_runs.status` to `completed` and
writes counts.

That's three handler types total. All ride the existing queue,
heartbeat, stale-reaper, and retry machinery — no new infrastructure.

---

## QoL features (all default-safe, all opt-in upgrades)

### Politeness

- **Per-host rate limiter** (`tools/harvest/rate_limit.py`) — token
  bucket keyed by host. Default 1 req/s per host. Configurable per
  host via the per-host config endpoint.
- **robots.txt** (`tools/harvest/robots.py`) — fetched once per host
  per 24h, cached in memory. Disable per-host for internal systems.
- **User-Agent declared** — `mst-harness/{version} (+contact)`. Some
  sites block unknown UAs.

### Conditional GET (cheap re-checks)

- The fetcher stores `etag` + `last_modified` for each URL in the
  existing `scrape_targets` row's metadata field (or a new minor
  one). Refresh jobs send `If-None-Match` / `If-Modified-Since` and
  short-circuit on `304`. Slashes the cost of `competitor_watcher`,
  `stale_refresher`, `feed_watcher`.

### Headless fallback

- When the cheap fetcher returns 403 / 0-byte / detects a JS-redirect
  page, the fetcher retries through Playwright once. Per-host decision
  cached so we don't pay headless on sites that don't need it.
- Strict global cap (default 1 headless fetch / minute / host) so a
  bug doesn't blow the budget.
- Off by default; opt-in via per-host config or a fetcher param.

### Format loaders (selected by content-type)

| Content type | Loader | Returns |
|---|---|---|
| `text/html` | existing scrape | text + links |
| `application/pdf` | pdfminer / unstructured | text + (optional) tables |
| `application/rss+xml`, `application/atom+xml` | feedparser | one entry per item |
| `application/xml` (sitemap) | lxml | list of `<loc>` URLs |
| `application/json`, `text/csv` | direct parse | structured data |
| YouTube URL | `youtube-transcript-api` | transcript as text |

All return the same `FetchResult` shape so the rest of the pipeline is
agnostic to source format.

### Authenticated fetch via existing `api_connections`

- The fetcher accepts a `connection_id` param. Pulls auth headers from
  the existing connectors registry. Unlocks scraping behind paywalls
  and internal dashboards.
- Body is passed through a `_redact_secrets` filter before any LLM
  call — known patterns (API keys, tokens, SSNs, credit cards) get
  masked. The LLM only ever sees redacted content.

### Per-URL content cache + near-dup merge

- Cache key: `(url, content_hash)`. If we've fetched this exact
  content before, skip the LLM and return the cached extract.
- During extract, embed the text and check cosine to existing
  `knowledge` rows. If > 0.95 similarity, **upsert** the existing row
  (bumping `last_seen`) rather than insert a duplicate.
- Cuts repeat cost dramatically and prevents corpus bloat from
  re-runs.

### Retry budgets + cool-off

- Per-host failure counter in memory + persisted on the per-host
  config. After N consecutive failures (default 5), the host enters
  cool-off for an hour. Logs once per cool-off; existing scraper's
  `failures` column already tracks per-URL.
- Per-job cost cap (`policy.max_cost_usd`) — the runner aborts cleanly
  if the next LLM call would exceed it. The harvest is marked
  `completed_partial` with the artifacts gathered up to that point.

### Manual overrides

- `POST /harvest/scrape-now` — bypasses the suggestion / approval
  flow, scrapes immediately, ingests to `knowledge`.
- `POST /harvest/bulk-upload` — bulk URL list, queues one harvest run
  with `seed_strategy="url_list"`.
- `POST /harvest/runs/{id}/cancel` — soft cancel (current sub-job
  finishes; no new sub-jobs enqueued; run marked `cancelled`).
- `POST /harvest/runs/{id}/retry` — re-runs with the same params,
  chained via `parent_run_id` so you can audit retry history.

### Observability

- Every run logs the same boundary lines we established in the queue
  refactor: `harvest:<policy>:<run_id> START`, `FETCH START/RETURN`,
  `EXTRACT START/RETURN`, `PERSIST`, `DONE` with elapsed and counts.
- A stuck harvest is greppable in 30 seconds.
- `GET /harvest/runs/{id}` returns the full counts + latest log
  excerpt, suitable for a UI status panel.

---

## Implementation patterns (concrete)

### `runner.run_harvest(run_id)`

```python
def run_harvest(run_id: int) -> dict:
    client, run = _load_run(run_id)
    if not run:
        return {"status": "not_found", "run_id": run_id}

    policy = REGISTRY.get(run["policy"])
    if not policy:
        _fail(client, run_id, f"unknown policy: {run['policy']}")
        return {"status": "failed"}

    _patch_run(client, run_id, {"status": "planning",
                                "started_at": _now_iso()})

    seeds = _resolve_seeds(policy, run["seed"], run["params_json"])
    _patch_run(client, run_id, {"urls_planned": len(seeds),
                                "status": "fetching"})

    # Enqueue per-URL sub-jobs through the existing scrape_page queue.
    # Each sub-job is small and bounded; runner doesn't loop over them.
    sub_job_ids = []
    for url in seeds[:policy.max_pages]:
        sub_id = _enqueue_scrape(url, run_id, policy)
        sub_job_ids.append(sub_id)

    # Final job depends on all sub-jobs and finalises the run.
    _enqueue_finalise(run_id, depends_on=sub_job_ids)
    return {"status": "in_flight", "run_id": run_id,
            "scheduled_urls": len(sub_job_ids)}
```

### Per-URL sub-job (rides existing `scrape_page` + adds extractor)

The existing `scrape_page` handler is wrapped with a thin
`harvest_scrape_page` adapter that, after the fetch + summarise, also:

1. Optionally walks links per `policy.walk_*` and enqueues child URLs
   (subject to depth + page caps tracked on the run).
2. Runs the structured extractor if `policy.extract_schema` is set.
3. Persists per `policy.persist_target` and `policy.persist_mode`.
4. Increments the relevant `urls_*` counter on `harvest_runs`.

### Persisters

```python
def persist(client, run, policy, fetch_result, extract_result):
    if policy.persist_target == "knowledge":
        _persist_knowledge(client, run, fetch_result, extract_result)
    elif policy.persist_target == "knowledge_update":
        _persist_knowledge_update(client, run, fetch_result, extract_result)
    elif policy.persist_target == "graph_node":
        _persist_graph_node(client, run, fetch_result, extract_result)
    elif policy.persist_target == "artifacts":
        _persist_artifact(client, run, policy, fetch_result, extract_result)
```

`_persist_artifact` appends a row-shaped object into
`harvest_runs.artifacts_json` (under a key matching the policy) — same
pattern as research's `schema._artifacts`. No new table needed; if a
particular policy's artifacts later prove valuable enough to query
heavily, *then* promote to a dedicated table.

### Finalisation

The `harvest_finalise` handler runs after all sub-jobs complete. It:

1. Aggregates counts from sub-jobs.
2. Marks `status=completed` (or `completed_partial` if the cost cap
   was hit, or `failed` if zero URLs persisted).
3. Writes `finished_at`.
4. Logs the closing summary line.

---

## Migration phases (no breakage)

### Phase 1 — extract primitives (no behaviour change)

- New files: `tools/harvest/{fetcher,walker}.py`.
- Move pure utilities from `scraper.py` and `pathfinder.py` into them.
- Existing modules import from new ones. Public functions and
  handlers unchanged.
- Validate against the existing cron pipeline producing identical
  output on a known seed.
- **Ships in: 1 PR. ~300 LOC moved, no logic changes.**

### Phase 2 — QoL (default-safe)

- `rate_limit.py` + `robots.py`. Defaults match current
  behaviour (no rate limit if no per-host config; respect robots is on
  by default but can be disabled per host).
- Conditional GET (writes etag/last-modified into existing metadata;
  reads on next fetch).
- Headless fallback (off by default; opt-in per host).
- **Ships in: 1 PR. Zero schema changes if the existing metadata
  field is JSON-shaped; otherwise one minor column.**

### Phase 3 — harvest runner + first three policies

- Create `harvest_runs` table.
- Implement `runner.py` + `policy.py` + `extractor.py` + `persister.py`.
- Register `harvest_run` and `harvest_finalise` queue handlers.
- Ship policies: `topic_seeder`, `url_column_backfill`,
  `competitor_watcher`. These three exercise all three persist
  targets (knowledge / knowledge_update / artifacts).
- Endpoints: `POST /harvest/{policy}`, `POST /harvest/scrape-now`,
  `GET /harvest/runs[/...]`.
- **Ships in: 1 PR. One new table, ~800 LOC of new code.**

### Phase 4 — full breadth

- Remaining 17 policies. Each is one small file in
  `tools/harvest/policies/`.
- Format loaders (PDF, RSS, sitemap, YouTube, JSON/CSV).
- Authenticated fetch via `api_connections`.
- Per-host config endpoint + UI.
- Failed-run review queue UI.
- **Ships incrementally — one policy or loader per PR.**

---

## What stays out of v1

- **Per-policy dedicated tables** (e.g. `competitor_changes`,
  `job_postings`, `funding_rounds`). They live in `artifacts_json`
  until usage volume warrants a real table.
- **A new metadata column on `knowledge`** if the existing one already
  accepts JSON (which it does for research).
- **Cross-run analytics** (total cost by policy, success rate, etc.).
  Calculable from `harvest_runs` rows on demand; no aggregate table
  needed.
- **A new dashboard page**. The existing `/tool-queue/*` endpoints
  already show queue depth and per-job status; harvest runs surface
  there via the new `harvest_run` job type. A dedicated UI is Phase 5
  if desired.

---

## Bottom line

- **One new table** (`harvest_runs`).
- **Zero new columns** on existing tables in the optimistic case
  (existing `metadata` JSON columns absorb the new fields). Worst
  case, one column on `knowledge` for `harvest_run_id`.
- **Five small modules** in a new `tools/harvest/` package, plus one
  small file per policy.
- **Two new queue handler types** (`harvest_run`, `harvest_finalise`),
  reusing the existing scraper handler for per-URL work.
- **Twenty harvest policies** as concrete `HarvestPolicy` configs;
  adding the 21st is one file.
- **All QoL** layered on the new fetcher/walker — benefits the
  existing scraper as a side effect.
- **Existing pipeline runs unchanged** the whole time.

The end state: anything that was reachable as "scrape one approved
URL" is still reachable that way; everything that was reachable as
"discover, suggest, approve, scrape" is still reachable that way; and
twenty new patterns (topic seed, domain crawl, sitemap harvest,
citation crawl, competitor watch, URL backfill, …) become available
through `POST /harvest/{policy}`.
