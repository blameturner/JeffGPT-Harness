# Major Uplift Plan

Comprehensive review of every user-facing surface, with 3–6 enhancements
each, plus 10 net-new features that fit the existing architecture.

## Per-feature enhancements

### Home Chat (PA chat)
1. **Token-aware history truncation** — current cap is char-based
   (`MAX_HISTORY_CHARS=8000`). Switch to a token estimator and pack to budget.
2. **"Why did you say that?"** — `GET /home/chat/{message_id}/grounding`
   returns the slices that grounded the reply (digest preface, PA loops
   cited, graph edges injected, RAG snippets, search sources).
3. **Inline citation markers** — when web search or graph context was used,
   append `[1]`, `[2]` markers and return a `sources[]` payload.
4. **Conversation forking** — `POST /home/chat/fork` clones rolling summary
   + last N msgs into a new conversation.
5. **Per-conversation memory pin** — pin a fact/loop/edge force-injected
   into CONTEXT every turn for a given conversation.
6. **Streaming context-refresh event** — push an SSE `context_refreshed`
   event when the post-turn extractor invalidates the cache.

### Memory (Chroma)
1. **Cross-collection reranker** — small LLM rerank pass over top-N for
   precision in `memory_ask`.
2. **Write-time near-dup detection** — query cosine > 0.95 first; merge
   into existing chunk's metadata instead of duplicating.
3. **Forget API** — `DELETE /home/memory/items/{chunk_id}` surfaced in
   the `memory_ask` sources panel.
4. **Provenance ledger** — every chunk carries `produced_by` (`chat`,
   `research_agent`, `digest`, `scraper`) so the UI can filter.
5. **Recall replay** — `GET /home/memory/recall/{request_id}` returns the
   exact hits a previous ask used.
6. **Memory health metrics** — dead chunks, redundant clusters, growth
   rate. New `GET /home/memory/health`.

### Knowledge Graph
1. **Edge confidence score** — combine `hits` + `weight` + recency into
   a `confidence ∈ [0,1]` returned from `/home/graph/search`.
2. **Path queries** — `POST /home/graph/path {from, to}` shortest weighted
   path between two entities.
3. **Manual merge / split** — `/home/graph/entity/merge` and `/split` so
   the user fixes alias-resolver misses inline.
4. **Entity timeline** — per-edge first/last seen + sparkline of activity.
   "This entity went quiet 30 days ago".
5. **Subgraph export** — `GET /home/graph/export?seed=X&hops=2&format=...`
   for visualisation tools.
6. **Graph diff** — what entities/edges changed in last N days; becomes
   a daily-digest section.

### Agents
1. **Dry-run** — `POST /agents/{name}/dry-run` returns rendered prompt +
   estimated tokens without calling the model.
2. **Versioning** — version `persona` and `system_prompt_template`;
   one-click revert.
3. **Eval harness** — `POST /agents/{name}/eval` runs against a registered
   test set, stores rubric scores in `agent_evals`.
4. **Capability badges** — declared `tools_allowed` + `output_schema`
   in `GET /agents/{name}`.
5. **Cost attribution** — `estimated_cost_usd` on `agent_runs` from
   per-model pricing config.
6. **Failure replay with edits** — retry button that opens a small editor
   for model/temp/prompt overlay.

### Daily Digest
1. **Per-cluster feedback** — add `cluster_id` to `digest_feedback`.
2. **Settings (cadence, time-of-day, exclusions)** — per-org config.
3. **Digest "ask back"** — synthesiser picks 1–2 questions whose answers
   would sharpen tomorrow's digest; flows into `assistant_questions`.
4. **Backfill** — for new orgs, retro-generate last 7 days from existing
   data.
5. **Cluster → insight threading** — link insights to the digest cluster
   they came out of.
6. **Delivery channels** — email/Slack via the existing SMTP shim.

### Insights
1. **Faceting** — `category`, `urgency`, `audience` columns; UI filters.
2. **Scoring** — `actionability`, `novelty`, `time_sensitivity` 0–1
   scores drive ordering.
3. **Subscriptions** — user names topics; topic_picker biases toward them.
4. **Convert to loop** — `POST /home/insights/{id}/to-loop` creates a
   PA open loop anchored on the insight.
5. **Insight clustering** — group related insights from last 30 days.
6. **Stale / superseded** — mark insights stale when underlying data is
   overtaken; show "superseded by N".

### Research (plans + agent)
1. **Plan templates** — competitive teardown / trend tracker / primer /
   due-diligence presets.
2. **Mid-flight steering** — pause / edit sub-topics / resume.
3. **Citation sanity check** — pre-completion LLM reviewer verifies every
   cited claim resolves to a real fetched chunk.
4. **Per-paragraph confidence** — synthesised papers carry a 0–1
   confidence and source chunk ids per paragraph.
5. **Auto-followup proposals** — propose 2–3 follow-up plans on completion.
6. **Source allow/deny lists** — per-plan or per-org domain rules.

### Planned Search
1. **Per-query approval** — accept/reject individual queries from the
   proposal, not all-or-nothing.
2. **Mid-execution progress** — stream per-query status (`searxng:running`,
   `scrape:N/M`).
3. **Pre-synthesis result review** — show fetched sources with relevance
   scores; user prunes before LLM synthesis.
4. **Repeat that search** — `POST /planned_search/repeat {message_id}`
   refires the same plan with optional refinement.
5. **Inline suggestion in chat** — when chat detects a search-shaped
   question, suggest the queries inline.

### Enrichment (pathfinder + scraper)
1. **Source health dashboard** — per-domain success rate, avg TTFB,
   last-error. New `GET /enrichment/sources/health`.
2. **Per-domain extractor presets** — config map (article/forum/docs/pdf)
   → extractor settings.
3. **Re-scrape TTL** — per-source TTL; auto-requeue stale targets where
   the page changes often.
4. **Canonical URL + content-hash dedupe** — kills `?utm=` and other
   variant duplicates.
5. **Robots / politeness window** — per-domain rate limit on
   `scrape_targets`.
6. **Activity-aware priority** — if user just chatted about X, bump
   pending scrape_targets matching X to priority 1.

### Tool Queue
1. **Per-type concurrency caps** — currently effectively global.
2. **DAG endpoint** — `GET /tool-queue/dag` returns dependency graph for
   in-flight `depends_on` chains.
3. **Dead-letter queue** — jobs failing N times in M hours go to a
   `tool_jobs_dead` bucket the user reviews.
4. **ETA estimates** — track median duration per type; expose
   `eta_seconds` on queued jobs.
5. **Backoff explainability** — return `{backoff: "active", reason:
   "chat_idle < 600s", clears_at: ...}`.
6. **Per-org fairness** — round-robin across orgs in the claim loop.

### PA (now folded into chat — finish the dispatcher)
1. **Move dispatcher implementation** — implement closure / connect /
   news_watch / serendipity / resurface as concrete handlers.
2. **Loop snooze** — new status `snoozed` with `snooze_until`.
3. **Mute confirmation surface** — `/home/pa/snapshot.muted[]` and
   `/home/pa/mute {phrase}` direct API.
4. **Per-org warmth-decay tuning** — config knobs (currently hardcoded).
5. **Weekly review move** — Sunday-only PA move that rolls up the week's
   resolved loops, completed research, dominant topics.
6. **PA explainability** — `/home/pa/last-move/{id}/explain` returns why
   a given proactive surface fired.

### Conversations / Chat (non-home)
1. **Tags + cross-conversation search**.
2. **Pin conversations**.
3. **Reply regeneration** — `POST /messages/{id}/regenerate`.
4. **Reply variants** — return 2 candidate replies.
5. **Voice input** — stub `/chat/transcribe` for Whisper.
6. **On-demand summarise** — `POST /conversations/{id}/summarise`.

### Code agent
1. **Per-codebase RAG defaults** — `rag_collection` bound to codebase id.
2. **Test runner integration** — run tests after edits; attach output
   as a tool result.
3. **`/code/ask`** — one-shot Q&A over a codebase.
4. **PR draft from conversation** — turn a code conversation into a PR
   description + commit message.
5. **Workspace diff replay** — visual replay of file diffs.
6. **Webhook re-index** — re-index on commit/PR.

### Stats / Ops
1. **SLO panel** — targets (chat p50 < 4s, queue depth < 50); breaches
   over last 24h.
2. **Anomaly detection** — flag days where tokens 3× baseline or error
   rate spikes.
3. **Cost projection** — 30d spend forecast per model.
4. **Per-agent cost roll-up**.
5. **Feature health matrix** — `GET /health/features` consolidates the
   5 partial health views.

### Scheduler
1. **Backfill on restart** — optionally run schedules that should have
   fired in the last N hours.
2. **Conditional schedules** — only fire if predicate holds.
3. **One-off schedules** — `POST /schedule/once {at, ...}` for "remind
   me to X tomorrow".
4. **Configured-vs-live diff** — between `agent_schedules` and
   apscheduler's live jobs.

### Home Dashboard
1. **Customisable layout** — user toggles panels; persisted on
   `user_settings`.
2. **SSE deltas** — push panel-level updates instead of forcing polls.
3. **Quick-action buttons per panel** — surface inline.
4. **"Today" / date toggle** — retarget dashboard to a chosen day.
5. **Multi-org switcher**.

### Assistant Questions
1. **Categories** — `kind` (preference / factual / scheduling).
2. **Auto-drafts** — pre-draft likely answers the user accepts/overrides.
3. **Auto-expire** — dismiss after N days; record `expired_at`.
4. **Bulk actions**.

---

## 10 net-new features

1. **Email triage agent** — wire the deferred Gmail MCP tool into an
   agent that triages inbox into `needs_reply` / `FYI` / `ignore`, with
   a draft reply for the first bucket. New dashboard panel.

2. **Calendar-aware PA briefs** — Google Calendar MCP tool feeds an agent
   that 30 min before each event assembles a "prep brief" pulling
   relevant graph entities, prior conversations, open loops involving
   the same people.

3. **Decisions log** — first-class table of *decisions* (rationale,
   alternatives, outcome), distinct from facts. Extractor sniffs chat
   for "we decided to X because Y" patterns.

4. **Watchlist** — user pins topics; a daily background agent runs light
   web search on each and only surfaces something when there's a genuine
   new development (delta against last week's results).

5. **Context packs** — named bundle of memory collections + graph
   entities + facts for a project. One toggle scopes chat / memory_ask
   / graph_search to that pack.

6. **Memory garden** — visualisation surface (Cytoscape) showing graph +
   memory + insights as a coherent map. Click a node → drill into the
   underlying chunk(s).

7. **Conversation playback** — given any conversation, generate a 30-second
   narrated summary for quick re-orientation.

8. **Smart paste** — paste an email / doc / transcript into chat; an
   agent classifies it (meeting notes / proposal / research / random)
   and routes it to the right table with the right metadata.

9. **Question pre-bake** — for recurring user patterns, pre-compute the
   answer overnight via the scheduler and surface it Monday morning.

10. **Audio digest ("briefing radio")** — TTS the daily digest to a
    2–3 min mp3 the user plays during commute.

---

## Suggested sequence

PA move dispatcher → decisions log → context packs → email/calendar
agents → memory garden → watchlist → audio digest → the rest. The first
three compound; the agents need MCP wiring; everything else is incremental.
