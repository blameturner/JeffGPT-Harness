# Agent Architecture

A small runtime that turns NocoDB rows into autonomous staff. Five loop **types** describe runtime behaviour; everything else (role, prompt, schedule, tools, APIs, artifacts) is a row edit.

---

## Principles

- **No code per role.** A new "Architect" or "Strategist" is a row in `agents`, not a file.
- **One inbox.** All triggers write to `assignments`. The runtime only polls one table.
- **Long-text in NocoDB.** Artifacts are rows with long-text columns, not files.
- **Tools and APIs are configured, not coded.** `connected_apis`, `connected_smtp`, `allowed_tools` are CSVs of references.
- **Lean.** Five small files: one base, five type loops, one dispatcher.

---

## File layout

```
workers/user_agents/
├── agent.py            # existing — stays for one-shot LLM calls
├── runtime.py          # NEW — assignment poll loop + dispatcher
├── context.py          # NEW — builds prompt context (persona, brief, RAG, APIs, vars)
├── types/
│   ├── __init__.py     # registry: TYPE -> handler
│   ├── base.py         # AgentTypeBase + helpers (tool loop, budgets)
│   ├── document.py     # Document type
│   ├── queue.py        # Queue type
│   ├── producer.py     # Producer type
│   ├── responder.py    # Responder type
│   └── supervisor.py   # Supervisor type
└── triggers/
    ├── cron.py         # extends scheduler.py — fires assignments
    ├── email_inbound.py # IMAP poll → assignments
    ├── api_endpoint.py # FastAPI route → assignments
    ├── webhook.py      # signed inbound → assignments
    └── table_watch.py  # diff poll → assignments
```

Total expected size: ~800 lines.

---

## Lifecycle of a run

```
trigger fires
  → assignments INSERT (status=queued)
  → runtime claims (status=running, claimed_at)
  → context.build(agent_row, assignment)
       persona + brief + pinned_context + variables
       + connected_apis usage_prompts
       + RAG context
       + type-specific inputs (target row, queue items, inbox message)
  → types[agent.type].run(context)
       loops up to max_iterations:
         plan → tool calls (queued via tool_queue) → observe → continue/stop
       writes artifact according to type_config_json
       respects requires_approval_for (writes to agent_approvals, pauses)
  → assignments UPDATE (status=completed, result_*)
  → notify_on_complete dispatch
  → trigger_on_completion_of children created
```

---

## Five type contracts

Each type implements one method:

```python
def run(ctx: RunContext) -> RunResult: ...
```

`RunContext` carries: `agent_row`, `assignment_row`, `tools`, `apis`, `secrets`, `db (NocodbClient)`, `model_call(messages) -> str`, `budgets`. All five share the **tool loop** (in `base.py`):

```python
def tool_loop(ctx, system_prompt, user_prompt) -> str:
    messages = [system, user]
    for i in range(ctx.budgets.max_iterations):
        response = ctx.model_call(messages)
        actions = parse_tool_calls(response)
        if not actions:
            return final_text(response)
        results = run_actions_via_tool_queue(ctx, actions)
        messages += [response, format_results(results)]
    return last_text(messages)
```

### Document
- Reads target row+column.
- Tool-loops with `body=current_text` injected into context.
- Writes back according to `edit_mode` (replace | append | patch_section).
- One assignment = one revision pass.

### Queue
- Selects N rows by `filter`, ordered by `done_column ASC NULLS FIRST`.
- For each: builds row context, tool-loops, writes `output_column`, sets `done_column = now`.
- One assignment = one batch.

### Producer
- Tool-loops with the assignment task as input.
- Parses output (LLM emits JSON if `output_schema_json` set, else single field).
- INSERTs a new row using `column_map`.
- One assignment = one new row.

### Responder
- Loads inbound message from `assignment.source_meta_json`.
- Tool-loops to draft a reply.
- If `reply_mode == "auto"`: send via configured channel (SMTP / api).
- If `reply_mode == "approval"`: write `agent_approvals`, pause. Approval webhook resumes and sends.
- Logs to `log_table`.

### Supervisor
- Receives task, doesn't loop on its own — reads its `team_agent_ids`, picks targets, INSERTs new `assignments` for each (with `parent_assignment_id`).
- Optionally waits (poll) for children to complete, then synthesises.
- Or fire-and-forget: completes immediately after dispatch.

---

## Context assembly (`context.py`)

The single boring function that determines output quality:

```python
def build_context(agent_row, assignment) -> str:
    parts = []
    parts.append(persona_with_variables(agent_row))         # system_prompt + interpolation
    if pinned := agent_row.get("pinned_context"):
        parts.append(pinned)
    parts.append(f"BRIEF: {agent_row.get('brief','')}")
    for api in resolve_apis(agent_row.get("connected_apis")):
        parts.append(f"\n# API: {api['name']}\n{api['usage_prompt']}")
    if rag_enabled(agent_row):
        parts.append(retrieve_rag(query=assignment.task, agent=agent_row))
    parts.append(type_specific_input(agent_row, assignment))
    return "\n\n".join(parts)
```

`api['usage_prompt']` is the LLM-friendly text written by `inspect_api()` — this is why API inspection matters: it produces the very text that goes into every agent prompt that uses the API.

---

## Tool integration

- Existing `tools/dispatcher.py` + `tools/contract.py` stays.
- New `ToolName.HTTP_REQUEST` accepts `{ connection, method, path, params, body, headers }`. Resolves the connection by id or name; injects auth, base_url, default headers; enforces `allowed_methods` and `allowed_paths_regex`; rate-limits.
- New `ToolName.SEND_EMAIL` accepts `{ smtp_account, to, subject, body_text, body_html }`. Resolves the account, fetches the password secret, sends.
- Both honour `agents.requires_approval_for` — if listed, the tool dispatcher writes `agent_approvals` and raises `ApprovalPending`, which the type loop catches and propagates as a paused assignment.

---

## Triggers wiring

Cron path already exists (`scheduler.py` reads `agent_schedules`). Refactor minimally so cron fires write `assignments` instead of POSTing to `/run`. Other triggers add files under `workers/user_agents/triggers/` and register at app startup.

```
agent_schedules row → cron tick → INSERT assignment(source=cron) → runtime claims
inbound IMAP → match address to agent → INSERT assignment(source=email)
POST /agents/{slug}/run → INSERT assignment(source=api)
POST /agents/{id}/webhook (signed) → INSERT assignment(source=webhook)
table_watch poll → diff → INSERT assignment(source=table_watch) per change
```

The runtime is unchanged regardless of trigger — it just pulls rows.

---

## Implementation order

1. **Tables** — create the schemas in `docs/new-tables.md`. Stop the runtime work until they exist.
2. **`runtime.py` + `types/base.py`** — claim, dispatch, tool loop, budgets. Skeleton handlers that return a stub.
3. **`types/document.py`** — first concrete type. Good test bed because it's read-modify-write on one row.
4. **`context.py`** — wire APIs (`connected_apis` → `usage_prompt`), RAG, variables.
5. **HTTP_REQUEST tool** — minimum viable shape, no approval gate yet. Test by configuring a connection (e.g. JSONPlaceholder) and giving a Document agent the brief "fetch /todos/1 and append the title to your body."
6. **`types/producer.py`, `queue.py`** — extend tests.
7. **Cron trigger refactor** — point at `assignments` instead of `/run`.
8. **`types/responder.py`** + SEND_EMAIL tool + email inbound trigger.
9. **`types/supervisor.py`** + chained assignments.
10. **Approval gate** — `agent_approvals` write path + resume path.

Stop after each step and run a real agent end-to-end before moving on.

---

## Robustness features

**Concurrency.** Workers claim assignments atomically (`claimed_by_worker`, `claimed_at`). While running they patch `heartbeat_at` every 15s. Stale claims (`heartbeat_at` older than `heartbeat_ttl_seconds`) are reclaimable — recovers from crashed workers without manual intervention. `agents.max_concurrent_runs` caps how many of one agent run at once.

**Retry + circuit breaker.** Each assignment carries `attempts`, `max_attempts`, `next_retry_at` (exponential backoff). On error, agent's `consecutive_failures` counter increments; once it crosses `circuit_breaker_threshold`, the agent flips to `active=false` and writes an incident to `agent_incidents`. A successful run resets the counter.

**Output validation + reflection.** If `output_schema_json` is set, LLM output is JSON-validated; on failure, the runtime reprompts with the validator error, up to `max_validation_retries`. If `reflect=true`, output passes through a self-critique step before commit. `confidence_threshold` lets the agent self-rate; below threshold routes to `agent_approvals` instead of writing.

**Cost + rate limits.** Per-run ceilings (`max_iterations`, `max_runtime_seconds`, `max_tokens_per_run`) and per-day caps (`max_runs_per_day`, `max_tokens_per_day`, `max_cost_usd_per_day`) are checked before claim. Counters live on the agent row and reset nightly.

**Audit log.** `agent_runs` gets `events_jsonl` (one line per LLM call, tool call, approval, write), `prompt_snapshot`, `prompt_version`, `tokens_*`, `cost_usd`. Full traceability without a separate logging system.

**Artifact versioning.** Every write to a target row also writes a row to `artifact_versions` (`agent_id`, `assignment_id`, `table`, `row_id`, `column`, `before_text`, `after_text`). Rollback = pick a version and reapply.

**Dedup + memoization.** `assignments.dedup_key` blocks duplicate inserts. `agents.memoize_ttl_seconds`: if a recent completed assignment has the same dedup_key within TTL, return its result without re-running. Tool results cached by params for `tool_cache_ttl_seconds` (HTTP GET only).

**Sandboxing.** `allowed_outbound_hosts_regex` restricts where HTTP_REQUEST can call. `forbidden_tables` overrides any allowlist. `dry_run` logs intended writes/sends instead of executing them. `test_mode` runs end-to-end but never persists artifacts — returns the would-be diff.

**Hooks + fallbacks.** `pre_run_hook`, `post_run_hook` (dotted Python paths) for custom validation or postprocessing. `fallback_model` and `fallback_agent_id` kick in when primary fails. `on_error_action`: `retry | escalate | pause | fallback`.

**Templates + cloning.** `agent_templates` row → `clone_template(template_id, overrides)` produces a configured agent with one call. Lets you ship preset roles (Architect, Strategist, etc.) without code.

**Surfaces.** `surface_kind` like `conversation:42` makes the agent's completion message appear in that conversation thread. Same agent can also stream chunks during a long run.

---

## What this avoids

- No per-role code. One row creates an "Architect", another creates a "Strategist".
- No background daemons per agent. One runtime polls `assignments`.
- No bespoke LLM call per agent. The shared `tool_loop` does it.
- No new schedule mechanism. `assignments` row generated by existing cron, plus new trigger files that all do the same insert.
