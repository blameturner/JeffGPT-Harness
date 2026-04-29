# New NocoDB Tables

Tables to create in NocoDB before the integration registries and the agent runtime can be used. All tables are scoped by `org_id`.

Column types use NocoDB names: `SingleLineText`, `LongText`, `Number`, `Checkbox`, `JSON`, `DateTime`.

---

## `secrets`

Credential store. Referenced by name from other tables.

| Column | Type | Notes |
|---|---|---|
| `name` | SingleLineText | unique per org, e.g. `stripe_live_key` |
| `org_id` | Number | |
| `kind` | SingleLineText | `api_key` \| `password` \| `oauth_token` \| `bearer` |
| `value` | LongText | the secret itself (consider an encryption column later) |
| `description` | LongText | what it is, where issued |
| `expires_at` | DateTime | optional |
| `rotated_at` | DateTime | optional |

---

## `api_connections`

One row per external API. Populated by `tools/integrations/api_registry.py`.

| Column | Type | Notes |
|---|---|---|
| `name` | SingleLineText | unique per org |
| `org_id` | Number | |
| `description` | LongText | human description |
| `base_url` | SingleLineText | no trailing slash |
| `auth_type` | SingleLineText | `none` \| `bearer` \| `basic` \| `api_key_header` \| `api_key_query` \| `oauth2` |
| `auth_secret_ref` | SingleLineText | name in `secrets` |
| `auth_extra_json` | JSON | `{header_name, query_name, username, scopes}` |
| `default_headers_json` | JSON | always-sent headers |
| `default_query_json` | JSON | always-sent query params |
| `openapi_url` | SingleLineText | optional explicit OpenAPI/Swagger URL |
| `allowed_methods` | SingleLineText | csv, default `GET` |
| `allowed_paths_regex` | SingleLineText | optional path allowlist |
| `rate_limit_per_min` | Number | |
| `timeout_seconds` | Number | |
| `usage_prompt` | LongText | written by `inspect_api()` — what the LLM reads to use this API |
| `inspection_summary_json` | JSON | raw probe output |
| `verified_at` | DateTime | |
| `verification_status` | SingleLineText | `unverified` \| `verified` \| `failed` |

---

## `smtp_accounts`

One row per outbound mailbox. Populated by `tools/integrations/smtp_registry.py`.

| Column | Type | Notes |
|---|---|---|
| `name` | SingleLineText | unique per org |
| `org_id` | Number | |
| `description` | LongText | |
| `host` | SingleLineText | smtp host |
| `port` | Number | |
| `username` | SingleLineText | |
| `password_secret_ref` | SingleLineText | name in `secrets` |
| `from_email` | SingleLineText | |
| `use_tls` | Checkbox | implicit TLS (port 465 style) |
| `use_starttls` | Checkbox | upgrade after EHLO (port 587 style) |
| `imap_host` | SingleLineText | optional, enables receipt-confirmation test |
| `imap_port` | Number | default 993 |
| `imap_username` | SingleLineText | defaults to `username` |
| `imap_password_secret_ref` | SingleLineText | defaults to `password_secret_ref` |
| `verified_at` | DateTime | |
| `verification_status` | SingleLineText | `unverified` \| `send_only` \| `verified` \| `failed` |
| `verification_note` | LongText | last test result detail |
| `last_test_message_id` | SingleLineText | RFC Message-ID of the last self-test |

---

## `agents` — additions to existing table

Add these columns to the existing `agents` table. Existing columns (name, model, system_prompt_template, persona, temperature, max_tokens, rag_*, products, org_id) stay as-is.

| Column | Type | Notes |
|---|---|---|
| `type` | SingleLineText | `document` \| `queue` \| `producer` \| `responder` \| `supervisor` |
| `description` | LongText | UI summary |
| `brief` | LongText | current focus, rewritable = reassignment |
| `pinned_context` | LongText | always prepended |
| `prompt_variables_json` | JSON | `{key: value}` interpolated as `{key}` into prompt + brief |
| `output_format` | SingleLineText | `markdown` \| `json` \| `html` \| `plain` |
| `output_schema_json` | JSON | optional, for structured output |
| `allowed_tools` | SingleLineText | csv |
| `tool_config_json` | JSON | per-tool overrides |
| `connected_apis` | SingleLineText | csv of `api_connections.id` |
| `connected_smtp` | SingleLineText | csv of `smtp_accounts.id` |
| `connected_secrets` | SingleLineText | csv of secret names |
| `allowed_tables_read` | SingleLineText | csv |
| `allowed_tables_write` | SingleLineText | csv |
| `type_config_json` | JSON | type-specific config — see below |
| `trigger_cron` | SingleLineText | optional cron expr |
| `trigger_timezone` | SingleLineText | default `Australia/Sydney` |
| `trigger_interval_minutes` | Number | optional |
| `trigger_email_address` | SingleLineText | optional inbound address |
| `trigger_api_slug` | SingleLineText | optional, exposes `POST /agents/{slug}/run` |
| `trigger_webhook_secret` | SingleLineText | optional |
| `trigger_supervisor` | Checkbox | accepts handoffs |
| `trigger_table_watch_json` | JSON | `{table, filter, on}` |
| `trigger_on_completion_of` | SingleLineText | csv of upstream agent ids |
| `run_window` | SingleLineText | e.g. `Mon-Fri 09:00-18:00` |
| `pause_until` | DateTime | |
| `active` | Checkbox | |
| `dry_run` | Checkbox | |
| `requires_approval_for` | SingleLineText | csv of action kinds |
| `approval_route` | SingleLineText | `user:N` or `agent:M` |
| `max_iterations` | Number | default 5 |
| `max_runtime_seconds` | Number | default 300 |
| `max_runs_per_day` | Number | optional |
| `parent_agent_id` | Number | chief link |
| `can_delegate_to` | SingleLineText | csv of agent ids |
| `notify_on_complete_json` | JSON | `[{channel, target}]` |
| `notify_on_error_json` | JSON | |
| `last_run_at` | DateTime | maintained by runtime |
| `last_run_status` | SingleLineText | |
| `last_run_summary` | LongText | |
| `prompt_version` | Number | bumped on system_prompt change; recorded per run |
| `max_concurrent_runs` | Number | default 1 |
| `max_tokens_per_run` | Number | optional |
| `max_tokens_per_day` | Number | optional |
| `max_cost_usd_per_day` | Number | optional |
| `runs_today` | Number | runtime-maintained, resets nightly |
| `tokens_today` | Number | runtime-maintained |
| `cost_usd_today` | Number | runtime-maintained |
| `consecutive_failures` | Number | runtime-maintained |
| `circuit_breaker_threshold` | Number | default 5 |
| `memoize_ttl_seconds` | Number | 0 = off |
| `tool_cache_ttl_seconds` | Number | 0 = off |
| `allowed_outbound_hosts_regex` | SingleLineText | restricts HTTP_REQUEST |
| `forbidden_tables` | SingleLineText | csv |
| `test_mode` | Checkbox | run end-to-end without persisting |
| `reflect` | Checkbox | self-critique before commit |
| `confidence_threshold` | Number | 0–1; below = approval queue |
| `max_validation_retries` | Number | default 2 |
| `on_error_action` | SingleLineText | `retry` \| `escalate` \| `pause` \| `fallback` |
| `fallback_model` | SingleLineText | optional |
| `fallback_agent_id` | Number | optional |
| `pre_run_hook` | SingleLineText | dotted python path |
| `post_run_hook` | SingleLineText | dotted python path |
| `surface_kind` | SingleLineText | e.g. `conversation:42` |
| `heartbeat_ttl_seconds` | Number | default 60 |

### `type_config_json` shape per type

```jsonc
// document
{ "target_table": "design_docs", "target_row_id": 42, "target_column": "body", "edit_mode": "replace" }

// queue
{ "target_table": "contacts", "filter": "(status,eq,new)", "output_column": "notes_body", "done_column": "last_processed_at", "batch_size": 5 }

// producer
{ "target_table": "weekly_recaps", "column_map": { "body": "<llm.body>", "title": "<llm.title>" } }

// responder
{ "inbox_kind": "email", "reply_mode": "approval", "log_table": "email_triage" }

// supervisor
{ "team_agent_ids": [12, 13, 14], "escalate_to_user_id": 1 }
```

---

## `assignments`

Universal inbox. Every trigger writes here; the runtime polls.

| Column | Type | Notes |
|---|---|---|
| `agent_id` | Number | |
| `org_id` | Number | |
| `source` | SingleLineText | `cron` \| `email` \| `api` \| `webhook` \| `supervisor` \| `table_watch` \| `manual` |
| `source_meta_json` | JSON | trigger-specific (e.g. inbound email envelope) |
| `task` | LongText | the prompt for this run |
| `priority` | Number | default 3 |
| `status` | SingleLineText | `queued` \| `running` \| `completed` \| `failed` \| `awaiting_approval` |
| `dedup_key` | SingleLineText | dedup at insert |
| `parent_assignment_id` | Number | for chained runs |
| `claimed_at` | DateTime | |
| `claimed_by_worker` | SingleLineText | worker id |
| `heartbeat_at` | DateTime | runtime-maintained while running |
| `attempts` | Number | starts at 0 |
| `max_attempts` | Number | default 3 |
| `next_retry_at` | DateTime | optional |
| `completed_at` | DateTime | |
| `result_summary` | LongText | |
| `result_ref_json` | JSON | pointers to written rows |
| `error` | LongText | |

---

## `agent_approvals`

For actions where `requires_approval_for` matches.

| Column | Type | Notes |
|---|---|---|
| `assignment_id` | Number | |
| `agent_id` | Number | |
| `org_id` | Number | |
| `action_kind` | SingleLineText | `send_email` \| `http_post` \| `sql_write` \| ... |
| `action_payload_json` | JSON | full call about to be made |
| `status` | SingleLineText | `pending` \| `approved` \| `rejected` |
| `decided_by` | SingleLineText | user or agent ref |
| `decided_at` | DateTime | |
| `note` | LongText | |

---

## `agent_artifacts`

Optional registry — lets one agent own multiple rows across tables.

| Column | Type | Notes |
|---|---|---|
| `agent_id` | Number | |
| `table_name` | SingleLineText | |
| `row_id` | Number | |
| `column_name` | SingleLineText | for long-text targets |
| `role` | SingleLineText | `primary` \| `reference` |
| `last_revised_at` | DateTime | |

---

---

## `agent_runs` — additions

Existing `agent_runs` table gets:

| Column | Type | Notes |
|---|---|---|
| `assignment_id` | Number | link |
| `prompt_version` | Number | snapshot of agent.prompt_version at run time |
| `prompt_snapshot` | LongText | full assembled context |
| `events_jsonl` | LongText | one line per LLM call / tool call / approval / write |
| `cost_usd` | Number | |
| `iteration_count` | Number | |
| `worker_id` | SingleLineText | |

---

## `artifact_versions`

One row per artifact write — enables rollback.

| Column | Type | Notes |
|---|---|---|
| `agent_id` | Number | |
| `assignment_id` | Number | |
| `table_name` | SingleLineText | |
| `row_id` | Number | |
| `column_name` | SingleLineText | |
| `before_text` | LongText | |
| `after_text` | LongText | |
| `created_at` | DateTime | |

---

## `agent_incidents`

Auto-written when circuit breaker trips.

| Column | Type | Notes |
|---|---|---|
| `agent_id` | Number | |
| `org_id` | Number | |
| `kind` | SingleLineText | `circuit_breaker` \| `budget_exceeded` \| `manual_pause` |
| `reason` | LongText | |
| `created_at` | DateTime | |
| `resolved_at` | DateTime | |
| `resolved_by` | SingleLineText | |

---

## `agent_templates`

Preset role definitions; clone into a real agent row.

| Column | Type | Notes |
|---|---|---|
| `name` | SingleLineText | e.g. `architect`, `strategist` |
| `description` | LongText | |
| `defaults_json` | JSON | column → default value, applied on clone |

---

## `chat_memory`

Per-conversation structured memory. The summariser proposes items here after each turn that crosses the threshold; the user accepts/edits/pins via the Properties panel. Pinned items are prepended verbatim to every turn's system prompt.

| Column | Type | Notes |
|---|---|---|
| `conversation_id` | Number | FK to `conversations.Id` |
| `org_id` | Number | scoping |
| `category` | SingleLineText | `fact` \| `decision` \| `thread` |
| `text` | LongText | the item itself, one fact/decision/thread per row |
| `pinned` | Checkbox | pinned items are always prepended verbatim |
| `status` | SingleLineText | `proposed` \| `active` \| `rejected` (proposed = awaiting user review) |
| `source_message_id` | Number | optional FK to `messages.Id` — the turn that produced this |
| `confidence` | Number | 0–100, summariser self-assessed (informational only) |
| `last_seen_at` | DateTime | bumped each time the summariser sees the topic again |

Indexes/queries:
- list by `conversation_id` + `status` (Properties panel)
- list `pinned=true` for prompt injection (every turn)

---

## `conversations` — additions

Add these to the existing `conversations` table. Existing columns (model, title, rag_enabled, rag_collection, knowledge_enabled, etc.) stay as-is. All new fields are per-conversation Properties-tab defaults.

| Column | Type | Notes |
|---|---|---|
| `system_note` | LongText | one editable line — sticky context prepended to every turn (`"Strategic work for Altitude Group, peer-mode, push back hard"`) |
| `default_response_style` | SingleLineText | one of the `CHAT_STYLES` keys; falls back to `companion` |
| `polish_pass_default` | Checkbox | when true, every reply runs critique→revise unless overridden per turn |
| `strict_grounding_default` | Checkbox | when true, the model must cite a source from RAG/web or admit it can't |
| `ask_back_default` | Checkbox | when true, the model is required to ask one clarifying question on ambiguity |
| `memory_extract_every_n_turns` | Number | default 6; structured extraction cadence |
| `memory_token_budget` | Number | default 800; cap on tokens spent on prepended pinned memory |
| `saved_fragments_json` | JSON | `[{label, text}]` — composer "saved fragments" dropdown |

---

## Creation order

1. `secrets`
2. `api_connections`, `smtp_accounts` (depend on `secrets`)
3. `agents` column additions, `agent_runs` column additions
4. `assignments`
5. `agent_approvals`
6. `agent_artifacts`
7. `artifact_versions`
8. `agent_incidents`
9. `agent_templates`
10. `chat_memory`, `conversations` column additions
