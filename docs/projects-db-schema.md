# Projects feature — NocoDB schema

All tables live in the existing NocoDB base alongside `agents`, `agent_runs`,
`code_conversations`, etc.

Tables are split into **required** (the feature is non-functional without
them) and **optional** (the corresponding endpoints gracefully no-op when the
table is missing — see `_safe_*` helpers in `infra/nocodb_client.py`).

## NocoDB type cheatsheet (used throughout)

| NocoDB UI type | Stored as | Used for |
|---|---|---|
| `Number` | int64 | every `*_id` foreign key, counters, line numbers, version numbers |
| `SingleLineText` | varchar(255) | short identifiers (slugs, tokens, sha hex, kind, mime, severity, rule, status enums) |
| `LongText` | text (unbounded) | file `content`, `path` (paths cap at 512 chars > SingleLineText 255 limit), descriptions, summaries, anything potentially > 255 chars |
| `Checkbox` | int 0/1 | `pinned`, `locked`, `watermark` — code writes `1`/`0`, reads via `bool()` |
| `JSON` | json text | `payload`, `tags`, `retrieval_scope`, `steps`, `concerns`, `anchor`, `project_ids`, `precommit_chain` — code sends raw Python `dict`/`list`, NocoDB serialises |
| `DateTime` | ISO8601 | `archived_at`, `verified_at`, `expires_at`, `revoked_at`, `resolved_at`, `gitea_last_synced_at` — code writes `datetime.now(timezone.utc).isoformat()`, reads via `datetime.fromisoformat()` |

NocoDB auto-adds `Id` (PK, Number), `CreatedAt` (DateTime), `UpdatedAt`
(DateTime) to every table. Don't define them — and **don't rename them**, the
code reads `row["Id"]`, `row["CreatedAt"]`, `row["UpdatedAt"]` literally.

Foreign keys here are **plain `Number` columns**, not NocoDB
LinkToAnotherRecord. The code joins via where-filters
(e.g. `(project_id,eq,123)`), not via NocoDB link traversal.

---

## Required

### `projects`

The unit of code work — a stable context plus a versioned virtual filesystem.

| column | NocoDB type | notes |
|---|---|---|
| `org_id` | Number | FK to org |
| `name` | SingleLineText | display name |
| `slug` | SingleLineText | URL-safe; capped 80 chars; unique per org when active |
| `description` | LongText | one-liner; LongText so longer descriptions don't truncate |
| `system_note` | LongText | always prepended to agent prompts |
| `default_model` | SingleLineText | e.g. `code` |
| `retrieval_scope` | JSON | array of Chroma collection names |
| `chroma_collection` | SingleLineText | scoped collection for project-internal embeddings |
| `parent_project_id` | Number | nullable; set on branched projects |
| `gitea_origin` | SingleLineText | e.g. `mike/marketing-site@main`; nullable |
| `gitea_last_synced_sha` | SingleLineText | 40-char sha; nullable |
| `gitea_last_synced_at` | DateTime | nullable |
| `precommit_chain` | JSON | array of check names |
| `archived_at` | DateTime | nullable; soft-delete |

### `project_files`

Pointer rows in the workspace. One row per (active) file path.

| column | NocoDB type | notes |
|---|---|---|
| `project_id` | Number | FK |
| `path` | LongText | POSIX, leading slash, no `..`; capped 512 chars (> SingleLineText 255 limit) |
| `current_version_id` | Number | FK to `project_file_versions` |
| `kind` | SingleLineText | `code` / `doc` / `note` / `adr` |
| `mime` | SingleLineText | `text/typescript`, `text/markdown`, … |
| `size_bytes` | Number | of the current version |
| `pinned` | Checkbox | always-include in agent prompt |
| `locked` | Checkbox | agent cannot overwrite without `allow_overwrite_locked` |
| `tags` | JSON | array of strings |
| `created_by` | SingleLineText | `user` / `agent:{conversation_id}` / `gitea:{owner}/{repo}@{sha}` / `branch:{src_project_id}` / `ai:{flow}` |
| `summary` | LongText | nullable; populated by `POST /fs/file/summary` |
| `preferred_model` | SingleLineText | nullable; per-file model override |
| `watermark` | Checkbox | optional footer-comment marker on agent edits |
| `archived_at` | DateTime | soft-delete; 30-day grace |

### `project_file_versions`

Append-only version log. `project_files.current_version_id` points to the head.

| column | NocoDB type | notes |
|---|---|---|
| `file_id` | Number | FK to `project_files` |
| `version` | Number | monotonic per file, 1-based |
| `content` | LongText | file content; capped at 100 KB by `PROJECT_MAX_FILE_BYTES` |
| `content_hash` | SingleLineText | sha256 hex (64 chars); used for idempotent-write check |
| `parent_version_id` | Number | nullable; the version this one was edited from |
| `edit_summary` | LongText | free-text from agent or user |
| `conversation_id` | Number | nullable; the code-conversation that produced this version |
| `created_by_message_id` | Number | nullable; the assistant message that wrote it |
| `pushed_to_sha` | SingleLineText | nullable; the Gitea commit sha this version was last pushed in |

---

## Optional (per-feature)

These tables are referenced via `_safe_*` helpers. Endpoints that read them
return empty lists when missing; endpoints that write them either silently
no-op (audit) or 500 with `missing table 'X'` (CRUD endpoints).

### `project_audit`

State-change ledger; powers the audit tab and the change feed.

| column | NocoDB type | notes |
|---|---|---|
| `project_id` | Number | FK |
| `actor` | SingleLineText | `user` / `org:{id}` / `agent:{conversation_id}` / `gitea:pull@{sha}` / `ai:{flow}` |
| `kind` | SingleLineText | see "Audit kinds" below |
| `payload` | JSON | event-specific |

**Audit kinds emitted by the backend:** `file_write`, `file_archive`,
`file_pin`, `file_lock`, `file_move`, `file_restore`, `file_preferred_model`,
`permission_request`, `snapshot_create`, `branch_create`, `gitea_import`,
`gitea_pull`, `gitea_pull_skip`, `cache_drop`, `adr_create`,
`readme_regenerate`, `faq_append`, `spec_regen`, `playbook_generate`,
`ai_review`, `symbol_rename`.

### `project_snapshots` + `project_snapshot_files`

Frozen named pointers into `project_file_versions`.

`project_snapshots`

| column | NocoDB type | notes |
|---|---|---|
| `project_id` | Number | FK |
| `label` | SingleLineText | unique per project; `[A-Za-z0-9._- ]{1,80}` |
| `description` | LongText | optional |
| `created_by` | SingleLineText | actor |

`project_snapshot_files`

| column | NocoDB type | notes |
|---|---|---|
| `snapshot_id` | Number | FK |
| `file_id` | Number | FK |
| `path` | LongText | snapshot of the path at capture time |
| `version_id` | Number | FK to `project_file_versions` |

### `project_lint_results`

| column | NocoDB type | notes |
|---|---|---|
| `project_id` | Number | FK |
| `file_id` | Number | FK |
| `version` | Number | the version the lint ran against |
| `line` | Number | nullable |
| `col` | Number | nullable |
| `severity` | SingleLineText | `info` / `warning` / `error` / `security` |
| `rule` | SingleLineText | `py-syntax`, `unused-import`, `json-parse`, … |
| `message` | LongText | human-readable |
| `kind` | SingleLineText | `lint` / `format` / `typecheck` / `security` |

### `project_symbols`

| column | NocoDB type | notes |
|---|---|---|
| `project_id` | Number | FK |
| `file_id` | Number | FK |
| `path` | LongText | denormalised for fast palette search |
| `name` | SingleLineText | `loginUser`, `Foo`, `useAuth`, … |
| `kind` | SingleLineText | `class` / `function` / `method` / `const` / `headingN` |
| `line` | Number | 1-based |
| `signature` | LongText | e.g. `(name: string)` — LongText because some signatures are long |

### `project_dependencies`

| column | NocoDB type | notes |
|---|---|---|
| `project_id` | Number | FK |
| `file_id` | Number | FK (the importer) |
| `depends_on` | LongText | the imported module / path |
| `edge_type` | SingleLineText | `import` (default) / `require` / `link` / `script` |

### `project_share_tokens`

| column | NocoDB type | notes |
|---|---|---|
| `project_id` | Number | FK |
| `token` | SingleLineText | `secrets.token_urlsafe(18)`; unique |
| `snapshot_id` | Number | nullable |
| `expires_at` | DateTime | nullable |
| `revoked_at` | DateTime | nullable |

### `project_bookmarks`

| column | NocoDB type | notes |
|---|---|---|
| `project_id` | Number | FK |
| `target_kind` | SingleLineText | `file` / `version` / `diff` |
| `target_ref` | LongText | path / `path@v3` / `path@v2..v3` |
| `label` | SingleLineText | optional |
| `color` | SingleLineText | optional |

### `project_saved_queries`

| column | NocoDB type | notes |
|---|---|---|
| `project_id` | Number | FK |
| `name` | SingleLineText | display |
| `query` | LongText | substring or regex |
| `kind` | SingleLineText | `search` / `symbols` / `lint` |

### `project_recipes`

| column | NocoDB type | notes |
|---|---|---|
| `project_id` | Number | FK |
| `name` | SingleLineText | display |
| `steps` | JSON | `[{prompt, target_glob?, expected_kind?}]` |

### `project_pins`

| column | NocoDB type | notes |
|---|---|---|
| `project_id` | Number | FK |
| `kind` | SingleLineText | `conversation_message` / `question` / `snippet` |
| `target_ref` | SingleLineText | message id / file path |
| `body` | LongText | inline content (subject to budget) |

### `workspaces`

Multi-project navigational sets. **Org-scoped**, not project-scoped.

| column | NocoDB type | notes |
|---|---|---|
| `org_id` | Number | FK |
| `name` | SingleLineText | display |
| `project_ids` | JSON | array of int |

### `project_pending_changes`

Staged-changes queue.

| column | NocoDB type | notes |
|---|---|---|
| `project_id` | Number | FK |
| `file_id` | Number | FK |
| `version_id` | Number | FK to the staged `project_file_versions` row |
| `conversation_id` | Number | nullable |
| `status` | SingleLineText | `pending` / `accepted` / `discarded` |
| `resolved_at` | DateTime | nullable |

### `project_playbooks`

| column | NocoDB type | notes |
|---|---|---|
| `project_id` | Number | FK |
| `goal` | LongText | e.g. "React 17 → 18" |
| `steps` | JSON | `[{title, description, scope_paths?, risk}]` |
| `current_step` | Number | 0-based pointer |
| `status` | SingleLineText | `active` / `done` / `aborted` |

### `project_reviews`

| column | NocoDB type | notes |
|---|---|---|
| `project_id` | Number | FK |
| `conversation_id` | Number | nullable |
| `from_snapshot` | SingleLineText | nullable |
| `summary` | LongText | model-generated overview |
| `concerns` | JSON | `[{path, line, severity, comment}]` |
| `suggested_followups` | JSON | array of strings |

### `project_file_comments`

| column | NocoDB type | notes |
|---|---|---|
| `project_id` | Number | FK |
| `file_id` | Number | FK |
| `version` | Number | the version the comment was authored against |
| `anchor` | JSON | `{line}` / `{range:[start,end]}` / `{symbol:"name"}` |
| `body` | LongText | comment text |
| `author` | SingleLineText | actor |

### `gitea_connections`

One row per org. NocoDB doesn't natively encrypt at rest — restrict access at
the deployment layer. The token is never echoed in API responses (replaced
with `***` by `_redact()` in `app/routers/gitea.py`).

| column | NocoDB type | notes |
|---|---|---|
| `org_id` | Number | one row per org |
| `base_url` | SingleLineText | e.g. `http://gitea:3000` |
| `username` | SingleLineText | the PAT's owning user |
| `access_token` | LongText | Gitea PAT; LongText so future longer token formats fit |
| `default_branch` | SingleLineText | usually `main` |
| `verified_at` | DateTime | last successful `/user` ping |

---

## Existing tables this feature reads from / extends

`code_conversations` — gains three optional columns:

| column | NocoDB type | notes |
|---|---|---|
| `project_id` | Number | nullable; set on project-scoped conversations |
| `interactive_fs` | Checkbox | nullable; opts into multi-turn `fs_*` tool loop |
| `scope_paths` | JSON | nullable; array of allowed path prefixes for `apply` mode |
| `code_checklist` | JSON | nullable; array of plan steps parsed from `plan` mode output |

Ad-hoc conversations leave these null and continue to work unchanged.

`code_messages`, `knowledge_sources` — untouched.

---

## Setup order

1. Required trio: `projects`, `project_files`, `project_file_versions`. The
   feature flag `features.code_v2.enabled` in `config.json` already controls
   whether the routes are exposed.
2. Add `code_conversations.project_id` (+ `interactive_fs`, `scope_paths`,
   `code_checklist`) for project-scoped chat.
3. Add `project_audit` early — every mutating endpoint logs to it.
4. Add `project_snapshots` + `project_snapshot_files` together.
5. Add the rest as you ship the corresponding UI tab.

## Indexes (recommended)

NocoDB exposes indexes via the table-meta UI. Add for hot read paths:

- `project_files (project_id, path)` — read-by-path is the hottest path
- `project_file_versions (file_id, version DESC)` — version history per file
- `project_audit (project_id, CreatedAt DESC)` — feed/timeline
- `project_snapshot_files (snapshot_id)`
- `project_lint_results (project_id, severity)`
- `project_symbols (project_id, name)` — palette lookups
- `project_share_tokens (token)` — anonymous fetch
- `gitea_connections (org_id)`

## Column-size guidance

- `path` cap: 512 chars (`infra.paths.normalize_project_path`) — needs LongText
- `content` cap: 100 KB (`PROJECT_MAX_FILE_BYTES`) — needs LongText
- `slug` cap: 80 chars (`_slugify`) — SingleLineText is fine
- `label` (snapshots) cap: 80 chars — SingleLineText is fine
- File-count per project cap: 5000 (`PROJECT_MAX_FILES`) — enforced in code
- `content_hash`: 64-char hex sha256 — SingleLineText

## Code ↔ NocoDB type mapping (what to expect)

- Code writes `True`/`False` for Checkbox columns? **No** — code writes `1` /
  `0` (`{field: 1 if value else 0}`). Reads via `bool(row.get(...))` which
  coerces both ints and bools.
- Code writes raw `dict`/`list` for JSON columns. NocoDB serialises them on
  insert and returns parsed objects on read.
- Code writes ISO 8601 strings for DateTime columns
  (`datetime.now(timezone.utc).isoformat()`). NocoDB returns ISO strings
  (sometimes with trailing `Z`); read code uses
  `datetime.fromisoformat(str(ts).replace("Z", "+00:00"))`.
- For Number columns, code uses Python `int`. Don't define these as Decimal
  — sort comparisons and `(col,eq,123)` filters assume integer semantics.
