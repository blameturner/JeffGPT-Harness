# Code feature uplift — "Projects" and a virtual filesystem

## Implementation status (2026-05-01)

Backend progress checkpoint in this repo:

- Implemented: project CRUD + major FS endpoints (`list/read/write/import/pin/lock/delete/restore/move/diff/search/export`) in `app/routers/projects.py`.
- Implemented: project-aware code runs (`project_id` support), workspace persistence via fenced file parsing, and per-change stream events (`file_changed`, `workspace_changed`, `permission_request`).
- Implemented: dedicated stream endpoints (`/code/stream/{job_id}`, `/projects/{id}/chat/stream/{job_id}`) and shared launcher (`app/routers/code_launch.py`).
- Implemented: code mode/style uplift foundations in `workers/code/config.py` and prompt assembler in `infra/prompts/__init__.py`.
- Implemented: migration script scaffold (`scripts/migrate_codebases_to_projects.py`) and regression tests for parser/config/fs helpers.
- Implemented: project audit endpoint (`GET /projects/{id}/audit`) with mutation hooks on FS write/archive/lock/pin/move/restore paths.
- Implemented: plan preview endpoint (`POST /projects/{id}/plans/{message_id}/preview`) for pre-apply affected-file inspection.
- Implemented: context inspector endpoint (`POST /projects/{id}/turns/{message_id}/context-inspector`) returning assembled prompt/context metadata and token estimates.
- Implemented: `interactive_fs` directive execution path (`workers/code/fs_tools.py`) with `fs_read/fs_write/fs_list/fs_delete` parsing plus `tool_result`/`permission_request` events.
- Implemented: context inspector summary endpoint (`POST /projects/{id}/turns/{message_id}/context-inspector/summary`) for lightweight UI list surfaces.
- Implemented: project chat history search endpoint (`GET /projects/{id}/chats/search?q=...`) with substring fallback.
- Implemented: open-work overview endpoint (`GET /projects/{id}/open-work`) aggregating TODO markers and recent permission requests.
- Implemented: `permission_request` audit persistence when locked-file writes are attempted by the agent.
- Implemented: cross-project import endpoint (`POST /projects/{id}/fs/import-from`) for drag/copy-style file transfer workflows.
- Implemented: backend endpoint contract doc (`docs/projects-api-contract.md`) for frontend integration.
- Implemented: idempotent fs writes (`PUT .../fs/file` accepts `if_content_hash`; mismatch returns 409 with `{expected,actual}`) — backend hygiene §28-26.
- Implemented: snapshots (`POST/GET /projects/{id}/snapshots`, `GET .../snapshots/{label}/diff`) — annotations §3, with dedup on (file_id, version_id) pairs.
- Implemented: graveyard listing (`GET /projects/{id}/graveyard`) for soft-deleted files with age (restore via existing `POST .../fs/file/restore`).
- Implemented: cross-project diff (`GET /projects/{id}/diff?against={other_id}`) — annotations §4 supporting endpoint.
- Implemented: full Gitea integration backend — `infra/gitea_client.py` REST wrapper + `app/routers/gitea.py` covering connection CRUD, repo browse/import, create-repo, push, status, pull preview, pull apply (with `theirs_into_new_branch` conflict path).
- Implemented: lint backend (`infra/lint_runners.py` with deterministic stdlib rules: trailing-whitespace, no-final-newline, py-syntax/unused-imports/security-smells, json-parse, yaml-parse, md-heading-depth) + `POST /projects/{id}/lint`, `GET /projects/{id}/issues`.
- Implemented: code analysis (`infra/code_analysis.py`) — symbols, imports, complexity, doc-coverage, test discovery, dependency parsing (npm/pip/pyproject), glossary; surfaced via `GET /projects/{id}/{symbols,symbol/{name}/refs,graph,complexity,doc-coverage,tests,dependencies,glossary}`.
- Implemented: branches (`POST /projects/{id}/branch` from current state or named snapshot), share tokens (`POST /projects/{id}/share`, anonymous `GET /projects/p/{token}`), bookmarks, saved queries, recipes, pinboard, multi-project workspaces, file comments, playbooks, ADRs (auto-pinned), per-language file templates.
- Implemented: time-travel diff (`GET /projects/{id}/fs/file/at-time?path=&at=`), find/replace with dry-run, deterministic cross-file rename, auto CHANGES.md append, watermarks, per-file model override, per-conversation `scope_paths`, pre-commit chain config + `POST /preflight`, hot-reload cache drop, staged-diff queue.
- Implemented: multi-turn `fs_*` tool loop in `workers/code/agent.py` (cap = 3 hops), feeds tool results back to the model and continues.
- Implemented: scaffold-from-spec stub, conversation-aggregated multi-file diff (`GET /projects/{id}/diff/conversation/{conversation_id}`).
- Implemented: frontend project workspace — `/projects` (list/create/archive) and `/projects/$id` (file tree + viewer/editor with version history & restore + project chat with mode/interactive toggle + tabbed side panel: open-work / issues / snapshots / gitea / metrics / audit / analysis) wired into `JeffGPT-gw-ui` (TS clean, prod build OK).
- Implemented: 79 unit tests passing across `test_code_analysis`, `test_lint_runners`, `test_file_templates`, `test_gitea_client`, `test_project_snapshots`, plus all prior project tests.

Still pending for full plan completion:

- Frontend project workspace routes and UI (`/projects`, `/projects/$id`) including file tree/editor/chat integration.
- Full interactive `fs_*` tool-loop mode (beyond fenced-output default).
- Complete module split under `app/routers/projects/*` and worker split under `workers/projects/*`.
- Settings/history tabs parity and broader operations surfaces (audit dashboard, full lint/review/snapshots stack).
- Gitea integration and advanced lifecycle/stretch feature sets.

Near-term backend next steps:

1. Add project-aware live progress/audit UI wiring (backend endpoints now present).
2. Expand interactive `fs_*` from directive parsing to true multi-turn tool-loop orchestration when latency budget allows.
3. Split `projects` router into modular sub-routers (`crud/fs/chat/metrics`) to reduce risk as scope grows.

## What's there today

Backend (`app/routers/code.py`):

| Endpoint | Behaviour |
|---|---|
| `POST /code` | Chat turn against `CodeAgent` (mode plan/apply, files attached on the request). |
| `GET /codebases` | Lists `knowledge_sources` rows where `type=codebase`, plus Chroma record count. |
| `POST /codebases` | Inserts a `knowledge_sources` row + reserves a scoped Chroma collection name. Does not create the collection or index anything. |
| `POST /codebases/{id}/index` | Embeds uploaded files into the codebase's Chroma collection. |
| `GET /code/conversations/{id}/workspace` | Returns the `files_json` from the **latest** user message — i.e. the most recent attachment, not a persistent file set. |

Frontend (`features/code/`): `CodebaseManager` lets the user create a codebase and upload a directory; the toggle in `CodePage` picks a `codebase_collection` so RAG retrieval is scoped. Output renders as code blocks with diff view.

What this means in practice:
- The "codebase" is **a Chroma collection of inputs** — useful only for retrieval grounding. It doesn't survive as a *project* the user can return to with state.
- The "workspace" is **the last file attachment** — refresh the page and it's gone unless re-attached.
- The agent produces code blocks in chat. Those blocks evaporate into history; nothing the agent wrote is browsable, editable, or version-tracked.
- The codebase toggle changes retrieval but not output destination — files the agent generates don't go anywhere durable.

## What we're building

A **Project** is the unit of code work: a stable context + a persistent virtual filesystem.

```
Project (e.g. "marketing-site")
├── Context
│   ├── system_note          ← always-on instructions for this project
│   ├── pinned_files[]       ← "always include these in the agent's prompt"
│   ├── retrieval_scope      ← which Chroma collections to RAG over
│   └── default_model
└── Workspace
    └── files/
        ├── /spec/auth.md         (v3)  ← agent-edited, latest version
        ├── /src/login.tsx        (v2)
        ├── /notes/decisions.md   (v1)  ← user-uploaded
        └── … each file has versions, an edit log, and per-version diffs
```

The toggle the user sees ("scope this conversation to *marketing-site*") then means three concrete things:

1. The agent's system prompt is prefixed with the project's `system_note` and the contents of every `pinned_file`.
2. RAG retrieval pulls from the project's `retrieval_scope`.
3. When the agent emits files (in `apply` mode), they're written to the project's workspace as new versions, not just shown in chat.

No real repo, no git, no S3. Everything lives in NocoDB (metadata) + Chroma (embeddings) + content fields in NocoDB rows. Files capped at ~100 KB each, ~5,000 per project.

---

## Schema

Three new NocoDB tables. Naming chosen to coexist with the existing `knowledge_sources type=codebase` rows during migration.

### `projects`

| column | type | notes |
|---|---|---|
| `Id` | int | PK |
| `org_id` | int | |
| `name` | str | display name |
| `slug` | str | URL-safe, unique per org |
| `description` | text | one-liner |
| `system_note` | text | always prepended to agent prompts on this project |
| `default_model` | str | e.g. `code` |
| `retrieval_scope` | text (JSON) | `["chat_knowledge","research_knowledge","codebase_marketing-site"]` |
| `chroma_collection` | str | scoped collection for project-internal embeddings |
| `archived_at` | timestamp | nullable |
| `CreatedAt`, `UpdatedAt` | auto | |

Migration: a one-shot script copies every `knowledge_sources type=codebase` into `projects` with `retrieval_scope=[chroma_collection]`. The old rows stay (to not break existing chat conversations referencing them).

### `project_files`

| column | type | notes |
|---|---|---|
| `Id` | int | PK |
| `project_id` | int | FK |
| `path` | str | `/src/login.tsx` — POSIX, leading slash, no `..` |
| `current_version_id` | int | FK to `project_file_versions` |
| `kind` | str | `code` / `doc` / `note` (free-form, drives icon + syntax) |
| `mime` | str | `text/typescript`, `text/markdown`, … |
| `size_bytes` | int | of `current_version` |
| `pinned` | bool | always-include in agent prompt |
| `locked` | bool | agent cannot overwrite without `--force` flag (UI confirm) |
| `tags` | str (JSON) | `["spec","auth"]` |
| `created_by` | str | `user` / `agent:{conversation_id}` |
| `archived_at` | timestamp | soft-delete |
| `CreatedAt`, `UpdatedAt` | auto | |

Uniqueness: `(project_id, path)` where `archived_at IS NULL`.

### `project_file_versions`

| column | type | notes |
|---|---|---|
| `Id` | int | PK |
| `file_id` | int | FK |
| `version` | int | monotonic per file, 1-based |
| `content` | text | the file's text. Capped at 100 KB; over that → reject with 413. |
| `content_hash` | str | sha256, used for "no-op edit" detection |
| `parent_version_id` | int | nullable; the version this one was edited from |
| `edit_summary` | str | free-text from the agent or user; "added auth provider" |
| `conversation_id` | int | nullable; the code-conversation that produced this version |
| `created_by_message_id` | int | nullable; the assistant message that wrote it |
| `CreatedAt` | auto | |

When the agent rewrites a file: insert a new `project_file_versions` row, update `project_files.current_version_id`. Old versions remain for the diff history.

---

## Backend endpoints

All under `/projects/*` or `/projects/{id}/*`. Owns its router; doesn't touch `code.py`.

### Project CRUD

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/projects` | `?org_id&archived?` | `{projects: [...]}` |
| `POST` | `/projects` | `{name, description?, system_note?, default_model?, retrieval_scope?}` | `{project}` |
| `GET` | `/projects/{id}` | — | `{project, file_count, latest_activity_at}` |
| `PATCH` | `/projects/{id}` | partial | `{project}` |
| `POST` | `/projects/{id}/archive` | — | `{ok}` |

### File system

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/projects/{id}/fs` | `?prefix=/src` | `{files: [{path, kind, size, current_version, updated_at, pinned, locked}]}` |
| `GET` | `/projects/{id}/fs/file?path=/src/x.tsx` | — | `{file, current_version: {content, version, edit_summary, ...}}` |
| `GET` | `/projects/{id}/fs/file/versions?path=…` | — | `{versions: [...]}` |
| `GET` | `/projects/{id}/fs/file/diff?path=…&from=v2&to=v3` | — | `{unified: "..."}` |
| `PUT` | `/projects/{id}/fs/file` | `{path, content, edit_summary?, kind?, mime?}` | `{file, version}` — creates new version. User-attributed. |
| `POST` | `/projects/{id}/fs/file/pin` | `{path, pinned}` | `{ok}` |
| `POST` | `/projects/{id}/fs/file/lock` | `{path, locked}` | `{ok}` |
| `DELETE` | `/projects/{id}/fs/file?path=…` | — | `{ok}` (sets `archived_at`; recoverable) |
| `POST` | `/projects/{id}/fs/file/restore` | `{path, version}` | `{file, version}` — copies an old version forward as a new current version |
| `POST` | `/projects/{id}/fs/move` | `{from, to}` | `{file}` — rename |
| `GET` | `/projects/{id}/fs/search?q=…` | — | `{hits: [{path, version, snippet}]}` — Chroma-backed across the project's own collection |
| `GET` | `/projects/{id}/fs/export?format=zip` | — | binary stream of all current versions |
| `POST` | `/projects/{id}/fs/import` | `{files: [{path, content}]}` | `{written: N, skipped: M}` — bulk upload, used by the existing dropdown |

### Chat scoped to a project

`POST /code` already exists. Add an optional `project_id` to `CodeRequest`. When set, the agent:
- Prepends the project's `system_note` to its system prompt.
- Loads every `pinned=True` file as system context.
- Receives a virtual-filesystem tool (see below).
- Writes any file outputs to the project's workspace, not just to the chat blob.
- The conversation row stores `project_id` so the workspace endpoint can join.

`GET /code/conversations/{id}/workspace` becomes a thin wrapper: if the conversation has a `project_id`, return `{project_id, files: [from project_files]}`; otherwise keep the legacy "last user attachment" behaviour for backwards compat.

---

## Agent integration: a virtual-filesystem tool

The code agent currently sees a `files: [{name, content}]` block in its prompt and emits a free-form response. Replace that, when `project_id` is set, with explicit tool calls:

```python
# tools/contract.py addition
class ToolName(str, Enum):
    ...
    FS_READ = "fs_read"
    FS_WRITE = "fs_write"
    FS_LIST = "fs_list"
    FS_DELETE = "fs_delete"

# tools/fs/handlers.py — new module
def fs_read(project_id: int, path: str) -> dict:
    file = _get(project_id, path)
    if not file or file["locked"]:
        ...
    return {"path": path, "content": ..., "version": ..., "kind": ...}

def fs_write(project_id: int, path: str, content: str, edit_summary: str = "",
             allow_overwrite_locked: bool = False) -> dict:
    ...
    new_version = _insert_version(file_id, content, edit_summary,
                                  conversation_id=ctx.conversation_id)
    _update_pointer(file_id, new_version.id)
    return {"path": path, "version": new_version.version, "diff_lines": ...}

def fs_list(project_id: int, prefix: str = "/") -> dict:
    return {"paths": [...]}
```

The agent's prompt is rewritten so it emits **tool calls**, not raw file contents:

```
You are working in project: marketing-site (slug: marketing-site)
System note: <project.system_note>

Pinned context (always loaded):
  /spec/auth.md (v3): <inlined content>
  /spec/styles.md (v2): <inlined content>

Tools available:
  fs_list(prefix)       → list workspace files
  fs_read(path)         → read a file's current content
  fs_write(path, content, edit_summary)  → create/update a file
  fs_delete(path)       → archive a file

Ground-rules:
  - Always fs_read before fs_write on an existing file.
  - One fs_write per logical change; meaningful edit_summary.
  - Never write outside /. Paths must be absolute.
  - Locked files require explicit user permission; surface a request.

When you've finished, end your reply with a one-paragraph "what I did".
```

The reply that flows to the UI is the natural-language summary; the actual file changes are surfaced through the workspace panel, which polls the project's FS and animates new/changed files. Diffs render against the previous version automatically.

This decouples the chat history from the file state. Re-running a turn doesn't duplicate file content in messages; it just emits a new version row.

### What about the existing inline-attachment flow?

Keep it. `POST /code` without `project_id` works exactly as today — files go in the request, the agent operates on them in memory, output is a code block. The project flow is opt-in and additive.

---

## Frontend

### Routing

- New route `/projects` — list / create / archive.
- New route `/projects/$id` — project workspace, owns the URL state for which file is open.
- `/code` stays for ad-hoc, project-less code chat.
- Sidebar: collapse "Code" and "Codebases" into a single "Code & Projects" entry; clicking expands a tree of recent projects.

### `/projects/$id` layout

Three columns, resizable:

```
┌────────────────┬─────────────────────────────┬─────────────────────────┐
│  File tree     │   File viewer / editor      │   Chat                  │
│                │                             │                         │
│  ▾ /spec       │   /spec/auth.md  (v3)       │   ┌───────────────────┐ │
│    auth.md  •  │   ┌─────────────────────┐   │   │ assistant:        │ │
│    styles.md   │   │ # Auth flow         │   │   │ I updated auth.md │ │
│  ▾ /src        │   │ ...                 │   │   │ to use the new    │ │
│    login.tsx*  │   │                     │   │   │ provider.         │ │
│                │   └─────────────────────┘   │   └───────────────────┘ │
│  + new file    │   [versions] [diff] [pin]   │   > _your message_      │
└────────────────┴─────────────────────────────┴─────────────────────────┘
```

- **File tree** — virtualised, indented by `/` segments. Pinned files marked `•`. Locked marked `🔒`. Files modified in the last hour have a small dot.
- **File viewer** — Monaco editor, read-only by default with an `Edit` toggle that flips to write-mode (saves create a user-attributed version). Above the editor: version dropdown, diff-against-previous toggle, pin/lock buttons, edit-history drawer.
- **Chat** — same `CodePage`-style chat but pinned to the project's `conversation_id`. New file events render inline as `📄 created /path` / `✎ /path (v2 → v3)` chips that, when clicked, focus the file in the viewer. Animations: a brief highlight-pulse on the file in the tree.

### Settings drawer (top-right)

Edit `system_note`, `default_model`, `retrieval_scope` (multi-select over the org's Chroma collections), and an "Import files" button that drops a folder upload into the workspace.

### History tab

A timeline view: `2026-05-04 14:32 · agent edited /src/login.tsx (v3) — "added MFA gate"`. Each row links to the diff view for that version.

---

## Stretch features (only if foundation lands clean)

1. **Project templates** — `New project from template: React app / spec workspace / research notebook`. Templates are themselves projects (system_note + a few seed files); creation deep-copies them.
2. **AI-driven refactor** — a one-shot `POST /projects/{id}/refactor {goal: "..."}` that does N agent turns under one umbrella, ending in a multi-file diff for review.
3. **Inline edit comments** — user can highlight a region of a file and ask "redo this paragraph" → tool call writes a partial version with just that hunk replaced.
4. **Project export to a real repo** — when the user is ready, a one-shot push to a GitHub repo via the deferred GitHub MCP. Out of scope for v1.
5. **File-level RAG embeddings** — every `fs_write` triggers an embed of the new content into the project's Chroma collection so future turns can retrieve their own outputs by content. Cheap, high value once volumes grow.
6. **Locked-file permission flow** — when the agent wants to overwrite a locked file, it surfaces a `tool_status` event the UI renders as a "permission requested" toast with `Allow` / `Deny`.

---

## Sequencing (build order)

1. **Schema + migration** — create the three tables, copy existing codebases over.
2. **Project CRUD + FS read endpoints** — list / create / get / list-files / read-file / list-versions.
3. **Frontend projects list + read-only project viewer** — file tree, Monaco viewer, version dropdown, diff toggle. No editing yet.
4. **FS write endpoints + user-side editing** — PUT file, pin/lock/move/delete/restore. UI flips to writeable.
5. **Bulk import endpoint + UI dropzone** — folder upload from the existing `CodebaseManager` flow.
6. **Agent integration** — `fs_*` tools, project-aware system prompt, project_id on `POST /code`, workspace endpoint update.
7. **Live workspace updates during chat** — SSE that the project page subscribes to, plus the inline file-event chips.
8. **Settings drawer + history tab.**
9. **Stretch list above, item by item.**

Each step is independently deployable; the agent integration (step 6) is the inflection point where the toggle starts paying off, and earlier steps are usable on their own (manual project workspaces).

---

## Local CPU constraints — what changes

This runs on local CPU models. That has three concrete consequences and the
plan above bends to fit them.

### 1. Avoid multi-turn tool loops

A real `fs_read` → think → `fs_write` round-trip is several model calls,
each replaying full context through a CPU prompt-eval. On a workstation
that's 10–60s per hop. The honest answer is to make tool calls **optional
and singular**, not the default flow.

**Default flow: single-pass "fenced output".** The agent receives pinned
files inlined, plus a path manifest of the rest, and emits a single reply
where any file changes are wrapped in path-tagged fences:

````markdown
I refactored auth into a provider:

```file path=/src/auth/provider.tsx mode=replace summary="extract provider"
import …
```

```file path=/src/login.tsx mode=patch summary="use new provider"
@@ -12,7 +12,7 @@
- import { signIn } from './auth';
+ import { useAuth } from './auth/provider';
```
````

A server-side parser (`workers/code/fs_parser.py`) scans the assistant
message after streaming, pulls out every ```` ```file ```` block, and
writes each as a new version on the project. Modes:

- `replace` — full new content, new version.
- `patch` — unified-diff applied against the current version.
- `append` — appended to current.
- `delete` — archive the file.

This is one model call, total. The "tool" is just a parser. The user can
switch to **interactive mode** (real `fs_*` tool calls) per-conversation
when they want surgical multi-step work, accepting the latency cost.

### 2. Cap context aggressively

Pinned files inlined verbatim is dangerous on local CPU — a 50KB pin adds
~12K tokens of prompt-eval to every turn.

- **Per-file cap**: 8 KB pinned, the rest stored as path manifest only.
  Files larger than that get summarised (via `tool` model) on first pin
  and the summary is what's inlined; full content fetched only on
  explicit `fs_read`.
- **Pin budget**: total pinned content capped at 30 KB across all pinned
  files; UI shows the budget in the settings drawer with a per-file
  contribution bar so the user sees what's expensive.
- **Path manifest only**: every other file is exposed as
  `/src/login.tsx (v2, 4.2 KB, last edited 2h ago)` so the agent knows
  what exists and can request reads in interactive mode.

### 3. Don't auto-embed on every write

Embedding is CPU-bound (the same backend as chat). Auto-embedding every
`fs_write` will fight the chat for the model.

- Embedding into the project Chroma collection is **off by default**.
- A nightly per-project background job re-embeds files modified since
  last run, gated behind `yield_to_chat()` so it pauses when chat is
  active (uses the cooperative checkpoint we already have).
- `GET /projects/{id}/fs/search` can fall back to **substring + token
  match** until embeddings catch up. That's already serviceable for
  human-readable code at the volumes we're talking about.

### 4. Diffs and parsing are CPU-cheap; lean on them

What we *can* afford:

- Server-side unified-diff generation (`difflib`), so the UI never has to
  ask the model to produce diffs.
- `mime`-based syntax detection from the path extension; no model needed.
- File `kind` inference (`code` / `doc` / `note`) from extension + first
  100 chars; deterministic.
- Token counting (`tiktoken` offline) for the pin-budget meter.

### 5. Keep the tool-call mode but make it a flag

Add `interactive_fs: bool = false` to `CodeRequest`. When false (default),
agent uses fenced-output mode + parser. When true, the agent gets the
real `fs_*` tools and we accept the multi-hop cost.

Per-project default lives in `projects.default_interactive_fs` (column
addition), so a "deep refactor" project can opt in once and stay there.

### 6. Streaming + parser hand-off

Don't wait for the full assistant message before parsing — stream chunks
to the client AND to a fence-parser running in a daemon thread. As soon
as a complete ```` ```file … ``` ```` block is closed, write its version
and emit a `file_changed` SSE event. The UI's chip ("✎ /src/login.tsx
v2→v3") appears mid-stream, which makes a slow CPU model feel responsive
because the user sees concrete progress before the full reply lands.

### 7. Sequencing tweak

Re-ordered for CPU realities — fenced parser before tool-call agent:

1. Schema + migration.
2. Project CRUD + FS read endpoints.
3. Frontend projects list + read-only viewer + diff.
4. FS write endpoints + user editing.
5. Bulk import.
6. **Fenced-output parser** + project_id on `POST /code` + workspace endpoint update — this is the inflection point on CPU.
7. Streaming fence detection + live `file_changed` SSE.
8. Settings drawer + history tab + pin budget meter.
9. **Interactive `fs_*` tools (opt-in flag).**
10. Background re-embedding (chat-yielding) + project fs search.
11. Stretch list.

Net: on a CPU model the user gets project-shaped persistence, file
versioning, and live workspace updates with **one model call per turn**.
The full tool-loop agent is there when they want it, on a flag.

---

## Robustness features — beyond the core

Hard rule: **no code execution, no sandbox.** Linters and other static
analysis are fine because they only read text. Everything below works on
the file content as data.

### Quality & static analysis

1. **Per-file linter pass** — pluggable linters keyed on file extension:
   - `.py` → ruff
   - `.js`/`.ts`/`.tsx` → eslint
   - `.md` → markdownlint
   - `.json`/`.yaml` → schema-aware parse + jsonschema if a schema is pinned
   - `.css`/`.scss` → stylelint
   - All linters run in subprocess against a temp file, output captured as
     `{file_id, version, line, col, severity, rule, message}` rows in a
     new `project_lint_results` table. Lint runs on demand
     (`POST /projects/{id}/lint?path=…`) and as a background job after
     `fs_write` (gated by `yield_to_chat()`).
   - UI: gutter markers in the Monaco viewer, a project-wide "issues"
     panel grouped by severity, click-to-jump-to-line.

2. **Format-on-save (read-only check)** — `prettier --check` /
   `ruff format --check`. Reports diff but does not auto-apply unless the
   user clicks "Apply formatter". Same plumbing as the linter, distinct
   table column for `kind=format` so the issues panel separates style
   from correctness.

3. **Type-check pass** — `mypy --no-incremental` / `tsc --noEmit` against
   the project's current versions. Heavy, opt-in. Run as a tool-queue
   job (`project_typecheck`) so it doesn't block the user. Output stored
   like lint results; same UI surface.

4. **Security scan** — `bandit` (Python), `semgrep` with a default
   ruleset, both static. Same job pattern. Surfaced as a "security"
   severity on the issues panel.

5. **Dead-code / unused-import detection** — leverages whatever the
   linters already detect. Adds an `unused` filter to the issues panel
   so cleanup PRs are easy to scope.

### Search and structure

1. **AST-aware symbol search** — `tree-sitter` (read-only parser) builds
   a per-project symbol index: `class`, `function`, `interface`,
   `export`. New endpoint `GET /projects/{id}/symbols?q=loginUser` returns
   `{path, kind, line, signature}`. Cheaper and more precise than
   embedding search for "where is X defined". Update on `fs_write`.

2. **Definition / references jump** — using the same tree-sitter index:
   `GET /projects/{id}/symbol/{name}/refs`. UI: cmd-click on an
   identifier to open a references drawer. Read-only — no LSP, no
   execution.

3. **Import graph** — parse `import` / `require` / `from … import` /
   `<link>` / `<script src>` per file; build a dependency DAG. Stored
   as `project_dependencies (file_id, depends_on_file_id, edge_type)`.
   New endpoint `GET /projects/{id}/graph` returns Cytoscape elements;
   UI gets a small Mermaid-style overview on the project page so the
   user sees the project's shape at a glance.

4. **Project-wide find/replace** — `POST /projects/{id}/fs/replace
   {pattern, replacement, paths?: glob, regex: bool, dry_run: bool}`.
   In dry-run mode returns a multi-file diff for preview; on commit it
   creates one new version per affected file with shared `edit_summary`
   (`bulk replace: …`). All purely string-level.

5. **Saved searches / queries** — name a search ("untyped functions",
   "TODO markers"), pin it to the project sidebar, re-run on demand.
   Backed by `project_saved_queries` table.

### AI workflows

1. **AI code review on diff** — `POST /projects/{id}/review
    {from_version: …, to_version: …}` (or pass a conversation id to
    review everything that conversation produced). Single agent call
    that reads the unified diff and returns a structured response:
    `{summary, concerns:[{path, line, severity, comment}], suggested_followups:[]}`.
    Stored in `project_reviews`. UI: review drawer in the History tab,
    inline comments rendered as gutter notes in the file viewer.

2. **Architectural Decision Records (ADR)** — first-class file kind.
    When the user (or the agent) makes a choice, an ADR template is
    written to `/decisions/NNN-slug.md` with `Status / Context /
    Decision / Consequences`. ADRs are auto-pinned. The agent's system
    prompt includes a one-line summary of every ADR so "why we picked
    X" persists across sessions.

3. **Auto-summary / project README maintenance** — a nightly background
    agent regenerates a `/README.md` that's a living summary of the
    project (file count by kind, recent decisions, open todos, what the
    project does). Locked by default; user can unlock to edit. Uses the
    cheap `tool` model role, runs once per day.

4. **TODO / FIXME aggregator** — periodic scan of all files for
    `TODO`, `FIXME`, `XXX`, `HACK`, `NOTE` markers. Surface in a project
    "open work" panel with file + line + author + age. Agent gets a
    line in its system prompt: `there are 14 outstanding TODOs across
    the workspace, oldest is 23 days`.

5. **Per-file ask** — "ask a question grounded only in this file" —
    fast single-call flow that pre-fills the chat with `Re: /path/x.tsx`
    and constrains retrieval to that file. UI: a question-mark button
    in the file viewer header.

6. **Workflow recipes** — saved multi-prompt sequences: "do a security
    pass" / "extract this module" / "rename and update all references".
    Stored as `project_recipes (id, name, steps:[{prompt, target_glob,
    expected_kind}])`. Run with one click; each step becomes a chat
    turn the user can approve before the next step runs.

7. **Agent personas per project** — `projects.persona_template` column
    so a "spec workspace" speaks differently from a "frontend code"
    workspace. The persona is injected into the system prompt before
    the project's `system_note`. Templates ship with sensible defaults.

8. **Smart paste** — paste a chunk of text / a URL / a file path into
    the chat, the agent classifies and routes it: code → write to a
    suggested path, doc → add to `/notes`, URL → scrape and embed.
    Confirmation step before any FS write so the user stays in control.

### Context awareness

1. **Token cost meter** — every chat input shows live token estimate
    of what will be sent (system prompt + pinned files + retrieved RAG +
    user message), with a budget bar. Uses tiktoken offline. Click to
    expand into the detailed breakdown panel below.

2. **Context inspector** — for any past assistant turn, "show me
    exactly what the model saw" — opens a drawer with the full
    serialized prompt: system note, pinned files inlined, RAG hits,
    tool definitions, history window. Critical for debugging "why did
    it answer that way?" especially on a CPU model where token
    pressure forces a lot of trimming.

3. **Pin budget meter** — already mentioned; expand: a per-pinned-file
    contribution bar so you can see `/spec/auth.md takes 6 KB of your
    8 KB pin budget`. Click to summarise (replace pinned content with
    its summary, freeing budget) or to unpin.

4. **Project conventions file** — `/.project/conventions.md` always
    pinned (no count toward the 30 KB budget — capped at 4 KB,
    hard-truncated). Default contents seeded by template; the agent
    reads them as authoritative style/architecture rules. Ergonomic
    place to record "we use Tailwind, not styled-components" without
    burying it in a long system note.

### Annotations and collaboration (single-user, but as if not)

1. **Line-level comments / pins** — `project_file_comments (id, file_id,
    version, anchor: {line | range | symbol}, body, author)`. Comments
    survive across versions when the line still has matching content
    (fuzzy-anchor); mark "stale" when they don't. UI: pinned in the
    gutter, threaded panel on the right.

2. **File annotations / "stickers"** — emoji or string tags applied to
    a file (`status:wip`, `status:reviewed`, `area:auth`). Indexed for
    filtering in the tree (`status:wip` shows only those). Stored in
    `project_files.tags` (already in core schema).

3. **Read-only "frozen" snapshots** — `POST /projects/{id}/snapshot
    {label}` captures every file's current_version into a named
    snapshot. `GET /projects/{id}/snapshots/{label}/diff?ref=current`
    returns a multi-file diff. Useful for "before this refactor" baselines.
    Cheap because we just record `(file_id, version_id)` pairs.

4. **Project branching** — `POST /projects/{id}/branch {name, from_snapshot?}`
    deep-copies a project's file pointers into a sibling project marked
    `parent_project_id`. Useful for "try this approach in parallel".
    The two projects can be diffed (`GET
    /projects/{id}/diff?against=other_id`) or merged file-by-file.

### Previews and rendering

1. **Markdown preview** — split-view in the viewer for `.md` files:
    source on the left, rendered HTML on the right. Mermaid blocks
    render as diagrams. Read-only; no script execution.

2. **JSON/YAML schema validation + tree view** — for any
    `.json`/`.yaml` file, a tree explorer panel and, if a schema is
    pinned (referenced via `$schema` or `project_files.schema_path`),
    inline validation markers. Pure parser work.

3. **HTML preview (sandboxed iframe)** — render `.html` files in a
    `sandbox=""` iframe with NO scripts allowed (`sandbox="allow-same-origin"`
    only). Strictly visual; this is rendering, not execution.

4. **Diagram-as-code rendering** — Mermaid, PlantUML (rendered
    client-side via a JS library, no server-side process). For files
    `.mmd`, `.puml` show the rendered diagram in the viewer.

### Project lifecycle

1. **Project templates** — already in core stretch list. Concretely:
    `react-frontend`, `python-package`, `spec-workspace`,
    `research-notebook`, `decision-log`. Each is a project with seed
    files + a tuned `system_note` + a tuned `persona_template`.

2. **Project import from ZIP / Git URL** — `POST /projects/{id}/import
    {kind: "zip", file: …}` or `{kind: "git_archive", url: …}` (HTTPS
    fetch only, no clone). Walks the archive in-memory, writes one
    version per file. No `.git` history is preserved; this is a
    snapshot import. Honours `.gitignore` and a project-level
    `.projectignore` (built-in defaults: `node_modules`, `__pycache__`,
    `.env*`, `*.lock`, …).

3. **Project export** — `GET /projects/{id}/export?format=zip|tar|json`.
    JSON format is the full versioned history (round-trippable via
    import). ZIP is current-versions only.

4. **Cross-project file references** — `[[ref:proj-slug:/path/x.md]]`
    syntax inside any pinned doc resolves to a live link in the viewer
    and is followed by the agent (the referenced file's current content
    is inlined, subject to the pin budget).

5. **Project archive vs delete** — archive sets `archived_at` and
    hides from default list, files preserved. Delete is a separate,
    confirm-twice action that removes versions and the Chroma
    collection. Versions are deduplicated by `content_hash` across the
    project so a re-archive doesn't bloat storage.

### Operations and metering

1. **Project metrics dashboard** — per project: file count by kind,
    bytes total, edits per day (user vs agent), open todos, lint
    pass-rate trend, last activity. Endpoint
    `GET /projects/{id}/metrics?period=30d`. UI: small panel on the
    project page header, full panel under a "Metrics" tab.

2. **Edit attribution timeline** — every version shows who wrote it
    (`user` vs `agent:{conversation_id}`) and what it cost
    (tokens_in / tokens_out / duration_seconds, pulled from the
    `agent_runs` row). Lets the user see which conversations were
    expensive vs productive.

3. **Quota & budgets** — per-project caps on file count, total bytes,
    and (optional) daily token spend on agent calls scoped to the
    project. `projects.budget_*` columns. Soft warnings + hard stop.

4. **Project audit log** — every state change (file write, lock,
    unlock, archive, persona change, system_note edit) recorded in
    `project_audit (id, project_id, actor, kind, payload, ts)`. Read
    via `GET /projects/{id}/audit`. Useful for "what changed last
    Tuesday".

5. **Cache controls** — extend the existing System Controls panel:
    "Drop project caches" → resets pinned-file resolution + symbol
    index for one project. Cheap.

### Backend module layout

To keep this from becoming a god-router, split:

```
app/routers/projects/
    __init__.py        — registers sub-routers
    crud.py            — list/create/patch/archive
    fs.py              — file CRUD, versions, diff, search, replace
    symbols.py         — symbols, references, import graph
    lint.py            — POST .../lint, GET .../issues
    review.py          — AI review on diff
    snapshots.py       — snapshots, branches, exports
    metrics.py         — metrics, audit log
    chat.py            — project-aware /code wrapper

workers/projects/
    fs_parser.py       — fenced-output → fs_write parser
    symbol_indexer.py  — tree-sitter sweep
    linter_jobs.py     — handlers registered with the tool queue
    nightly_summary.py — README maintenance, TODO sweep
    embedder.py        — yield-to-chat-aware project embedder
```

Each is its own tool-queue handler where applicable; nothing runs in the
chat thread.

---

## Prompts, modes, and styles — aligned to projects

### What's there today

`workers/code/agent.py` defines four **modes**:

- `plan` — structured plan, no code.
- `execute` — write working code.
- `review` — code review feedback.
- `explain` — explain how code works.

`workers/code/config.py` defines ten **styles** that overlap and conflict
with the modes (`review` and `explain` exist in both). The agent doesn't
know about projects, pinned files, ADRs, conventions, or the workspace.

### The redesign

Cleanly separate **mode** (the *flow*) from **style** (the *focus area*).
Each is one orthogonal axis.

#### Modes (the flow)

| Mode | Purpose | Output shape |
|---|---|---|
| `chat` | open-ended conversation, no file changes | prose only |
| `plan` | propose what to do; never writes files | structured plan with checklist |
| `apply` | execute against a (possibly approved) plan; emits fenced `file` blocks | file changes + summary |
| `review` | analyse a diff or a file; never writes | structured concerns list |
| `explain` | walk through code; never writes | prose with line refs |
| `decide` | **new** — propose an ADR | `/decisions/NNN-slug.md` ADR file |
| `scaffold` | **new** — generate a stub set from a prompt | multi-file fenced output |
| `refine` | **new** — take an inline selection and rewrite it; replaces only that hunk | `mode=patch` fenced block |

`apply` replaces `execute` (clearer name; matches the plan/apply terminology
the UI uses).

#### Styles (the focus)

Trimmed to non-overlapping focus areas — the things modes can't carry:

- `bug_fix`
- `tests`
- `security`
- `optimize`
- `refactor`
- `document` (code comments, docstrings, READMEs)
- `new_feature`
- `accessibility` (new)
- `migration` (new — schema/dep upgrades)
- `none` (default; pure mode-driven)

`review` / `explain` styles dropped — they're modes now.

The product of `mode × style` gives the system prompt's spine. Mode picks
the verb shape; style picks the lens.

### Project-aware system prompt

When `project_id` is set on the request, the prompt is assembled in this
order so the model sees identity → context → rules → task:

```
[ persona_template ]                            ← project.persona_template (default: senior engineer)
[ project header ]                              ← name, slug, one-line description
[ system_note ]                                 ← project.system_note
[ conventions ]                                 ← /.project/conventions.md (always pinned, ≤4KB)
[ recent decisions ]                            ← bullet list: ADR-001 …, ADR-002 …
[ pinned files: full content ]                  ← capped by pin budget (30KB total)
[ path manifest of remaining files ]            ← `(/src/login.tsx, v2, 4.2KB)` per line
[ open todos summary ]                          ← `14 outstanding TODOs across the workspace`
[ mode prompt ]                                 ← from the mode table above
[ style prompt ]                                ← from the style table above
[ retrieval block ]                             ← RAG hits with citations
[ fence-output instructions ]                   ← only when mode=apply or scaffold
[ user message ]
```

For each ADR included, only the **Decision** line goes in the recent-decisions
list, not the full body. The full ADR is reachable via fs_read.

### Mode-specific prompt sketches

#### `decide` mode

```
You are proposing an ARCHITECTURAL DECISION RECORD.

Output exactly one fenced file block:

```file path=/decisions/NNN-{slug}.md mode=replace summary="propose: {short title}"
# ADR-NNN: {title}

## Status
Proposed

## Context
{why this needs deciding now; link to specific files / past ADRs}

## Decision
{the choice, in one sentence, then a paragraph of justification}

## Consequences
{what this enables / forecloses; trade-offs}
```

Pick NNN as the next free integer in /decisions/. Keep total length under 600 words.
Do NOT write any other files. Do NOT include implementation details — that's
for a follow-up `apply`.
```

The server-side fence parser writes the file as a new version. The UI auto-pins ADRs.

#### `plan` mode (project-aware)

```
You are in PLAN mode. Produce a plan that another instance of you (in apply
mode) can execute without further questions. Be concrete and specific to
THIS project — reference actual files in the path manifest. Do NOT write
code in this turn.

Required sections:
  Context     ← what you understand about the request
  Approach    ← the chosen direction; why it fits the project's conventions
  Files       ← list of paths (existing or new) and what changes for each
  Steps       ← numbered checklist; each step ≤ 1 file's worth of work
  Open questions ← things the user must answer before you proceed
  Risks       ← what could go wrong; what to verify after

If a decision should outlive this plan (architectural, irreversible), say
"This warrants an ADR." and stop — don't carry on.
```

The plan's `Files` section is parsed (regex against `path=`) so the UI can
preview affected files before the user clicks Apply.

#### `apply` mode (project-aware)

```
You are in APPLY mode for project: {slug}. Execute the approved plan (above).
The conventions file is authoritative; deviate only if the plan explicitly
overrides it.

Output fenced ```file ``` blocks per the project's protocol:
  mode=replace : full new content
  mode=patch   : unified diff against the current version
  mode=append  : appended to current
  mode=delete  : remove the file

Rules:
  - Always state path= and summary=
  - Ground every change in a step from the plan
  - Locked files: if you must touch one, don't write it; emit a `permission_required`
    block instead and stop
  - End with a one-paragraph "What I did". No file blocks after that paragraph.
```

#### `refine` mode (inline edit)

The UI passes `{path, anchor: {start_line, end_line}}` along with the user's
prompt. The model sees only the surrounding hunk plus the conventions file:

```
You are in REFINE mode. The user selected lines {S}–{E} of {path}. Your job
is to rewrite ONLY those lines.

Output exactly one fenced file block in `mode=patch`. The patch must touch
ONLY the selected range (you may add/remove lines within that range, but
you may not edit code outside it).
```

This keeps refines fast (small context) and safe (no surprise edits elsewhere).

### Style prompts — concrete

Each style is a short paragraph appended after the mode prompt. Keep them ≤ 120
words so prompt-eval cost stays low. Examples (paraphrased):

- `accessibility` — "Treat WCAG 2.2 AA as the floor. For every UI change,
  consider keyboard nav, focus order, screen-reader semantics, contrast.
  Cite the rule (e.g. WCAG 2.4.7) when explaining a fix."
- `migration` — "You are moving from {from} to {to}. Preserve behaviour.
  Update tests in the same change. List any deprecation paths. Surface
  breaking changes in the summary."
- `tests` — "Write tests as documentation. Cover the happy path, the
  obvious failure mode, and one weird edge. Use the project's existing
  testing library (check imports). No mocks unless the conventions file
  permits."

Per-project override: a `/.project/styles/{style}.md` file (if present)
replaces the default style paragraph. Lets a project tune the tone without
touching the platform.

### Frontend prompt-meta surface

- **Mode pill** in the chat input (was a dropdown). Cycles `chat → plan → apply
  → review → explain → decide → scaffold`. `apply` and `scaffold` show a
  warning dot — "this will write files". Keyboard: `⌘M`.
- **Style chip** beside it. Click to swap focus area. Saved per-project as
  `default_style`.
- **"Why this answer" inspector** (already mentioned) renders the assembled
  prompt with each section collapsible: persona / system_note / conventions /
  decisions / pinned files / manifest / mode / style / retrieval. Critical
  for tuning the project on a CPU model where every section costs.
- **"Promote to ADR"** action — when a chat exchange landed on a decision,
  click it on the assistant message to re-run that turn in `decide` mode
  with the prior context as the Context section.
- **Plan → Apply hand-off** — after a `plan` turn lands, a "Apply this plan"
  button at the bottom of the plan; one click queues an `apply` turn with
  the plan injected as the approved_plan.

### Backend wiring

- `workers/code/config.py`: rewrite — separate `MODES`, `STYLES`,
  `STYLE_META`, helpers; drop `review`/`explain` from styles; add the
  three new modes.
- `workers/code/agent.py`: prompt assembler honours the layered order
  above; `mode_prompt(mode, project)` and `style_prompt(style, project)`
  helpers; new `apply` mode wiring with the fence-parser.
- `app/routers/code.py`: `CodeRequest.mode` accepts the new modes;
  validates against the catalogue.
- `app/routers/projects/chat.py`: project-aware wrapper that builds the
  full prompt before delegating.
- `infra/prompts/` (new module): the assembler — small, pure, testable.
  Functions: `assemble_prompt(project, mode, style, user_message,
  history, retrieval) -> str`. Trivial to unit-test without the model.

---

## 10 more features (with frontend impact)

These are **incremental** to the 40 already listed; they target concrete UX
gaps that surfaced when sketching the prompt overhaul. Frontend impact
called out per feature.

1. **Plan → Apply preview drawer**
   - Backend: `POST /projects/{id}/plans/{plan_msg_id}/preview` parses the
     plan's `Files` section and returns `{path, action: create|edit|delete,
     existing_size}` for each.
   - Frontend: opens a drawer titled "Apply this plan?" with a checklist
     of files about to change. User can deselect any to constrain the
     `apply` turn. Clicking Apply hands the trimmed list to the agent as
     `scope_paths: [...]`. Belt-and-braces against runaway applies on a
     CPU model.

2. **Permission-required modal**
   - Backend: when the agent emits a `permission_required` fence
     (locked file, scope outside `scope_paths`), the streaming parser
     emits an SSE event `permission_request {path, reason}`.
   - Frontend: modal with `Allow once` / `Allow + unlock` / `Deny` —
     answer is sent back as a follow-up tool result; the agent resumes.

3. **Diff queue ("staged changes")**
   - Backend: optional `auto_commit: false` flag on `apply`; when set,
     versions are written but `current_version_id` is *not* advanced.
     A new table `project_pending_changes` holds them.
   - Frontend: a "Staged" tab on the project page that lists each
     pending change with a per-file Accept / Discard / View diff. Lets
     the user gate every agent edit before it becomes the current
     version. Default off; per-project flip.

4. **Live workspace tree with edit beacons**
   - Backend: SSE on the existing project channel emits
     `file_changed {path, version, by, summary}` as the parser writes.
   - Frontend: the tree pulses the changed path; the tree node carries
     a tiny `v3` badge that fades over 30s. Multiple files changed in
     one turn render as a counter ("5 files just changed — review").

5. **"Open in chat" from the file viewer**
   - Backend: nothing new — uses existing chat endpoint.
   - Frontend: button on the file viewer header that pre-fills the
     chat with `Re: /path/x.tsx` and a quoted snippet of any selected
     range. Sets `mode=refine` if a selection is active, else `chat`.

6. **Symbol palette (⌘P)**
   - Backend: uses the symbols index from feature §search-and-structure-1.
   - Frontend: command palette like VS Code's Go-to-Symbol; fuzzy match
     against `{path, symbol}` pairs; `Enter` opens the file at the
     symbol's line. Critical for navigation once a project has > 30
     files.

7. **Per-conversation file scope**
   - Backend: `code_conversations.scope_paths` (JSON array) constrains
     all `apply` writes in that conversation to those paths or
     prefixes; everything else is read-only for that conversation.
   - Frontend: chip under the chat header — "scoped to /src/auth/*";
     click to add or remove paths. Lets you say "this conversation is
     auth work" so the agent can't drift.

8. **Tabbed editor**
   - Backend: nothing — pure UI state.
   - Frontend: clicking a file opens it in a tab bar above the viewer
     (max 8 tabs, oldest evicted). Cmd-W closes a tab. A modified-
     locally indicator keeps unsaved user edits visible. Restores tab
     state per-project on reload.

8. **Inline AI inline-edit ("ghost" diffs)**
   - Backend: same as `refine` mode but the response is delivered as a
     tentative edit rather than a committed version.
   - Frontend: the patch renders as a translucent overlay on the
     selected range; `⌘↵` accepts (writes a new version), `Esc`
     dismisses. Feels like Copilot inline accept/reject.

10. **Conversation pinboard**
    - Backend: `project_pins.kind = 'conversation_message'` — pin any
      assistant message to the project. Pinned messages get included
      verbatim (subject to budget) at the bottom of the system prompt
      under "Pinned exchanges".
    - Frontend: thumbtack icon on every assistant turn; pinboard panel
      lists them with the conversation context. Dragging a pinned
      message into the file viewer creates a new file from its
      content. Useful for "the agent finally got it right; remember
      this approach forever".

11. **Multi-file diff review viewer**
    - Backend: `GET /projects/{id}/diff?from=snapshot:foo&to=current`
      returns a unified diff per file (already the snapshot endpoint;
      this is the multi-file aggregator).
    - Frontend: GitHub-style multi-file review screen — file list on
      the left, side-by-side or unified diff on the right, comment
      anchors per line. Used after a long apply turn or when comparing
      a branch.

12. **Drag-and-drop folder upload everywhere**
    - Backend: bulk import endpoint already in core.
    - Frontend: any project page accepts a folder drag — drops a
      visual overlay with the file count + size and an Import button;
      respects `.projectignore` defaults visibly. Removes the modal-
      hidden upload from the current `CodebaseManager`.

(The list went 12 because two of them — tabbed editor and inline ghost
diffs — felt small to spend a slot each but earn separate UI surfaces.)

---

## Gitea integration — minimal

Goal: let a project optionally connect to a Gitea repo so you can pull a
snapshot in or push a snapshot out. Nothing more. No env vars, no webhooks,
no merge state, no CI hooks, no token mucking on every action.

### One-time setup

A single row in a new `gitea_connections` table:

| column | type |
|---|---|
| `Id` | int |
| `org_id` | int |
| `base_url` | str — `https://gitea.example.com` |
| `username` | str |
| `access_token` | encrypted str — Gitea Personal Access Token, repo scope |
| `default_branch` | str — usually `main` |
| `verified_at` | timestamp — last successful API ping |

UI lives in **System → Console → Connectors** as a single panel:

```text
┌─ Gitea ──────────────────────────────────────────┐
│ ( Base URL    ) https://gitea.example.com        │
│ ( Username    ) mike                             │
│ ( Access token) •••••••••••••••• (paste)         │
│                                                  │
│ ✓ Connected as @mike   ·   17 repos visible      │
│ ( Test connection )  ( Disconnect )              │
└──────────────────────────────────────────────────┘
```

That's the entire setup surface. No env file touches, no per-project
re-entry of credentials. One Gitea connection per org. The token is stored
encrypted in NocoDB; never echoed in API responses (replaced with `***`).

### What it does

Three actions, all manual, all explicit. None of them watch, sync, or
listen.

#### 1. Browse + import a Gitea repo as a snapshot

```
POST /projects/import-from-gitea
{
  "owner": "mike",
  "repo":  "marketing-site",
  "ref":   "main",            // branch, tag, or commit
  "name":  "marketing-site",  // new project name
  "ignore": ["node_modules/", "*.lock"]   // optional, on top of defaults
}
```

Backend uses Gitea's archive endpoint (`/api/v1/repos/{owner}/{repo}/archive/{ref}.zip`),
streams the zip into the existing zip-import path, creates a new project,
and tags the project with `gitea_origin = "{owner}/{repo}@{ref}"`. No
clone, no `.git`, no history. One commit's worth of files. Done.

The Gitea origin is stored on the project so the UI can show "snapshot of
mike/marketing-site@main, imported 3 days ago".

#### 2. Browse Gitea from inside a project

A "Browse Gitea" tab on any project page lists repos accessible to the
configured user, plus search. Each repo expands to its file tree
(read-only, lazy fetch). The user can:

- **Copy file → workspace** — drops the file (current ref) into the
  project's workspace as a new version with `created_by =
  "gitea:{owner}/{repo}@{sha}"`.
- **Open in browser** — link out to the file on Gitea.

This isn't synced and won't update. It's a curated copy-paste with a
provenance tag. Cleaner mental model than a partial mirror.

#### 3. Push a project snapshot to a Gitea repo

```
POST /projects/{id}/push-to-gitea
{
  "owner": "mike",
  "repo":  "marketing-site",
  "branch": "from-jeff",          // creates if missing
  "message": "Snapshot from Jeff project marketing-site",
  "include_versions": false       // current versions only (default)
}
```

Strategy is one-tree, one-commit:

- Walk the project's current versions.
- For each file, call Gitea's `PUT /repos/{owner}/{repo}/contents/{path}`
  (creates or updates).
- Commit message + branch as supplied. Branch is created if missing.
- Returns `{commit_url, branch_url}`.

Optional flag `include_versions: true` writes a follow-on commit per file
in version order so the Gitea history mirrors the project's edit log. Off
by default because most users want a clean snapshot.

No PR creation, no diff review, no merge logic. The user opens Gitea in
their browser if they want a PR — the UI surfaces the branch URL after
push as a one-click link.

### Frontend surface

- **Connectors panel** — the setup card above. Single source of truth.
- **Per-project header** — when `projects.gitea_origin` is set, a small
  badge `gitea: mike/marketing-site@main` sits beside the project name.
  Click → drawer with two actions: "Refresh from Gitea" (re-imports the
  same ref into a new snapshot, keeping versions) and "Push current
  snapshot" (the push action above, pre-filled with the origin's owner +
  repo).
- **Project import flow** — the existing `New project` flow gains a
  "from Gitea" tab next to "from ZIP" and "blank". Same UI, different
  source.
- **No live indicators**, no SSE, no webhook listeners. The user is in
  control of every sync moment.

### 4. Create a new Gitea repo from a project

```
POST /projects/{id}/create-gitea-repo
{
  "owner":      "mike",         // user or org
  "owner_kind": "user",          // "user" | "org"
  "repo":       "marketing-site",
  "description": "Snapshot from Jeff",
  "private":    true,
  "default_branch": "main",
  "init_readme": false           // we push our own README, don't init blank
}
```

Calls Gitea's `POST /api/v1/user/repos` (or `/api/v1/orgs/{org}/repos`),
captures the returned `clone_url` + `default_branch`, sets the project's
`gitea_origin = "{owner}/{repo}@{default_branch}"`, then immediately
performs an initial push of the project's current versions as one commit.
Idempotent if the repo already exists with no commits — it just pushes
into it.

UI: a "Create new repo" affordance in the Gitea drawer alongside "Push to
existing repo". Pre-fills `repo` from the project slug, `description`
from the project description.

### IDE-like push: working tree → branch

Treat the project's workspace as the working tree, the linked Gitea ref
as the remote tracking branch. The user wants a familiar "ahead / behind /
push / pull" flow, not a snapshot dump.

#### State we track

Two new columns on `projects`:

- `gitea_last_synced_sha` — the Gitea commit sha we last imported from
  or pushed to.
- `gitea_last_synced_at` — timestamp.

One new column on `project_file_versions`:

- `pushed_to_sha` — nullable. When this version was pushed in a Gitea
  commit, the resulting sha goes here. Stays null for unpushed versions.

That's the entire model. `pushed_to_sha IS NULL on a current_version`
means "this file is ahead of remote".

#### Status indicator (like an IDE status bar)

Endpoint: `GET /projects/{id}/gitea/status` returns

```jsonc
{
  "linked": true,
  "origin": "mike/marketing-site@main",
  "ahead":  ["/src/login.tsx", "/spec/auth.md"],
  "behind_count": 2,                    // remote commits since last sync
  "remote_head_sha": "a1b2c3...",
  "last_synced_sha": "9988ff...",
  "last_synced_at": "2026-05-01T12:30:00Z",
  "state": "diverged"                   // in_sync | ahead | behind | diverged | unlinked
}
```

`behind_count` is computed by calling Gitea
`/repos/{o}/{r}/commits?sha={ref}&limit=10&since={last_synced_at}` and
counting until we hit `last_synced_sha` (cap at 10 — beyond that we say
`10+`). One cheap REST call. Cached in-process for 60 s, drop-on-action.

UI surfaces this as a badge beside the project name:

```text
marketing-site   ↑2  ↓ —     in sync ✓     diverged ↕      ↑3 ↓2
```

Click → drawer with the file list (ahead) and the recent remote commits
list (behind), each with a one-click action.

#### Push flow ("commit and push")

`POST /projects/{id}/gitea/push`

```
{
  "branch":  "main",                    // defaults to origin's ref
  "message": "Update auth and styles",  // single commit message
  "paths":   null,                      // optional path filter; default = all "ahead" files
  "scope":   "current"                  // "current" (one combined commit) | "per_file" (one per file)
}
```

Strategy is one round-trip to Gitea per file with `PUT contents`, all
within a single user-issued action. After every successful PUT, we
update the version's `pushed_to_sha`. The last call's commit sha
becomes the project's new `gitea_last_synced_sha`.

If `behind_count > 0` at push time, **block by default** with HTTP 409
and `{reason: "remote_diverged", remote_head_sha, behind_count}`. The
UI then offers two paths:

1. **Pull first** (open the conflict drawer below).
2. **Force push** — write Gitea contents with the version's current
   sha-on-record marked stale. We don't actually do `--force` (REST
   API doesn't); we re-fetch each file's latest sha and overwrite.
   Behaves like "their changes lost". Confirmed via a destructive
   modal ("This will overwrite remote changes to N files").

UI behaviour:

- Hover on the ↑N badge → tooltip lists the files, each with a small
  "View diff" link.
- "Push" button in the drawer; disabled while behind unless
  user opts into force.
- Per-file "unstage" toggles let the user push only some files this
  round (uses `paths: [...]` in the request).
- After push, badge animates to ✓ and remote head sha displayed.

#### Pull flow ("fetch and merge")

`GET /projects/{id}/gitea/pull/preview` returns a per-file conflict map:

```
{
  "remote_head_sha": "a1b2c3…",
  "files": [
    {
      "path": "/src/login.tsx",
      "state": "remote_only" | "local_only" | "both_modified" | "identical",
      "remote_sha": "…",
      "remote_size": 2148,
      "local_version": 4,
      "diff": "...unified..."     // null when state=identical
    }
  ]
}
```

`POST /projects/{id}/gitea/pull/apply` accepts a per-file decision:

```
{
  "decisions": [
    { "path": "/src/login.tsx", "choice": "ours" },
    { "path": "/spec/auth.md",  "choice": "theirs" },
    { "path": "/notes/x.md",    "choice": "skip" },
    { "path": "/styles/a.css",  "choice": "theirs_into_new_branch" }
  ],
  "set_synced_to": "a1b2c3…"     // the remote head we just inspected
}
```

`choice` values:

- `ours` — keep the local version; mark it `pushed_to_sha = null` so
  the next push includes it.
- `theirs` — write the remote content as a new local version
  attributed to `gitea:pull@{sha}`; pre-marked as `pushed_to_sha = sha`
  (already in remote).
- `skip` — record the decision in audit; leave both untouched.
- `theirs_into_new_branch` — write the remote content into a new
  branched project (the user can come back and merge by hand).

There is **no auto-merge**. Conflicts are resolved file-by-file, by
the user, with both contents visible side-by-side. Same UI as the
"multi-file diff review" feature already in the plan, with three
buttons per file. Cheap on CPU because there's no model involvement —
this is plain text routing.

#### What an IDE has that we don't (and don't need)

| IDE feature | Our equivalent |
|---|---|
| Local commit history (git log) | `project_file_versions` rows |
| Working tree | Current versions |
| Index / staging area | The `paths` filter on push (transient) |
| `.gitignore` | `/.projectignore`, on import only |
| Branches | Branch a project (already in plan); pull-into-new-branch on conflict |
| Stash | Snapshots (already in plan) |
| Merge | **Manual file-by-file decisions** (above). Deliberate; CPU-friendly. |
| Rebase | None. Don't pretend. |
| Tags | Snapshot labels (already in plan) |
| Remote tracking | Single `gitea_origin` per project |

The mental model is: **the project is a working tree linked to one
Gitea branch**. Push exports your current versions; pull imports
remote changes as new local versions you decide to keep or drop. No
hidden state, no surprise merges.

### What we deliberately don't build

- **No clone**, no `.git`. We don't ship git on the harness; we use the
  REST API only.
- **No env-var configuration**. Token entry is UI-only.
- **No auto-pull on file open** — every Gitea fetch is user-initiated.
- **No merge / rebase / conflict UI** — we only ever do straight
  overwrites at commit time. If a remote file has diverged, the push
  fails and surfaces "remote has changed since import; refresh first?"
- **No multi-account, no per-project token override.** One org, one
  Gitea account. Keeps the data model trivial.
- **No webhook receiver** for inbound events. Out of scope; could come
  later as a "watch repo for changes" cron job that polls daily.

### Why this shape

The user-stated constraint was "super simple design, no env variables,
no mucking up of finding data points." The model above:

- Stores one row, encrypted, behind a one-time UI step.
- Exposes three explicit actions; everything else is opt-in browsing.
- Treats Gitea as an **archive source/sink**, not a sync partner. That
  means no merge state, no conflict UI, no listener processes.
- Carries a provenance tag on every imported file/project so the
  history is preserved without mirroring `.git`.

Sequencing-wise this slots in **after** the FS write endpoints land
(step 4 in the build order above) and **before** AI tool integration.
A user can do useful work — pull a repo, edit files, push a branch back
— before the agent ever touches the project.

---

## Plan review — gaps and additional features

After re-reading the whole plan, the following are high-value additions
that aren't covered yet. Each is annotated with its **CPU cost
profile** because that's the binding constraint.

CPU cost legend:

- 🟢 free — pure plumbing, no model calls
- 🟡 cheap — single targeted model call with small context
- 🔴 expensive — multi-call or large-context flow; gate behind explicit
  user action and `yield_to_chat()`

### Editor & navigation (UI-only mostly)

1. 🟢 **Split view / two-pane editor** — open two files side-by-side or
   the same file at two versions. Critical when the agent edits one
   file and you want to keep the spec open in the other pane. Pure UI
   state; no extra requests.

2. 🟢 **Tabbed editor with peek** — already mentioned; add `cmd-click`
   on a path in the tree opens the file in a *peek* tab (italic, single
   click elsewhere closes it) so navigation doesn't accumulate stale
   tabs. This is how VS Code does it; cheap and habit-forming.

3. 🟢 **File breadcrumbs** — above the editor: `marketing-site /
   spec / auth.md  v3`, each segment clickable. Click `v3` to open the
   version dropdown; click a folder to filter the tree.

4. 🟢 **Distraction-free / focus mode** — `⌘.` collapses tree, chat,
   tabs into a single editor pane. ESC restores. Useful when reading
   long generated files on a small screen.

5. 🟢 **Bookmarks** — pin a file, a specific version, or a diff range
   to a project-wide bookmarks panel. Stored as `project_bookmarks
   (id, project_id, target_kind, target_ref, label, color)`. Useful
   for "the version that worked".

6. 🟢 **Per-file blame** — for any file, render a left-gutter strip
   coloured by version. Hover shows version number + edit summary +
   `created_by` (`agent:{conversation_id}` clickable). Computed
   server-side using `difflib.SequenceMatcher` over consecutive
   versions; stored as `project_file_blame (file_id, version, line_no,
   originating_version)`. Recomputed on `fs_write`, async.

### Agent workflows (carefully gated)

7. 🟡 **Quick-fix lightbulb on lint issues** — every lint marker in
   the gutter gets a small lightbulb. Click → opens a one-turn
   `apply` mini-chat where the user message is pre-filled
   `Fix: {rule}: {message}` and the agent only sees that file. One
   single-file call, bounded context. Result lands as a normal patch
   version.

8. 🟡 **"Describe this file" one-click summary** — button on the file
   header runs the cheap `tool` model role with `{path, content}` as
   the only context, asks for a one-paragraph summary, stores it on
   `project_files.summary`. Used by the path manifest + the README
   maintainer so other turns benefit. Recompute is debounced (no auto-
   rerun until content_hash changes substantially).

9. 🔴 **"What if?" trial mode** — a chat toggle that forces every
   `apply` turn to run as `plan` instead, no matter what. Lets you
   scope-check what the agent would do before flipping the switch.
   Free in cost (it just rewrites the request); valuable in chats
   where you're nervous about file writes.

10. 🟡 **Linked tasks panel** — per-file `tasks` (free-form bullets the
    user pins) live in `/.project/tasks/{path-slug}.md`. Always
    inlined when that file is open in the editor. Lets you say "note
    to self: refactor the auth provider" against the file without
    polluting the file's own TODO comments.

### Visibility & state

11. 🟢 **Project change feed (24h activity)** — a slim panel on the
    project page listing every event in the last 24h: file edits,
    ADRs added, lint runs, pushes, pulls. Same shape as the existing
    `project_audit` log; rendered as a vertical timeline.

12. 🟢 **Open work overview** — single panel that aggregates: pending
    TODOs (count by file), unresolved lint issues (by severity),
    permission requests, staged changes awaiting review. The
    "homepage" for a project after the README. Computed server-side
    by joining the existing tables.

13. 🟢 **File age / staleness halo** — visual hint on the tree:
    files unedited > 30 days get a gentle desaturation; files edited
    in the last hour get a fresh-mint dot. Tells you what's "live"
    work vs "old stable" at a glance.

14. 🟢 **Throttle indicator (per-project)** — a small chip at the top
    of the chat input showing live model state: "ready", "queued
    behind 1 task", "model busy 12s". Same data the Ops queue shows;
    repurposed at the chat level so you don't have to switch tabs to
    know why a turn is slow.

15. 🟢 **Resume last session banner** — top of the project page on
    return: "You were last working on `/src/login.tsx (v3)` — open it?".
    Stores `last_focus_*` columns on `projects` per user (cookie-scoped
    in single-user mode).

### Sharing & artefacts

16. 🟢 **Shareable read-only project URL** — `POST /projects/{id}/share
    {snapshot_id?, expires_in_days?}` returns a slug; visiting
    `/p/{slug}` renders a read-only viewer of that snapshot's files.
    Stored in `project_share_tokens (token, project_id,
    snapshot_id?, expires_at, revoked_at)`. No auth gate; revoke any
    time. Useful for showing a teammate / customer a frozen state.

17. 🟢 **Image / asset paste** — paste an image into chat or an `.md`
    file → stored as `/assets/YYYY-MM-DD-{hash}.png` (or `.jpg`,
    `.svg`). Markdown previews render them inline. The image bytes
    live in a NocoDB attachment column on `project_file_versions`
    rather than the `content` text field. Cap at 1 MB / image.

18. 🟢 **Per-file model override** — column `project_files.preferred_model`
    nullable. When the agent edits this file, the override wins over
    the project default. Cheap plumbing; lets you say "always use
    the larger model when touching `/src/auth/*`" without making
    every turn expensive.

### Cross-project & multi-window

19. 🟢 **Cross-project drag** — drag a file from one project's tree to
    another's. UI-only; calls a server-side `POST
    /projects/{dst}/fs/import-from {src_project_id, paths}` that copies
    the *current* version (no version history transfer; provenance
    tag `created_by = "copy:{src_slug}:/path"`).

20. 🟢 **Recent edits sidebar** — global panel (sidebar drawer)
    showing the last N file edits across **all** projects. Lets you
    flip between active projects without losing your place. Reads
    `project_audit` rows, `kind=file_write`, sorted by ts.

21. 🟢 **Project chat history search** — search across every
    conversation message in a project (using the project's Chroma
    collection if embeddings are warm, falling back to substring).
    `GET /projects/{id}/chats/search?q=...`. Lets you find "where
    did the agent suggest using JWT?" weeks later.

### Input & accessibility

22. 🟡 **Voice input** — `/code/transcribe` endpoint accepts an audio
    blob, returns text via Whisper.cpp running on CPU (small / base
    models — `~200 ms/s` on a modern Mac CPU). Mic button next to
    the chat input. Single-shot transcription per recording; no
    streaming partial-results to keep CPU bounded.

23. 🟢 **Per-language file templates** — when creating a new file via
    UI, the Path → Kind detection offers a starter (`.tsx → React
    component stub`, `.md → frontmatter + heading`, `.py → module
    docstring + main`). Templates ship in `infra/file_templates/` as
    plain text. Saves the agent from being asked to write boilerplate.

24. 🟢 **Keyboard-first command palette (⌘K)** — superset of the
    `⌘P` symbol palette: actions ("apply this plan", "push to
    Gitea", "create ADR", "lint /src"), files, and symbols all in
    one fuzzy matcher. Single source of truth for everything the user
    can do, keyboard-only. Critical on a workstation where every
    extra click compounds.

### Backend hygiene

25. 🟢 **Path normalisation guard** — single helper used everywhere
    paths are accepted. Rejects: empty path, no leading `/`, `..`,
    duplicate slashes, NUL bytes, paths > 512 chars. Standardises
    case-folding behaviour (POSIX-strict). One small `infra/fs/path.py`
    module, unit-tested. Prevents a lot of bug surface across all
    the FS endpoints.

26. 🟢 **Idempotent fs_write** — accept an optional
    `if_content_hash` parameter; if the current version's hash
    differs, return 409. Lets the agent (and the UI editor) retry
    safely on transient errors without producing duplicate versions
    when the request actually succeeded the first time.

27. 🟢 **Soft-delete grace window** — `archived_at` is a 30-day
    tombstone; an undelete button restores the file with a new
    version. After 30 days a nightly job hard-deletes. Saves a class
    of "I didn't mean to delete that" panic.

28. 🟢 **Single feature flag for the whole stack** — `features.code_v2`
    in `config.json`. While off, the new project routes 404 and the
    UI hides the projects nav. Lets the work land incrementally without
    breaking the existing ad-hoc Code page.

### Why not (rejected ideas)

For the record, things I considered and rejected for CPU + scope reasons:

- **Inline AI autocomplete** (Copilot-style ghost text) — needs a fast
  small model with very low first-token latency. CPU can't deliver
  that without a dedicated model role and special tuning. The "inline
  ghost diff" feature on selection (already in the plan) is the
  acceptable substitute.
- **Real-time collaborative editing** — out of scope; this is single-
  user. CRDTs would balloon the schema.
- **Built-in code execution / REPL** — explicitly excluded by the
  user. Linters cover the validation need.
- **Live websocket file-system watcher** — every project change
  already flows through SSE on the project channel; adding a
  second wire would be redundant.

### Updated build sequence

Inserting the additions where they slot most naturally, **after the
existing 11-step sequence** (which itself ended at "stretch list").
Keep all of these incremental — every step is independently
deployable and shippable as soon as it's working.

12. Path normalisation guard, idempotent fs_write, soft-delete
    grace window, single feature flag (backend hygiene; trivial,
    high payoff).
13. Bookmarks, breadcrumbs, distraction-free mode, recent-edits
    sidebar (UI-only state).
14. Project change feed, open-work overview, throttle indicator
    (visibility — pure DB queries).
15. Quick-fix lightbulb, "describe this file", linked tasks
    (cheap agent integrations).
16. Per-file blame, image/asset paste, per-language templates.
17. Shareable read-only URL, per-file model override.
18. Voice input, command palette (⌘K).
19. Cross-project drag, project chat history search.
20. Gitea: create-repo + IDE-like push/pull (lands once the
    underlying push/pull lands).

That's a clean 20-step ladder from schema to a fully-fleshed-out
project workspace, with CPU-cheap items front-loaded and any expensive
agent-driven flows clearly gated.

---

## Powerful tools we haven't considered

These are higher-leverage features that don't fit the earlier groupings.
Each has a **CPU cost profile** (🟢 free / 🟡 cheap / 🔴 expensive)
and explicit gating notes for the expensive ones.

### Tools that change how the agent thinks

1. 🟡 **Spec-first scaffolding** — drop a YAML/JSON schema into
   `/.project/specs/{name}.yaml`, then `POST /projects/{id}/scaffold-from-spec
   {spec_path, targets: [...]}` runs `scaffold` mode against the spec
   to emit aligned files (types, tests, fixtures). On spec change, a
   diff-mode regeneration: the agent regenerates only the lines that
   diverge from the new spec, preserving user edits elsewhere. The
   spec is the source of truth; the agent's role is to keep code in
   sync with it. CPU: one call per scaffold; on regen, one call per
   target file. Bounded.

2. 🟡 **Test-first apply mode** — toggle on the `apply` request:
   when set, the agent's prompt is forced to "write the tests first,
   verify they capture intent, then write the implementation". Output
   is two fenced blocks: tests file + implementation file, in that
   order. The streaming parser writes the test file as `v_n`, then
   the implementation as `v_n` of its own file. UI shows them as
   linked changes. Single call; cost identical to a normal `apply`.

3. 🔴 **Migration playbook** — for goals like "React 17 → 18",
   "Pydantic v1 → v2", "switch from class components to hooks",
   the agent emits a multi-step playbook upfront, then executes one
   step at a time with checkpoints. Each step is a small `apply`
   with `scope_paths` derived from the playbook, followed by a
   user-confirm gate. Stored as `project_playbooks (id, project_id,
   goal, steps, current_step, status)`. CPU profile: **N+1 calls
   total** (one to plan, N to execute), but each is small. The user
   sees concrete, reviewable progress; aborts safely between steps.

4. 🟢 **Compose action** — select N files (or N regions of files)
   in the tree, click "Compose for next prompt". The selection is
   serialised as the agent's pinned context for the next single
   turn only — overrides the project pin budget temporarily. Lets
   you ask cross-file questions ("how does login.tsx relate to
   auth/provider.tsx and the spec?") without permanently pinning
   anything. Pure UI + a one-shot context override.

5. 🟢 **Diff-minimise** — after any agent-emitted change, a "Tighten"
   action runs *no model call* — it pure-CPU re-applies the agent's
   patch but strips whitespace-only and comment-only lines from the
   diff, surfaces the smaller patch as a draft alternative. Lets the
   user accept the minimal change instead of the full reformatting
   the agent might have done.

6. 🟡 **AI naming assistant** — right-click on any identifier in the
   editor → "Propose better names". One small model call with the
   identifier plus the file's symbols index returns 3 alternatives
   with one-line justifications. User picks → triggers the cross-file
   rename below. Cost capped at one call per rename request.

7. 🟢 **Cross-file symbol rename** — uses the tree-sitter symbol
   index. `POST /projects/{id}/rename {old, new, kind}` returns a
   diff preview across all files. Confirm → one new version per
   affected file, all sharing an `edit_summary` of `rename: old → new`.
   No model needed once the user picks the target — the rename is
   deterministic.

### Tools that ground the project

1. 🟢 **Project glossary** — periodic background scan extracts
   nouns from docstrings, ADR titles, conventions, and pinned files;
   builds a per-project term dictionary. The agent receives the top-N
   terms as a "vocabulary" line in its system prompt so usage stays
   consistent (`Auth Provider` not `auth_provider` mid-paragraph).
   Cost: one cheap `tool` call nightly; no per-turn cost.

2. 🟢 **Documentation coverage** — counts public symbols with /
   without docstrings (Python, TypeScript exports), surfaces a
   coverage % per file and per project. Pure tree-sitter walk.
   Drives a "Doc gap" panel; clicking a file's gap lights up the
   undocumented symbols in the gutter.

3. 🟡 **Doc fill batch** — companion to coverage: "Fill missing
   docstrings in /src" runs *one* model call with all undocumented
   symbol signatures (no bodies, just signatures + immediate context),
   returns a single block of `path:line | docstring` pairs that get
   merged in deterministically. Avoids the N-call trap.

4. 🟢 **Cyclomatic complexity per file** — cheap CPU walk
   (radon for Python, eslint-plugin for JS). Stored as a
   `complexity` column on files. Surfaced as a tree badge for files
   over a threshold. Click → go to the most complex function.

5. 🟢 **Dependency vulnerability scan** — read `package.json`,
   `requirements.txt`, `pyproject.toml`. Match against a local OSV
   advisory mirror (no network call per scan if pre-synced). Outputs
   a vulnerabilities table. Manual refresh or daily cron. No agent
   needed.

6. 🟢 **Test discovery (no execution)** — pure structural scan:
   files matching `*_test.py` / `*.test.ts` / `__tests__/`. Counts
   `def test_*`, `it()`, `test()` blocks. Surfaces "untested
   function" markers when a function has zero callers in test files.
   Trivial to compute from the symbols index + import graph.

### Time and history

1. 🟢 **Time-travel diff** — "show me what /src/login.tsx looked
   like 3 days ago" → endpoint resolves the version active at a
   given timestamp via `project_file_versions` ordering. Produces
   a diff against current. UI: a date picker in the file viewer.
   No new schema; uses what versions already record.

2. 🟢 **Project graveyard** — `/projects/{id}/graveyard` lists
   archived (soft-deleted) files with their last version's content.
   Restore button rolls them forward. After 30 days they hard-delete;
   the graveyard surfaces age so the user can rescue before purge.

3. 🟢 **Auto CHANGES.md** — every successful `apply` appends one
   line to `/CHANGES.md`: `2026-05-04 14:32 · agent · /src/login.tsx
   v3 — added MFA gate (conv 412)`. The file is auto-pinned (small).
   Free running ledger of what changed and why.

4. 🟢 **Generated artifact watermarks** — when the agent writes a
   file, append a footer comment marking it agent-generated with
   the conversation id (`# generated by jeff/conv-412`,
   commented appropriately for the language). Off by default per
   project; on per file with `project_files.watermark = true`.
   Useful for "this is the file the AI maintains; don't hand-edit".
   Removed from the diff renderer so the user doesn't see noise.

### Project shape and orchestration

1. 🟢 **Multi-project workspace** — `workspaces (id, org_id, name,
   project_ids[])`. A workspace is a saved set of projects (e.g.
   `frontend + backend + spec`). When active, chat retrieval can
   span all member projects' Chroma collections; the file tree
   shows all member trees with project-prefix separators; ⌘P
   palette searches across them. No new conversation model — chat
   still belongs to one project; the workspace is a navigational
   convenience.

2. 🟢 **Pin Q&A as FAQ** — any chat exchange can be pinned to
   `/FAQ.md`. The agent maintains FAQ.md: dedupes near-duplicates,
   keeps the most recent answer when contradicted, sorts by
   recency. One cheap `tool` call per pinned-message addition. The
   FAQ is auto-pinned, so the next conversation knows what's
   already been answered.

3. 🟢 **Pre-flight checks before push** — push action blocks
   on a configurable check chain: `lint` → `typecheck` → `ai_review`
   (optional). Each check is opt-in per project. Failing check
   stops the push, shows results inline. "Push anyway" requires
   confirmation. CPU profile depends on which checks are enabled —
   user-controlled.

4. 🟢 **Pre-commit hook chain (per-project)** — same plumbing as
   pre-flight, runs on `fs_write` (debounced 5s after last write).
   Configurable: format-only (cheap, every write) vs lint
   (per-write) vs typecheck (per-batch). Stored on
   `projects.precommit_chain`.

5. 🟡 **Hot-reload of conventions** — when `/.project/conventions.md`
   is written or imported, drop the project's prompt-cache entry so
   the next chat turn sees the new rules immediately. Same plumbing
   as the cache-controls panel I already wired earlier.

### Editor superpowers

1. 🟢 **Scratch buffers** — untitled, ephemeral files (`scratch:1`,
   `scratch:2`) that don't persist past the session. Useful for
   trying agent output before deciding to commit. "Save as" turns
   them into proper files at a chosen path. Pure UI state.

2. 🟢 **Visual import graph** — Cytoscape view of the project's
   files; node = file, edge = import. Click a node → open file.
   Click an edge → see what symbols cross. Uses the existing
   import-graph table. Useful for orientation in unfamiliar
   projects.

3. 🟢 **Sticky breadcrumb context** — when scrolling deep into a
   long file, the breadcrumbs sticky-pin to show
   `function > if-branch > nested-block`. Tree-sitter walk on cursor
   move. Helps in long files, especially where the agent has
   produced a lot of code.

4. 🟢 **Diff-aware mini-map** — Monaco's existing minimap, but
   coloured to show which lines changed in the most recent version
   (green: agent added, blue: user added, red: removed). At-a-glance
   "what's new in this file". Pure render-side.

5. 🟢 **Per-line history bar** — `cmd-hover` on any line shows a
   tiny floating "blame trail" with the last 3 versions that
   touched it: `v3 (agent · 2h) | v2 (user · 1d) | v1 (import)`.
   Built from the blame index already proposed.

### Smart input affordances

1. 🟡 **Slash commands in the chat** — `/plan`, `/apply`, `/decide`,
   `/explain`, `/scaffold`, `/refine` switch mode for the next turn
   only. `/file /src/login.tsx` quickly attaches a file. `/diff
   from=v2 to=v4 path=…` opens the diff viewer. `/lint` runs lint
   on the project. Power-user shortcut for keyboard-driven flow.
   Cost: zero except where the underlying mode runs a model call.

2. 🟢 **Prompt templates / variables** — saved prompts with
   `${file}`, `${selection}`, `${todays_date}`, `${recent_decision}`
   placeholders. Insert with `⌘'`. Useful for repeated patterns
   like "review this PR style: ${diff}".

3. 🟢 **Reuse last reply** — quick action on any assistant message:
   "use this as the starter for next turn". Pre-fills the chat
   composer with the message, edit-and-send to refine. No model
   call to set up.

### Sequencing

These slot **after** the build-sequence already in the plan.
Suggested ordering, with CPU-cheap items front-loaded:

- Tier A (no model cost): time-travel diff, project graveyard, auto
  CHANGES.md, generated artifact watermarks, scratch buffers, visual
  import graph, mini-map, slash commands, prompt templates,
  pre-commit hook chain, pre-flight checks, multi-project workspace,
  cross-file symbol rename, diff-minimise, sticky breadcrumb context,
  per-line history bar, complexity badges, dependency vulnerability
  scan, test discovery, documentation coverage.
- Tier B (one cheap call): doc fill batch, spec-first scaffolding
  (regen path), naming assistant, FAQ maintenance, hot-reload of
  conventions, project glossary.
- Tier C (multi-call, gated): test-first apply (one call but bigger
  output), migration playbook (N+1 calls).

Net: even if the user only ships Tier A, they end up with an editor
that feels close to a real IDE — minus execution, plus first-class
versioning, time-travel, and project shape. Tier B and C are the
"agent really does work for you" tools, capped at controllable
cost.

---

## What this stops being

- A bag of inputs you re-attach every conversation.
- A toggle that only changes RAG and nothing else.
- Output that vanishes the moment the chat scrolls.

…and starts being a place where the AI's work *accumulates*, version by version, retrievable in a tree, diffable against history, and scoped behind a single context-pack you carry between sessions.
