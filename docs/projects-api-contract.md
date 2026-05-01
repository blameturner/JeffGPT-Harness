# Projects API contract (backend snapshot)

This document tracks implemented `/projects/*` endpoints for frontend wiring.

## Core

- `GET /projects?org_id=&archived=` -> `{projects: [...]}`
- `POST /projects` -> `{project}`
- `GET /projects/{project_id}?org_id=` -> `{project, file_count, latest_activity_at}`
- `PATCH /projects/{project_id}?org_id=` -> `{project}`
- `POST /projects/{project_id}/archive?org_id=` -> `{ok}`

## Filesystem

- `GET /projects/{project_id}/fs?org_id=&prefix=` -> `{files:[...]}`
- `GET /projects/{project_id}/fs/file?org_id=&path=` -> `{file,current_version}`
- `GET /projects/{project_id}/fs/file/versions?org_id=&path=` -> `{versions:[...]}`
- `GET /projects/{project_id}/fs/file/diff?org_id=&path=&from_version=&to_version=` -> `{unified}`
- `PUT /projects/{project_id}/fs/file?org_id=` -> `{file,version,changed}` (optional body field `if_content_hash`; mismatch returns 409 with `{error,expected,actual}`)
- `POST /projects/{project_id}/fs/import?org_id=` -> `{written,skipped}`
- `POST /projects/{project_id}/fs/import-from?org_id=` -> `{written,skipped,missing}`
- `POST /projects/{project_id}/fs/file/pin?org_id=` -> `{ok}`
- `POST /projects/{project_id}/fs/file/lock?org_id=` -> `{ok}`
- `DELETE /projects/{project_id}/fs/file?org_id=&path=` -> `{ok}`
- `POST /projects/{project_id}/fs/file/restore?org_id=` -> `{file,version,changed}`
- `POST /projects/{project_id}/fs/move?org_id=` -> `{file}`
- `GET /projects/{project_id}/fs/search?org_id=&q=` -> `{hits:[...]}`
- `GET /projects/{project_id}/fs/export?org_id=&format=zip` -> zip stream

## Chat + streams

- `POST /projects/{project_id}/chat?org_id=` -> `{job_id}`
- `GET /projects/{project_id}/chat/stream/{job_id}?org_id=&cursor=` -> SSE stream

## Snapshots & cross-project diff

- `POST /projects/{project_id}/snapshots?org_id=` body `{label,description?}` -> `{snapshot}` (label `[A-Za-z0-9._- ]{1,80}`, must be unique per project)
- `GET /projects/{project_id}/snapshots?org_id=&limit=` -> `{snapshots:[...]}`
- `GET /projects/{project_id}/snapshots/{label}/diff?org_id=` -> `{snapshot, files:[{path,state,from_version,to_version,unified}]}`
- `GET /projects/{project_id}/diff?org_id=&against={other_project_id}` -> `{project_id,against,files:[{path,state,left_version,right_version,unified}]}`
- `GET /projects/{project_id}/graveyard?org_id=&limit=` -> `{files:[{path,kind,size_bytes,archived_at,age_days,current_version_id}], count}` (use existing `POST .../fs/file/restore` to bring one back)

## Ops / history

- `GET /projects/{project_id}/audit?org_id=&kind=&limit=` -> `{events:[...]}`
- `GET /projects/{project_id}/history?org_id=&limit=` -> `{events,total_versions}`
- `GET /projects/{project_id}/metrics?org_id=&period=30d` -> metrics payload
- `GET /projects/{project_id}/open-work?org_id=` -> open todo + permission request summary
- `GET /projects/{project_id}/chats/search?org_id=&q=&limit=` -> `{hits:[...]}`

## Planning / inspector

- `POST /projects/{project_id}/plans/{message_id}/preview?org_id=` -> `{items:[{path,action,existing_size}]}`
- `POST /projects/{project_id}/turns/{message_id}/context-inspector?org_id=` -> full context inspector payload
- `POST /projects/{project_id}/turns/{message_id}/context-inspector/summary?org_id=` -> compact inspector payload

## Branches & sharing

- `POST /projects/{project_id}/branch?org_id=` body `{name, from_snapshot?}` -> `{project_id, files_written, parent_project_id}`
- `POST /projects/{project_id}/share?org_id=` body `{snapshot_id?, expires_in_days?}` -> `{token, url_path, expires_at}`
- `DELETE /projects/share/{token}` -> `{revoked}`
- `GET /projects/p/{token}` (anonymous) -> `{project, files}` (read-only)

## Bookmarks / saved queries / recipes / pins

- `POST/GET /projects/{project_id}/bookmarks?org_id=`
- `DELETE /projects/{project_id}/bookmarks/{bookmark_id}?org_id=`
- `POST/GET /projects/{project_id}/saved-queries?org_id=`
- `POST/GET /projects/{project_id}/recipes?org_id=`
- `POST/GET /projects/{project_id}/pins?org_id=`

## Workspaces

- `POST /projects/workspaces` body `{org_id, name, project_ids}`
- `GET /projects/workspaces?org_id=`

## Templates / ADRs / change log

- `GET /projects/_templates/file?path=&name=`
- `GET /projects/_templates/conventions`
- `POST /projects/{project_id}/adrs?org_id=` body `{title, context?, decision?, consequences?}` -> `{path, number, file, version}`
- `POST /projects/{project_id}/changes/append?org_id=&line=`

## Time-travel / find-replace / rename

- `GET /projects/{project_id}/fs/file/at-time?org_id=&path=&at=` -> `{path, found, version_at_time, content_at_time, current_version, unified}`
- `POST /projects/{project_id}/fs/replace?org_id=` body `{pattern, replacement, paths?, regex, dry_run}`
- `POST /projects/{project_id}/rename?org_id=` body `{old, new, kind?, paths?}` -> `{old, new, files, file_count}`

## Workflow controls

- `PUT /projects/{project_id}/precommit-chain?org_id=` body `{chain: [...]}`
- `GET /projects/{project_id}/precommit-chain?org_id=`
- `POST /projects/{project_id}/cache/drop?org_id=` (hot-reload trigger; emits audit `cache_drop`)
- `POST /projects/{project_id}/preflight?org_id=` -> `{ok, checks: [...]}`
- `GET /projects/{project_id}/staged?org_id=` / `POST .../staged/{pending_id}` body `{accept}`

## Per-conversation / per-file

- `PUT /projects/{project_id}/conversations/{conversation_id}/scope?org_id=` body `{scope_paths}`
- `PUT /projects/{project_id}/fs/file/preferred-model?org_id=` body `{path, preferred_model}`
- `PUT /projects/{project_id}/fs/file/watermark?org_id=` body `{path, watermark}`

## Playbooks / scaffolding

- `POST/GET /projects/{project_id}/playbooks?org_id=`
- `PATCH /projects/{project_id}/playbooks/{playbook_id}?org_id=` body `{current_step, status}`
- `POST /projects/{project_id}/scaffold-from-spec?org_id=` body `{spec_path, targets?}`

## File comments / change feed

- `POST/GET /projects/{project_id}/fs/file/comments?org_id=` (POST body `{path, version, anchor, body}`)
- `GET /projects/{project_id}/feed?org_id=&hours=24`

## Analysis

- `POST /projects/{project_id}/lint?org_id=&path=` -> `{files, issues_total, files_scanned}`
- `GET /projects/{project_id}/issues?org_id=&severity=&limit=`
- `POST /projects/{project_id}/symbols/reindex?org_id=`
- `GET /projects/{project_id}/symbols?org_id=&q=`
- `GET /projects/{project_id}/symbol/{name}/refs?org_id=`
- `GET /projects/{project_id}/graph?org_id=` -> Cytoscape `{elements:{nodes,edges}}`
- `GET /projects/{project_id}/complexity?org_id=&threshold=`
- `GET /projects/{project_id}/doc-coverage?org_id=`
- `GET /projects/{project_id}/tests?org_id=`
- `GET /projects/{project_id}/dependencies?org_id=`
- `GET /projects/{project_id}/glossary?org_id=&top_n=`
- `GET /projects/{project_id}/diff/conversation/{conversation_id}?org_id=`

## Gitea

- `GET /gitea/connection?org_id=` / `PUT /gitea/connection` / `DELETE /gitea/connection?org_id=`
- `GET /gitea/repos?org_id=&limit=`
- `GET /gitea/repos/{owner}/{repo}/contents?org_id=&path=&ref=`
- `POST /projects/import-from-gitea` body `{org_id, owner, repo, ref, name, ignore?}` -> `{project_id, written, skipped, head_sha}` (head_sha is captured at import time so the project starts `in_sync` with the imported ref)
- `POST /projects/{project_id}/create-gitea-repo?org_id=` body `{owner, owner_kind, repo, description?, private, default_branch, init_readme}`
- `POST /projects/{project_id}/push-to-gitea?org_id=` body `{branch?, message, paths?, scope, force?}` -> `{branch, pushed, skipped, failures, head_sha}`. By default, only files whose current version isn't already on remote (i.e. `pushed_to_sha` is empty) are pushed; pass `force=true` or explicit `paths` to override. Returns `409 {detail:{reason:"remote_diverged", remote_head_sha, behind_count, hint}}` when the remote has unsynced commits and `force=false`.
- `GET /projects/{project_id}/gitea/status?org_id=` -> `{linked, origin, ahead, behind_count, remote_head_sha, last_synced_sha, last_synced_at, state}`
- `GET /projects/{project_id}/gitea/pull/preview?org_id=`
- `POST /projects/{project_id}/gitea/pull/apply?org_id=` body `{decisions:[{path,choice}], set_synced_to}`

## Notes

- `features.code_v2.enabled` must be true for projects routes.
- Most endpoints require `org_id` query param for scope validation.
- Path inputs must be absolute (e.g. `/src/app.py`) and pass `normalize_project_path`.
- Optional NocoDB tables (no-op when missing): `project_audit`, `project_snapshots`, `project_snapshot_files`, `project_lint_results`, `project_symbols`, `project_dependencies`, `project_share_tokens`, `project_bookmarks`, `project_saved_queries`, `project_recipes`, `project_pins`, `workspaces`, `project_pending_changes`, `project_playbooks`, `project_reviews`, `project_file_comments`, `gitea_connections`. Mutating endpoints that hit a missing table return 500 with `missing table`; read endpoints return empty lists.

