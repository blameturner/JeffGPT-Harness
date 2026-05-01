import time
import logging
import hashlib
from datetime import datetime, timezone
import requests
from infra.config import NOCODB_URL, NOCODB_TOKEN, NOCODB_BASE_ID

_log = logging.getLogger("nocodb")

PROJECT_MAX_FILES = 5000
PROJECT_MAX_FILE_BYTES = 100 * 1024


class ConflictError(Exception):
    def __init__(self, message: str, expected: str = "", actual: str = ""):
        super().__init__(message)
        self.expected = expected
        self.actual = actual


class NocodbClient:
    def __init__(self):
        self.url = f"{NOCODB_URL}/api/v1/db/data/noco/{NOCODB_BASE_ID}"
        self.headers = {
            "xc-token": NOCODB_TOKEN,
            "Content-Type": "application/json"
        }
        self.tables = self._load_tables()

    def _load_tables(self) -> dict:
        for attempt in range(15):
            try:
                response = requests.get(
                    f"{NOCODB_URL}/api/v1/db/meta/projects/{NOCODB_BASE_ID}/tables",
                    headers={"xc-token": NOCODB_TOKEN},
                    timeout=10
                )
                response.raise_for_status()
                tables = response.json()["list"]
                table_map = {table["title"]: table["id"] for table in tables}
                if not hasattr(NocodbClient, "_tables_logged"):
                    _log.info("tables loaded  count=%d", len(table_map))
                    NocodbClient._tables_logged = True
                return table_map
            except Exception:
                _log.warning("not ready, retrying (%d/15)", attempt + 1)
                time.sleep(2)
        raise RuntimeError("Could not connect to Nocodb after 30 seconds")

    def _get(self, table: str, params: dict = None) -> dict:
        response = requests.get(
            f"{self.url}/{self.tables[table]}",
            headers=self.headers,
            params=params,
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    def _get_paginated(self, table: str, params: dict, page_size: int = 50) -> list[dict]:
        # NocoDB expands M2M links as one UNION ALL subquery per row. Large pages
        # hit SQLite's compound-SELECT cap, so we fetch in small chunks.
        out: list[dict] = []
        offset = 0
        target = int(params.get("limit", 500))
        base = {k: v for k, v in params.items() if k not in ("limit", "offset")}
        while len(out) < target:
            chunk_size = min(page_size, target - len(out))
            chunk = self._get(table, params={**base, "limit": chunk_size, "offset": offset})
            rows = chunk.get("list", [])
            out.extend(rows)
            page_info = chunk.get("pageInfo") or {}
            if page_info.get("isLastPage") or len(rows) < chunk_size:
                break
            offset += len(rows)
        return out

    def _post(self, table: str, data: dict) -> dict:
        _log.debug("db write  %s", table)
        response = requests.post(
            f"{self.url}/{self.tables[table]}",
            headers=self.headers,
            json=data,
            timeout=10
        )
        if response.status_code >= 400:
            _log.error(
                "db write %s failed  %d  body=%s  payload_keys=%s",
                table, response.status_code, response.text[:2000], sorted(data.keys()),
            )
        response.raise_for_status()
        result = response.json()
        _log.debug("db write ok  %s id=%s", table, result.get("Id"))
        return result

    def _patch(self, table: str, row_id: int, data: dict) -> dict:
        _log.debug("db update  %s/%d", table, row_id)
        response = requests.patch(
            f"{self.url}/{self.tables[table]}/{row_id}",
            headers=self.headers,
            json=data,
            timeout=10
        )
        if response.status_code >= 400:
            _log.error(
                "db update %s/%d failed  %d  body=%s  payload_keys=%s",
                table, row_id, response.status_code, response.text[:2000], sorted(data.keys()),
            )
        response.raise_for_status()
        return response.json()

    def _delete(self, table: str, row_id: int) -> bool:
        """Hard-delete a row by primary key. Returns True on success."""
        _log.debug("db delete  %s/%d", table, row_id)
        response = requests.delete(
            f"{self.url}/{self.tables[table]}/{row_id}",
            headers=self.headers,
            timeout=10,
        )
        if response.status_code >= 400:
            _log.error(
                "db delete %s/%d failed  %d  body=%s",
                table, row_id, response.status_code, response.text[:2000],
            )
        response.raise_for_status()
        return True

    def _has_table(self, table: str) -> bool:
        return table in self.tables

    def _require_table(self, table: str) -> None:
        if not self._has_table(table):
            raise RuntimeError(f"missing table '{table}'")

    def list_agents(self, org_id: int, limit: int = 200) -> list[dict]:
        rows = self._get_paginated("agents", params={
            "where": f"(org_id,eq,{org_id})~and(deleted_at,is,null)",
            "limit": limit,
        })
        _log.debug("list_agents  org=%d count=%d", org_id, len(rows))
        return rows

    def get_agent(self, name: str, org_id: int) -> dict | None:
        data = self._get("agents", params={
            "where": f"(name,eq,{name})~and(org_id,eq,{org_id})~and(deleted_at,is,null)",
            "limit": 1
        })
        records = data.get("list", [])
        found = records[0] if records else None
        _log.debug("get_agent  name=%s org=%d found=%s", name, org_id, bool(found))
        return found

    def create_run(self, agent: dict, org_id: int, task_description: str, product: str) -> dict:
        _log.info("create_run  agent=%s org=%d", agent.get("name"), org_id)
        return self._post("agent_runs", {
            "agent_id": agent["Id"],
            "agent_name": agent["name"],
            "agent_version": agent.get("version", 1),
            "org_id": org_id,
            "project_id": agent.get("project_id"),
            "product": product,
            "task_description": task_description,
            "status": "running"
        })

    def complete_run(
        self,
        run_id: int,
        summary: str,
        tokens_input: int,
        tokens_output: int,
        context_tokens: int,
        duration_seconds: float,
        quality_score: int,
        model_name: str
    ) -> dict:
        _log.info("complete_run  id=%d tokens_in=%d tokens_out=%d %.1fs", run_id, tokens_input, tokens_output, duration_seconds)
        return self._patch("agent_runs", run_id, {
            "status": "complete",
            "summary": summary,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "context_tokens": context_tokens,
            "duration_seconds": duration_seconds,
            "quality_score": quality_score,
            "model_name": model_name
        })

    def fail_run(self, run_id: int, error_message: str) -> dict:
        _log.warning("fail_run  id=%d error=%s", run_id, error_message[:200])
        return self._patch("agent_runs", run_id, {
            "status": "failed",
            "error_message": error_message
        })

    def save_output(self, run: dict, full_text: str, chroma_ids: list) -> dict:
        _log.debug("save_output  run=%d text_len=%d chroma_ids=%d", run["Id"], len(full_text), len(chroma_ids))
        return self._post("agent_outputs", {
            "run_id": run["Id"],
            "agent_id": run["agent_id"],
            "agent_name": run["agent_name"],
            "org_id": run["org_id"],
            "project_id": run.get("project_id"),
            "full_text": full_text,
            "chroma_ids": chroma_ids
        })

    def create_conversation(
        self,
        org_id: int,
        model: str,
        title: str = "",
        rag_enabled: bool = False,
        rag_collection: str | None = None,
        knowledge_enabled: bool = False,
    ) -> dict:
        _log.info("create_conversation  org=%d model=%s title=%s rag=%s knowledge=%s", org_id, model, title[:40], rag_enabled, knowledge_enabled)
        return self._post("conversations", {
            "org_id": org_id,
            "model": model,
            "title": title or "New chat",
            "rag_enabled": 1 if rag_enabled else 0,
            "rag_collection": rag_collection or "",
            "knowledge_enabled": 1 if knowledge_enabled else 0,
        })

    def get_conversation(self, conversation_id: int, org_id: int | None = None) -> dict | None:
        where = f"(Id,eq,{conversation_id})"
        if org_id is not None:
            where = f"{where}~and(org_id,eq,{int(org_id)})"
        data = self._get("conversations", params={
            "where": where,
            "limit": 1
        })
        records = data.get("list", [])
        return records[0] if records else None

    def update_conversation(self, conversation_id: int, data: dict) -> dict:
        return self._patch("conversations", conversation_id, {"Id": conversation_id, **data})

    def list_conversations(self, org_id: int, limit: int = 50) -> list[dict]:
        # Exclude the home dashboard's rolling conversation — it's surfaced
        # via /home/overview, not the chat list. Filter in Python rather than
        # NocoDB's where-clause because NocoDB drops NULL `kind` rows under
        # both `neq` and `~not(...eq...)`. Strings duplicated from
        # shared/home_conversation.py (HOME_KIND, HOME_TITLE) to avoid a
        # circular import.
        # Over-fetch so the home-row exclusion doesn't shrink the page.
        rows = self._get_paginated("conversations", params={
            "where": f"(org_id,eq,{org_id})",
            "sort": "-CreatedAt",
            "limit": limit + 1,
        })
        filtered = [
            r for r in rows
            if r.get("kind") != "home" and r.get("title") != "Home — ongoing"
        ]
        return filtered[:limit]

    def list_messages(self, conversation_id: int, limit: int = 500, org_id: int | None = None) -> list[dict]:
        where = f"(conversation_id,eq,{conversation_id})"
        if org_id is not None:
            where = f"{where}~and(org_id,eq,{int(org_id)})"
        return self._get_paginated("messages", params={
            "where": where,
            "sort": "CreatedAt",
            "limit": limit,
        })

    def add_message(
        self,
        conversation_id: int,
        org_id: int,
        role: str,
        content: str,
        model: str = "",
        tokens_input: int = 0,
        tokens_output: int = 0,
        response_style: str = "",
        search_used: bool = False,
        search_status: str = "",
        search_confidence: str = "",
        search_source_count: int = 0,
        search_context_text: str = "",
        **extra_fields,
    ) -> dict:
        # nocodb silently drops unknown columns — schema-optional fields are safe to pass
        _log.info("add_message  conv=%d role=%s model=%s content_len=%d", conversation_id, role, model, len(content))
        payload = {
            "conversation_id": conversation_id,
            "org_id": org_id,
            "role": role,
            "content": content,
            "model": model,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
        }
        if response_style:
            payload["response_style"] = response_style
        if search_used:
            payload["search_used"] = 1
        if search_status:
            payload["search_status"] = search_status
        if search_confidence:
            payload["search_confidence"] = search_confidence
        if search_source_count:
            payload["search_source_count"] = search_source_count
        if search_context_text:
            payload["search_context_text"] = search_context_text
        for key, value in extra_fields.items():
            if value is None or value == "":
                continue
            payload[key] = value
        return self._post("messages", payload)

    def add_message_search_sources(
        self,
        message_id: int,
        conversation_id: int,
        org_id: int,
        sources: list[dict],
    ) -> list[dict]:
        rows: list[dict] = []
        for i, src in enumerate(sources):
            payload = {
                "message_id": message_id,
                "conversation_id": conversation_id,
                "org_id": org_id,
                "source_index": i,
                "title": (src.get("title") or "")[:255],
                "url": src.get("url") or "",
                "relevance": src.get("relevance") or "unknown",
                "source_type": src.get("source_type") or "unknown",
                "content_type": src.get("content_type") or "UNCLEAR",
                "snippet": src.get("snippet") or src.get("summary") or "",
                "used_in_answer": 1 if src.get("used_in_answer") else 0,
            }
            try:
                row = self._post("message_search_sources", payload)
                rows.append(row)
            except Exception:
                _log.error("message_search_sources write failed  msg=%d idx=%d", message_id, i, exc_info=True)
        return rows

    def list_message_search_sources(
        self,
        message_id: int | None = None,
        conversation_id: int | None = None,
        org_id: int | None = None,
    ) -> list[dict]:
        parts = []
        if message_id is not None:
            parts.append(f"(message_id,eq,{message_id})")
        if conversation_id is not None:
            parts.append(f"(conversation_id,eq,{conversation_id})")
        if org_id is not None:
            parts.append(f"(org_id,eq,{int(org_id)})")
        params: dict = {"sort": "source_index", "limit": 500}
        if parts:
            params["where"] = "~and".join(parts)
        return self._get_paginated("message_search_sources", params=params)

    def create_code_conversation(
        self,
        org_id: int,
        model: str,
        title: str = "",
        mode: str = "plan",
        knowledge_enabled: bool = False,
        project_id: int | None = None,
        interactive_fs: bool = False,
    ) -> dict:
        _log.info("create_code_conversation  org=%d model=%s mode=%s knowledge=%s title=%s", org_id, model, mode, knowledge_enabled, title[:40])
        payload = {
            "org_id": org_id,
            "model": model,
            "title": title or "New code session",
            "rag_enabled": 0,
            "rag_collection": mode,
            "knowledge_enabled": 1 if knowledge_enabled else 0,
        }
        if project_id is not None:
            payload["project_id"] = int(project_id)
        if interactive_fs:
            payload["interactive_fs"] = 1
        return self._post("code_conversations", payload)

    def get_code_conversation(self, conversation_id: int, org_id: int | None = None) -> dict | None:
        where = f"(Id,eq,{conversation_id})"
        if org_id is not None:
            where = f"{where}~and(org_id,eq,{int(org_id)})"
        data = self._get("code_conversations", params={
            "where": where,
            "limit": 1,
        })
        records = data.get("list", [])
        return records[0] if records else None

    def update_code_conversation(self, conversation_id: int, data: dict) -> dict:
        return self._patch("code_conversations", conversation_id, {"Id": conversation_id, **data})

    def list_code_conversations(self, org_id: int, limit: int = 50) -> list[dict]:
        return self._get_paginated("code_conversations", params={
            "where": f"(org_id,eq,{org_id})",
            "sort": "-CreatedAt",
            "limit": limit,
        })

    def list_code_messages(self, conversation_id: int, limit: int = 500, org_id: int | None = None) -> list[dict]:
        where = f"(conversation_id,eq,{conversation_id})"
        if org_id is not None:
            where = f"{where}~and(org_id,eq,{int(org_id)})"
        return self._get_paginated("code_messages", params={
            "where": where,
            "sort": "CreatedAt",
            "limit": limit,
        })

    def get_code_message(self, message_id: int, org_id: int | None = None) -> dict | None:
        where = f"(Id,eq,{message_id})"
        if org_id is not None:
            where = f"{where}~and(org_id,eq,{int(org_id)})"
        rows = self._get("code_messages", params={"where": where, "limit": 1}).get("list", [])
        return rows[0] if rows else None

    def add_code_message(
        self,
        conversation_id: int,
        org_id: int,
        role: str,
        content: str,
        model: str = "",
        tokens_input: int = 0,
        tokens_output: int = 0,
        mode: str = "",
        files_json: list | None = None,
        response_style: str = "",
    ) -> dict:
        _log.info("add_code_message  conv=%d role=%s mode=%s content_len=%d", conversation_id, role, mode, len(content))
        payload = {
            "conversation_id": conversation_id,
            "org_id": org_id,
            "role": role,
            "content": content,
            "model": model,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
        }
        if mode:
            payload["mode"] = mode
        if files_json:
            payload["files_json"] = files_json
        if response_style:
            payload["response_style"] = response_style
        return self._post("code_messages", payload)

    def _list_by_conversation(
        self,
        table: str,
        conversation_id: int,
        limit: int = 200,
        org_id: int | None = None,
    ) -> list[dict]:
        try:
            where = f"(conversation_id,eq,{conversation_id})"
            if org_id is not None:
                where = f"{where}~and(org_id,eq,{int(org_id)})"
            return self._get_paginated(table, params={
                "where": where,
                "limit": limit,
            })
        except (requests.HTTPError, KeyError):
            _log.debug("_list_by_conversation  table=%s conv=%d returned empty (table may lack column)", table, conversation_id)
            return []

    def list_runs_for_conversation(self, conversation_id: int, limit: int = 200, org_id: int | None = None) -> list[dict]:
        return self._list_by_conversation("agent_runs", conversation_id, limit, org_id=org_id)

    def list_outputs_for_conversation(self, conversation_id: int, limit: int = 200, org_id: int | None = None) -> list[dict]:
        return self._list_by_conversation("agent_outputs", conversation_id, limit, org_id=org_id)

    def list_tasks_for_conversation(self, conversation_id: int, limit: int = 200, org_id: int | None = None) -> list[dict]:
        return self._list_by_conversation("tasks", conversation_id, limit, org_id=org_id)

    def list_observations_for_conversation(self, conversation_id: int, limit: int = 200, org_id: int | None = None) -> list[dict]:
        try:
            where = f"(conversation_id,eq,{conversation_id})"
            if org_id is not None:
                where = f"{where}~and(org_id,eq,{int(org_id)})"
            return self._get_paginated("observations", params={
                "where": where,
                "limit": limit,
            })
        except requests.HTTPError:
            _log.debug("list_observations  conv=%d returned empty (table may lack column)", conversation_id)
            return []

    def save_observation(
        self,
        run: dict,
        title: str,
        content: str,
        obs_type: str,
        domain: str,
        confidence: str = "medium"
    ) -> dict:
        _log.debug("save_observation  run=%d type=%s domain=%s", run["Id"], obs_type, domain)
        return self._post("observations", {
            "title": title,
            "content": content,
            "type": obs_type,
            "domain": domain,
            "confidence": confidence,
            "status": "open",
            "source_run_id": run["Id"],
            "agent_id": run["agent_id"],
            "agent_name": run["agent_name"],
            "org_id": run["org_id"],
            "project_id": run.get("project_id")
        })

    # Projects + virtual filesystem
    def list_projects(self, org_id: int, archived: bool = False, limit: int = 200) -> list[dict]:
        self._require_table("projects")
        where = f"(org_id,eq,{org_id})"
        where = f"{where}~and(archived_at,isnot,null)" if archived else f"{where}~and(archived_at,is,null)"
        return self._get_paginated("projects", params={"where": where, "sort": "-UpdatedAt", "limit": limit})

    def get_project(self, project_id: int, org_id: int | None = None) -> dict | None:
        self._require_table("projects")
        where = f"(Id,eq,{project_id})"
        if org_id is not None:
            where = f"{where}~and(org_id,eq,{int(org_id)})"
        rows = self._get("projects", params={"where": where, "limit": 1}).get("list", [])
        return rows[0] if rows else None

    def create_project(
        self,
        org_id: int,
        name: str,
        description: str = "",
        slug: str = "",
        system_note: str = "",
        default_model: str = "code",
        retrieval_scope: list[str] | None = None,
        chroma_collection: str = "",
    ) -> dict:
        self._require_table("projects")
        return self._post(
            "projects",
            {
                "org_id": org_id,
                "name": name,
                "slug": slug,
                "description": description,
                "system_note": system_note,
                "default_model": default_model,
                "retrieval_scope": retrieval_scope or [],
                "chroma_collection": chroma_collection,
            },
        )

    def update_project(self, project_id: int, data: dict) -> dict:
        self._require_table("projects")
        return self._patch("projects", project_id, {"Id": project_id, **data})

    def add_project_audit_event(
        self,
        project_id: int,
        actor: str,
        kind: str,
        payload: dict | None = None,
    ) -> dict | None:
        table = "project_audit"
        if table not in self.tables:
            return None
        return self._post(
            table,
            {
                "project_id": project_id,
                "actor": actor,
                "kind": kind,
                "payload": payload or {},
            },
        )

    def list_project_audit_events(
        self,
        project_id: int,
        kind: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        table = "project_audit"
        if table not in self.tables:
            return []
        where = f"(project_id,eq,{project_id})"
        if kind:
            where = f"{where}~and(kind,eq,{kind})"
        return self._get_paginated(table, params={"where": where, "sort": "-CreatedAt", "limit": limit})

    def list_project_files(
        self,
        project_id: int,
        prefix: str | None = None,
        include_archived: bool = False,
        archived_only: bool = False,
        limit: int = 5000,
    ) -> list[dict]:
        self._require_table("project_files")
        parts = [f"(project_id,eq,{project_id})"]
        if archived_only:
            parts.append("(archived_at,isnot,null)")
        elif not include_archived:
            parts.append("(archived_at,is,null)")
        if prefix:
            parts.append(f"(path,like,{prefix}%)")
        where = "~and".join(parts)
        return self._get_paginated(
            "project_files",
            params={"where": where, "sort": "path", "limit": limit},
            page_size=100,
        )

    def get_project_file(self, project_id: int, path: str, include_archived: bool = False) -> dict | None:
        self._require_table("project_files")
        where = f"(project_id,eq,{project_id})~and(path,eq,{path})"
        if not include_archived:
            where = f"{where}~and(archived_at,is,null)"
        rows = self._get("project_files", params={"where": where, "limit": 1}).get("list", [])
        return rows[0] if rows else None

    def get_project_file_version(self, version_id: int) -> dict | None:
        self._require_table("project_file_versions")
        rows = self._get("project_file_versions", params={"where": f"(Id,eq,{version_id})", "limit": 1}).get("list", [])
        return rows[0] if rows else None

    def list_project_file_versions(self, file_id: int, limit: int = 100) -> list[dict]:
        self._require_table("project_file_versions")
        return self._get_paginated(
            "project_file_versions",
            params={"where": f"(file_id,eq,{file_id})", "sort": "-version", "limit": limit},
        )

    def archive_project_file(
        self,
        project_id: int,
        path: str,
        audit_actor: str | None = None,
    ) -> dict:
        row = self.get_project_file(project_id, path, include_archived=True)
        if not row:
            return {"ok": True, "already_absent": True}
        if row.get("archived_at"):
            return row
        archived_at = datetime.now(timezone.utc).isoformat()
        out = self._patch("project_files", int(row["Id"]), {"Id": int(row["Id"]), "archived_at": archived_at})
        if audit_actor:
            self.add_project_audit_event(project_id, audit_actor, "file_archive", {"path": path})
        return out

    def set_project_file_flag(
        self,
        project_id: int,
        path: str,
        field: str,
        value: bool,
        audit_actor: str | None = None,
    ) -> dict:
        if field not in {"pinned", "locked"}:
            raise ValueError("unsupported field")
        row = self.get_project_file(project_id, path)
        if not row:
            raise KeyError("file not found")
        out = self._patch(
            "project_files",
            int(row["Id"]),
            {"Id": int(row["Id"]), field: 1 if value else 0},
        )
        if audit_actor:
            self.add_project_audit_event(project_id, audit_actor, f"file_{field}", {"path": path, field: value})
        return out

    def move_project_file(
        self,
        project_id: int,
        from_path: str,
        to_path: str,
        audit_actor: str | None = None,
    ) -> dict:
        src = self.get_project_file(project_id, from_path)
        if not src:
            raise KeyError("source file not found")
        dst = self.get_project_file(project_id, to_path)
        if dst:
            raise ValueError("destination path already exists")
        out = self._patch("project_files", int(src["Id"]), {"Id": int(src["Id"]), "path": to_path})
        if audit_actor:
            self.add_project_audit_event(project_id, audit_actor, "file_move", {"from": from_path, "to": to_path})
        return out

    def restore_project_file_version(
        self,
        project_id: int,
        path: str,
        version: int,
        audit_actor: str | None = None,
    ) -> tuple[dict, dict, bool]:
        file_row = self.get_project_file(project_id, path, include_archived=True)
        if not file_row:
            raise KeyError("file not found")
        versions = self.list_project_file_versions(int(file_row["Id"]), limit=500)
        target = None
        for row in versions:
            if int(row.get("version") or 0) == int(version):
                target = row
                break
        if not target:
            raise KeyError("version not found")
        return self.write_project_file_version(
            project_id=project_id,
            path=path,
            content=target.get("content") or "",
            edit_summary=f"restore v{version}",
            kind=file_row.get("kind") or "",
            mime=file_row.get("mime") or "",
            created_by="user",
            audit_actor=audit_actor,
            audit_kind="file_restore",
        )

    def write_project_file_version(
        self,
        project_id: int,
        path: str,
        content: str,
        edit_summary: str = "",
        kind: str = "",
        mime: str = "",
        pinned: bool | None = None,
        created_by: str = "user",
        conversation_id: int | None = None,
        created_by_message_id: int | None = None,
        allow_overwrite_locked: bool = False,
        audit_actor: str | None = None,
        audit_kind: str = "file_write",
        if_content_hash: str | None = None,
    ) -> tuple[dict, dict, bool]:
        self._require_table("project_files")
        self._require_table("project_file_versions")

        content = content or ""
        size_bytes = len(content.encode("utf-8"))
        if size_bytes > PROJECT_MAX_FILE_BYTES:
            raise ValueError("file exceeds 100KB limit")
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        file_row = self.get_project_file(project_id, path)
        current_version: dict | None = None
        if file_row and file_row.get("current_version_id"):
            current_version = self.get_project_file_version(int(file_row["current_version_id"]))

        if file_row and bool(file_row.get("locked")) and not allow_overwrite_locked:
            raise PermissionError(f"file is locked: {path}")

        if if_content_hash is not None:
            existing_hash = (current_version or {}).get("content_hash") or ""
            if existing_hash != if_content_hash:
                raise ConflictError(
                    f"if_content_hash mismatch for {path}",
                    expected=if_content_hash,
                    actual=existing_hash,
                )

        if current_version and (current_version.get("content_hash") == content_hash):
            return file_row, current_version, False

        if file_row is None:
            active_count = len(self.list_project_files(project_id=project_id, limit=PROJECT_MAX_FILES + 1))
            if active_count >= PROJECT_MAX_FILES:
                raise ValueError("project exceeds file limit")
            file_row = self._post(
                "project_files",
                {
                    "project_id": project_id,
                    "path": path,
                    "kind": kind or "code",
                    "mime": mime or "text/plain",
                    "size_bytes": size_bytes,
                    "created_by": created_by,
                    "pinned": 1 if pinned else 0,
                },
            )
            parent_version_id = None
            next_version = 1
        else:
            parent_version_id = int(current_version["Id"]) if current_version else None
            next_version = int(current_version.get("version") or 0) + 1 if current_version else 1

        payload: dict = {
            "file_id": int(file_row["Id"]),
            "version": next_version,
            "content": content,
            "content_hash": content_hash,
            "edit_summary": edit_summary,
        }
        if parent_version_id is not None:
            payload["parent_version_id"] = parent_version_id
        if conversation_id is not None:
            payload["conversation_id"] = int(conversation_id)
        if created_by_message_id is not None:
            payload["created_by_message_id"] = int(created_by_message_id)

        new_version = self._post("project_file_versions", payload)

        file_patch: dict = {
            "Id": int(file_row["Id"]),
            "current_version_id": int(new_version["Id"]),
            "size_bytes": size_bytes,
            "created_by": created_by,
            "archived_at": None,
        }
        if kind:
            file_patch["kind"] = kind
        if mime:
            file_patch["mime"] = mime
        if pinned is not None:
            file_patch["pinned"] = 1 if pinned else 0

        file_row = self._patch("project_files", int(file_row["Id"]), file_patch)
        if audit_actor:
            self.add_project_audit_event(
                project_id,
                audit_actor,
                audit_kind,
                {
                    "path": path,
                    "version": new_version.get("version"),
                    "changed": True,
                    "edit_summary": edit_summary,
                },
            )
        return file_row, new_version, True

    # Gitea connections (one per org, encrypted token field handled at write site).
    def get_gitea_connection(self, org_id: int) -> dict | None:
        if "gitea_connections" not in self.tables:
            return None
        rows = self._get(
            "gitea_connections",
            params={"where": f"(org_id,eq,{org_id})", "limit": 1, "sort": "-UpdatedAt"},
        ).get("list", [])
        return rows[0] if rows else None

    def upsert_gitea_connection(
        self,
        org_id: int,
        base_url: str,
        username: str,
        access_token: str,
        default_branch: str = "main",
    ) -> dict:
        self._require_table("gitea_connections")
        existing = self.get_gitea_connection(org_id)
        payload = {
            "org_id": org_id,
            "base_url": base_url,
            "username": username,
            "access_token": access_token,
            "default_branch": default_branch,
        }
        if existing:
            return self._patch("gitea_connections", int(existing["Id"]), {"Id": int(existing["Id"]), **payload})
        return self._post("gitea_connections", payload)

    def delete_gitea_connection(self, org_id: int) -> bool:
        existing = self.get_gitea_connection(org_id)
        if not existing:
            return False
        # NocoDB rest API doesn't expose hard delete here uniformly; soft-clear token.
        self._patch(
            "gitea_connections",
            int(existing["Id"]),
            {"Id": int(existing["Id"]), "access_token": "", "verified_at": None},
        )
        return True

    def mark_gitea_verified(self, org_id: int) -> None:
        existing = self.get_gitea_connection(org_id)
        if not existing:
            return
        self._patch(
            "gitea_connections",
            int(existing["Id"]),
            {"Id": int(existing["Id"]), "verified_at": datetime.now(timezone.utc).isoformat()},
        )

    def mark_version_pushed(self, version_id: int, sha: str) -> None:
        if "project_file_versions" not in self.tables:
            return
        try:
            self._patch(
                "project_file_versions",
                int(version_id),
                {"Id": int(version_id), "pushed_to_sha": sha},
            )
        except Exception:
            _log.debug("mark_version_pushed skipped  v=%s", version_id, exc_info=True)

    def update_project_gitea_state(self, project_id: int, last_synced_sha: str, origin: str | None = None) -> None:
        patch: dict = {
            "Id": project_id,
            "gitea_last_synced_sha": last_synced_sha,
            "gitea_last_synced_at": datetime.now(timezone.utc).isoformat(),
        }
        if origin:
            patch["gitea_origin"] = origin
        try:
            self._patch("projects", project_id, patch)
        except Exception:
            _log.debug("update_project_gitea_state skipped", exc_info=True)

    # Snapshots — frozen named pointers into project_file_versions.
    def list_project_snapshots(self, project_id: int, limit: int = 200) -> list[dict]:
        if "project_snapshots" not in self.tables:
            return []
        return self._get_paginated(
            "project_snapshots",
            params={"where": f"(project_id,eq,{project_id})", "sort": "-CreatedAt", "limit": limit},
        )

    def get_project_snapshot(self, project_id: int, label: str) -> dict | None:
        if "project_snapshots" not in self.tables:
            return None
        rows = self._get(
            "project_snapshots",
            params={"where": f"(project_id,eq,{project_id})~and(label,eq,{label})", "limit": 1},
        ).get("list", [])
        return rows[0] if rows else None

    def list_project_snapshot_files(self, snapshot_id: int) -> list[dict]:
        if "project_snapshot_files" not in self.tables:
            return []
        return self._get_paginated(
            "project_snapshot_files",
            params={"where": f"(snapshot_id,eq,{snapshot_id})", "limit": 5000},
        )

    def create_project_snapshot(
        self,
        project_id: int,
        label: str,
        actor: str,
        description: str = "",
    ) -> dict:
        self._require_table("project_snapshots")
        self._require_table("project_snapshot_files")
        if self.get_project_snapshot(project_id, label):
            raise ValueError(f"snapshot label '{label}' already exists for project")
        snap = self._post(
            "project_snapshots",
            {
                "project_id": project_id,
                "label": label,
                "description": description,
                "created_by": actor,
            },
        )
        snap_id = int(snap["Id"])
        captured = 0
        for file_row in self.list_project_files(project_id=project_id):
            version_id = file_row.get("current_version_id")
            if not version_id:
                continue
            self._post(
                "project_snapshot_files",
                {
                    "snapshot_id": snap_id,
                    "file_id": int(file_row["Id"]),
                    "path": file_row.get("path") or "",
                    "version_id": int(version_id),
                },
            )
            captured += 1
        self.add_project_audit_event(
            project_id,
            actor,
            "snapshot_create",
            {"label": label, "files": captured},
        )
        return {**snap, "file_count": captured}

    # Generic helpers for new optional tables — graceful no-op when missing.
    def _safe_post(self, table: str, payload: dict) -> dict | None:
        if table not in self.tables:
            return None
        return self._post(table, payload)

    def _safe_list(self, table: str, where: str, sort: str = "-CreatedAt", limit: int = 200) -> list[dict]:
        if table not in self.tables:
            return []
        return self._get_paginated(table, params={"where": where, "sort": sort, "limit": limit})

    def _safe_get(self, table: str, where: str) -> dict | None:
        if table not in self.tables:
            return None
        rows = self._get(table, params={"where": where, "limit": 1}).get("list", [])
        return rows[0] if rows else None

    # Lint results
    def add_lint_results(self, project_id: int, file_id: int, version: int, issues: list[dict]) -> int:
        if "project_lint_results" not in self.tables:
            return 0
        # Clear previous issues for this file/version, then insert.
        n = 0
        for issue in issues:
            self._post(
                "project_lint_results",
                {
                    "project_id": project_id,
                    "file_id": file_id,
                    "version": version,
                    "line": issue.get("line"),
                    "col": issue.get("col"),
                    "severity": issue.get("severity") or "info",
                    "rule": issue.get("rule") or "",
                    "message": issue.get("message") or "",
                    "kind": issue.get("kind") or "lint",
                },
            )
            n += 1
        return n

    def list_lint_issues(self, project_id: int, severity: str | None = None, limit: int = 1000) -> list[dict]:
        where = f"(project_id,eq,{project_id})"
        if severity:
            where = f"{where}~and(severity,eq,{severity})"
        return self._safe_list("project_lint_results", where, sort="-CreatedAt", limit=limit)

    # Symbols + dependencies
    def replace_project_symbols(self, project_id: int, file_id: int, symbols: list[dict]) -> int:
        if "project_symbols" not in self.tables:
            return 0
        n = 0
        for s in symbols:
            self._post(
                "project_symbols",
                {
                    "project_id": project_id,
                    "file_id": file_id,
                    "name": s.get("name"),
                    "kind": s.get("kind"),
                    "line": s.get("line"),
                    "signature": s.get("signature") or "",
                    "path": s.get("path") or "",
                },
            )
            n += 1
        return n

    def search_project_symbols(self, project_id: int, q: str, limit: int = 200) -> list[dict]:
        where = f"(project_id,eq,{project_id})~and(name,like,%{q}%)"
        return self._safe_list("project_symbols", where, sort="path", limit=limit)

    def replace_project_dependencies(self, project_id: int, file_id: int, edges: list[dict]) -> int:
        if "project_dependencies" not in self.tables:
            return 0
        n = 0
        for e in edges:
            self._post(
                "project_dependencies",
                {
                    "project_id": project_id,
                    "file_id": file_id,
                    "depends_on": e.get("depends_on"),
                    "edge_type": e.get("edge_type") or "import",
                },
            )
            n += 1
        return n

    def list_project_dependencies(self, project_id: int, limit: int = 5000) -> list[dict]:
        return self._safe_list("project_dependencies", f"(project_id,eq,{project_id})", sort="file_id", limit=limit)

    # Share tokens
    def create_share_token(self, project_id: int, token: str, snapshot_id: int | None, expires_at: str | None) -> dict | None:
        return self._safe_post(
            "project_share_tokens",
            {
                "project_id": project_id,
                "token": token,
                "snapshot_id": snapshot_id,
                "expires_at": expires_at,
            },
        )

    def get_share_token(self, token: str) -> dict | None:
        return self._safe_get("project_share_tokens", f"(token,eq,{token})~and(revoked_at,is,null)")

    def revoke_share_token(self, token: str) -> bool:
        row = self._safe_get("project_share_tokens", f"(token,eq,{token})")
        if not row:
            return False
        self._patch(
            "project_share_tokens",
            int(row["Id"]),
            {"Id": int(row["Id"]), "revoked_at": datetime.now(timezone.utc).isoformat()},
        )
        return True

    # Bookmarks
    def add_bookmark(self, project_id: int, target_kind: str, target_ref: str, label: str = "", color: str = "") -> dict | None:
        return self._safe_post("project_bookmarks", {
            "project_id": project_id, "target_kind": target_kind, "target_ref": target_ref,
            "label": label, "color": color,
        })

    def list_bookmarks(self, project_id: int) -> list[dict]:
        return self._safe_list("project_bookmarks", f"(project_id,eq,{project_id})")

    def remove_bookmark(self, bookmark_id: int) -> bool:
        if "project_bookmarks" not in self.tables:
            return False
        try:
            requests.delete(
                f"{self.url}/{self.tables['project_bookmarks']}/{bookmark_id}",
                headers=self.headers, timeout=10,
            )
            return True
        except Exception:
            return False

    # Saved queries
    def add_saved_query(self, project_id: int, name: str, query: str, kind: str = "search") -> dict | None:
        return self._safe_post("project_saved_queries", {
            "project_id": project_id, "name": name, "query": query, "kind": kind,
        })

    def list_saved_queries(self, project_id: int) -> list[dict]:
        return self._safe_list("project_saved_queries", f"(project_id,eq,{project_id})")

    # Recipes
    def add_recipe(self, project_id: int, name: str, steps: list[dict]) -> dict | None:
        return self._safe_post("project_recipes", {"project_id": project_id, "name": name, "steps": steps})

    def list_recipes(self, project_id: int) -> list[dict]:
        return self._safe_list("project_recipes", f"(project_id,eq,{project_id})")

    # Pinned exchanges (assistant messages)
    def add_pin(self, project_id: int, kind: str, target_ref: str, body: str = "") -> dict | None:
        return self._safe_post("project_pins", {
            "project_id": project_id, "kind": kind, "target_ref": target_ref, "body": body,
        })

    def list_pins(self, project_id: int) -> list[dict]:
        return self._safe_list("project_pins", f"(project_id,eq,{project_id})")

    # Workspaces (multi-project navigational sets)
    def create_workspace(self, org_id: int, name: str, project_ids: list[int]) -> dict | None:
        return self._safe_post("workspaces", {"org_id": org_id, "name": name, "project_ids": project_ids})

    def list_workspaces(self, org_id: int) -> list[dict]:
        return self._safe_list("workspaces", f"(org_id,eq,{org_id})")

    # Pending changes (staged diff queue)
    def add_pending_change(self, project_id: int, file_id: int, version_id: int, conversation_id: int | None) -> dict | None:
        return self._safe_post("project_pending_changes", {
            "project_id": project_id, "file_id": file_id, "version_id": version_id,
            "conversation_id": conversation_id, "status": "pending",
        })

    def list_pending_changes(self, project_id: int) -> list[dict]:
        return self._safe_list("project_pending_changes", f"(project_id,eq,{project_id})~and(status,eq,pending)")

    def resolve_pending_change(self, pending_id: int, status: str) -> dict | None:
        if "project_pending_changes" not in self.tables:
            return None
        return self._patch("project_pending_changes", pending_id, {
            "Id": pending_id, "status": status, "resolved_at": datetime.now(timezone.utc).isoformat(),
        })

    # Playbooks
    def create_playbook(self, project_id: int, goal: str, steps: list[dict]) -> dict | None:
        return self._safe_post("project_playbooks", {
            "project_id": project_id, "goal": goal, "steps": steps,
            "current_step": 0, "status": "active",
        })

    def list_playbooks(self, project_id: int) -> list[dict]:
        return self._safe_list("project_playbooks", f"(project_id,eq,{project_id})")

    def update_playbook(self, playbook_id: int, data: dict) -> dict | None:
        if "project_playbooks" not in self.tables:
            return None
        return self._patch("project_playbooks", playbook_id, {"Id": playbook_id, **data})

    # Reviews (AI code review records)
    def create_review(self, project_id: int, payload: dict) -> dict | None:
        return self._safe_post("project_reviews", {"project_id": project_id, **payload})

    def list_reviews(self, project_id: int) -> list[dict]:
        return self._safe_list("project_reviews", f"(project_id,eq,{project_id})")

    # File comments
    def add_file_comment(self, project_id: int, file_id: int, version: int, anchor: dict, body: str, author: str) -> dict | None:
        return self._safe_post("project_file_comments", {
            "project_id": project_id, "file_id": file_id, "version": version,
            "anchor": anchor, "body": body, "author": author,
        })

    def list_file_comments(self, project_id: int, file_id: int) -> list[dict]:
        return self._safe_list("project_file_comments", f"(project_id,eq,{project_id})~and(file_id,eq,{file_id})")


