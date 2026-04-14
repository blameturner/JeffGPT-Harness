import time
import logging
import requests
from infra.config import NOCODB_URL, NOCODB_TOKEN, NOCODB_BASE_ID

_log = logging.getLogger("nocodb")


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

    def list_agents(self, org_id: int, limit: int = 200) -> list[dict]:
        data = self._get("agents", params={
            "where": f"(org_id,eq,{org_id})~and(deleted_at,is,null)",
            "limit": limit,
        })
        rows = data.get("list", [])
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

    def get_conversation(self, conversation_id: int) -> dict | None:
        data = self._get("conversations", params={
            "where": f"(Id,eq,{conversation_id})",
            "limit": 1
        })
        records = data.get("list", [])
        return records[0] if records else None

    def update_conversation(self, conversation_id: int, data: dict) -> dict:
        return self._patch("conversations", conversation_id, {"Id": conversation_id, **data})

    def list_conversations(self, org_id: int, limit: int = 50) -> list[dict]:
        data = self._get("conversations", params={
            "where": f"(org_id,eq,{org_id})",
            "sort": "-CreatedAt",
            "limit": limit,
        })
        return data.get("list", [])

    def list_messages(self, conversation_id: int, limit: int = 500) -> list[dict]:
        data = self._get("messages", params={
            "where": f"(conversation_id,eq,{conversation_id})",
            "sort": "CreatedAt",
            "limit": limit,
        })
        return data.get("list", [])

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

    def list_message_search_sources(self, message_id: int | None = None, conversation_id: int | None = None) -> list[dict]:
        parts = []
        if message_id is not None:
            parts.append(f"(message_id,eq,{message_id})")
        if conversation_id is not None:
            parts.append(f"(conversation_id,eq,{conversation_id})")
        params: dict = {"sort": "source_index", "limit": 500}
        if parts:
            params["where"] = "~and".join(parts)
        return self._get("message_search_sources", params=params).get("list", [])

    def create_code_conversation(
        self,
        org_id: int,
        model: str,
        title: str = "",
        mode: str = "plan",
        knowledge_enabled: bool = False,
    ) -> dict:
        _log.info("create_code_conversation  org=%d model=%s mode=%s knowledge=%s title=%s", org_id, model, mode, knowledge_enabled, title[:40])
        return self._post("code_conversations", {
            "org_id": org_id,
            "model": model,
            "title": title or "New code session",
            "rag_enabled": 0,
            "rag_collection": mode,
            "knowledge_enabled": 1 if knowledge_enabled else 0,
        })

    def get_code_conversation(self, conversation_id: int) -> dict | None:
        data = self._get("code_conversations", params={
            "where": f"(Id,eq,{conversation_id})",
            "limit": 1,
        })
        records = data.get("list", [])
        return records[0] if records else None

    def update_code_conversation(self, conversation_id: int, data: dict) -> dict:
        return self._patch("code_conversations", conversation_id, {"Id": conversation_id, **data})

    def list_code_conversations(self, org_id: int, limit: int = 50) -> list[dict]:
        data = self._get("code_conversations", params={
            "where": f"(org_id,eq,{org_id})",
            "sort": "-CreatedAt",
            "limit": limit,
        })
        return data.get("list", [])

    def list_code_messages(self, conversation_id: int, limit: int = 500) -> list[dict]:
        data = self._get("code_messages", params={
            "where": f"(conversation_id,eq,{conversation_id})",
            "sort": "CreatedAt",
            "limit": limit,
        })
        return data.get("list", [])

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

    def _list_by_conversation(self, table: str, conversation_id: int, limit: int = 200) -> list[dict]:
        try:
            data = self._get(table, params={
                "where": f"(conversation_id,eq,{conversation_id})",
                "limit": limit,
            })
            return data.get("list", [])
        except (requests.HTTPError, KeyError):
            _log.debug("_list_by_conversation  table=%s conv=%d returned empty (table may lack column)", table, conversation_id)
            return []

    def list_runs_for_conversation(self, conversation_id: int, limit: int = 200) -> list[dict]:
        return self._list_by_conversation("agent_runs", conversation_id, limit)

    def list_outputs_for_conversation(self, conversation_id: int, limit: int = 200) -> list[dict]:
        return self._list_by_conversation("agent_outputs", conversation_id, limit)

    def list_tasks_for_conversation(self, conversation_id: int, limit: int = 200) -> list[dict]:
        return self._list_by_conversation("tasks", conversation_id, limit)

    def list_observations_for_conversation(self, conversation_id: int, limit: int = 200) -> list[dict]:
        try:
            data = self._get("observations", params={
                "where": f"(conversation_id,eq,{conversation_id})",
                "limit": limit,
            })
            return data.get("list", [])
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
