import time
import requests
from config import NOCODB_URL, NOCODB_TOKEN, NOCODB_BASE_ID


class NocodbClient:
    def __init__(self):
        self.url = f"{NOCODB_URL}/api/v1/db/data/noco/{NOCODB_BASE_ID}"
        self.headers = {
            "xc-token": NOCODB_TOKEN,
            "Content-Type": "application/json"
        }
        self.tables = self._load_tables()

# runs at startup to get all tableIDs for db functions
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
                return {table["title"]: table["id"] for table in tables}
            except Exception:
                print(f"Nocodb not ready, retrying... ({attempt + 1}/15)")
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
        response = requests.post(
            f"{self.url}/{self.tables[table]}",
            headers=self.headers,
            json=data,
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    def _patch(self, table: str, row_id: int, data: dict) -> dict:
        response = requests.patch(
            f"{self.url}/{self.tables[table]}/{row_id}",
            headers=self.headers,
            json=data,
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    def list_agents(self, org_id: int, limit: int = 200) -> list[dict]:
        data = self._get("agents", params={
            "where": f"(org_id,eq,{org_id})~and(deleted_at,is,null)",
            "limit": limit,
        })
        return data.get("list", [])

    def get_agent(self, name: str, org_id: int) -> dict | None:
        data = self._get("agents", params={
            "where": f"(name,eq,{name})~and(org_id,eq,{org_id})~and(deleted_at,is,null)",
            "limit": 1
        })
        records = data.get("list", [])
        if not records:
            return None
        return records[0]

    def create_run(self, agent: dict, org_id: int, task_description: str, product: str) -> dict:
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
        return self._patch("agent_runs", run_id, {
            "status": "failed",
            "error_message": error_message
        })

    def save_output(self, run: dict, full_text: str, chroma_ids: str) -> dict:
        return self._post("agent_outputs", {
            "run_id": run["Id"],
            "agent_id": run["agent_id"],
            "agent_name": run["agent_name"],
            "org_id": run["org_id"],
            "project_id": run.get("project_id"),
            "full_text": full_text,
            "chroma_ids": chroma_ids
        })

    def create_conversation(self, org_id: int, model: str, title: str = "") -> dict:
        return self._post("conversations", {
            "org_id": org_id,
            "model": model,
            "title": title or "New chat",
        })

    def get_conversation(self, conversation_id: int) -> dict | None:
        data = self._get("conversations", params={
            "where": f"(Id,eq,{conversation_id})",
            "limit": 1
        })
        records = data.get("list", [])
        return records[0] if records else None

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
    ) -> dict:
        return self._post("messages", {
            "conversation_id": conversation_id,
            "org_id": org_id,
            "role": role,
            "content": content,
            "model": model,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
        })

    def save_observation(
        self,
        run: dict,
        title: str,
        content: str,
        obs_type: str,
        domain: str,
        confidence: str = "medium"
    ) -> dict:
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