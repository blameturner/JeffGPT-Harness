import requests

from config import NOCODB_URL, NOCODB_TOKEN, NOCODB_BASE_ID, TABLE_AGENTS, TABLE_AGENT_RUNS, TABLE_AGENT_OUTPUTS, TABLE_OBSERVATIONS

class nocodbClient:
    def __init__(self):
        self.url = f"{NOCODB_URL}/api/v1/db/data/noco/{NOCODB_BASE_ID}"
        self.headers = {
            "xc-token": NOCODB_TOKEN,
            "Content-Type": "application/json"
        }

    def _get(self, table: str, params: dict = None) -> dict:
        response = requests.get(
            f"{self.url}/{table}",
            headers=self.headers,
            params=params,
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    def _post(self, table: str, data: dict) -> dict:
        response = requests.post(
            f"{self.url}/{table}",
            headers=self.headers,
            json=data,
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    def _patch(self, table: str, row_id: int, data: dict ) -> dict:
        response = requests.patch(
            f"{self.url}/{table}/{row_id}",
            headers=self.headers,
            json=data,
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    def get_agent(self, name: str, org_id: int) -> dict | None:
        data = self._get(TABLE_AGENTS, params={
            "where": f"(name,eq,{name})~and(org_id,eq,{org_id})~and(deleted_at,is,null)",
            "limit": 1
        })

        records = data.get("list", [])
        if not records:
            return None

        return records[0]

    def create_run(self, agent: dict, org_id: int, task_description: str, product: str) -> dict:
        return self._post(TABLE_AGENT_RUNS, {
            "agent_id": agent["Id"],
            "agent_name": agent["name"],
            "agent_version": agent.get("version", 1),
            "org_id": org_id,
            "project_id": agent.get("project_id"),
            "model_name": agent["model"],
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
        return self._patch(TABLE_AGENT_RUNS, run_id, {
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
        return self._patch(TABLE_AGENT_RUNS, run_id, {
            "status": "failed",
            "error_message": error_message
        })


    def save_output(self, run: dict, full_text: str, chroma_ids: str) -> dict:

        return self._post(TABLE_AGENT_OUTPUTS, {
            "run_id": run["Id"],
            "agent_id": run["agent_id"],
            "agent_name": run["agent_name"],
            "org_id": run["org_id"],
            "project_id": run.get("project_id"),
            "full_text": full_text,
            "chroma_ids": chroma_ids
        })

    def save_observation(self, run: dict, title: str, content: str,
                         obs_type: str, domain: str,
                         confidence: str = "medium") -> dict:
        return self._post(TABLE_OBSERVATIONS, {
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




