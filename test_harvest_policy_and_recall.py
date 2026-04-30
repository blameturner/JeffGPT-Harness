import unittest
from datetime import datetime, timedelta, timezone

import requests

from shared.pa import recall
from tools.harvest import policy as harvest_policy
from tools.research_seeder import agent as research_seeder


class _RecallClient:
    def __init__(self, now: datetime):
        self.now = now
        self.calls: list[tuple[str, dict]] = []

    def _get_paginated(self, table: str, params: dict, page_size: int = 50):
        self.calls.append((table, dict(params)))
        where = params.get("where", "")
        if table == "messages" and "CreatedAt,gt" in where:
            response = requests.Response()
            response.status_code = 422
            response.url = "http://nocodb.invalid/messages"
            raise requests.HTTPError("422 Client Error", response=response)
        if table == "messages":
            return [
                {
                    "conversation_id": 10,
                    "role": "assistant",
                    "content": "Latest useful reply",
                    "CreatedAt": (self.now - timedelta(hours=2)).isoformat(),
                },
                {
                    "conversation_id": 10,
                    "role": "user",
                    "content": "Recent question",
                    "CreatedAt": (self.now - timedelta(hours=4)).isoformat(),
                },
                {
                    "conversation_id": 10,
                    "role": "user",
                    "content": "Too old to keep",
                    "CreatedAt": (self.now - timedelta(hours=60)).isoformat(),
                },
            ]
        if table == "conversations":
            return [{"Id": 10, "kind": "project", "title": "Alpha"}]
        return []


class RecallFallbackTests(unittest.TestCase):
    def test_build_tails_falls_back_to_python_cutoff_after_nocodb_422(self):
        now = datetime(2026, 4, 30, 5, 25, 54, tzinfo=timezone.utc)
        client = _RecallClient(now)

        tails, thread = recall._build_tails(client, org_id=1, now=now)

        self.assertEqual(len(tails), 1)
        self.assertIsNotNone(thread)
        self.assertEqual(tails[0].conversation_id, 10)
        self.assertEqual([m["content"] for m in tails[0].messages], ["Recent question", "Latest useful reply"])
        self.assertTrue(any("CreatedAt,gt" in (params.get("where") or "") for table, params in client.calls if table == "messages"))
        self.assertTrue(any((params.get("where") or "") == "(org_id,eq,1)" for table, params in client.calls if table == "messages"))


class HarvestPolicyBootstrapTests(unittest.TestCase):
    def test_list_policies_bootstraps_registry_without_startup_side_effects(self):
        original = dict(harvest_policy.REGISTRY)
        try:
            harvest_policy.REGISTRY.clear()
            names = [p["name"] for p in harvest_policy.list_policies()]
            self.assertIn("topic_seeder", names)
            self.assertIn("single_url", names)
            self.assertIsNotNone(harvest_policy.get_policy("topic_seeder"))
        finally:
            harvest_policy.REGISTRY.clear()
            harvest_policy.REGISTRY.update(original)


class _ResearchSeederClient:
    def __init__(self, now: datetime):
        self.now = now
        self.tables = {research_seeder.NOCODB_TABLE_PA_OPEN_LOOPS: 1}
        self.calls: list[tuple[str, dict]] = []

    def _get_paginated(self, table: str, params: dict, page_size: int = 50):
        self.calls.append((table, dict(params)))
        where = params.get("where", "")
        if "CreatedAt,gt" in where:
            response = requests.Response()
            response.status_code = 422
            response.url = "http://nocodb.invalid/open_loops"
            raise requests.HTTPError("422 Client Error", response=response)
        return [
            {
                "text": "Decide CRM vendor",
                "intent": research_seeder.LOOP_INTENT_DECISION,
                "status": research_seeder.LOOP_STATUS_OPEN,
                "CreatedAt": (self.now - timedelta(hours=3)).isoformat(),
            },
            {
                "text": "Too old",
                "intent": research_seeder.LOOP_INTENT_DECISION,
                "status": research_seeder.LOOP_STATUS_OPEN,
                "CreatedAt": (self.now - timedelta(hours=60)).isoformat(),
            },
        ]


class ResearchSeederFallbackTests(unittest.TestCase):
    def test_candidate_decisions_falls_back_to_python_cutoff_after_422(self):
        now = datetime(2026, 4, 30, 5, 25, 54, tzinfo=timezone.utc)
        client = _ResearchSeederClient(now)

        decisions = research_seeder._candidate_decisions(client, org_id=1, now=now)

        self.assertEqual(decisions, ["Decide CRM vendor"])
        self.assertTrue(any("CreatedAt,gt" in (params.get("where") or "") for _, params in client.calls))
        self.assertTrue(any("CreatedAt,gt" not in (params.get("where") or "") for _, params in client.calls))


if __name__ == "__main__":
    unittest.main()


