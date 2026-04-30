import unittest
import sys
import types
from unittest.mock import patch

# Keep tests independent from optional runtime deps.
if "falkordb" not in sys.modules:
    class _FakeFalkorDBClient:
        def __init__(self, *args, **kwargs):
            pass

        def select_graph(self, name):
            return None

    fake_falkordb = types.ModuleType("falkordb")
    fake_falkordb.FalkorDB = _FakeFalkorDBClient
    sys.modules["falkordb"] = fake_falkordb

from infra import graph
from shared import graph_recall


class _FakeResult:
    def __init__(self, result_set):
        self.result_set = result_set


class _FakeGraph:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def query(self, q, params=None, timeout=None):
        self.calls.append({"query": q, "params": params or {}, "timeout": timeout})
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class GraphNeighbourhoodTests(unittest.TestCase):
    def test_max_hops_1_uses_seed_anchored_query_with_timeout(self):
        fake = _FakeGraph([
            _FakeResult([
                ["Alice", "Person", "KNOWS", "Bob", "Person", 3, 1.2, None, []],
            ])
        ])

        with patch("infra.graph.get_graph", return_value=fake):
            edges = graph.get_weighted_neighbourhood(
                org_id=1,
                seed_names=["Alice"],
                max_hops=1,
                edge_limit=30,
                timeout_ms=900,
            )

        self.assertEqual(len(edges), 1)
        self.assertEqual(fake.calls[0]["timeout"], 900)
        self.assertIn("MATCH (seed)-[edge]-(nbr)", fake.calls[0]["query"])

    def test_timeout_retries_with_smaller_seed_and_limit(self):
        fake = _FakeGraph([
            Exception("Query timed out"),
            _FakeResult([
                ["A", "Concept", "REL", "B", "Concept", 1, 1.0, None, []],
            ]),
        ])

        with patch("infra.graph.get_graph", return_value=fake):
            edges = graph.get_weighted_neighbourhood(
                org_id=1,
                seed_names=["one", "two", "three", "four", "five"],
                max_hops=1,
                edge_limit=80,
                timeout_ms=500,
            )

        self.assertEqual(len(edges), 1)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[1]["params"]["limit"], 20)
        self.assertLessEqual(len(fake.calls[1]["params"]["seeds"]), 3)


class GraphRecallSeedFilterTests(unittest.TestCase):
    def test_filter_graph_seeds_drops_low_signal_tokens(self):
        seeds = graph_recall._filter_graph_seeds([
            "any", "developers", "AuthService", "AI", "AuthService", "ML Platform"
        ])
        self.assertEqual(seeds, ["AuthService", "ML Platform"])


if __name__ == "__main__":
    unittest.main()



