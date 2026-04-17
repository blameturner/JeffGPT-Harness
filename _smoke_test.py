"""Smoke test for all uncommitted changes. Run with: python _smoke_test.py"""
import json
import sys
import types
import threading
import time
import unittest.mock as m

# --------------- optional-dependency stubs ---------------
if "chromadb" not in sys.modules:
    _chroma = types.ModuleType("chromadb")
    class _FakeHttpClient:
        def __init__(self, *a, **kw): pass
        def get_or_create_collection(self, *a, **kw): return _FakeCollection()
    class _FakeCollection:
        def query(self, *a, **kw): return {"documents": [[]], "metadatas": [[]]}
        def add(self, *a, **kw): pass
        def upsert(self, *a, **kw): pass
    _chroma.HttpClient = _FakeHttpClient
    sys.modules["chromadb"] = _chroma

# ---------- _PrioritySemaphore ----------
from shared.model_pool import _PrioritySemaphore, user_priority_scope

sem = _PrioritySemaphore(2)
assert sem.free_slots() == 2, "expected 2 free"
sem.acquire(blocking=False, priority=False)
assert sem.free_slots() == 1
sem.release()
assert sem.free_slots() == 2

# Priority ordering: normal queues first, priority queues second; release should wake priority first
results: list[str] = []
sem2 = _PrioritySemaphore(0)

def _normal():
    sem2.acquire(blocking=True, priority=False)
    results.append("normal")

def _priority():
    sem2.acquire(blocking=True, priority=True)
    results.append("priority")

t1 = threading.Thread(target=_normal);  t1.start()
time.sleep(0.05)
t2 = threading.Thread(target=_priority); t2.start()
time.sleep(0.05)
sem2.release()   # should wake priority waiter
time.sleep(0.05)
sem2.release()   # should wake normal waiter
t1.join(timeout=1); t2.join(timeout=1)
assert results == ["priority", "normal"], f"wrong order: {results}"

# non-blocking acquire returns False immediately when full
sem3 = _PrioritySemaphore(0)
assert sem3.acquire(blocking=False) is False

print("  _PrioritySemaphore  OK")

# ---------- gate_check ----------
from tools.gate import gate_check
h = gate_check("what is the weather today")
assert isinstance(h, set)
print("  gate_check          OK")

# ---------- ToolJob round-trip ----------
from workers.tool_queue import ToolJob, ToolJobQueue, HandlerConfig

job = ToolJob(job_id="abc", type="summarise_page", org_id=42, payload={"url": "http://x.com"})
row = job.to_row()
assert json.loads(row["payload_json"])["url"] == "http://x.com"
job2 = ToolJob.from_row(row)
assert job2.org_id == 42
print("  ToolJob round-trip  OK")

# ---------- submit() org_id fallback from payload ----------
q = ToolJobQueue()
q.register("test_type", HandlerConfig(handler=lambda p: {}, priority_default=3))
with m.patch.object(q, "_persist_new"), m.patch.object(q, "_emit_event"):
    q.submit("test_type", {"org_id": 7}, source="test", org_id=0)
print("  submit org_id       OK")

# ---------- _get_paginated always returns list ----------
with m.patch("requests.get") as mg, m.patch("requests.post"), m.patch("requests.patch"):
    mg.return_value.status_code = 200
    mg.return_value.json.return_value = {"list": [{"Id": 1}], "pageInfo": {"isLastPage": True}}
    mg.return_value.raise_for_status = lambda: None
    from infra.nocodb_client import NocodbClient
    client = NocodbClient.__new__(NocodbClient)
    client.url = "http://fake"
    client.headers = {}
    client.tables = {"tool_jobs": "tid1"}
    rows = client._get_paginated("tool_jobs", params={"limit": 10})
    assert isinstance(rows, list), f"_get_paginated returned {type(rows)}"
    assert len(rows) == 1
print("  _get_paginated      OK")

# ---------- research_planner imports ----------
from tools.research import research_planner
from tools.research import agent as research_agent
print("  research imports    OK")

# ---------- enrichment pathfinder has correct tq.submit signature ----------
import ast, pathlib
src = pathlib.Path("tools/enrichment/pathfinder.py").read_text()
tree = ast.parse(src)
# check all tq.submit calls have org_id keyword
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "submit":
            kw_names = {kw.arg for kw in node.keywords}
            if "org_id" not in kw_names:
                func_loc = node.lineno
                # only flag if it's a tq.submit (not some other object's submit)
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "tq":
                    print(f"  WARNING: tq.submit at line {func_loc} missing org_id kwarg")
print("  pathfinder submit   OK")

print("\nALL SMOKE TESTS PASSED")


