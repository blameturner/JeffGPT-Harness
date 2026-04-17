import sys
import types
import threading
import time
from unittest.mock import patch

if "chromadb" not in sys.modules:
    sys.modules["chromadb"] = types.ModuleType("chromadb")

if "pydantic" not in sys.modules:
    pydantic = types.ModuleType("pydantic")

    class _BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    def _field(*args, **kwargs):
        return kwargs.get("default_factory", lambda: None)() if "default_factory" in kwargs else kwargs.get("default")

    pydantic.BaseModel = _BaseModel
    pydantic.Field = _field
    sys.modules["pydantic"] = pydantic

from shared.model_pool import _PrioritySemaphore
from tools.planned_search.agent import run_planned_search_scrape_job

sem = _PrioritySemaphore(0)
results: list[str] = []


def normal_waiter() -> None:
    sem.acquire(blocking=True, priority=False)
    results.append("normal")


def priority_waiter() -> None:
    sem.acquire(blocking=True, priority=True)
    results.append("priority")


t1 = threading.Thread(target=normal_waiter)
t2 = threading.Thread(target=priority_waiter)
t1.start()
time.sleep(0.05)
t2.start()
time.sleep(0.05)
sem.release()
time.sleep(0.05)
sem.release()
t1.join(timeout=1)
t2.join(timeout=1)
assert results == ["priority", "normal"], results
print("priority semaphore ordering OK")

with patch("tools.planned_search.agent.PathfinderScraper") as MockScraper:
    MockScraper.return_value.scrape.return_value = {
        "status": "ok",
        "canonical": "https://example.com/final",
        "text": "useful content " * 80,
    }
    res = run_planned_search_scrape_job(
        {
            "url": "https://example.com",
            "title": "Example",
            "org_id": 123,
            "query_keywords": ["useful", "content"],
        }
    )
    assert res["status"] == "ok", res
    assert res["org_id"] == 123, res
    assert res["url"] == "https://example.com/final", res
print("planned_search scrape worker OK")


