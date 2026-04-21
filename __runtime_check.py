import sys
import types
import threading
import time

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
