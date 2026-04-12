"""Test what llama.cpp returns with and without thinking params."""
import httpx, time, json

URL = "http://mst-ag-t3-tool:8085/v1/chat/completions"
BASE = {"model": "x", "messages": [{"role": "user", "content": "Say hi"}], "max_tokens": 5, "temperature": 0}

for label, extra in [
    ("no param", {}),
    ("enable_thinking=false", {"enable_thinking": False}),
    ("think=false", {"think": False}),
]:
    payload = {**BASE, **extra}
    t = time.time()
    try:
        r = httpx.post(URL, json=payload, timeout=120)
        elapsed = round(time.time() - t, 1)
        d = r.json()
        msg = d["choices"][0]["message"]
        print(f"\n=== {label} === {elapsed}s status={r.status_code}")
        print(f"  keys: {list(msg.keys())}")
        print(f"  content: {repr(msg.get('content', '')[:200])}")
        if msg.get("reasoning_content"):
            print(f"  reasoning: {repr(msg['reasoning_content'][:100])}")
    except Exception as e:
        print(f"\n=== {label} === FAILED: {e}")
