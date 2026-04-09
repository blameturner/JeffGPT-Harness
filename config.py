import os
import re
import time
import subprocess
import requests
from dotenv import load_dotenv

load_dotenv()

# How long to wait for model containers to become reachable at startup.
MODEL_DISCOVERY_TIMEOUT_S = int(os.getenv("MODEL_DISCOVERY_TIMEOUT_S", "60"))
MODEL_DISCOVERY_INTERVAL_S = 2


def _get_host() -> str:
    """Fallback: discover the Docker host gateway for port-scanning mode."""
    host = os.getenv("MODEL_HOST")
    if host:
        return host

    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        gateway = result.stdout.split("via ")[1].split()[0]
        print(f"[config] Discovered Docker gateway: {gateway}")
        return gateway
    except (subprocess.SubprocessError, IndexError, OSError):
        print("[config] Could not discover gateway, falling back to 0.0.0.0")
        return "0.0.0.0"


def _clean_model_id(model_id: str) -> str:
    name = model_id.replace(".gguf", "")
    name = re.sub(r"-Q\d+.*$", "", name, flags=re.IGNORECASE)
    return name


def _query_model_id(url: str) -> str | None:
    """Single attempt: return cleaned model ID from the server, or None."""
    try:
        response = requests.get(f"{url}/v1/models", timeout=3)
        response.raise_for_status()
        data = response.json()
        if data.get("data"):
            return _clean_model_id(data["data"][0]["id"])
    except (requests.RequestException, ValueError, KeyError):
        pass
    return None


def _wait_for_model(url: str, label: str) -> str | None:
    """Poll a model host until it responds or the discovery timeout elapses."""
    deadline = time.time() + MODEL_DISCOVERY_TIMEOUT_S
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        model_id = _query_model_id(url)
        if model_id:
            print(
                f"[config] OK  {label}: {model_id} @ {url} "
                f"(after {attempt} attempt{'s' if attempt > 1 else ''})"
            )
            return model_id
        time.sleep(MODEL_DISCOVERY_INTERVAL_S)
    print(f"[config] FAIL {label}: no response from {url} after {MODEL_DISCOVERY_TIMEOUT_S}s")
    return None


def _collect_role_env_vars() -> list[tuple[str, str]]:
    """Return [(role, url)] for every MODEL_<ROLE>_URL env var that is set."""
    pattern = re.compile(r"^MODEL_(.+)_URL$")
    pairs: list[tuple[str, str]] = []
    for key, value in os.environ.items():
        match = pattern.match(key)
        if not match or not value.strip():
            continue
        role = match.group(1).lower()
        # Skip the embedder/reranker — they are not chat models.
        if role in {"embedder", "reranker", "host"}:
            continue
        pairs.append((role, value.strip()))
    return pairs


def _register(catalog: dict, entry: dict) -> None:
    """Register a catalog entry under its role key and model_id alias."""
    catalog[entry["role"]] = entry
    model_id = entry.get("model_id")
    if model_id and model_id not in catalog:
        catalog[model_id] = entry


def discover_models() -> dict:
    """Build the model catalog.

    Preferred: explicit per-role env vars (MODEL_<ROLE>_URL).
    Fallback 1: MODEL_HOSTS comma-separated list (role inferred from hostname).
    Fallback 2: port-scan the Docker gateway (dev mode).
    """
    catalog: dict = {}

    # --- Preferred: per-role env vars ---
    role_pairs = _collect_role_env_vars()
    if role_pairs:
        print(f"[config] Found {len(role_pairs)} MODEL_<ROLE>_URL env vars")
        for role, url in role_pairs:
            model_id = _wait_for_model(url, f"role={role}")
            if not model_id:
                continue
            _register(catalog, {"role": role, "url": url, "model_id": model_id})
        return catalog

    # --- Fallback: MODEL_HOSTS ---
    hosts_env = os.getenv("MODEL_HOSTS", "").strip()
    if hosts_env:
        print("[config] Using MODEL_HOSTS fallback")
        for url in hosts_env.split(","):
            url = url.strip()
            if not url:
                continue
            role = _infer_role_from_url(url)
            model_id = _wait_for_model(url, f"role={role}")
            if not model_id:
                continue
            _register(catalog, {"role": role, "url": url, "model_id": model_id})
        return catalog

    # --- Fallback: port scan (dev mode on host) ---
    print("[config] Using port-scan discovery fallback")
    host = _get_host()
    port_start = int(os.getenv("MODEL_PORT_START", "8080"))
    port_end = int(os.getenv("MODEL_PORT_END", "8090"))
    exclude_ports = {
        int(os.getenv("EMBEDDER_PORT", "8083")),
        int(os.getenv("RERANKER_PORT", "8084")),
    }
    for port in range(port_start, port_end + 1):
        if port in exclude_ports:
            continue
        url = f"http://{host}:{port}"
        model_id = _query_model_id(url)
        if not model_id:
            continue
        role = model_id
        _register(catalog, {"role": role, "url": url, "model_id": model_id})
    return catalog


def _infer_role_from_url(url: str) -> str:
    """Best-effort role from a hostname like mst-ag-reasoner-gemma4-e4b."""
    host = url.split("://")[-1].split("/")[0].split(":")[0]
    parts = host.split("-")
    if len(parts) >= 3 and parts[0] == "mst" and parts[1] == "ag":
        return parts[2]
    return host


# Populate at import; harness re-checks lazily if this ends up empty.
MODELS: dict = discover_models()
print(
    f"[config] Model catalog ready: "
    f"{sorted({v['role'] for v in MODELS.values() if isinstance(v, dict)})}"
)


def refresh_models() -> dict:
    """Lazy re-discovery — called by the /models endpoint if the catalog is empty."""
    global MODELS
    MODELS = discover_models()
    return MODELS


def get_model_url(key: str) -> str | None:
    """Resolve a role OR model_id to a URL. Used by agents."""
    entry = MODELS.get(key) or MODELS.get(key.lower())
    if isinstance(entry, dict):
        return entry.get("url")
    return None


EMBEDDER_URL = os.getenv("EMBEDDER_URL") or f"http://{_get_host()}:{os.getenv('EMBEDDER_PORT', '8083')}"
RERANKER_URL = os.getenv("RERANKER_URL") or f"http://{_get_host()}:{os.getenv('RERANKER_PORT', '8084')}"

CHROMA_URL = os.getenv("CHROMA_URL")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://mst-ag-searxng:8080")
FALKORDB_HOST = os.getenv("FALKORDB_HOST")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))

NOCODB_URL = os.getenv("NOCODB_URL")
NOCODB_TOKEN = os.getenv("NOCODB_TOKEN")
NOCODB_BASE_ID = os.getenv("NOCODB_BASE_ID")

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


ENRICHMENT_TOKEN_BUDGET = int(os.getenv("ENRICHMENT_TOKEN_BUDGET", "50000"))
ENRICHMENT_LOG_RETENTION_DAYS = int(os.getenv("ENRICHMENT_LOG_RETENTION_DAYS", "30"))
MAX_SUMMARY_INPUT_CHARS = 15000
PROACTIVE_BUDGET_THRESHOLD = 5000
CATEGORY_COLLECTIONS = {
    "documentation": "scraped_documentation",
    "news": "scraped_news",
    "competitive": "scraped_competitive",
    "regulatory": "scraped_regulatory",
    "research": "scraped_research",
    "security": "scraped_security",
    "model_releases": "scraped_model_releases",
}


def scoped_collection(org_id: int, collection_name: str) -> str:
    return f"org_{org_id}_{collection_name}"


def scoped_graph(org_id: int) -> str:
    return f"org_{org_id}_mst_ag"
