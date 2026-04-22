import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

_log = logging.getLogger("config")

load_dotenv()


def _collect_nested_models(features: dict, models: dict) -> None:
    """Walk feature sections; any sub-dict with a 'models' child contributes those
    entries to the flat models registry. Lets each domain co-locate its model defs
    with its feature flags, while keeping get_function_config(name) flat."""
    for section_name, section in features.items():
        if not isinstance(section, dict):
            continue
        nested = section.get("models")
        if isinstance(nested, dict):
            for model_name, model_cfg in nested.items():
                if model_name in models:
                    _log.warning(
                        "model %r defined in both top-level models and features.%s.models — features wins",
                        model_name, section_name,
                    )
                models[model_name] = model_cfg


def load_platform_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.json"
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        _log.info("platform config loaded from %s", config_path)
    except FileNotFoundError:
        _log.error("config.json not found at %s — cannot start without it", config_path)
        raise SystemExit(1)
    except Exception as e:
        _log.error("failed to parse config.json: %s", e)
        raise SystemExit(1)

    models = dict(cfg.get("models", {}))
    features = cfg.get("features", {})

    _collect_nested_models(features, models)

    if not models:
        _log.error("config.json has no models defined (top-level or nested)")
        raise SystemExit(1)

    return {"models": models, "features": features}


PLATFORM: dict = load_platform_config()


def get_function_config(function_name: str) -> dict:
    cfg = PLATFORM.get("models", {}).get(function_name)
    if cfg:
        return cfg
    raise KeyError(
        f"function {function_name!r} not defined in config.json — "
        "add it to the 'models' section"
    )


def is_feature_enabled(name: str) -> bool:
    features = PLATFORM.get("features", {})
    if name not in features:
        _log.warning("feature %r not in config.json, defaulting to True", name)
        return True
    val = features[name]
    if isinstance(val, dict):
        return bool(val.get("enabled", True))
    return bool(val)


def get_feature_value(name: str, default: Any = None) -> Any:
    val = PLATFORM.get("features", {}).get(name, default)
    if isinstance(val, dict):
        return val.get("enabled", default)
    return val


def get_feature(section: str, key: str, default: Any = None) -> Any:
    """Read a nested feature value, e.g. get_feature('research', 'max_queries', 20)."""
    sec = PLATFORM.get("features", {}).get(section)
    if not isinstance(sec, dict):
        return default
    return sec.get(key, default)

MODEL_DISCOVERY_TIMEOUT_S = int(os.getenv("MODEL_DISCOVERY_TIMEOUT_S", "60"))
MODEL_DISCOVERY_INTERVAL_S = 2


def _get_host() -> str:
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
        _log.debug("discovered docker gateway: %s", gateway)
        return gateway
    except (subprocess.SubprocessError, IndexError, OSError):
        _log.warning("could not discover gateway, falling back to 0.0.0.0")
        return "0.0.0.0"


def _clean_model_id(model_id: str) -> str:
    name = model_id.replace(".gguf", "")
    name = re.sub(r"-Q\d+.*$", "", name, flags=re.IGNORECASE)
    return name


def _query_model_id(url: str) -> str | None:
    try:
        response = requests.get(f"{url}/v1/models", timeout=3)
        response.raise_for_status()
        data = response.json()
        if data.get("data"):
            return _clean_model_id(data["data"][0]["id"])
    except (requests.RequestException, ValueError, KeyError) as e:
        _log.debug("model probe %s: %s", url, e)
    return None


def _wait_for_model(url: str, label: str) -> str | None:
    deadline = time.time() + MODEL_DISCOVERY_TIMEOUT_S
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        model_id = _query_model_id(url)
        if model_id:
            _log.info("model ready  %s: %s @ %s (attempt %d)", label, model_id, url, attempt)
            return model_id
        time.sleep(MODEL_DISCOVERY_INTERVAL_S)
    _log.error("model unreachable  %s: no response from %s after %ds", label, url, MODEL_DISCOVERY_TIMEOUT_S)
    return None


def _collect_role_env_vars() -> list[tuple[str, str]]:
    pattern = re.compile(r"^MODEL_(.+)_URL$")
    pairs: list[tuple[str, str]] = []
    for key, value in os.environ.items():
        match = pattern.match(key)
        if not match or not value.strip():
            continue
        role = match.group(1).lower()
        if role in {"embedder", "reranker", "host"}:
            continue
        pairs.append((role, value.strip()))
    return pairs


def _register(catalog: dict, entry: dict) -> None:
    catalog[entry["role"]] = entry
    model_id = entry.get("model_id")
    if model_id and model_id not in catalog:
        catalog[model_id] = entry


def discover_models() -> dict:
    catalog: dict = {}

    role_pairs = _collect_role_env_vars()
    if role_pairs:
        _log.info("found %d MODEL_<ROLE>_URL env vars", len(role_pairs))
        for role, url in role_pairs:
            model_id = _wait_for_model(url, f"role={role}")
            if not model_id:
                continue
            _register(catalog, {"role": role, "url": url, "model_id": model_id})
        return catalog

    hosts_env = os.getenv("MODEL_HOSTS", "").strip()
    if hosts_env:
        _log.info("using MODEL_HOSTS fallback")
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

    _log.info("using port-scan discovery fallback")
    host = _get_host()
    port_start = int(os.getenv("MODEL_PORT_START", "8080"))
    port_end = int(os.getenv("MODEL_PORT_END", "8090"))
    # port-scan must skip non-model SVC_* ports or they'll register as models
    exclude_ports: set[int] = set()
    for _svc_var in ("SVC_EMBEDDER_URL", "SVC_RERANKER_URL", "SVC_WHISPER_URL", "SVC_SEARXNG_URL"):
        _svc_url = os.getenv(_svc_var, "")
        if _svc_url:
            try:
                exclude_ports.add(int(_svc_url.rstrip("/").rsplit(":", 1)[-1]))
            except ValueError:
                pass
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
    host = url.split("://")[-1].split("/")[0].split(":")[0]
    parts = host.split("-")
    if len(parts) >= 3 and parts[0] == "mst" and parts[1] == "ag":
        return parts[2]
    return host


MODELS: dict = discover_models()
_log.info("model catalog ready: %s", sorted({v['role'] for v in MODELS.values() if isinstance(v, dict)}))


def refresh_models() -> dict:
    _log.info("refreshing model catalog")
    global MODELS
    MODELS = discover_models()
    return MODELS


def get_model_url(key: str) -> str | None:
    entry = MODELS.get(key) or MODELS.get(key.lower())
    if isinstance(entry, dict):
        return entry.get("url")
    _log.debug("model not found in catalog  key=%s available=%s", key, list(MODELS.keys()))
    return None


EMBEDDER_URL = os.getenv("SVC_EMBEDDER_URL")
RERANKER_URL = os.getenv("SVC_RERANKER_URL")
WHISPER_URL = os.getenv("SVC_WHISPER_URL")

CHROMA_URL = os.getenv("DB_CHROMADB_URL")
SEARXNG_URL = os.getenv("SVC_SEARXNG_URL", "http://mst-ag-searxng:8080")
BROWSER_URL = os.getenv("SVC_BROWSER_URL", "http://localhost:3800/browser")
SANDBOX_URL = os.getenv("SVC_SANDBOX_URL", "http://localhost:3800/sandbox")
FALKORDB_HOST = os.getenv("DB_FALKORDB_HOST")
FALKORDB_PORT = int(os.getenv("DB_FALKORDB_PORT", "6379"))

NOCODB_URL = os.getenv("DB_NOCODB_URL")
NOCODB_TOKEN = os.getenv("DB_NOCODB_TOKEN")
NOCODB_BASE_ID = os.getenv("DB_NOCODB_BASE_ID")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


HUEY_ENABLED = _env_bool("HUEY_ENABLED", True)
HUEY_SQLITE_PATH = os.getenv("HUEY_SQLITE_PATH", "/app/data/huey_queue.db")
HUEY_CONSUMER_WORKERS = int(os.getenv("HUEY_CONSUMER_WORKERS", "1"))
HUEY_TASK_RETRIES = int(os.getenv("HUEY_TASK_RETRIES", "2"))
HUEY_TASK_RETRY_DELAY_S = int(os.getenv("HUEY_TASK_RETRY_DELAY_S", "5"))

NOCODB_TABLE_ORGANISATION = "organisation"
NOCODB_TABLE_AGENT_RUNS = "agent_runs"
NOCODB_TABLE_ENRICHMENT_AGENTS = "enrichment_agents"
NOCODB_TABLE_SCRAPE_TARGETS = "scrape_targets"
NOCODB_TABLE_ENRICHMENT_LOG = "enrichment_log"
NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS = "suggested_scrape_targets"
NOCODB_TABLE_MESSAGES = "messages"
NOCODB_TABLE_MESSAGE_SEARCH_SOURCES = "message_search_sources"
NOCODB_TABLE_CONVERSATIONS = "conversations"
NOCODB_TABLE_DAILY_DIGESTS = "daily_digests"
NOCODB_TABLE_ASSISTANT_QUESTIONS = "assistant_questions"

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

TOOLS_FRAMEWORK_ENABLED = is_feature_enabled("tools_framework")

BASE_SYSTEM_PROMPT = (
    "You are JeffGPT, a direct and capable AI assistant. "
    "You are running on local models hosted by the user. "
    "Be honest, be concise, and match the tone of the conversation."
)

CHAT_TIMEZONE = os.getenv("CHAT_TIMEZONE", "Australia/Sydney")


ENRICHMENT_TOKEN_BUDGET = int(os.getenv("ENRICHMENT_TOKEN_BUDGET", "50000"))
ENRICHMENT_LOG_RETENTION_DAYS = int(os.getenv("ENRICHMENT_LOG_RETENTION_DAYS", "30"))
MAX_SUMMARY_INPUT_CHARS = 6000  # tool model has 8k context
PROACTIVE_BUDGET_THRESHOLD = 5000

# reasoner reserved for interactive chat — background/tool paths must never resolve to it
REASONER_ROLE = "reasoner"

# must match llama.cpp --parallel N
MODEL_PARALLEL_SLOTS = int(os.getenv("MODEL_PARALLEL_SLOTS", "2"))

ROLE_PARALLEL_SLOTS: dict[str, int] = {
    "t3_tool": int(os.getenv("MODEL_PARALLEL_SLOTS_T3_TOOL", "1")),
}

JOB_QUEUE_POLL_INTERVAL = float(os.getenv("JOB_QUEUE_POLL_INTERVAL", "300"))
JOB_QUEUE_STALE_TIMEOUT = int(os.getenv("JOB_QUEUE_STALE_TIMEOUT", "300"))
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


def no_think_params(model_id: str | None = None) -> dict:
    # chat_template_kwargs flows through jinja template to skip thinking blocks
    if model_id:
        from model_profiles import no_think_params_for
        params = no_think_params_for(model_id)
        if params:
            return params
    return {"chat_template_kwargs": {"enable_thinking": False}}
