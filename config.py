import os
import re
import subprocess
import requests
from dotenv import load_dotenv

load_dotenv()


#gets docker for env variable
def _get_host() -> str:
    host = os.getenv("MODEL_HOST")
    if host:
        return host

    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True,
            text=True,
            timeout=5
        )
        gateway = result.stdout.split("via ")[1].split()[0]
        print(f"Discovered Docker gateway: {gateway}")
        return gateway
    except Exception:
        print("Could not discover gateway, check Docker Gateway")
        return "0.0.0.0"


def _discover_models(host: str) -> dict:
    port_start = int(os.getenv("MODEL_PORT_START", "8080"))
    port_end = int(os.getenv("MODEL_PORT_END", "8090"))

    exclude_ports = {
        int(os.getenv("EMBEDDER_PORT", "8083")),
        int(os.getenv("RERANKER_PORT", "8084"))
    }

    def _model_name(model_id: str) -> str:
        name = model_id.replace(".gguf", "")
        name = re.sub(r'-Q\d+.*$', '', name, flags=re.IGNORECASE)
        return name

    models = {}
    for port in range(port_start, port_end + 1):
        if port in exclude_ports:
            continue
        url = f"http://{host}:{port}"
        try:
            response = requests.get(f"{url}/v1/models", timeout=2)
            response.raise_for_status()
            data = response.json()
            if data.get("data"):
                model_id = data["data"][0]["id"]
                name = _model_name(model_id)
                models[name] = url
                print(f"Discovered model: {name} on port {port}")
        except Exception:
            pass

    return models


_host = _get_host()

MODELS = _discover_models(_host)

EMBEDDER_URL = f"http://{_host}:{os.getenv('EMBEDDER_PORT', '8083')}"
RERANKER_URL = f"http://{_host}:{os.getenv('RERANKER_PORT', '8084')}"

CHROMA_URL = os.getenv("CHROMA_URL")
FALKORDB_HOST = os.getenv("FALKORDB_HOST")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))

NOCODB_URL = os.getenv("NOCODB_URL")
NOCODB_TOKEN = os.getenv("NOCODB_TOKEN")
NOCODB_BASE_ID = os.getenv("NOCODB_BASE_ID")

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


def scoped_collection(org_id: int, collection_name: str) -> str:
    return f"org_{org_id}_{collection_name}"


def scoped_graph(org_id: int) -> str:
    return f"org_{org_id}_mst_ag"