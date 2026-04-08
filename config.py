import os
import requests
from dotenv import load_dotenv

load_dotenv()


# build for scanning llama.cpp models
def _discover_models() -> dict:
    host = os.getenv("MODEL_HOST")
    port_start = int(os.getenv("MODEL_PORT_START"))
    port_end = int(os.getenv("MODEL_PORT_END"))

    # exclude embedder and reranker ports from model discovery
    exclude_ports = {
        int(os.getenv("EMBEDDER_PORT", "8083")),
        int(os.getenv("RERANKER_PORT", "8084"))
    }

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
                models[model_id] = url
                print(f"Discovered model: {model_id} on port {port}")
        except Exception:
            pass

    return models


MODELS = _discover_models()

_host = os.getenv("MODEL_HOST")
EMBEDDER_URL = f"http://{_host}:{os.getenv('EMBEDDER_PORT')}"
RERANKER_URL = f"http://{_host}:{os.getenv('RERANKER_PORT')}"

CHROMA_URL = os.getenv("CHROMA_URL")
FALKORDB_HOST = os.getenv("FALKORDB_HOST")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT"))

NOCODB_URL = os.getenv("NOCODB_URL")
NOCODB_TOKEN = os.getenv("NOCODB_TOKEN")
NOCODB_BASE_ID = os.getenv("NOCODB_BASE_ID")

ENVIRONMENT = os.getenv("ENVIRONMENT")

def scoped_collection(org_id: int, collection_name: str) -> str:
    return f"org_{org_id}_{collection_name}"


def scoped_graph(org_id: int) -> str:
    return f"org_{org_id}_mst_ag"
