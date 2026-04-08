import os
from dotenv import load_dotenv

load_dotenv()

MODELS = {
    key.replace("MODEL_", "").lower(): value
    for key, value in os.environ.items()
    if key.startswith("MODEL_")
}

EMBEDDER_URL = os.getenv("EMBEDDER_URL")
RERANKER_URL = os.getenv("RERANKER_URL")

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
