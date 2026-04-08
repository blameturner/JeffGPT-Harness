import os
from dotenv import load_dotenv

load_dotenv()

# eventually rebuild from env variables to using dynamic from a db

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

TABLE_ORGANISATIONS = os.getenv("NOCODB_TABLE_ORGANISATIONS")
TABLE_USERS = os.getenv("NOCODB_TABLE_USERS")
TABLE_PROJECTS = os.getenv("NOCODB_TABLE_PROJECTS")
TABLE_AGENTS = os.getenv("NOCODB_TABLE_AGENTS")
TABLE_AGENT_SCHEDULES = os.getenv("NOCODB_TABLE_AGENT_SCHEDULES")
TABLE_AGENT_RUNS = os.getenv("NOCODB_TABLE_AGENT_RUNS")
TABLE_AGENT_OUTPUTS = os.getenv("NOCODB_TABLE_AGENT_OUTPUTS")
TABLE_AGENT_TRIGGERS = os.getenv("NOCODB_TABLE_AGENT_TRIGGERS")
TABLE_AGENT_MEMORY = os.getenv("NOCODB_TABLE_AGENT_MEMORY")
TABLE_OBSERVATIONS = os.getenv("NOCODB_TABLE_OBSERVATIONS")
TABLE_TASKS = os.getenv("NOCODB_TABLE_TASKS")
TABLE_CONVERSATIONS = os.getenv("NOCODB_TABLE_CONVERSATIONS")
TABLE_MESSAGES = os.getenv("NOCODB_TABLE_MESSAGES")
TABLE_KNOWLEDGE_SOURCES = os.getenv("NOCODB_TABLE_KNOWLEDGE_SOURCES")
TABLE_SCRAPE_TARGETS = os.getenv("NOCODB_TABLE_SCRAPE_TARGETS")
TABLE_MODEL_PERFORMANCE = os.getenv("NOCODB_TABLE_MODEL_PERFORMANCE")
TABLE_SYSTEM_HEALTH = os.getenv("NOCODB_TABLE_SYSTEM_HEALTH")
TABLE_TRAINING_EXAMPLES = os.getenv("NOCODB_TABLE_TRAINING_EXAMPLES")
TABLE_NOTIFICATIONS = os.getenv("NOCODB_TABLE_NOTIFICATIONS")
TABLE_PROJECT_MEMBERS = os.getenv("NOCODB_TABLE_PROJECT_MEMBERS")
TABLE_AUDIT_LOG = os.getenv("NOCODB_TABLE_AUDIT_LOG")

ENVIRONMENT = os.getenv("ENVIRONMENT")

def scoped_collection(org_id: int, collection_name: str) -> str:
    return f"org_{org_id}_{collection_name}"


def scoped_graph(org_id: int) -> str:
    return f"org_{org_id}_mst_ag"
