import logging

from fastapi import APIRouter, HTTPException

from infra.config import MODELS, refresh_models
from shared.styles import (
    CHAT_DEFAULT_STYLE,
    CODE_DEFAULT_STYLE,
    list_chat_styles,
    list_code_styles,
)

_log = logging.getLogger("main.health")

router = APIRouter()


WORKER_TYPES = [
    {"id": "generator", "name": "Generator", "description": "Produces structured output for a task."},
    {"id": "monitor", "name": "Monitor", "description": "Watches a data source and flags changes."},
    {"id": "evaluator", "name": "Evaluator", "description": "Scores or critiques outputs against criteria."},
    {"id": "researcher", "name": "Researcher", "description": "Performs multi-source web research."},
    {"id": "memory", "name": "Memory", "description": "Writes and retrieves knowledge from ChromaDB / FalkorDB."},
    {"id": "code", "name": "Code", "description": "Plans, writes, and debugs code."},
]


@router.get("/health")
async def health():
    return {"status": "ok", "service": "MSTAG Harness"}


@router.get("/models")
def list_models():
    catalog = MODELS or refresh_models()
    seen: set[str] = set()
    models: list[dict] = []
    for entry in catalog.values():
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        if not role or role in seen:
            continue
        seen.add(role)
        models.append({
            "name": role,
            "role": role,
            "model_id": entry.get("model_id"),
            "url": entry.get("url"),
        })
    return {"models": models}


@router.get("/styles")
def get_styles(surface: str | None = None):
    out: dict = {}
    if surface in (None, "chat"):
        out["chat"] = {"default": CHAT_DEFAULT_STYLE, "styles": list_chat_styles()}
    if surface in (None, "code"):
        out["code"] = {"default": CODE_DEFAULT_STYLE, "styles": list_code_styles()}
    if not out:
        raise HTTPException(status_code=400, detail="surface must be 'chat' or 'code'")
    return out


@router.get("/workers/types")
def worker_types():
    return {"types": WORKER_TYPES}
