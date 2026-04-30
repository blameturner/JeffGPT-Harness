import json
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.schemas import ConversationUpdate
from infra.nocodb_client import NocodbClient
from shared.jobs import STORE, run_in_background
from workers.code.agent import CodeAgent

_log = logging.getLogger("main.code")

router = APIRouter()


class CodeRequest(BaseModel):
    org_id: int
    model: str
    message: str
    mode: str = "plan"
    approved_plan: str | None = None
    files: list[dict] | None = None
    conversation_id: int | None = None
    title: str | None = None
    codebase_collection: str | None = None
    response_style: str | None = None
    knowledge_enabled: bool | None = None
    search_enabled: bool = False
    temperature: float = 0.2
    max_tokens: int = 8192


class CodebaseCreate(BaseModel):
    org_id: int
    name: str
    description: str | None = None


class CodebaseFileUpload(BaseModel):
    files: list[dict]


@router.post("/code")
def code(request: CodeRequest):
    _log.info(
        "POST /code  model=%s org=%d mode=%s conv=%s",
        request.model,
        request.org_id,
        request.mode,
        request.conversation_id,
    )
    try:
        agent = CodeAgent(
            model=request.model,
            org_id=request.org_id,
            mode=request.mode,  # type: ignore[arg-type]
            approved_plan=request.approved_plan,
            files=request.files,
            search_enabled=request.search_enabled,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    job = STORE.create()
    run_in_background(
        job,
        lambda j: agent.run_job(
            j,
            user_message=request.message,
            conversation_id=request.conversation_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            title=request.title,
            codebase_collection=request.codebase_collection,
            response_style=request.response_style,
            knowledge_enabled=request.knowledge_enabled,
        ),
    )
    return {"job_id": job.id}


@router.get("/codebases")
def list_codebases(org_id: int):
    from infra.memory import client

    try:
        db = NocodbClient()
        codebases: list[dict] = []
        if "knowledge_sources" in db.tables:
            rows = db._get_paginated(
                "knowledge_sources",
                params={
                    "where": f"(org_id,eq,{org_id})~and(type,eq,codebase)",
                    "limit": 200,
                },
            )
            for row in rows:
                collection_name = row.get("collection_name") or ""
                record_count = 0
                try:
                    col = client.get_or_create_collection(collection_name)
                    record_count = col.count()
                except Exception:
                    pass
                codebases.append(
                    {
                        "id": row["Id"],
                        "name": row.get("name"),
                        "description": row.get("description"),
                        "collection_name": collection_name,
                        "records": record_count,
                        "source": row.get("source"),
                        "created_at": row.get("CreatedAt"),
                    }
                )
        return {"codebases": codebases}
    except Exception as e:
        _log.error("list_codebases failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/codebases")
def create_codebase(body: CodebaseCreate):
    from infra.config import scoped_collection

    try:
        db = NocodbClient()
        collection_name = scoped_collection(
            body.org_id,
            f"codebase_{body.name.lower().replace(' ', '_')}",
        )
        row = db._post(
            "knowledge_sources",
            {
                "org_id": body.org_id,
                "name": body.name,
                "description": body.description or "",
                "type": "codebase",
                "collection_name": collection_name,
                "source": "manual",
            },
        )
        _log.info("codebase created  name=%s collection=%s", body.name, collection_name)
        return row
    except Exception as e:
        _log.error("create_codebase failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/codebases/{codebase_id}/index")
def index_codebase_files(codebase_id: int, body: CodebaseFileUpload, org_id: int | None = None):
    import base64

    from infra.memory import remember

    try:
        db = NocodbClient()
        if "knowledge_sources" not in db.tables:
            raise HTTPException(status_code=404, detail="knowledge_sources table not found")
        where = f"(Id,eq,{codebase_id})"
        if org_id is not None:
            where = f"{where}~and(org_id,eq,{int(org_id)})"
        rows = db._get("knowledge_sources", params={"where": where, "limit": 1}).get("list", [])
        if not rows:
            raise HTTPException(status_code=404, detail="codebase not found")
        row = rows[0]
        collection_name = row["collection_name"]
        row_org_id = int(row.get("org_id") or 0)

        indexed = 0
        for f in body.files:
            name = f.get("name", "unknown")
            content = f.get("content") or ""
            if f.get("content_b64"):
                content = base64.b64decode(f["content_b64"]).decode("utf-8", errors="replace")
            if not content.strip():
                continue
            text = f"FILE: {name}\n\n{content}"
            try:
                remember(
                    text=text,
                    metadata={"file": name, "codebase_id": codebase_id, "type": "codebase"},
                    org_id=row_org_id,
                    collection_name=collection_name,
                )
                indexed += 1
            except Exception:
                _log.error("index failed for file %s in codebase %d", name, codebase_id, exc_info=True)

        _log.info("codebase indexed  id=%d files=%d/%d", codebase_id, indexed, len(body.files))
        return {"indexed": indexed, "total": len(body.files)}
    except HTTPException:
        raise
    except Exception as e:
        _log.error("index_codebase_files failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/code/conversations")
def list_code_conversations(org_id: int, limit: int = 50):
    try:
        db = NocodbClient()
        return {"conversations": db.list_code_conversations(org_id, limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/code/conversations/{conversation_id}")
def get_code_conversation(conversation_id: int, org_id: int | None = None):
    try:
        db = NocodbClient()
        convo = db.get_code_conversation(conversation_id, org_id=org_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Code conversation not found")
        return convo
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/code/conversations/{conversation_id}/messages")
def get_code_messages(conversation_id: int, limit: int = 500, org_id: int | None = None):
    try:
        db = NocodbClient()
        convo = db.get_code_conversation(conversation_id, org_id=org_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Code conversation not found")
        return {"messages": db.list_code_messages(conversation_id, limit, org_id=org_id)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/code/conversations/{conversation_id}/workspace")
def get_code_workspace(conversation_id: int, org_id: int | None = None):
    try:
        db = NocodbClient()
        convo = db.get_code_conversation(conversation_id, org_id=org_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Code conversation not found")
        msgs = db.list_code_messages(conversation_id, org_id=org_id)
        for m in reversed(msgs):
            if m.get("role") != "user":
                continue
            raw = m.get("files_json")
            if not raw:
                continue
            if isinstance(raw, list):
                return {"files": raw}
            if isinstance(raw, str):
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                if isinstance(data, list):
                    return {"files": data}
        return {"files": []}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/code/conversations/{conversation_id}")
def update_code_conversation(conversation_id: int, body: ConversationUpdate, org_id: int | None = None):
    try:
        db = NocodbClient()
        convo = db.get_code_conversation(conversation_id, org_id=org_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Code conversation not found")
        updates: dict = {}
        if body.title is not None:
            updates["title"] = body.title.strip() or "Untitled"
        if body.code_checklist is not None:
            updates["code_checklist"] = body.code_checklist
        if not updates:
            return convo
        return db.update_code_conversation(conversation_id, updates)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
