from __future__ import annotations

import mimetypes
import re
import difflib
import io
import zipfile
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.routers.code_launch import start_code_job, stream_job_events_response
from infra.config import is_feature_enabled
from infra.nocodb_client import ConflictError, NocodbClient
from infra.project_copy import import_from_project
from infra.paths import normalize_project_path
from infra.plan_preview import extract_plan_file_intents
from infra.project_context import (
    build_context_inspector_metadata,
    build_context_inspector_summary,
    build_project_context_pack,
    coerce_retrieval_scope,
    find_query_snippet,
)
from infra.project_metrics import count_todo_markers, parse_period_days
from infra.prompts import assemble_code_system_prompt
from workers.code.config import CODE_MODES, code_style_prompt, resolve_code_mode

router = APIRouter(prefix="/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    org_id: int
    name: str
    description: str | None = None
    system_note: str | None = None
    default_model: str | None = "code"
    retrieval_scope: list[str] | None = None


class ProjectPatch(BaseModel):
    name: str | None = None
    description: str | None = None
    system_note: str | None = None
    default_model: str | None = None
    retrieval_scope: list[str] | None = None


class ProjectFileWrite(BaseModel):
    path: str
    content: str = ""
    edit_summary: str | None = ""
    kind: str | None = None
    mime: str | None = None
    if_content_hash: str | None = None


class SnapshotCreate(BaseModel):
    label: str
    description: str | None = ""


class ProjectBulkImport(BaseModel):
    files: list[ProjectFileWrite] = Field(default_factory=list)


class ProjectFileFlag(BaseModel):
    path: str
    pinned: bool | None = None
    locked: bool | None = None


class ProjectFileRestore(BaseModel):
    path: str
    version: int


class ProjectFileMove(BaseModel):
    from_path: str = Field(alias="from")
    to_path: str = Field(alias="to")


class ProjectCodeChatRequest(BaseModel):
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
    interactive_fs: bool = False


class PlanPreviewRequest(BaseModel):
    scope_paths: list[str] | None = None


class ContextInspectorRequest(BaseModel):
    mode: str = "plan"
    response_style: str | None = None
    message: str = ""
    conversation_id: int | None = None
    codebase_collection: str | None = None
    include_history: bool = True


class ProjectImportFromRequest(BaseModel):
    src_project_id: int
    paths: list[str] = Field(default_factory=list)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug[:80] or "project"


def _guess_mime(path: str, provided: str | None) -> str:
    if provided:
        return provided
    guessed, _ = mimetypes.guess_type(path)
    return guessed or "text/plain"


def _require_enabled() -> None:
    if not is_feature_enabled("code_v2"):
        raise HTTPException(status_code=404, detail="projects feature disabled")


def _require_project(db: NocodbClient, project_id: int, org_id: int) -> dict:
    project = db.get_project(project_id, org_id=org_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    return project


def _normalize_prefix(prefix: str | None) -> str | None:
    if prefix is None:
        return None
    if prefix.strip() == "/":
        return "/"
    return normalize_project_path(prefix)


def _actor(org_id: int) -> str:
    return f"org:{org_id}"


@router.get("")
def list_projects(org_id: int, archived: bool = False):
    try:
        _require_enabled()
        db = NocodbClient()
        return {"projects": db.list_projects(org_id=org_id, archived=archived)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("")
def create_project(body: ProjectCreate):
    try:
        _require_enabled()
        db = NocodbClient()
        slug = _slugify(body.name)
        row = db.create_project(
            org_id=body.org_id,
            name=body.name.strip(),
            slug=slug,
            description=body.description or "",
            system_note=body.system_note or "",
            default_model=body.default_model or "code",
            retrieval_scope=body.retrieval_scope or [],
            chroma_collection=f"codebase_{slug}",
        )
        db.add_project_audit_event(int(row.get("Id") or 0), _actor(body.org_id), "project_create", {"name": row.get("name")})
        return {"project": row}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}")
def get_project(project_id: int, org_id: int):
    try:
        _require_enabled()
        db = NocodbClient()
        project = _require_project(db, project_id, org_id)
        files = db.list_project_files(project_id=project_id, limit=1)
        latest_activity_at = files[0].get("UpdatedAt") if files else project.get("UpdatedAt")
        return {
            "project": project,
            "file_count": len(db.list_project_files(project_id=project_id)),
            "latest_activity_at": latest_activity_at,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{project_id}")
def patch_project(project_id: int, body: ProjectPatch, org_id: int):
    try:
        _require_enabled()
        db = NocodbClient()
        project = _require_project(db, project_id, org_id)
        updates = body.model_dump(exclude_none=True)
        if "name" in updates:
            updates["name"] = updates["name"].strip()
        if not updates:
            return {"project": project}
        out = db.update_project(project_id, updates)
        db.add_project_audit_event(project_id, _actor(org_id), "project_patch", updates)
        return {"project": out}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/archive")
def archive_project(project_id: int, org_id: int):
    try:
        _require_enabled()
        db = NocodbClient()
        _require_project(db, project_id, org_id)
        db.update_project(project_id, {"archived_at": datetime.now(timezone.utc).isoformat()})
        db.add_project_audit_event(project_id, _actor(org_id), "project_archive", {})
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/fs")
def list_project_fs(project_id: int, org_id: int, prefix: str | None = None):
    try:
        _require_enabled()
        db = NocodbClient()
        _require_project(db, project_id, org_id)
        normalized_prefix = _normalize_prefix(prefix)
        rows = db.list_project_files(project_id=project_id, prefix=normalized_prefix)
        files = [
            {
                "path": r.get("path"),
                "kind": r.get("kind"),
                "size": r.get("size_bytes") or 0,
                "current_version": r.get("current_version_id"),
                "updated_at": r.get("UpdatedAt"),
                "pinned": bool(r.get("pinned")),
                "locked": bool(r.get("locked")),
            }
            for r in rows
        ]
        return {"files": files}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/fs/file")
def read_project_file(project_id: int, org_id: int, path: str):
    try:
        _require_enabled()
        db = NocodbClient()
        _require_project(db, project_id, org_id)
        normalized_path = normalize_project_path(path)
        file_row = db.get_project_file(project_id=project_id, path=normalized_path)
        if not file_row:
            raise HTTPException(status_code=404, detail="file not found")
        current = None
        if file_row.get("current_version_id"):
            current = db.get_project_file_version(int(file_row["current_version_id"]))
        return {"file": file_row, "current_version": current}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/fs/file/versions")
def list_project_file_versions(project_id: int, org_id: int, path: str):
    try:
        _require_enabled()
        db = NocodbClient()
        _require_project(db, project_id, org_id)
        normalized_path = normalize_project_path(path)
        file_row = db.get_project_file(project_id=project_id, path=normalized_path, include_archived=True)
        if not file_row:
            raise HTTPException(status_code=404, detail="file not found")
        return {"versions": db.list_project_file_versions(int(file_row["Id"]))}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{project_id}/fs/file")
def write_project_file(project_id: int, body: ProjectFileWrite, org_id: int):
    try:
        _require_enabled()
        db = NocodbClient()
        _require_project(db, project_id, org_id)
        normalized_path = normalize_project_path(body.path)
        size_bytes = len((body.content or "").encode("utf-8"))
        if size_bytes > 100 * 1024:
            raise HTTPException(status_code=413, detail="file exceeds 100KB limit")

        file_row, version_row, changed = db.write_project_file_version(
            project_id=project_id,
            path=normalized_path,
            content=body.content,
            edit_summary=body.edit_summary or "",
            kind=body.kind or "code",
            mime=_guess_mime(normalized_path, body.mime),
            created_by="user",
            audit_actor=_actor(org_id),
            if_content_hash=body.if_content_hash,
        )
        return {"file": file_row, "version": version_row, "changed": changed}
    except ConflictError as e:
        raise HTTPException(
            status_code=409,
            detail={"error": str(e), "expected": e.expected, "actual": e.actual},
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/fs/import")
def bulk_import_project_files(project_id: int, body: ProjectBulkImport, org_id: int):
    try:
        _require_enabled()
        db = NocodbClient()
        _require_project(db, project_id, org_id)
        written = 0
        skipped = 0
        for item in body.files:
            normalized_path = normalize_project_path(item.path)
            content = item.content or ""
            if len(content.encode("utf-8")) > 100 * 1024:
                skipped += 1
                continue
            _, _, changed = db.write_project_file_version(
                project_id=project_id,
                path=normalized_path,
                content=content,
                edit_summary=item.edit_summary or "bulk import",
                kind=item.kind or "code",
                mime=_guess_mime(normalized_path, item.mime),
                created_by="user",
                audit_actor=_actor(org_id),
            )
            if changed:
                written += 1
            else:
                skipped += 1
        return {"written": written, "skipped": skipped}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/fs/file/pin")
def set_project_file_pin(project_id: int, body: ProjectFileFlag, org_id: int):
    try:
        _require_enabled()
        db = NocodbClient()
        _require_project(db, project_id, org_id)
        if body.pinned is None:
            raise HTTPException(status_code=400, detail="pinned is required")
        path = normalize_project_path(body.path)
        db.set_project_file_flag(project_id, path, "pinned", bool(body.pinned), audit_actor=_actor(org_id))
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError:
        raise HTTPException(status_code=404, detail="file not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/fs/file/lock")
def set_project_file_lock(project_id: int, body: ProjectFileFlag, org_id: int):
    try:
        _require_enabled()
        db = NocodbClient()
        _require_project(db, project_id, org_id)
        if body.locked is None:
            raise HTTPException(status_code=400, detail="locked is required")
        path = normalize_project_path(body.path)
        db.set_project_file_flag(project_id, path, "locked", bool(body.locked), audit_actor=_actor(org_id))
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError:
        raise HTTPException(status_code=404, detail="file not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{project_id}/fs/file")
def delete_project_file(project_id: int, org_id: int, path: str):
    try:
        _require_enabled()
        db = NocodbClient()
        _require_project(db, project_id, org_id)
        normalized_path = normalize_project_path(path)
        db.archive_project_file(project_id, normalized_path, audit_actor=_actor(org_id))
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/fs/file/diff")
def diff_project_file(
    project_id: int,
    org_id: int,
    path: str,
    from_version: int | None = None,
    to_version: int | None = None,
):
    try:
        _require_enabled()
        db = NocodbClient()
        _require_project(db, project_id, org_id)
        normalized_path = normalize_project_path(path)
        file_row = db.get_project_file(project_id=project_id, path=normalized_path, include_archived=True)
        if not file_row:
            raise HTTPException(status_code=404, detail="file not found")
        versions = db.list_project_file_versions(int(file_row["Id"]), limit=200)
        by_version = {int(v.get("version") or 0): v for v in versions if v.get("version") is not None}
        if to_version is None:
            to_row = db.get_project_file_version(int(file_row["current_version_id"])) if file_row.get("current_version_id") else None
        else:
            to_row = by_version.get(int(to_version))
        if not to_row:
            raise HTTPException(status_code=404, detail="target version not found")
        if from_version is None:
            parent_id = to_row.get("parent_version_id")
            from_row = db.get_project_file_version(int(parent_id)) if parent_id else None
        else:
            from_row = by_version.get(int(from_version))

        before = (from_row or {}).get("content") or ""
        after = (to_row or {}).get("content") or ""
        unified = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"{normalized_path}@v{(from_row or {}).get('version') or 0}",
                tofile=f"{normalized_path}@v{to_row.get('version')}",
            )
        )
        return {"unified": unified}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/fs/file/restore")
def restore_project_file(project_id: int, body: ProjectFileRestore, org_id: int):
    try:
        _require_enabled()
        db = NocodbClient()
        _require_project(db, project_id, org_id)
        normalized_path = normalize_project_path(body.path)
        file_row, version_row, changed = db.restore_project_file_version(
            project_id,
            normalized_path,
            body.version,
            audit_actor=_actor(org_id),
        )
        return {"file": file_row, "version": version_row, "changed": changed}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/fs/move")
def move_project_file(project_id: int, body: ProjectFileMove, org_id: int):
    try:
        _require_enabled()
        db = NocodbClient()
        _require_project(db, project_id, org_id)
        from_path = normalize_project_path(body.from_path)
        to_path = normalize_project_path(body.to_path)
        file_row = db.move_project_file(project_id, from_path, to_path, audit_actor=_actor(org_id))
        return {"file": file_row}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/fs/search")
def search_project_files(project_id: int, org_id: int, q: str):
    try:
        _require_enabled()
        if not q.strip():
            return {"hits": []}
        db = NocodbClient()
        _require_project(db, project_id, org_id)
        query = q.strip().lower()
        hits: list[dict] = []
        for row in db.list_project_files(project_id=project_id):
            version_id = row.get("current_version_id")
            if not version_id:
                continue
            version = db.get_project_file_version(int(version_id))
            if not version:
                continue
            content = version.get("content") or ""
            pos = content.lower().find(query)
            if pos < 0:
                continue
            start = max(0, pos - 80)
            end = min(len(content), pos + len(query) + 120)
            snippet = content[start:end]
            hits.append({"path": row.get("path"), "version": version.get("version"), "snippet": snippet})
            if len(hits) >= 200:
                break
        return {"hits": hits}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/fs/export")
def export_project_files(project_id: int, org_id: int, format: str = "zip"):
    try:
        _require_enabled()
        if format != "zip":
            raise HTTPException(status_code=400, detail="only format=zip is supported")
        db = NocodbClient()
        _require_project(db, project_id, org_id)

        mem = io.BytesIO()
        with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for row in db.list_project_files(project_id=project_id):
                version_id = row.get("current_version_id")
                if not version_id:
                    continue
                version = db.get_project_file_version(int(version_id))
                if not version:
                    continue
                path = (row.get("path") or "").lstrip("/")
                if not path:
                    continue
                zf.writestr(path, version.get("content") or "")
        mem.seek(0)
        headers = {"Content-Disposition": f"attachment; filename=project-{project_id}.zip"}
        return StreamingResponse(mem, media_type="application/zip", headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/chat")
def project_chat(project_id: int, org_id: int, body: ProjectCodeChatRequest):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)

    resolved_mode = resolve_code_mode(body.mode)
    if resolved_mode not in CODE_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode '{body.mode}'")

    return start_code_job(
        org_id=org_id,
        model=body.model,
        message=body.message,
        mode=resolved_mode,
        approved_plan=body.approved_plan,
        files=body.files,
        conversation_id=body.conversation_id,
        title=body.title,
        codebase_collection=body.codebase_collection,
        response_style=body.response_style,
        knowledge_enabled=body.knowledge_enabled,
        search_enabled=body.search_enabled,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        project_id=project_id,
        interactive_fs=body.interactive_fs,
    )


@router.get("/{project_id}/audit")
def project_audit(project_id: int, org_id: int, kind: str | None = None, limit: int = 200):
    try:
        _require_enabled()
        db = NocodbClient()
        _require_project(db, project_id, org_id)
        events = db.list_project_audit_events(project_id=project_id, kind=kind, limit=max(1, min(limit, 1000)))
        return {"events": events}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/plans/{message_id}/preview")
def preview_plan_apply(project_id: int, message_id: int, body: PlanPreviewRequest, org_id: int):
    try:
        _require_enabled()
        db = NocodbClient()
        _require_project(db, project_id, org_id)

        msg = db.get_code_message(message_id, org_id=org_id)
        if not msg:
            raise HTTPException(status_code=404, detail="plan message not found")
        if msg.get("role") != "assistant":
            raise HTTPException(status_code=400, detail="plan preview requires an assistant message")

        convo_id = int(msg.get("conversation_id") or 0)
        convo = db.get_code_conversation(convo_id, org_id=org_id) if convo_id else None
        if not convo or int(convo.get("project_id") or 0) != int(project_id):
            raise HTTPException(status_code=400, detail="plan message is not scoped to this project")

        files = db.list_project_files(project_id=project_id)
        existing = {str(f.get("path") or "") for f in files if f.get("path")}
        intents = extract_plan_file_intents(msg.get("content") or "", existing)

        allowed = None
        if body.scope_paths:
            allowed = {normalize_project_path(p) for p in body.scope_paths}

        out = []
        for it in intents:
            path = it["path"]
            if allowed is not None and path not in allowed:
                continue
            size = 0
            if path in existing:
                row = next((f for f in files if f.get("path") == path), None)
                size = int((row or {}).get("size_bytes") or 0)
            out.append({"path": path, "action": it["action"], "existing_size": size})

        return {"items": out}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/chats/search")
def project_chat_search(project_id: int, org_id: int, q: str, limit: int = 50):
    try:
        _require_enabled()
        db = NocodbClient()
        _require_project(db, project_id, org_id)
        query = (q or "").strip()
        if not query:
            return {"hits": []}

        hits: list[dict] = []
        convs = db.list_code_conversations(org_id=org_id, limit=500)
        project_convs = [c for c in convs if int(c.get("project_id") or 0) == int(project_id)]
        for convo in project_convs:
            convo_id = int(convo.get("Id") or 0)
            if not convo_id:
                continue
            msgs = db.list_code_messages(convo_id, limit=300, org_id=org_id)
            for m in msgs:
                role = (m.get("role") or "").strip()
                if role not in {"user", "assistant"}:
                    continue
                snippet = find_query_snippet(m.get("content") or "", query)
                if not snippet:
                    continue
                hits.append(
                    {
                        "conversation_id": convo_id,
                        "message_id": m.get("Id"),
                        "role": role,
                        "snippet": snippet,
                        "created_at": m.get("CreatedAt"),
                    }
                )
                if len(hits) >= max(1, min(limit, 200)):
                    return {"hits": hits}
        return {"hits": hits}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/fs/import-from")
def import_from_other_project(project_id: int, org_id: int, body: ProjectImportFromRequest):
    try:
        _require_enabled()
        db = NocodbClient()
        _require_project(db, project_id, org_id)
        _require_project(db, int(body.src_project_id), org_id)

        if int(body.src_project_id) == int(project_id):
            raise HTTPException(status_code=400, detail="src_project_id must differ from destination project")
        if not body.paths:
            return {"written": 0, "skipped": 0, "missing": 0}

        result = import_from_project(
            db,
            src_project_id=int(body.src_project_id),
            dst_project_id=project_id,
            paths=body.paths,
            actor=_actor(org_id),
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/turns/{message_id}/context-inspector")
def project_context_inspector(project_id: int, message_id: int, body: ContextInspectorRequest, org_id: int):
    try:
        _require_enabled()
        db = NocodbClient()
        project = _require_project(db, project_id, org_id)

        msg = db.get_code_message(message_id, org_id=org_id)
        if not msg:
            raise HTTPException(status_code=404, detail="message not found")
        if msg.get("role") != "assistant":
            raise HTTPException(status_code=400, detail="context inspector expects an assistant message")

        convo_id = int(msg.get("conversation_id") or 0)
        convo = db.get_code_conversation(convo_id, org_id=org_id) if convo_id else None
        if not convo or int(convo.get("project_id") or 0) != int(project_id):
            raise HTTPException(status_code=400, detail="message is not scoped to this project")

        mode = resolve_code_mode(body.mode)
        if mode not in CODE_MODES:
            raise HTTPException(status_code=400, detail=f"invalid mode '{body.mode}'")
        style_key, style_prompt = code_style_prompt(body.response_style)

        pack = build_project_context_pack(db, project_id)
        system_prompt = assemble_code_system_prompt(
            mode=mode,
            style_prompt=style_prompt,
            project_name=project.get("name") or "",
            project_slug=project.get("slug") or "",
            system_note=project.get("system_note") or "",
            pinned_context=pack.get("pinned_context") or "",
            path_manifest=pack.get("path_manifest") or "",
            context_notice=pack.get("context_notice") or "",
            interactive_fs=bool(convo.get("interactive_fs")),
            glossary_terms=list(pack.get("glossary_terms") or []),
        )

        retrieval_collections: list[str] = []
        if body.codebase_collection:
            retrieval_collections = [body.codebase_collection]
        else:
            retrieval_collections = coerce_retrieval_scope(project.get("retrieval_scope"))

        history = []
        if body.include_history:
            all_msgs = db.list_code_messages(convo_id, org_id=org_id)
            history = [
                {"role": m.get("role"), "content": m.get("content")}
                for m in all_msgs
                if m.get("content") and int(m.get("Id") or 0) < int(message_id)
            ][-20:]

        return build_context_inspector_metadata(
            project_id=project_id,
            conversation_id=convo_id,
            message_id=message_id,
            mode=mode,
            style=style_key,
            interactive_fs=bool(convo.get("interactive_fs")),
            retrieval_collections=retrieval_collections,
            system_prompt=system_prompt,
            history=history,
            user_message=body.message,
            context_pack=pack,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/turns/{message_id}/context-inspector/summary")
def project_context_inspector_summary(project_id: int, message_id: int, body: ContextInspectorRequest, org_id: int):
    full = project_context_inspector(project_id=project_id, message_id=message_id, body=body, org_id=org_id)
    return build_context_inspector_summary(full)


@router.get("/{project_id}/chat/stream/{job_id}")
def stream_project_chat(project_id: int, org_id: int, job_id: str, cursor: int = 0):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    return stream_job_events_response(job_id, cursor)


@router.get("/{project_id}/history")
def project_history(project_id: int, org_id: int, limit: int = 200):
    try:
        _require_enabled()
        db = NocodbClient()
        _require_project(db, project_id, org_id)
        rows: list[dict] = []
        for file_row in db.list_project_files(project_id=project_id, include_archived=True):
            path = file_row.get("path") or ""
            versions = db.list_project_file_versions(int(file_row["Id"]), limit=max(1, min(limit, 500)))
            for v in versions:
                rows.append(
                    {
                        "path": path,
                        "version": v.get("version"),
                        "edit_summary": v.get("edit_summary") or "",
                        "conversation_id": v.get("conversation_id"),
                        "created_by_message_id": v.get("created_by_message_id"),
                        "created_at": v.get("CreatedAt"),
                    }
                )
        rows.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        return {"events": rows[: max(1, min(limit, 1000))], "total_versions": len(rows)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/metrics")
def project_metrics(project_id: int, org_id: int, period: str = "30d"):
    try:
        _require_enabled()
        db = NocodbClient()
        _require_project(db, project_id, org_id)

        days = parse_period_days(period)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        files = db.list_project_files(project_id=project_id)
        file_count = len(files)
        bytes_total = sum(int(f.get("size_bytes") or 0) for f in files)
        latest_activity_at = max((f.get("UpdatedAt") or "" for f in files), default="")

        edits_by_day: dict[str, int] = {}
        agent_edits = 0
        user_edits = 0
        todos_open = 0

        for file_row in files:
            if file_row.get("current_version_id"):
                cur = db.get_project_file_version(int(file_row["current_version_id"]))
                todos_open += count_todo_markers((cur or {}).get("content") or "")

            versions = db.list_project_file_versions(int(file_row["Id"]), limit=400)
            for v in versions:
                ts = v.get("CreatedAt")
                if not ts:
                    continue
                try:
                    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                except Exception:
                    continue
                if dt < cutoff:
                    continue
                day = dt.date().isoformat()
                edits_by_day[day] = edits_by_day.get(day, 0) + 1
                if v.get("conversation_id"):
                    agent_edits += 1
                else:
                    user_edits += 1

        return {
            "period": f"{days}d",
            "file_count": file_count,
            "bytes_total": bytes_total,
            "edits_by_day": edits_by_day,
            "agent_edits": agent_edits,
            "user_edits": user_edits,
            "open_todos": todos_open,
            "last_activity": latest_activity_at or None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/graveyard")
def project_graveyard(project_id: int, org_id: int, limit: int = 200):
    try:
        _require_enabled()
        db = NocodbClient()
        _require_project(db, project_id, org_id)
        rows = db.list_project_files(
            project_id=project_id,
            archived_only=True,
            limit=max(1, min(limit, 1000)),
        )
        out: list[dict] = []
        for r in rows:
            archived_at = r.get("archived_at") or ""
            age_days: float | None = None
            if archived_at:
                try:
                    dt = datetime.fromisoformat(str(archived_at).replace("Z", "+00:00"))
                    age_days = round((datetime.now(timezone.utc) - dt).total_seconds() / 86400, 2)
                except Exception:
                    age_days = None
            out.append(
                {
                    "path": r.get("path"),
                    "kind": r.get("kind"),
                    "size_bytes": r.get("size_bytes"),
                    "archived_at": archived_at,
                    "age_days": age_days,
                    "current_version_id": r.get("current_version_id"),
                }
            )
        return {"files": out, "count": len(out)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/snapshots")
def create_project_snapshot(project_id: int, body: SnapshotCreate, org_id: int):
    try:
        _require_enabled()
        db = NocodbClient()
        _require_project(db, project_id, org_id)
        label = (body.label or "").strip()
        if not label or len(label) > 80 or not re.match(r"^[A-Za-z0-9._\- ]+$", label):
            raise HTTPException(status_code=400, detail="label must be 1–80 chars; letters/digits/._- only")
        snap = db.create_project_snapshot(
            project_id=project_id,
            label=label,
            actor=_actor(org_id),
            description=body.description or "",
        )
        return {"snapshot": snap}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/snapshots")
def list_project_snapshots(project_id: int, org_id: int, limit: int = 200):
    try:
        _require_enabled()
        db = NocodbClient()
        _require_project(db, project_id, org_id)
        snaps = db.list_project_snapshots(project_id=project_id, limit=max(1, min(limit, 1000)))
        return {"snapshots": snaps}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/snapshots/{label}/diff")
def diff_project_snapshot(project_id: int, label: str, org_id: int):
    try:
        _require_enabled()
        db = NocodbClient()
        _require_project(db, project_id, org_id)
        snap = db.get_project_snapshot(project_id, label)
        if not snap:
            raise HTTPException(status_code=404, detail="snapshot not found")
        snap_files = db.list_project_snapshot_files(int(snap["Id"]))
        snap_by_path = {row.get("path"): row for row in snap_files if row.get("path")}

        current_files = db.list_project_files(project_id=project_id)
        current_by_path = {row.get("path"): row for row in current_files if row.get("path")}

        diffs: list[dict] = []
        all_paths = sorted(set(snap_by_path.keys()) | set(current_by_path.keys()))
        for p in all_paths:
            snap_row = snap_by_path.get(p)
            cur_row = current_by_path.get(p)
            before = ""
            after = ""
            from_version = 0
            to_version = 0
            if snap_row and snap_row.get("version_id"):
                v = db.get_project_file_version(int(snap_row["version_id"]))
                before = (v or {}).get("content") or ""
                from_version = (v or {}).get("version") or 0
            if cur_row and cur_row.get("current_version_id"):
                v = db.get_project_file_version(int(cur_row["current_version_id"]))
                after = (v or {}).get("content") or ""
                to_version = (v or {}).get("version") or 0
            if before == after:
                continue
            unified = "".join(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    after.splitlines(keepends=True),
                    fromfile=f"{p}@snapshot:{label}",
                    tofile=f"{p}@current",
                )
            )
            state = (
                "added" if not before else
                "removed" if not after else
                "modified"
            )
            diffs.append(
                {
                    "path": p,
                    "state": state,
                    "from_version": from_version,
                    "to_version": to_version,
                    "unified": unified,
                }
            )
        return {"snapshot": snap, "files": diffs}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/diff")
def diff_against_other_project(project_id: int, org_id: int, against: int):
    try:
        _require_enabled()
        if int(against) == int(project_id):
            raise HTTPException(status_code=400, detail="against must differ from project_id")
        db = NocodbClient()
        _require_project(db, project_id, org_id)
        _require_project(db, int(against), org_id)

        def _current(pid: int) -> dict[str, tuple[str, int]]:
            out: dict[str, tuple[str, int]] = {}
            for row in db.list_project_files(project_id=pid):
                vid = row.get("current_version_id")
                if not vid:
                    continue
                v = db.get_project_file_version(int(vid))
                if not v:
                    continue
                out[row.get("path") or ""] = ((v.get("content") or ""), int(v.get("version") or 0))
            return out

        left = _current(project_id)
        right = _current(int(against))
        diffs: list[dict] = []
        for p in sorted(set(left.keys()) | set(right.keys())):
            l_content, l_v = left.get(p, ("", 0))
            r_content, r_v = right.get(p, ("", 0))
            if l_content == r_content:
                continue
            state = (
                "only_in_left" if not r_content else
                "only_in_right" if not l_content else
                "modified"
            )
            unified = "".join(
                difflib.unified_diff(
                    r_content.splitlines(keepends=True),
                    l_content.splitlines(keepends=True),
                    fromfile=f"{p}@project:{against}",
                    tofile=f"{p}@project:{project_id}",
                )
            )
            diffs.append(
                {
                    "path": p,
                    "state": state,
                    "left_version": l_v,
                    "right_version": r_v,
                    "unified": unified,
                }
            )
        return {"project_id": project_id, "against": int(against), "files": diffs}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/open-work")
def project_open_work(project_id: int, org_id: int):
    try:
        _require_enabled()
        db = NocodbClient()
        _require_project(db, project_id, org_id)

        files = db.list_project_files(project_id=project_id)
        todos_open = 0
        for file_row in files:
            if not file_row.get("current_version_id"):
                continue
            cur = db.get_project_file_version(int(file_row["current_version_id"]))
            todos_open += count_todo_markers((cur or {}).get("content") or "")

        permission_events = db.list_project_audit_events(project_id=project_id, kind="permission_request", limit=200)
        unresolved = [
            {
                "path": (e.get("payload") or {}).get("path"),
                "reason": (e.get("payload") or {}).get("reason"),
                "created_at": e.get("CreatedAt"),
            }
            for e in permission_events
        ]

        return {
            "open_todos": todos_open,
            "permission_requests": unresolved,
            "permission_request_count": len(unresolved),
            "file_count": len(files),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


