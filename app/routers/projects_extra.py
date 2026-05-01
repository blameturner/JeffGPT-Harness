"""Project extras: branches, share, bookmarks, recipes, pinboard, workspaces,
templates, ADRs, time-travel, hot-reload, find/replace, rename, etc."""
from __future__ import annotations

import difflib
import re
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from infra.config import is_feature_enabled
from infra.file_templates import adr_path, adr_template, conventions_template, template_for
from infra.nocodb_client import NocodbClient
from infra.paths import normalize_project_path

router = APIRouter(prefix="/projects", tags=["projects-extra"])

# Endpoints whose path would otherwise collide with `/projects/{project_id:int}`
# (e.g. /projects/workspaces, /projects/p/{token}, /projects/_templates/...).
# Registered BEFORE `projects.router` in main.py so the literal-segment routes
# resolve first; otherwise FastAPI would match `/projects/{project_id}` and
# 422 trying to coerce the literal segment to int.
public_router = APIRouter(prefix="/projects", tags=["projects-public"])


def _require_enabled() -> None:
    if not is_feature_enabled("code_v2"):
        raise HTTPException(status_code=404, detail="projects feature disabled")


def _require_project(db: NocodbClient, project_id: int, org_id: int) -> dict:
    project = db.get_project(project_id, org_id=org_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"project {project_id} not found")
    return project


def _actor(org_id: int) -> str:
    return f"org:{org_id}"


# ---------- Branches ----------
class BranchRequest(BaseModel):
    name: str
    from_snapshot: str | None = None


@router.post("/{project_id}/branch")
def branch_project(project_id: int, org_id: int, body: BranchRequest):
    _require_enabled()
    db = NocodbClient()
    parent = _require_project(db, project_id, org_id)
    slug = re.sub(r"[^a-z0-9]+", "-", body.name.lower()).strip("-")[:80] or "branch"
    new = db.create_project(
        org_id=org_id,
        name=body.name,
        slug=slug,
        description=f"Branch of {parent.get('slug') or parent.get('name')}",
        chroma_collection=f"project_{slug}",
    )
    new_id = int(new["Id"])
    try:
        db.update_project(new_id, {"parent_project_id": project_id})
    except Exception:
        pass

    files: list[dict] = []
    if body.from_snapshot:
        snap = db.get_project_snapshot(project_id, body.from_snapshot)
        if not snap:
            raise HTTPException(status_code=404, detail="snapshot not found")
        for sf in db.list_project_snapshot_files(int(snap["Id"])):
            v = db.get_project_file_version(int(sf.get("version_id") or 0)) if sf.get("version_id") else None
            if v:
                files.append({"path": sf.get("path") or "", "content": v.get("content") or ""})
    else:
        for fr in db.list_project_files(project_id=project_id):
            vid = fr.get("current_version_id")
            if not vid:
                continue
            v = db.get_project_file_version(int(vid))
            if v:
                files.append({"path": fr.get("path") or "", "content": v.get("content") or ""})

    written = 0
    for f in files:
        try:
            normalized = normalize_project_path(f["path"])
        except ValueError:
            continue
        try:
            db.write_project_file_version(
                project_id=new_id,
                path=normalized,
                content=f["content"],
                edit_summary=f"branched from project {project_id}",
                created_by=f"branch:{project_id}",
                audit_actor=_actor(org_id),
                audit_kind="branch_create",
            )
            written += 1
        except Exception:
            continue
    return {"project_id": new_id, "files_written": written, "parent_project_id": project_id}


# ---------- Share tokens ----------
class ShareRequest(BaseModel):
    snapshot_id: int | None = None
    expires_in_days: int | None = 30


@router.post("/{project_id}/share")
def create_share_link(project_id: int, org_id: int, body: ShareRequest):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    token = secrets.token_urlsafe(18)
    expires_at = None
    if body.expires_in_days and body.expires_in_days > 0:
        expires_at = (datetime.now(timezone.utc) + timedelta(days=int(body.expires_in_days))).isoformat()
    row = db.create_share_token(project_id, token, body.snapshot_id, expires_at)
    if not row:
        raise HTTPException(status_code=500, detail="share-tokens table missing")
    return {"token": token, "url_path": f"/p/{token}", "expires_at": expires_at}


@public_router.delete("/share/{token}")
def revoke_share_link(token: str):
    _require_enabled()
    db = NocodbClient()
    return {"revoked": db.revoke_share_token(token)}


@public_router.get("/p/{token}")
def get_shared_project(token: str):
    _require_enabled()
    db = NocodbClient()
    row = db.get_share_token(token)
    if not row:
        raise HTTPException(status_code=404, detail="share token not found or revoked")
    expires_at = row.get("expires_at")
    if expires_at:
        try:
            if datetime.fromisoformat(str(expires_at).replace("Z", "+00:00")) < datetime.now(timezone.utc):
                raise HTTPException(status_code=410, detail="share token expired")
        except ValueError:
            pass
    project_id = int(row["project_id"])
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="project missing")

    files: list[dict] = []
    if row.get("snapshot_id"):
        snap = db.list_project_snapshot_files(int(row["snapshot_id"]))
        for sf in snap:
            v = db.get_project_file_version(int(sf.get("version_id") or 0)) if sf.get("version_id") else None
            files.append({"path": sf.get("path") or "", "content": (v or {}).get("content") or ""})
    else:
        for fr in db.list_project_files(project_id=project_id):
            vid = fr.get("current_version_id")
            v = db.get_project_file_version(int(vid)) if vid else None
            files.append({"path": fr.get("path") or "", "content": (v or {}).get("content") or "", "kind": fr.get("kind")})
    return {"project": {"name": project.get("name"), "slug": project.get("slug")}, "files": files}


# ---------- Bookmarks ----------
class BookmarkCreate(BaseModel):
    target_kind: str  # file | version | diff
    target_ref: str
    label: str | None = ""
    color: str | None = ""


@router.post("/{project_id}/bookmarks")
def add_project_bookmark(project_id: int, org_id: int, body: BookmarkCreate):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    row = db.add_bookmark(project_id, body.target_kind, body.target_ref, body.label or "", body.color or "")
    if not row:
        raise HTTPException(status_code=500, detail="bookmarks table missing")
    return {"bookmark": row}


@router.get("/{project_id}/bookmarks")
def list_project_bookmarks(project_id: int, org_id: int):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    return {"bookmarks": db.list_bookmarks(project_id)}


@router.delete("/{project_id}/bookmarks/{bookmark_id}")
def delete_project_bookmark(project_id: int, bookmark_id: int, org_id: int):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    return {"removed": db.remove_bookmark(bookmark_id)}


# ---------- Saved queries ----------
class SavedQueryCreate(BaseModel):
    name: str
    query: str
    kind: str = "search"


@router.post("/{project_id}/saved-queries")
def add_saved_query(project_id: int, org_id: int, body: SavedQueryCreate):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    return {"query": db.add_saved_query(project_id, body.name, body.query, body.kind)}


@router.get("/{project_id}/saved-queries")
def list_saved_queries(project_id: int, org_id: int):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    return {"queries": db.list_saved_queries(project_id)}


# ---------- Recipes ----------
class RecipeCreate(BaseModel):
    name: str
    steps: list[dict]


@router.post("/{project_id}/recipes")
def add_recipe(project_id: int, org_id: int, body: RecipeCreate):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    return {"recipe": db.add_recipe(project_id, body.name, body.steps)}


@router.get("/{project_id}/recipes")
def list_recipes(project_id: int, org_id: int):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    return {"recipes": db.list_recipes(project_id)}


# ---------- Pins (FAQ / pinboard) ----------
class PinCreate(BaseModel):
    kind: str  # conversation_message | question | snippet
    target_ref: str
    body: str | None = ""


@router.post("/{project_id}/pins")
def add_project_pin(project_id: int, org_id: int, body: PinCreate):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    return {"pin": db.add_pin(project_id, body.kind, body.target_ref, body.body or "")}


@router.get("/{project_id}/pins")
def list_project_pins(project_id: int, org_id: int):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    return {"pins": db.list_pins(project_id)}


# ---------- Workspaces (multi-project) ----------
class WorkspaceCreate(BaseModel):
    org_id: int
    name: str
    project_ids: list[int]


@public_router.post("/workspaces")
def create_workspace(body: WorkspaceCreate):
    _require_enabled()
    db = NocodbClient()
    return {"workspace": db.create_workspace(body.org_id, body.name, body.project_ids)}


@public_router.get("/workspaces")
def list_workspaces(org_id: int):
    _require_enabled()
    db = NocodbClient()
    return {"workspaces": db.list_workspaces(org_id)}


# ---------- Templates ----------
@public_router.get("/_templates/file")
def file_template(path: str, name: str = ""):
    _require_enabled()
    return {"path": path, "content": template_for(path, name)}


@public_router.get("/_templates/conventions")
def conventions_template_endpoint():
    _require_enabled()
    return {"path": "/.project/conventions.md", "content": conventions_template()}


# ---------- ADRs ----------
class ADRCreate(BaseModel):
    title: str
    context: str | None = ""
    decision: str | None = ""
    consequences: str | None = ""


@router.post("/{project_id}/adrs")
def create_adr(project_id: int, org_id: int, body: ADRCreate):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    files = db.list_project_files(project_id=project_id, prefix="/decisions/")
    next_n = 1
    for f in files:
        m = re.match(r"^/decisions/(\d{1,4})-", f.get("path") or "")
        if m:
            next_n = max(next_n, int(m.group(1)) + 1)
    path = adr_path(next_n, body.title)
    content = adr_template(next_n, body.title)
    if body.context:
        content = content.replace("_What is the problem we're solving and why now?_", body.context)
    if body.decision:
        content = content.replace("_The choice in one sentence, then a paragraph._", body.decision)
    if body.consequences:
        content = content.replace("_What this enables / forecloses; trade-offs._", body.consequences)
    file_row, version_row, _ = db.write_project_file_version(
        project_id=project_id,
        path=normalize_project_path(path),
        content=content,
        edit_summary=f"propose ADR-{next_n}: {body.title}",
        kind="adr",
        mime="text/markdown",
        pinned=True,  # auto-pin per plan
        created_by="user",
        audit_actor=_actor(org_id),
        audit_kind="adr_create",
    )
    return {"path": path, "number": next_n, "file": file_row, "version": version_row}


# ---------- Time-travel ----------
@router.get("/{project_id}/fs/file/at-time")
def time_travel_diff(project_id: int, org_id: int, path: str, at: str):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    normalized = normalize_project_path(path)
    file_row = db.get_project_file(project_id=project_id, path=normalized, include_archived=True)
    if not file_row:
        raise HTTPException(status_code=404, detail="file not found")
    try:
        ts = datetime.fromisoformat(at.replace("Z", "+00:00"))
    except Exception:
        raise HTTPException(status_code=400, detail="`at` must be ISO 8601 timestamp")
    versions = db.list_project_file_versions(int(file_row["Id"]), limit=500)
    historical = None
    for v in versions:
        try:
            v_ts = datetime.fromisoformat(str(v.get("CreatedAt") or "").replace("Z", "+00:00"))
        except Exception:
            continue
        if v_ts <= ts:
            historical = v
            break
    if not historical:
        return {"path": normalized, "found": False, "content": ""}
    current = db.get_project_file_version(int(file_row["current_version_id"])) if file_row.get("current_version_id") else None
    before = historical.get("content") or ""
    after = (current or {}).get("content") or ""
    unified = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"{normalized}@{at}",
            tofile=f"{normalized}@current",
        )
    )
    return {
        "path": normalized,
        "found": True,
        "version_at_time": historical.get("version"),
        "content_at_time": before,
        "current_version": (current or {}).get("version"),
        "unified": unified,
    }


# ---------- Find / replace ----------
class FindReplaceRequest(BaseModel):
    pattern: str
    replacement: str
    paths: list[str] | None = None  # path prefixes
    regex: bool = False
    dry_run: bool = True


@router.post("/{project_id}/fs/replace")
def project_find_replace(project_id: int, org_id: int, body: FindReplaceRequest):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    rx = re.compile(body.pattern) if body.regex else None

    def matches(path: str) -> bool:
        if not body.paths:
            return True
        return any(path.startswith(p) for p in body.paths)

    affected: list[dict] = []
    for fr in db.list_project_files(project_id=project_id):
        path = fr.get("path") or ""
        if not matches(path):
            continue
        vid = fr.get("current_version_id")
        if not vid:
            continue
        v = db.get_project_file_version(int(vid))
        if not v:
            continue
        before = v.get("content") or ""
        after = rx.sub(body.replacement, before) if rx else before.replace(body.pattern, body.replacement)
        if before == after:
            continue
        affected.append({"path": path, "before": before, "after": after, "file_id": int(fr["Id"])})

    if body.dry_run:
        return {
            "dry_run": True,
            "files": [
                {
                    "path": a["path"],
                    "unified": "".join(
                        difflib.unified_diff(
                            a["before"].splitlines(keepends=True),
                            a["after"].splitlines(keepends=True),
                            fromfile=a["path"], tofile=a["path"] + " (after)",
                        )
                    ),
                }
                for a in affected
            ],
            "count": len(affected),
        }

    written = 0
    for a in affected:
        try:
            db.write_project_file_version(
                project_id=project_id,
                path=a["path"],
                content=a["after"],
                edit_summary=f"bulk replace: {body.pattern} -> {body.replacement}",
                created_by="user",
                audit_actor=_actor(org_id),
            )
            written += 1
        except Exception:
            continue
    return {"dry_run": False, "written": written}


# ---------- Cross-file rename (deterministic) ----------
class RenameRequest(BaseModel):
    old: str
    new: str
    kind: str | None = None  # function | class | symbol | identifier
    paths: list[str] | None = None


@router.post("/{project_id}/rename")
def project_rename(project_id: int, org_id: int, body: RenameRequest):
    _require_enabled()
    if not body.old or not body.new:
        raise HTTPException(status_code=400, detail="old and new are required")
    if not re.match(r"^[A-Za-z_$][A-Za-z0-9_$]*$", body.old) or not re.match(r"^[A-Za-z_$][A-Za-z0-9_$]*$", body.new):
        raise HTTPException(status_code=400, detail="old/new must be valid identifiers")
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    rx = re.compile(rf"\b{re.escape(body.old)}\b")
    affected: list[dict] = []
    for fr in db.list_project_files(project_id=project_id):
        path = fr.get("path") or ""
        if body.paths and not any(path.startswith(p) for p in body.paths):
            continue
        vid = fr.get("current_version_id")
        if not vid:
            continue
        v = db.get_project_file_version(int(vid))
        if not v:
            continue
        before = v.get("content") or ""
        after = rx.sub(body.new, before)
        if before == after:
            continue
        try:
            db.write_project_file_version(
                project_id=project_id,
                path=path,
                content=after,
                edit_summary=f"rename: {body.old} -> {body.new}",
                created_by="user",
                audit_actor=_actor(org_id),
                audit_kind="symbol_rename",
            )
            affected.append({"path": path, "occurrences": len(rx.findall(before))})
        except Exception:
            continue
    return {"old": body.old, "new": body.new, "files": affected, "file_count": len(affected)}


# ---------- Auto CHANGES.md ----------
@router.post("/{project_id}/changes/append")
def append_change_log(project_id: int, org_id: int, line: str):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    path = "/CHANGES.md"
    existing = db.get_project_file(project_id=project_id, path=path)
    current = ""
    if existing and existing.get("current_version_id"):
        v = db.get_project_file_version(int(existing["current_version_id"]))
        current = (v or {}).get("content") or ""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    new_content = current + f"- {ts} · {line}\n"
    db.write_project_file_version(
        project_id=project_id,
        path=path,
        content=new_content,
        edit_summary="append change log entry",
        kind="doc",
        mime="text/markdown",
        pinned=True,
        created_by="user",
        audit_actor=_actor(org_id),
    )
    return {"ok": True}


# ---------- Pre-commit hook chain config ----------
class PrecommitChainSet(BaseModel):
    chain: list[str] = Field(default_factory=list)  # e.g. ["lint","format","typecheck"]


@router.put("/{project_id}/precommit-chain")
def set_precommit_chain(project_id: int, org_id: int, body: PrecommitChainSet):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    db.update_project(project_id, {"precommit_chain": body.chain})
    return {"chain": body.chain}


@router.get("/{project_id}/precommit-chain")
def get_precommit_chain(project_id: int, org_id: int):
    _require_enabled()
    db = NocodbClient()
    project = _require_project(db, project_id, org_id)
    return {"chain": project.get("precommit_chain") or []}


# ---------- Hot-reload conventions ----------
@router.post("/{project_id}/cache/drop")
def drop_project_cache(project_id: int, org_id: int):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    # Caches are in-process; emit an audit event so any hot-reload SSE consumer can react.
    db.add_project_audit_event(project_id, _actor(org_id), "cache_drop", {})
    return {"ok": True}


# ---------- Diff queue / staged changes ----------
@router.get("/{project_id}/staged")
def list_staged(project_id: int, org_id: int):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    rows = db.list_pending_changes(project_id)
    out = []
    for r in rows:
        v = db.get_project_file_version(int(r.get("version_id") or 0)) if r.get("version_id") else None
        out.append({
            "id": r.get("Id"),
            "file_id": r.get("file_id"),
            "version_id": r.get("version_id"),
            "version": (v or {}).get("version"),
            "edit_summary": (v or {}).get("edit_summary"),
            "conversation_id": r.get("conversation_id"),
        })
    return {"pending": out}


class StagedDecision(BaseModel):
    accept: bool


@router.post("/{project_id}/staged/{pending_id}")
def resolve_staged(project_id: int, pending_id: int, org_id: int, body: StagedDecision):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    rows = [r for r in db.list_pending_changes(project_id) if int(r.get("Id") or 0) == pending_id]
    if not rows:
        raise HTTPException(status_code=404, detail="pending change not found")
    pend = rows[0]
    if body.accept:
        # Advance file pointer to staged version.
        db._patch("project_files", int(pend["file_id"]), {
            "Id": int(pend["file_id"]),
            "current_version_id": int(pend["version_id"]),
        })
    db.resolve_pending_change(pending_id, "accepted" if body.accept else "discarded")
    return {"ok": True, "status": "accepted" if body.accept else "discarded"}


# ---------- Per-conversation file scope ----------
class ConversationScopeSet(BaseModel):
    scope_paths: list[str]


@router.put("/{project_id}/conversations/{conversation_id}/scope")
def set_conversation_scope(project_id: int, conversation_id: int, org_id: int, body: ConversationScopeSet):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    convo = db.get_code_conversation(conversation_id, org_id=org_id)
    if not convo or int(convo.get("project_id") or 0) != int(project_id):
        raise HTTPException(status_code=400, detail="conversation not in project")
    db.update_code_conversation(conversation_id, {"scope_paths": body.scope_paths})
    return {"scope_paths": body.scope_paths}


# ---------- Per-file model override ----------
class PerFileModelSet(BaseModel):
    path: str
    preferred_model: str | None


@router.put("/{project_id}/fs/file/preferred-model")
def set_per_file_model(project_id: int, org_id: int, body: PerFileModelSet):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    path = normalize_project_path(body.path)
    file_row = db.get_project_file(project_id=project_id, path=path)
    if not file_row:
        raise HTTPException(status_code=404, detail="file not found")
    db._patch("project_files", int(file_row["Id"]), {
        "Id": int(file_row["Id"]),
        "preferred_model": body.preferred_model or "",
    })
    db.add_project_audit_event(project_id, _actor(org_id), "file_preferred_model", {"path": path, "preferred_model": body.preferred_model})
    return {"ok": True}


# ---------- Watermark toggle ----------
class WatermarkSet(BaseModel):
    path: str
    watermark: bool


@router.put("/{project_id}/fs/file/watermark")
def set_watermark(project_id: int, org_id: int, body: WatermarkSet):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    path = normalize_project_path(body.path)
    file_row = db.get_project_file(project_id=project_id, path=path)
    if not file_row:
        raise HTTPException(status_code=404, detail="file not found")
    db._patch("project_files", int(file_row["Id"]), {"Id": int(file_row["Id"]), "watermark": 1 if body.watermark else 0})
    return {"ok": True}


# ---------- Change feed (24h) ----------
@router.get("/{project_id}/feed")
def project_feed(project_id: int, org_id: int, hours: int = 24, limit: int = 200):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, min(hours, 24 * 30)))
    events = db.list_project_audit_events(project_id=project_id, limit=max(1, min(limit, 1000)))
    out: list[dict] = []
    for e in events:
        try:
            ts = datetime.fromisoformat(str(e.get("CreatedAt") or "").replace("Z", "+00:00"))
        except Exception:
            continue
        if ts < cutoff:
            continue
        out.append({"actor": e.get("actor"), "kind": e.get("kind"), "payload": e.get("payload"), "created_at": e.get("CreatedAt")})
    return {"events": out}


# ---------- Pre-flight checks ----------
@router.post("/{project_id}/preflight")
def preflight(project_id: int, org_id: int):
    """Run the configured precommit_chain across the project. Returns a per-check report."""
    _require_enabled()
    from infra.lint_runners import lint_file
    db = NocodbClient()
    project = _require_project(db, project_id, org_id)
    chain: list[str] = list(project.get("precommit_chain") or [])
    if not chain:
        return {"ok": True, "checks": [], "note": "no precommit_chain configured"}
    files = db.list_project_files(project_id=project_id)
    file_contents: list[tuple[str, str]] = []
    for fr in files:
        vid = fr.get("current_version_id")
        if not vid:
            continue
        v = db.get_project_file_version(int(vid))
        if v:
            file_contents.append((fr.get("path") or "", (v.get("content") or "")))

    checks: list[dict] = []
    failed = False
    for check in chain:
        if check == "lint":
            issues = []
            for p, c in file_contents:
                issues.extend([{"path": p, **i} for i in lint_file(p, c)])
            errors = [i for i in issues if i.get("severity") == "error"]
            if errors:
                failed = True
            checks.append({"check": check, "issues": issues, "passed": not errors})
        elif check == "format":
            issues = []
            for p, c in file_contents:
                if c and not c.endswith("\n"):
                    issues.append({"path": p, "rule": "no-final-newline"})
            checks.append({"check": check, "issues": issues, "passed": True})
        else:
            checks.append({"check": check, "passed": True, "note": "no runner registered"})
    return {"ok": not failed, "checks": checks}


# ---------- Spec-first scaffolding (parsed from a YAML/JSON file in the project) ----------
class ScaffoldFromSpecRequest(BaseModel):
    spec_path: str
    targets: list[str] | None = None


@router.post("/{project_id}/scaffold-from-spec")
def scaffold_from_spec(project_id: int, org_id: int, body: ScaffoldFromSpecRequest):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    spec_path = normalize_project_path(body.spec_path)
    file_row = db.get_project_file(project_id=project_id, path=spec_path)
    if not file_row or not file_row.get("current_version_id"):
        raise HTTPException(status_code=404, detail="spec file not found")
    v = db.get_project_file_version(int(file_row["current_version_id"]))
    spec_content = (v or {}).get("content") or ""
    written: list[str] = []
    targets = body.targets or []
    for tpath in targets:
        try:
            normalized = normalize_project_path(tpath)
        except ValueError:
            continue
        # Skeleton ties spec content into a header comment for traceability.
        scaffold = template_for(normalized) + f"\n// scaffolded from {spec_path}\n"
        try:
            db.write_project_file_version(
                project_id=project_id,
                path=normalized,
                content=scaffold,
                edit_summary=f"scaffold from {spec_path}",
                created_by="scaffold",
                audit_actor=_actor(org_id),
            )
            written.append(normalized)
        except Exception:
            continue
    return {"spec": spec_path, "written": written, "spec_size": len(spec_content)}


# ---------- Migration playbook ----------
class PlaybookCreate(BaseModel):
    goal: str
    steps: list[dict]


@router.post("/{project_id}/playbooks")
def create_playbook(project_id: int, org_id: int, body: PlaybookCreate):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    return {"playbook": db.create_playbook(project_id, body.goal, body.steps)}


@router.get("/{project_id}/playbooks")
def list_playbooks(project_id: int, org_id: int):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    return {"playbooks": db.list_playbooks(project_id)}


class PlaybookAdvance(BaseModel):
    current_step: int
    status: str = "active"  # active | done | aborted


@router.patch("/{project_id}/playbooks/{playbook_id}")
def update_playbook(project_id: int, playbook_id: int, org_id: int, body: PlaybookAdvance):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    return {"playbook": db.update_playbook(playbook_id, {"current_step": body.current_step, "status": body.status})}


# ---------- File comments ----------
class FileCommentCreate(BaseModel):
    path: str
    version: int
    anchor: dict
    body: str


@router.post("/{project_id}/fs/file/comments")
def add_file_comment(project_id: int, org_id: int, body: FileCommentCreate):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    file_row = db.get_project_file(project_id=project_id, path=normalize_project_path(body.path))
    if not file_row:
        raise HTTPException(status_code=404, detail="file not found")
    return {"comment": db.add_file_comment(project_id, int(file_row["Id"]), body.version, body.anchor, body.body, _actor(org_id))}


@router.get("/{project_id}/fs/file/comments")
def list_file_comments(project_id: int, org_id: int, path: str):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    file_row = db.get_project_file(project_id=project_id, path=normalize_project_path(path))
    if not file_row:
        return {"comments": []}
    return {"comments": db.list_file_comments(project_id, int(file_row["Id"]))}
