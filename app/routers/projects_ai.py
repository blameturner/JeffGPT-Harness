"""AI-driven endpoints for projects: code review, file summary, README/FAQ
maintenance, smart paste classification, playbook generation, spec regen."""
from __future__ import annotations

import difflib
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from infra import ai_flows
from infra.config import is_feature_enabled
from infra.nocodb_client import NocodbClient
from infra.paths import normalize_project_path

router = APIRouter(prefix="/projects", tags=["projects-ai"])
_log = logging.getLogger("projects-ai")


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


# ------------- 1. AI code review on diff -------------
class ReviewRequest(BaseModel):
    from_snapshot: str | None = None
    to_snapshot: str | None = None
    conversation_id: int | None = None
    paths: list[str] | None = None


def _build_unified_diff_for_review(db: NocodbClient, project_id: int, body: ReviewRequest) -> str:
    """Build a multi-file unified diff based on the review scope."""
    parts: list[str] = []

    if body.conversation_id:
        for fr in db.list_project_files(project_id=project_id, include_archived=True):
            if body.paths and fr.get("path") not in set(body.paths):
                continue
            for v in db.list_project_file_versions(int(fr["Id"]), limit=400):
                if int(v.get("conversation_id") or 0) != int(body.conversation_id):
                    continue
                parent = db.get_project_file_version(int(v["parent_version_id"])) if v.get("parent_version_id") else None
                before = (parent or {}).get("content") or ""
                after = v.get("content") or ""
                parts.append("".join(difflib.unified_diff(
                    before.splitlines(keepends=True),
                    after.splitlines(keepends=True),
                    fromfile=f"{fr.get('path')}@v{(parent or {}).get('version') or 0}",
                    tofile=f"{fr.get('path')}@v{v.get('version')}",
                )))
                break
        return "\n".join(p for p in parts if p)

    # Snapshot diff
    if body.from_snapshot:
        snap = db.get_project_snapshot(project_id, body.from_snapshot)
        if not snap:
            raise HTTPException(status_code=404, detail="from_snapshot not found")
        snap_files = {f.get("path"): f for f in db.list_project_snapshot_files(int(snap["Id"]))}
        for fr in db.list_project_files(project_id=project_id):
            path = fr.get("path") or ""
            if body.paths and path not in set(body.paths):
                continue
            before = ""
            if path in snap_files and snap_files[path].get("version_id"):
                v = db.get_project_file_version(int(snap_files[path]["version_id"]))
                before = (v or {}).get("content") or ""
            after = ""
            if fr.get("current_version_id"):
                v = db.get_project_file_version(int(fr["current_version_id"]))
                after = (v or {}).get("content") or ""
            if before == after:
                continue
            parts.append("".join(difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"{path}@snapshot:{body.from_snapshot}",
                tofile=f"{path}@current",
            )))
        return "\n".join(p for p in parts if p)

    raise HTTPException(status_code=400, detail="provide conversation_id OR from_snapshot")


@router.post("/{project_id}/review")
def ai_review(project_id: int, org_id: int, body: ReviewRequest):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    diff = _build_unified_diff_for_review(db, project_id, body)
    if not diff.strip():
        return {"summary": "No diff to review.", "concerns": [], "suggested_followups": []}
    review = ai_flows.review_diff(diff)
    stored = db.create_review(project_id, {
        "conversation_id": body.conversation_id,
        "from_snapshot": body.from_snapshot,
        "summary": review.get("summary"),
        "concerns": review.get("concerns"),
        "suggested_followups": review.get("suggested_followups"),
    })
    db.add_project_audit_event(project_id, _actor(org_id), "ai_review", {
        "concerns": len(review.get("concerns") or []),
        "tokens": review.get("tokens"),
    })
    return {**review, "stored_id": (stored or {}).get("Id")}


# ------------- 2. File summary -------------
class FileSummaryRequest(BaseModel):
    path: str


@router.post("/{project_id}/fs/file/summary")
def ai_file_summary(project_id: int, org_id: int, body: FileSummaryRequest):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    path = normalize_project_path(body.path)
    file_row = db.get_project_file(project_id=project_id, path=path)
    if not file_row or not file_row.get("current_version_id"):
        raise HTTPException(status_code=404, detail="file not found")
    v = db.get_project_file_version(int(file_row["current_version_id"]))
    summary, tokens = ai_flows.summarise_file(path, (v or {}).get("content") or "")
    try:
        db._patch("project_files", int(file_row["Id"]), {"Id": int(file_row["Id"]), "summary": summary})
    except Exception:
        _log.debug("file summary persist skipped (column missing?)", exc_info=True)
    return {"path": path, "summary": summary, "tokens": tokens}


# ------------- 3. README maintenance -------------
@router.post("/{project_id}/readme/regenerate")
def ai_regenerate_readme(project_id: int, org_id: int):
    _require_enabled()
    db = NocodbClient()
    project = _require_project(db, project_id, org_id)
    files = db.list_project_files(project_id=project_id)
    manifest = [{"path": f.get("path"), "kind": f.get("kind"), "size_bytes": f.get("size_bytes")} for f in files]

    adrs: list[dict] = []
    for f in sorted(files, key=lambda r: r.get("path") or "", reverse=True):
        path = f.get("path") or ""
        if not path.startswith("/decisions/"):
            continue
        if not f.get("current_version_id"):
            continue
        v = db.get_project_file_version(int(f["current_version_id"]))
        content = (v or {}).get("content") or ""
        # title from first heading; decision = first sentence under "## Decision"
        import re as _re
        title_m = _re.search(r"^# (.+)$", content, _re.MULTILINE)
        decision_m = _re.search(r"## Decision\s*\n([^\n]+)", content)
        adrs.append({
            "title": title_m.group(1) if title_m else path,
            "decision": decision_m.group(1).strip() if decision_m else "",
        })
        if len(adrs) >= 10:
            break

    todos = sum(1 for f in files if f.get("current_version_id"))  # placeholder; open-work has the real count
    issues = db.list_lint_issues(project_id, limit=1)
    open_work = {"open_todos": todos, "issue_count": 1 if issues else 0}

    body, tokens = ai_flows.regenerate_readme(
        name=project.get("name") or "",
        description=project.get("description") or "",
        manifest=manifest,
        adrs=adrs,
        open_work=open_work,
    )
    if not body.strip():
        raise HTTPException(status_code=502, detail="model returned empty body")

    db.write_project_file_version(
        project_id=project_id,
        path="/README.md",
        content=body,
        edit_summary="auto-regenerate README",
        kind="doc",
        mime="text/markdown",
        pinned=True,
        created_by="ai:readme_maintain",
        audit_actor=_actor(org_id),
        audit_kind="readme_regenerate",
        allow_overwrite_locked=False,
    )
    return {"ok": True, "tokens": tokens, "bytes": len(body)}


# ------------- 4. FAQ maintenance -------------
class FAQAppendRequest(BaseModel):
    question: str
    answer: str


@router.post("/{project_id}/faq/append")
def ai_faq_append(project_id: int, org_id: int, body: FAQAppendRequest):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    existing_row = db.get_project_file(project_id=project_id, path="/FAQ.md")
    existing_content = ""
    if existing_row and existing_row.get("current_version_id"):
        v = db.get_project_file_version(int(existing_row["current_version_id"]))
        existing_content = (v or {}).get("content") or ""
    new_body, tokens = ai_flows.update_faq(existing_content, body.question, body.answer)
    if not new_body.strip():
        raise HTTPException(status_code=502, detail="model returned empty FAQ body")
    db.write_project_file_version(
        project_id=project_id,
        path="/FAQ.md",
        content=new_body,
        edit_summary=f"FAQ: append {body.question[:60]}",
        kind="doc",
        mime="text/markdown",
        pinned=True,
        created_by="ai:faq_maintain",
        audit_actor=_actor(org_id),
        audit_kind="faq_append",
    )
    return {"ok": True, "tokens": tokens}


# ------------- 5. Smart paste -------------
class SmartPasteRequest(BaseModel):
    text: str
    commit: bool = False  # if true, write to suggested_path


@router.post("/{project_id}/smart-paste")
def ai_smart_paste(project_id: int, org_id: int, body: SmartPasteRequest):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    cls = ai_flows.classify_paste(body.text)
    written = None
    if body.commit:
        try:
            path = normalize_project_path(cls.get("suggested_path") or "/notes/paste.md")
        except ValueError:
            path = "/notes/paste.md"
        file_row, version_row, _ = db.write_project_file_version(
            project_id=project_id,
            path=path,
            content=body.text,
            edit_summary=f"smart paste: {cls.get('reason') or cls.get('kind')}",
            kind=cls.get("kind") or "note",
            created_by="ai:smart_paste",
            audit_actor=_actor(org_id),
        )
        written = {"path": path, "file_id": file_row.get("Id"), "version": version_row.get("version")}
    return {"classification": cls, "written": written}


# ------------- 6. Playbook generation -------------
class PlaybookGenerateRequest(BaseModel):
    goal: str


@router.post("/{project_id}/playbooks/generate")
def ai_generate_playbook(project_id: int, org_id: int, body: PlaybookGenerateRequest):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    if not body.goal.strip():
        raise HTTPException(status_code=400, detail="goal is required")
    files = db.list_project_files(project_id=project_id)
    manifest = [{"path": f.get("path")} for f in files]
    pb = ai_flows.generate_playbook(body.goal, manifest)
    stored = db.create_playbook(project_id, body.goal, pb.get("steps") or [])
    db.add_project_audit_event(project_id, _actor(org_id), "playbook_generate", {
        "goal": body.goal[:200], "steps": len(pb.get("steps") or []), "tokens": pb.get("tokens"),
    })
    return {"playbook_id": (stored or {}).get("Id"), **pb}


# ------------- 7. Spec-first regeneration -------------
class SpecRegenRequest(BaseModel):
    spec_path: str
    targets: list[str]


@router.post("/{project_id}/scaffold-from-spec/regenerate")
def ai_regen_from_spec(project_id: int, org_id: int, body: SpecRegenRequest):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    spec_path = normalize_project_path(body.spec_path)
    spec_row = db.get_project_file(project_id=project_id, path=spec_path)
    if not spec_row or not spec_row.get("current_version_id"):
        raise HTTPException(status_code=404, detail="spec file not found")
    spec_v = db.get_project_file_version(int(spec_row["current_version_id"]))
    spec_content = (spec_v or {}).get("content") or ""

    out: list[dict] = []
    for tpath in body.targets:
        try:
            normalized = normalize_project_path(tpath)
        except ValueError:
            continue
        existing = db.get_project_file(project_id=project_id, path=normalized)
        current_content = ""
        if existing and existing.get("current_version_id"):
            v = db.get_project_file_version(int(existing["current_version_id"]))
            current_content = (v or {}).get("content") or ""
        new_content, tokens = ai_flows.regenerate_from_spec(spec_path, spec_content, normalized, current_content)
        if not new_content.strip():
            out.append({"path": normalized, "skipped": True, "tokens": tokens})
            continue
        try:
            _, version_row, changed = db.write_project_file_version(
                project_id=project_id,
                path=normalized,
                content=new_content,
                edit_summary=f"regen from {spec_path}",
                created_by="ai:spec_regen",
                audit_actor=_actor(org_id),
                audit_kind="spec_regen",
            )
            out.append({"path": normalized, "version": version_row.get("version"), "changed": changed, "tokens": tokens})
        except Exception as e:
            out.append({"path": normalized, "error": str(e), "tokens": tokens})
    return {"spec": spec_path, "files": out}
