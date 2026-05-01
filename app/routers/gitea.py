from __future__ import annotations

import io
import logging
import re
import zipfile

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from infra.config import is_feature_enabled
from infra.gitea_client import GiteaClient, GiteaError
from infra.nocodb_client import NocodbClient
from infra.paths import normalize_project_path

router = APIRouter(prefix="/gitea", tags=["gitea"])
projects_gitea = APIRouter(prefix="/projects", tags=["gitea"])

_log = logging.getLogger("gitea-router")

DEFAULT_IGNORE = (
    "node_modules/",
    "__pycache__/",
    ".git/",
    ".venv/",
    "dist/",
    "build/",
    ".DS_Store",
    "*.pyc",
    "*.lock",
    ".env",
)


class GiteaConnectionUpsert(BaseModel):
    org_id: int
    base_url: str
    username: str
    access_token: str
    default_branch: str = "main"


class ImportFromGiteaRequest(BaseModel):
    org_id: int
    owner: str
    repo: str
    ref: str = "main"
    name: str
    ignore: list[str] | None = None


class CreateGiteaRepoRequest(BaseModel):
    owner: str
    owner_kind: str = "user"
    repo: str
    description: str = ""
    private: bool = True
    default_branch: str = "main"
    init_readme: bool = False


class PushToGiteaRequest(BaseModel):
    branch: str | None = None
    message: str = "Update from Jeff"
    paths: list[str] | None = None
    scope: str = "current"  # "current" | "per_file"
    force: bool = False


class PullDecisionsRequest(BaseModel):
    decisions: list[dict]
    set_synced_to: str


def _require_enabled() -> None:
    if not is_feature_enabled("code_v2"):
        raise HTTPException(status_code=404, detail="projects feature disabled")


def _client(db: NocodbClient, org_id: int) -> GiteaClient:
    conn = db.get_gitea_connection(org_id)
    if not conn or not conn.get("access_token"):
        raise HTTPException(status_code=400, detail="gitea connection not configured")
    return GiteaClient(
        base_url=conn.get("base_url") or "",
        token=conn.get("access_token") or "",
        username=conn.get("username") or "",
    )


def _redact(conn: dict | None) -> dict | None:
    if not conn:
        return None
    out = dict(conn)
    if out.get("access_token"):
        out["access_token"] = "***"
    return out


def _matches_ignore(path: str, patterns: list[str]) -> bool:
    p = path.lstrip("/")
    for pat in patterns:
        if pat.endswith("/") and p.startswith(pat):
            return True
        if pat.startswith("*.") and p.endswith(pat[1:]):
            return True
        if pat == p or pat in p.split("/"):
            return True
    return False


@router.get("/connection")
def get_gitea_connection(org_id: int):
    _require_enabled()
    db = NocodbClient()
    return {"connection": _redact(db.get_gitea_connection(org_id))}


@router.put("/connection")
def upsert_gitea_connection(body: GiteaConnectionUpsert):
    _require_enabled()
    db = NocodbClient()
    try:
        row = db.upsert_gitea_connection(
            org_id=body.org_id,
            base_url=body.base_url.rstrip("/"),
            username=body.username,
            access_token=body.access_token,
            default_branch=body.default_branch,
        )
        # Verify token by calling /user + /version. If the username supplied
        # disagrees with the token's actual login, surface that immediately.
        try:
            client = GiteaClient(base_url=body.base_url, token=body.access_token, username=body.username)
            info = client.verify_credentials()
            actual_login = info.get("login") or ""
            if body.username and actual_login and actual_login != body.username:
                raise HTTPException(
                    status_code=400,
                    detail=f"token belongs to '{actual_login}', not '{body.username}'",
                )
            db.mark_gitea_verified(body.org_id)
            return {
                "connection": _redact(row),
                "verified_as": actual_login,
                "is_admin": info.get("is_admin"),
                "server_version": info.get("server_version"),
            }
        except GiteaError as e:
            raise HTTPException(status_code=400, detail=f"gitea verify failed: {e}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/connection")
def delete_gitea_connection(org_id: int):
    _require_enabled()
    db = NocodbClient()
    ok = db.delete_gitea_connection(org_id)
    return {"ok": ok}


@router.post("/connection/test")
def test_gitea_connection(org_id: int):
    """Re-validate the stored token + reachability of the Gitea server."""
    _require_enabled()
    db = NocodbClient()
    client = _client(db, org_id)
    try:
        info = client.verify_credentials()
        db.mark_gitea_verified(org_id)
        return {"ok": True, **info}
    except GiteaError as e:
        return {"ok": False, "error": str(e), "status": e.status_code}


@router.get("/orgs")
def list_gitea_orgs(org_id: int):
    _require_enabled()
    db = NocodbClient()
    client = _client(db, org_id)
    try:
        return {"orgs": client.list_user_orgs()}
    except GiteaError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/repos")
def list_gitea_repos(org_id: int, limit: int = 50):
    _require_enabled()
    db = NocodbClient()
    client = _client(db, org_id)
    try:
        return {"repos": client.list_repos(limit=limit)}
    except GiteaError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/repos/{owner}/{repo}/contents")
def gitea_browse(org_id: int, owner: str, repo: str, path: str = "", ref: str = ""):
    _require_enabled()
    db = NocodbClient()
    client = _client(db, org_id)
    try:
        return {"items": client.list_repo_contents(owner, repo, path=path, ref=ref)}
    except GiteaError as e:
        raise HTTPException(status_code=502, detail=str(e))


@projects_gitea.post("/import-from-gitea")
def import_from_gitea(body: ImportFromGiteaRequest):
    _require_enabled()
    db = NocodbClient()
    client = _client(db, body.org_id)
    ignore = list(DEFAULT_IGNORE) + list(body.ignore or [])

    # Resolve the head sha first so every imported version can be marked as
    # already-on-remote. Without this, every file appears "ahead" right after
    # import and the user has to push to get back to in_sync.
    try:
        commits = client.list_commits(body.owner, body.repo, sha=body.ref, limit=1)
        head_sha = commits[0].get("sha") if commits else ""
    except GiteaError as e:
        raise HTTPException(status_code=502, detail=f"resolve head sha: {e}")

    try:
        zip_bytes = client.archive_zip(body.owner, body.repo, body.ref)
    except GiteaError as e:
        raise HTTPException(status_code=502, detail=str(e))

    slug = re.sub(r"[^a-z0-9]+", "-", body.name.lower()).strip("-")[:80] or "project"
    project = db.create_project(
        org_id=body.org_id,
        name=body.name,
        slug=slug,
        description=f"Imported from gitea {body.owner}/{body.repo}@{body.ref}",
        chroma_collection=f"project_{slug}",
    )
    project_id = int(project["Id"])
    written = 0
    skipped = 0
    actor = f"gitea:{body.owner}/{body.repo}@{body.ref}"
    pushed_versions: list[int] = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            # Gitea archive prefixes entries with `{repo}/`; strip it.
            parts = info.filename.split("/", 1)
            rel = parts[1] if len(parts) > 1 else parts[0]
            if not rel:
                continue
            if _matches_ignore(rel, ignore):
                skipped += 1
                continue
            try:
                raw = zf.read(info)
                content = raw.decode("utf-8", errors="replace")
            except Exception:
                skipped += 1
                continue
            if len(content.encode("utf-8")) > 100 * 1024:
                skipped += 1
                continue
            try:
                normalized = normalize_project_path("/" + rel)
            except ValueError:
                skipped += 1
                continue
            try:
                _file_row, version_row, _changed = db.write_project_file_version(
                    project_id=project_id,
                    path=normalized,
                    content=content,
                    edit_summary=f"import from gitea {body.owner}/{body.repo}@{body.ref}",
                    created_by=actor,
                    audit_actor=actor,
                    audit_kind="gitea_import",
                )
                # Mark the imported version as already-on-remote so the project
                # starts in_sync with the imported ref.
                if head_sha and version_row.get("Id") is not None:
                    db.mark_version_pushed(int(version_row["Id"]), head_sha)
                    pushed_versions.append(int(version_row["Id"]))
                written += 1
            except Exception:
                skipped += 1

    # Persist origin + sync state so subsequent /gitea/status returns a clean baseline.
    db.update_project(
        project_id,
        {"gitea_origin": f"{body.owner}/{body.repo}@{body.ref}"},
    )
    if head_sha:
        db.update_project_gitea_state(project_id, head_sha)

    db.add_project_audit_event(
        project_id, actor, "gitea_import_complete",
        {"head_sha": head_sha, "files": written, "skipped": skipped, "ref": body.ref},
    )
    return {
        "project_id": project_id,
        "written": written,
        "skipped": skipped,
        "head_sha": head_sha,
    }


@projects_gitea.post("/{project_id}/create-gitea-repo")
def create_gitea_repo(project_id: int, org_id: int, body: CreateGiteaRepoRequest):
    _require_enabled()
    db = NocodbClient()
    project = db.get_project(project_id, org_id=org_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    client = _client(db, org_id)
    try:
        repo = client.create_repo(
            owner=body.owner,
            owner_kind=body.owner_kind,
            repo=body.repo,
            description=body.description or project.get("description") or "",
            private=body.private,
            default_branch=body.default_branch,
            init_readme=body.init_readme,
        )
    except GiteaError as e:
        raise HTTPException(status_code=502, detail=str(e))

    origin = f"{body.owner}/{body.repo}@{body.default_branch}"
    db.update_project(project_id, {"gitea_origin": origin})
    return _initial_push(db, client, project_id, body.owner, body.repo, body.default_branch, repo)


def _initial_push(db: NocodbClient, client: GiteaClient, project_id: int, owner: str, repo: str, branch: str, repo_meta: dict) -> dict:
    pushed = 0
    last_sha = ""
    failed = 0
    for file_row in db.list_project_files(project_id=project_id):
        version_id = file_row.get("current_version_id")
        if not version_id:
            continue
        v = db.get_project_file_version(int(version_id))
        if not v:
            continue
        path = (file_row.get("path") or "").lstrip("/")
        try:
            res = client.put_file(
                owner=owner,
                repo=repo,
                path=path,
                content_text=v.get("content") or "",
                message=f"Initial push from Jeff project {project_id}",
                branch=branch,
            )
            commit_sha = ((res.get("commit") or {}).get("sha")) or ""
            db.mark_version_pushed(int(version_id), commit_sha)
            last_sha = commit_sha or last_sha
            pushed += 1
        except GiteaError:
            failed += 1
            continue
    if last_sha:
        db.update_project_gitea_state(project_id, last_sha)
    db.add_project_audit_event(
        project_id, f"gitea:create:{owner}/{repo}", "gitea_create_repo",
        {"branch": branch, "pushed": pushed, "failed": failed, "head_sha": last_sha},
    )
    return {"repo": repo_meta, "pushed": pushed, "failed": failed, "head_sha": last_sha}


@projects_gitea.post("/{project_id}/push-to-gitea")
def push_to_gitea(project_id: int, org_id: int, body: PushToGiteaRequest):
    _require_enabled()
    db = NocodbClient()
    project = db.get_project(project_id, org_id=org_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    origin = (project.get("gitea_origin") or "").strip()
    if not origin:
        raise HTTPException(status_code=400, detail="project has no gitea_origin; set one or import-from-gitea first")
    m = re.match(r"^([^/]+)/([^@]+)@(.+)$", origin)
    if not m:
        raise HTTPException(status_code=400, detail=f"invalid gitea_origin: {origin}")
    owner, repo, default_ref = m.group(1), m.group(2), m.group(3)
    branch = body.branch or default_ref
    client = _client(db, org_id)

    # Divergence gate: if the remote has commits we haven't synced and the
    # caller didn't pass force=True, return 409 so the UI can prompt for pull.
    if not body.force:
        try:
            commits = client.list_commits(owner, repo, sha=branch, limit=10)
            last_synced = (project.get("gitea_last_synced_sha") or "").strip()
            behind = 0
            remote_head = ""
            for c in commits:
                if not remote_head:
                    remote_head = c.get("sha") or ""
                if c.get("sha") == last_synced:
                    break
                behind += 1
            if behind > 0 and last_synced:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "reason": "remote_diverged",
                        "remote_head_sha": remote_head,
                        "behind_count": behind,
                        "hint": "Pull first or pass force=true to overwrite remote.",
                    },
                )
        except (GiteaError, requests.RequestException):
            # Treat reachability failures as advisory only — let the push attempt
            # surface the real error per-file below.
            pass

    files = db.list_project_files(project_id=project_id)
    requested_paths = set(body.paths) if body.paths else None
    candidates: list[tuple[str, dict, dict]] = []
    for file_row in files:
        version_id = file_row.get("current_version_id")
        if not version_id:
            continue
        v = db.get_project_file_version(int(version_id))
        if not v:
            continue
        path = file_row.get("path") or ""
        if requested_paths is not None and path not in requested_paths:
            continue
        # When the caller didn't explicitly select paths or force, skip files
        # whose current version has already been pushed in a prior push. This
        # avoids creating noise commits for unchanged files on every iteration.
        if requested_paths is None and not body.force and v.get("pushed_to_sha"):
            continue
        candidates.append((path, file_row, v))

    # Ensure the target branch exists; create it from the repo's default if missing.
    try:
        if not client.branch_exists(owner, repo, branch):
            repo_meta = client.get_repo(owner, repo)
            base_branch = repo_meta.get("default_branch") or "main"
            base = client.get_branch(owner, repo, base_branch)
            base_sha = ((base or {}).get("commit") or {}).get("id") or ""
            if base_sha:
                client._post(
                    f"/repos/{owner}/{repo}/branches",
                    {"new_branch_name": branch, "old_branch_name": base_branch},
                )
    except (GiteaError, Exception):
        # If branch ensure fails (e.g. brand-new empty repo), Gitea will create
        # the branch from the first commit pushed below. Continue.
        pass

    pushed = 0
    last_sha = ""
    failures: list[dict] = []
    for path, _file_row, v in candidates:
        # Always look up the remote sha. `force` only governs the higher-level
        # "remote_diverged" gate (handled in /gitea/status before push) — at the
        # per-file PUT level we always need the current sha for an update.
        try:
            _, sha = client.get_file_content(owner, repo, path.lstrip("/"), ref=branch)
        except GiteaError:
            sha = ""
        try:
            res = client.put_file(
                owner=owner,
                repo=repo,
                path=path.lstrip("/"),
                content_text=v.get("content") or "",
                message=body.message,
                branch=branch,
                sha=sha,
            )
            commit_sha = ((res.get("commit") or {}).get("sha")) or ""
            db.mark_version_pushed(int(v["Id"]), commit_sha)
            last_sha = commit_sha or last_sha
            pushed += 1
        except GiteaError as e:
            failures.append({"path": path, "error": str(e), "status": e.status_code})

    if last_sha:
        db.update_project_gitea_state(project_id, last_sha)

    # Single summary audit row so the feed shows "X files pushed" rather than
    # one row per file (per-file rows live on each version's pushed_to_sha).
    db.add_project_audit_event(
        project_id, f"org:{org_id}", "gitea_push",
        {
            "branch": branch,
            "pushed": pushed,
            "failures": len(failures),
            "head_sha": last_sha,
            "force": bool(body.force),
            "scope": "selected" if requested_paths is not None else "ahead",
        },
    )
    return {
        "branch": branch,
        "pushed": pushed,
        "skipped": len(candidates) - pushed - len(failures),
        "failures": failures,
        "head_sha": last_sha,
    }


@projects_gitea.get("/{project_id}/gitea/status")
def gitea_status(project_id: int, org_id: int):
    _require_enabled()
    db = NocodbClient()
    project = db.get_project(project_id, org_id=org_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    origin = (project.get("gitea_origin") or "").strip()
    if not origin:
        return {"linked": False, "state": "unlinked"}
    m = re.match(r"^([^/]+)/([^@]+)@(.+)$", origin)
    if not m:
        return {"linked": False, "state": "unlinked", "error": f"invalid origin {origin}"}
    owner, repo, ref = m.group(1), m.group(2), m.group(3)

    ahead: list[str] = []
    for file_row in db.list_project_files(project_id=project_id):
        vid = file_row.get("current_version_id")
        if not vid:
            continue
        v = db.get_project_file_version(int(vid))
        if not v:
            continue
        if not v.get("pushed_to_sha"):
            ahead.append(file_row.get("path") or "")

    last_synced = (project.get("gitea_last_synced_sha") or "").strip()
    behind_count = 0
    remote_head_sha = ""
    try:
        client = _client(db, org_id)
        commits = client.list_commits(owner, repo, sha=ref, limit=10)
        if commits:
            remote_head_sha = commits[0].get("sha") or ""
        for c in commits:
            if c.get("sha") == last_synced:
                break
            behind_count += 1
    except (GiteaError, HTTPException):
        pass

    if not ahead and behind_count == 0:
        state = "in_sync"
    elif ahead and behind_count == 0:
        state = "ahead"
    elif not ahead and behind_count > 0:
        state = "behind"
    else:
        state = "diverged"

    return {
        "linked": True,
        "origin": origin,
        "ahead": ahead,
        "behind_count": behind_count,
        "remote_head_sha": remote_head_sha,
        "last_synced_sha": last_synced,
        "last_synced_at": project.get("gitea_last_synced_at"),
        "state": state,
    }


@projects_gitea.get("/{project_id}/gitea/pull/preview")
def gitea_pull_preview(project_id: int, org_id: int, max_files: int = 500):
    """Build a per-file diff between the project workspace and remote @ ref.

    Uses Gitea's `git/trees/{sha}?recursive=true` to fetch the entire tree in a
    single request (huge perf win vs walking `contents/` per directory). Then
    fetches file content only when local content differs in size or is missing.
    """
    _require_enabled()
    import difflib

    db = NocodbClient()
    project = db.get_project(project_id, org_id=org_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    origin = (project.get("gitea_origin") or "").strip()
    m = re.match(r"^([^/]+)/([^@]+)@(.+)$", origin)
    if not m:
        raise HTTPException(status_code=400, detail="project has no gitea_origin")
    owner, repo, ref = m.group(1), m.group(2), m.group(3)
    client = _client(db, org_id)

    try:
        commits = client.list_commits(owner, repo, sha=ref, limit=1)
        remote_head = commits[0].get("sha") if commits else ""
    except GiteaError as e:
        raise HTTPException(status_code=502, detail=str(e))
    if not remote_head:
        raise HTTPException(status_code=404, detail=f"ref '{ref}' has no commits")

    try:
        tree = client.git_tree_recursive(owner, repo, remote_head)
    except GiteaError as e:
        raise HTTPException(status_code=502, detail=f"git_tree fetch failed: {e}")

    files_out: list[dict] = []
    files_seen: set[str] = set()

    for entry in tree[:max_files]:
        if entry.get("type") != "blob":
            continue
        rel = entry.get("path") or ""
        if not rel:
            continue
        local_path = "/" + rel
        files_seen.add(local_path)
        local_row = db.get_project_file(project_id=project_id, path=local_path)
        local_content = ""
        local_version = 0
        if local_row and local_row.get("current_version_id"):
            lv = db.get_project_file_version(int(local_row["current_version_id"]))
            local_content = (lv or {}).get("content") or ""
            local_version = (lv or {}).get("version") or 0

        # Skip the content fetch entirely when sizes match exactly and the
        # tree's blob sha equals what we previously pushed.
        remote_size = int(entry.get("size") or 0)
        blob_sha = entry.get("sha") or ""
        local_size = len(local_content.encode("utf-8"))
        if local_row and remote_size == local_size:
            # Likely identical; fetch to confirm only when sizes match.
            try:
                text, _ = client.get_file_content(owner, repo, rel, ref=remote_head)
            except GiteaError:
                continue
            if text == local_content:
                files_out.append({
                    "path": local_path, "state": "identical",
                    "remote_sha": blob_sha, "remote_size": remote_size,
                    "local_version": local_version, "diff": None,
                })
                continue
        else:
            try:
                text, _ = client.get_file_content(owner, repo, rel, ref=remote_head)
            except GiteaError:
                continue

        if local_content == text:
            state = "identical"
            unified = None
        elif not local_row:
            state = "remote_only"
            unified = "".join(difflib.unified_diff(
                "".splitlines(keepends=True),
                text.splitlines(keepends=True),
                fromfile=f"{local_path}@local", tofile=f"{local_path}@gitea",
            ))
        else:
            state = "both_modified"
            unified = "".join(difflib.unified_diff(
                local_content.splitlines(keepends=True),
                text.splitlines(keepends=True),
                fromfile=f"{local_path}@local", tofile=f"{local_path}@gitea",
            ))

        files_out.append({
            "path": local_path, "state": state,
            "remote_sha": blob_sha, "remote_size": len(text.encode("utf-8")),
            "local_version": local_version, "diff": unified,
        })

    # Local-only files (present locally, absent from remote tree).
    for fr in db.list_project_files(project_id=project_id):
        p = fr.get("path") or ""
        if not p or p in files_seen:
            continue
        files_out.append({"path": p, "state": "local_only", "diff": None})

    return {
        "remote_head_sha": remote_head,
        "ref": ref,
        "tree_truncated": len(tree) >= max_files,
        "files": files_out,
    }


@projects_gitea.post("/{project_id}/gitea/pull/apply")
def gitea_pull_apply(project_id: int, org_id: int, body: PullDecisionsRequest):
    _require_enabled()
    db = NocodbClient()
    project = db.get_project(project_id, org_id=org_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    origin = (project.get("gitea_origin") or "").strip()
    m = re.match(r"^([^/]+)/([^@]+)@(.+)$", origin)
    if not m:
        raise HTTPException(status_code=400, detail="project has no gitea_origin")
    owner, repo, ref = m.group(1), m.group(2), m.group(3)
    client = _client(db, org_id)

    actor = f"gitea:pull@{body.set_synced_to[:8]}"
    applied: list[dict] = []
    for d in body.decisions:
        path = d.get("path") or ""
        choice = (d.get("choice") or "").lower()
        try:
            normalize_project_path(path)
        except ValueError:
            continue
        if choice == "skip":
            db.add_project_audit_event(project_id, actor, "gitea_pull_skip", {"path": path})
            continue
        if choice == "ours":
            row = db.get_project_file(project_id, path)
            if row and row.get("current_version_id"):
                db.mark_version_pushed(int(row["current_version_id"]), "")
            applied.append({"path": path, "choice": choice})
            continue
        if choice in ("theirs", "theirs_into_new_branch"):
            try:
                text, _sha = client.get_file_content(owner, repo, path.lstrip("/"), ref=ref)
            except GiteaError:
                continue
            target_project = project_id
            if choice == "theirs_into_new_branch":
                # Create a sibling branch project and write there.
                slug = (project.get("slug") or "branch") + "-pull-" + body.set_synced_to[:6]
                branch_proj = db.create_project(
                    org_id=org_id,
                    name=(project.get("name") or "branch") + " (pull)",
                    slug=slug,
                    description=f"Pull-into-branch from {origin}",
                    chroma_collection=f"project_{slug}",
                )
                target_project = int(branch_proj["Id"])
                try:
                    db.update_project(target_project, {"parent_project_id": project_id})
                except Exception:
                    pass
            file_row, version_row, _ = db.write_project_file_version(
                project_id=target_project,
                path=path,
                content=text,
                edit_summary=f"gitea pull {origin}",
                created_by=actor,
                audit_actor=actor,
                audit_kind="gitea_pull",
            )
            if choice == "theirs":
                db.mark_version_pushed(int(version_row["Id"]), body.set_synced_to)
            applied.append({"path": path, "choice": choice, "project_id": target_project})

    db.update_project_gitea_state(project_id, body.set_synced_to)
    return {"applied": applied}
