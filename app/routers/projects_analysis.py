"""Project analysis endpoints: lint, symbols, import graph, complexity, doc
coverage, test discovery, dependency listing, glossary."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from infra.code_analysis import (
    cyclomatic_complexity,
    detect_tests,
    doc_coverage,
    extract_glossary,
    extract_imports,
    extract_symbols,
    parse_dep_files,
)
from infra.config import is_feature_enabled
from infra.lint_runners import lint_file
from infra.nocodb_client import NocodbClient
from infra.paths import normalize_project_path

router = APIRouter(prefix="/projects", tags=["projects-analysis"])


def _require_enabled() -> None:
    if not is_feature_enabled("code_v2"):
        raise HTTPException(status_code=404, detail="projects feature disabled")


def _require_project(db: NocodbClient, project_id: int, org_id: int) -> dict:
    project = db.get_project(project_id, org_id=org_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"project {project_id} not found")
    return project


def _materialize_files(db: NocodbClient, project_id: int) -> list[dict]:
    out: list[dict] = []
    for fr in db.list_project_files(project_id=project_id):
        vid = fr.get("current_version_id")
        if not vid:
            continue
        v = db.get_project_file_version(int(vid))
        out.append({
            "path": fr.get("path") or "",
            "file_id": int(fr["Id"]),
            "version": (v or {}).get("version"),
            "content": (v or {}).get("content") or "",
        })
    return out


# ---------- Lint ----------
@router.post("/{project_id}/lint")
def run_lint(project_id: int, org_id: int, path: str | None = None):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    files = _materialize_files(db, project_id)
    if path:
        normalized = normalize_project_path(path)
        files = [f for f in files if f["path"] == normalized]
    issues_total = 0
    out: list[dict] = []
    for f in files:
        issues = lint_file(f["path"], f["content"])
        if issues:
            db.add_lint_results(project_id, f["file_id"], int(f.get("version") or 0), issues)
            issues_total += len(issues)
            out.append({"path": f["path"], "issues": issues})
    return {"files": out, "issues_total": issues_total, "files_scanned": len(files)}


@router.get("/{project_id}/issues")
def list_issues(project_id: int, org_id: int, severity: str | None = None, limit: int = 1000):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    return {"issues": db.list_lint_issues(project_id, severity=severity, limit=limit)}


# ---------- Symbols ----------
@router.post("/{project_id}/symbols/reindex")
def reindex_symbols(project_id: int, org_id: int):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    files = _materialize_files(db, project_id)
    total = 0
    for f in files:
        syms = [s.__dict__ for s in extract_symbols(f["path"], f["content"])]
        total += db.replace_project_symbols(project_id, f["file_id"], syms)
    return {"indexed_files": len(files), "symbols": total}


@router.get("/{project_id}/symbols")
def list_symbols(project_id: int, org_id: int, q: str = ""):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    if not q:
        # Re-extract live to avoid requiring prior reindex.
        files = _materialize_files(db, project_id)
        out: list[dict] = []
        for f in files:
            for s in extract_symbols(f["path"], f["content"]):
                out.append(s.__dict__)
        return {"symbols": out}
    files = _materialize_files(db, project_id)
    hits: list[dict] = []
    for f in files:
        for s in extract_symbols(f["path"], f["content"]):
            if q.lower() in s.name.lower():
                hits.append(s.__dict__)
    return {"symbols": hits}


@router.get("/{project_id}/symbol/{name}/refs")
def symbol_refs(project_id: int, name: str, org_id: int):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    files = _materialize_files(db, project_id)
    refs: list[dict] = []
    needle = name
    for f in files:
        for i, line in enumerate(f["content"].splitlines(), start=1):
            if needle in line:
                refs.append({"path": f["path"], "line": i, "snippet": line.strip()[:200]})
    return {"name": name, "refs": refs}


# ---------- Import graph ----------
@router.get("/{project_id}/graph")
def import_graph(project_id: int, org_id: int):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    files = _materialize_files(db, project_id)
    by_path = {f["path"]: f for f in files}
    nodes = [{"data": {"id": f["path"], "label": f["path"].rsplit("/", 1)[-1]}} for f in files]
    edges: list[dict] = []
    for f in files:
        for imp in extract_imports(f["path"], f["content"]):
            # Best-effort: try to resolve relative imports to project paths.
            for candidate in (imp, imp + ".py", imp + ".ts", imp + ".tsx"):
                target = "/" + candidate.lstrip("/")
                if target in by_path:
                    edges.append({"data": {"source": f["path"], "target": target}})
                    break
    return {"elements": {"nodes": nodes, "edges": edges}}


# ---------- Complexity ----------
@router.get("/{project_id}/complexity")
def complexity(project_id: int, org_id: int, threshold: int = 10):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    files = _materialize_files(db, project_id)
    out = []
    for f in files:
        score = cyclomatic_complexity(f["path"], f["content"])
        out.append({"path": f["path"], "complexity": score, "over_threshold": score >= threshold})
    out.sort(key=lambda x: x["complexity"], reverse=True)
    return {"files": out}


# ---------- Doc coverage ----------
@router.get("/{project_id}/doc-coverage")
def doc_coverage_endpoint(project_id: int, org_id: int):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    files = _materialize_files(db, project_id)
    per_file = []
    total_symbols = 0
    documented = 0
    for f in files:
        d = doc_coverage(f["path"], f["content"])
        if d["total"]:
            per_file.append({"path": f["path"], **d})
            total_symbols += d["total"]
            documented += d["documented"]
    overall = (documented / total_symbols) if total_symbols else 1.0
    return {"files": per_file, "total_symbols": total_symbols, "documented": documented, "overall_coverage": overall}


# ---------- Test discovery ----------
@router.get("/{project_id}/tests")
def test_discovery(project_id: int, org_id: int):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    return detect_tests(_materialize_files(db, project_id))


# ---------- Dependencies ----------
@router.get("/{project_id}/dependencies")
def dependencies(project_id: int, org_id: int):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    return {"dependencies": parse_dep_files(_materialize_files(db, project_id))}


# ---------- Glossary ----------
@router.get("/{project_id}/glossary")
def glossary(project_id: int, org_id: int, top_n: int = 50):
    _require_enabled()
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    files = _materialize_files(db, project_id)
    return {"terms": extract_glossary(files, top_n=top_n)}


# ---------- Multi-file diff between two versions of the project ----------
@router.get("/{project_id}/diff/conversation/{conversation_id}")
def conversation_diff(project_id: int, conversation_id: int, org_id: int):
    """Aggregate every version produced by a conversation."""
    _require_enabled()
    import difflib
    db = NocodbClient()
    _require_project(db, project_id, org_id)
    out: list[dict] = []
    for fr in db.list_project_files(project_id=project_id, include_archived=True):
        versions = db.list_project_file_versions(int(fr["Id"]), limit=400)
        # Find the latest version produced by this conversation, plus its parent.
        for v in versions:
            if int(v.get("conversation_id") or 0) != int(conversation_id):
                continue
            parent = db.get_project_file_version(int(v["parent_version_id"])) if v.get("parent_version_id") else None
            before = (parent or {}).get("content") or ""
            after = v.get("content") or ""
            unified = "".join(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    after.splitlines(keepends=True),
                    fromfile=f"{fr.get('path')}@v{(parent or {}).get('version') or 0}",
                    tofile=f"{fr.get('path')}@v{v.get('version')}",
                )
            )
            out.append({
                "path": fr.get("path"),
                "from_version": (parent or {}).get("version") or 0,
                "to_version": v.get("version"),
                "unified": unified,
            })
            break
    return {"conversation_id": conversation_id, "files": out}
