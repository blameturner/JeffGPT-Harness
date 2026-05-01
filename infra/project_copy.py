from __future__ import annotations

from infra.paths import normalize_project_path


def import_from_project(
    db,
    *,
    src_project_id: int,
    dst_project_id: int,
    paths: list[str],
    actor: str,
) -> dict:
    written = 0
    skipped = 0
    missing = 0

    for raw_path in paths:
        path = normalize_project_path(raw_path)
        src_file = db.get_project_file(src_project_id, path)
        if not src_file or not src_file.get("current_version_id"):
            missing += 1
            continue
        src_ver = db.get_project_file_version(int(src_file["current_version_id"]))
        if not src_ver:
            missing += 1
            continue

        _, _, changed = db.write_project_file_version(
            project_id=dst_project_id,
            path=path,
            content=src_ver.get("content") or "",
            edit_summary=f"import from project:{src_project_id}:{path}",
            kind=src_file.get("kind") or "code",
            mime=src_file.get("mime") or "text/plain",
            created_by=actor,
            audit_actor=actor,
            audit_kind="file_import_from",
        )
        if changed:
            written += 1
        else:
            skipped += 1

    return {"written": written, "skipped": skipped, "missing": missing}

