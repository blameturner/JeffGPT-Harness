from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from infra.paths import normalize_project_path


_FILE_FENCE_RE = re.compile(r"```file\s+(.*?)\n(.*?)(?:\n)?```", re.DOTALL | re.IGNORECASE)
_HEADER_KV_RE = re.compile(r"(\w+)=(\"[^\"]*\"|\S+)")


@dataclass
class FileFence:
    path: str
    mode: str
    summary: str
    content: str


def _parse_header(header: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, raw in _HEADER_KV_RE.findall(header):
        value = raw.strip()
        if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        out[key.lower()] = value
    return out


def parse_file_fences(text: str) -> list[FileFence]:
    fences: list[FileFence] = []
    for match in _FILE_FENCE_RE.finditer(text or ""):
        meta = _parse_header(match.group(1) or "")
        path = meta.get("path")
        if not path:
            continue
        mode = (meta.get("mode") or "replace").lower()
        summary = meta.get("summary") or ""
        try:
            normalized_path = normalize_project_path(path)
        except ValueError:
            continue
        fences.append(FileFence(path=normalized_path, mode=mode, summary=summary, content=match.group(2)))
    return fences


def fence_key(fence: FileFence) -> str:
    digest = hashlib.sha256(fence.content.encode("utf-8")).hexdigest()
    return f"{fence.path}|{fence.mode}|{fence.summary}|{digest}"


def _apply_unified_patch(base: str, patch: str) -> str:
    src = base.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    lines = patch.splitlines(keepends=True)
    p = 0

    while p < len(lines):
        line = lines[p]
        if line.startswith("--- ") or line.startswith("+++ "):
            p += 1
            continue
        if not line.startswith("@@"):
            p += 1
            continue

        m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line.strip())
        if not m:
            raise ValueError("invalid patch hunk header")
        old_start = int(m.group(1))

        while i < old_start - 1 and i < len(src):
            out.append(src[i])
            i += 1

        p += 1
        while p < len(lines) and not lines[p].startswith("@@"):
            hunk_line = lines[p]
            if hunk_line.startswith("\\"):
                p += 1
                continue
            marker = hunk_line[:1]
            payload = hunk_line[1:]
            if marker == " ":
                if i >= len(src) or src[i] != payload:
                    raise ValueError("patch context mismatch")
                out.append(src[i])
                i += 1
            elif marker == "-":
                if i >= len(src) or src[i] != payload:
                    raise ValueError("patch remove mismatch")
                i += 1
            elif marker == "+":
                out.append(payload)
            else:
                raise ValueError("invalid patch line")
            p += 1

    out.extend(src[i:])
    return "".join(out)


def apply_file_fences(
    db,
    project_id: int,
    response_text: str,
    conversation_id: int | None = None,
    assistant_message_id: int | None = None,
    seen_keys: set[str] | None = None,
) -> list[dict]:
    changes: list[dict] = []
    for fence in parse_file_fences(response_text):
        key = fence_key(fence)
        if seen_keys is not None and key in seen_keys:
            continue
        mode = fence.mode
        summary = fence.summary or "agent update"
        if mode == "delete":
            db.archive_project_file(
                project_id,
                fence.path,
                audit_actor=f"agent:{conversation_id}" if conversation_id else "agent",
            )
            changes.append({"path": fence.path, "mode": mode, "changed": True})
            if seen_keys is not None:
                seen_keys.add(key)
            continue

        if mode == "append":
            existing = db.get_project_file(project_id=project_id, path=fence.path)
            base = ""
            if existing and existing.get("current_version_id"):
                cur = db.get_project_file_version(int(existing["current_version_id"]))
                if cur:
                    base = cur.get("content") or ""
            content = base + fence.content
        elif mode == "patch":
            existing = db.get_project_file(project_id=project_id, path=fence.path)
            if not existing or not existing.get("current_version_id"):
                raise ValueError(f"patch target missing: {fence.path}")
            cur = db.get_project_file_version(int(existing["current_version_id"]))
            if not cur:
                raise ValueError(f"patch base version missing: {fence.path}")
            content = _apply_unified_patch(cur.get("content") or "", fence.content)
        else:
            content = fence.content

        # ADRs are auto-pinned per the project plan.
        auto_pin = True if fence.path.startswith("/decisions/") else None
        try:
            file_row, version_row, changed = db.write_project_file_version(
                project_id=project_id,
                path=fence.path,
                content=content,
                edit_summary=summary,
                kind="adr" if auto_pin else "",
                mime="text/markdown" if auto_pin else "",
                pinned=auto_pin,
                created_by=f"agent:{conversation_id}" if conversation_id else "agent",
                conversation_id=conversation_id,
                created_by_message_id=assistant_message_id,
                audit_actor=f"agent:{conversation_id}" if conversation_id else "agent",
            )
            changes.append(
                {
                    "path": fence.path,
                    "mode": mode,
                    "changed": changed,
                    "file_id": file_row.get("Id"),
                    "version": version_row.get("version"),
                }
            )
        except PermissionError as e:
            changes.append(
                {
                    "path": fence.path,
                    "mode": mode,
                    "changed": False,
                    "permission_required": True,
                    "reason": str(e),
                }
            )
        if seen_keys is not None:
            seen_keys.add(key)
    return changes

