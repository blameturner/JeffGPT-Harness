from __future__ import annotations

import re
from dataclasses import dataclass

from infra.paths import normalize_project_path

_TOOL_BLOCK_RE = re.compile(r"```tool\s+(.*?)\n(.*?)(?:\n)?```", re.DOTALL | re.IGNORECASE)
_HEADER_KV_RE = re.compile(r"(\w+)=(\"[^\"]*\"|\S+)")


@dataclass
class ToolDirective:
    name: str
    path: str | None
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


def parse_tool_directives(text: str) -> list[ToolDirective]:
    directives: list[ToolDirective] = []
    for m in _TOOL_BLOCK_RE.finditer(text or ""):
        meta = _parse_header(m.group(1) or "")
        name = (meta.get("name") or "").strip().lower()
        if not name:
            continue
        raw_path = (meta.get("path") or "").strip()
        path = None
        if raw_path:
            try:
                path = normalize_project_path(raw_path)
            except ValueError:
                continue
        summary = (meta.get("summary") or "").strip()
        directives.append(ToolDirective(name=name, path=path, summary=summary, content=m.group(2) or ""))
    return directives


def apply_tool_directives(
    db,
    project_id: int,
    response_text: str,
    conversation_id: int | None = None,
    assistant_message_id: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Apply interactive fs tool directives and return (changes, tool_events)."""
    changes: list[dict] = []
    tool_events: list[dict] = []

    for d in parse_tool_directives(response_text):
        if d.name == "fs_list":
            paths = [f.get("path") for f in db.list_project_files(project_id=project_id)]
            tool_events.append({"tool": d.name, "ok": True, "data": {"paths": paths}})
            continue

        if d.name == "fs_read":
            if not d.path:
                tool_events.append({"tool": d.name, "ok": False, "data": "path required"})
                continue
            row = db.get_project_file(project_id=project_id, path=d.path)
            if not row or not row.get("current_version_id"):
                tool_events.append({"tool": d.name, "ok": False, "data": "file not found"})
                continue
            ver = db.get_project_file_version(int(row["current_version_id"]))
            tool_events.append(
                {
                    "tool": d.name,
                    "ok": True,
                    "data": {
                        "path": d.path,
                        "version": (ver or {}).get("version"),
                        "content": (ver or {}).get("content") or "",
                    },
                }
            )
            continue

        if d.name == "fs_delete":
            if not d.path:
                tool_events.append({"tool": d.name, "ok": False, "data": "path required"})
                continue
            try:
                db.archive_project_file(
                    project_id,
                    d.path,
                    audit_actor=f"agent:{conversation_id}" if conversation_id else "agent",
                )
                change = {"path": d.path, "mode": "delete", "changed": True}
                changes.append(change)
                tool_events.append({"tool": d.name, "ok": True, "data": change})
            except Exception as e:
                tool_events.append({"tool": d.name, "ok": False, "data": str(e)})
            continue

        if d.name == "fs_write":
            if not d.path:
                tool_events.append({"tool": d.name, "ok": False, "data": "path required"})
                continue
            try:
                file_row, version_row, changed = db.write_project_file_version(
                    project_id=project_id,
                    path=d.path,
                    content=d.content,
                    edit_summary=d.summary or "interactive fs write",
                    created_by=f"agent:{conversation_id}" if conversation_id else "agent",
                    conversation_id=conversation_id,
                    created_by_message_id=assistant_message_id,
                    audit_actor=f"agent:{conversation_id}" if conversation_id else "agent",
                )
                change = {
                    "path": d.path,
                    "mode": "write",
                    "changed": changed,
                    "file_id": file_row.get("Id"),
                    "version": version_row.get("version"),
                }
                changes.append(change)
                tool_events.append({"tool": d.name, "ok": True, "data": change})
            except PermissionError as e:
                tool_events.append(
                    {
                        "tool": d.name,
                        "ok": False,
                        "data": str(e),
                        "permission_required": True,
                        "path": d.path,
                    }
                )
            except Exception as e:
                tool_events.append({"tool": d.name, "ok": False, "data": str(e)})
            continue

        tool_events.append({"tool": d.name, "ok": False, "data": "unknown tool"})

    return changes, tool_events

