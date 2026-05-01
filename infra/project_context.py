from __future__ import annotations

import json

PIN_PER_FILE_CHAR_CAP = 8 * 1024
PIN_TOTAL_CHAR_CAP = 30 * 1024
MANIFEST_LINE_LIMIT = 300


def coerce_retrieval_scope(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
        return [p.strip() for p in raw.split(",") if p.strip()]
    return []


def estimate_tokens(text: str) -> int:
    # Lightweight offline estimate for UI/debug; avoids tokenizer dependency.
    if not text:
        return 0
    return max(1, len(text) // 4)


def build_project_context_pack(
    db,
    project_id: int,
    *,
    pin_per_file_char_cap: int = PIN_PER_FILE_CHAR_CAP,
    pin_total_char_cap: int = PIN_TOTAL_CHAR_CAP,
    manifest_line_limit: int = MANIFEST_LINE_LIMIT,
) -> dict:
    all_rows = db.list_project_files(project_id)
    pinned_rows = [f for f in all_rows if bool(f.get("pinned")) and f.get("current_version_id")]

    pinned_chunks: list[str] = []
    notices: list[str] = []
    total_chars = 0
    truncated_files = 0

    for row in pinned_rows[:8]:
        version = db.get_project_file_version(int(row["current_version_id"]))
        if not version:
            continue
        content = version.get("content") or ""
        truncated = False
        if len(content) > pin_per_file_char_cap:
            content = content[:pin_per_file_char_cap]
            truncated = True

        remaining = pin_total_char_cap - total_chars
        if remaining <= 0:
            notices.append("Pinned context hit total budget; remaining pinned files omitted.")
            break
        if len(content) > remaining:
            content = content[:remaining]
            truncated = True

        total_chars += len(content)
        pinned_chunks.append(f"{row.get('path')} (v{version.get('version')}):\n{content}")
        if truncated:
            truncated_files += 1
            notices.append(f"{row.get('path')}: truncated for context budget")

    pinned_context = "\n\n".join(pinned_chunks) if pinned_chunks else "(none)"
    context_notice = "\n".join(notices)

    manifest_lines: list[str] = []
    for row in all_rows:
        path = row.get("path") or ""
        if not path:
            continue
        marker = " [pinned]" if bool(row.get("pinned")) else ""
        size = int(row.get("size_bytes") or 0)
        manifest_lines.append(
            f"{path} (v={row.get('current_version_id') or '-'}, {size}B, updated={row.get('UpdatedAt') or '-'}){marker}"
        )
        if len(manifest_lines) >= manifest_line_limit:
            manifest_lines.append("...manifest truncated...")
            break

    path_manifest = "\n".join(manifest_lines)

    glossary_terms: list[str] = []
    try:
        from infra.code_analysis import extract_glossary
        from infra.config import get_feature
        top_n = int(get_feature("code_v2", "glossary_top_n", 12) or 12)
        files_for_glossary: list[dict] = []
        for row in all_rows:
            if not bool(row.get("pinned")):
                continue
            if not row.get("current_version_id"):
                continue
            v = db.get_project_file_version(int(row["current_version_id"]))
            if not v:
                continue
            files_for_glossary.append({"content": (v.get("content") or "")[:8000]})
        if files_for_glossary:
            glossary_terms = [t["term"] for t in extract_glossary(files_for_glossary, top_n=top_n)]
    except Exception:
        glossary_terms = []

    return {
        "pinned_context": pinned_context,
        "path_manifest": path_manifest,
        "context_notice": context_notice,
        "pinned_file_count": len(pinned_rows),
        "truncated_files": truncated_files,
        "total_pinned_chars": total_chars,
        "manifest_count": len(manifest_lines),
        "glossary_terms": glossary_terms,
    }


def build_context_inspector_metadata(
    *,
    project_id: int,
    conversation_id: int,
    message_id: int,
    mode: str,
    style: str,
    interactive_fs: bool,
    retrieval_collections: list[str],
    system_prompt: str,
    history: list[dict],
    user_message: str,
    context_pack: dict,
) -> dict:
    history_text = "".join((m.get("content") or "") for m in history)
    total_chars = len(system_prompt) + len(history_text) + len(user_message or "")
    return {
        "project_id": project_id,
        "conversation_id": conversation_id,
        "message_id": message_id,
        "mode": mode,
        "style": style,
        "interactive_fs": interactive_fs,
        "retrieval_collections": retrieval_collections,
        "sections": {
            "system_prompt": system_prompt,
            "history_count": len(history),
            "history": history,
            "user_message": user_message,
        },
        "context_pack": {
            "pinned_file_count": context_pack.get("pinned_file_count"),
            "truncated_files": context_pack.get("truncated_files"),
            "total_pinned_chars": context_pack.get("total_pinned_chars"),
            "manifest_count": context_pack.get("manifest_count"),
        },
        "token_estimate": estimate_tokens(system_prompt) + estimate_tokens(history_text) + estimate_tokens(user_message or ""),
        "char_estimate": total_chars,
    }


def build_context_inspector_summary(metadata: dict) -> dict:
    sections = metadata.get("sections") or {}
    context_pack = metadata.get("context_pack") or {}
    retrieval = metadata.get("retrieval_collections") or []
    return {
        "project_id": metadata.get("project_id"),
        "conversation_id": metadata.get("conversation_id"),
        "message_id": metadata.get("message_id"),
        "mode": metadata.get("mode"),
        "style": metadata.get("style"),
        "interactive_fs": bool(metadata.get("interactive_fs")),
        "retrieval_collection_count": len(retrieval),
        "history_count": sections.get("history_count") or 0,
        "system_prompt_chars": len(sections.get("system_prompt") or ""),
        "token_estimate": int(metadata.get("token_estimate") or 0),
        "char_estimate": int(metadata.get("char_estimate") or 0),
        "pinned_file_count": int(context_pack.get("pinned_file_count") or 0),
        "truncated_files": int(context_pack.get("truncated_files") or 0),
        "manifest_count": int(context_pack.get("manifest_count") or 0),
    }


def find_query_snippet(text: str, query: str, radius: int = 120) -> str | None:
    if not text or not query:
        return None
    q = query.lower().strip()
    if not q:
        return None
    idx = text.lower().find(q)
    if idx < 0:
        return None
    start = max(0, idx - radius)
    end = min(len(text), idx + len(q) + radius)
    return text[start:end]




