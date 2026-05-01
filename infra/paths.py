from __future__ import annotations


def normalize_project_path(path: str) -> str:
    p = (path or "").strip()
    if not p:
        raise ValueError("path is required")
    if "\x00" in p:
        raise ValueError("path contains NUL byte")
    if len(p) > 512:
        raise ValueError("path exceeds 512 characters")
    if not p.startswith("/"):
        raise ValueError("path must start with '/'")
    if any(ch in p for ch in (",", "~", "(", ")")):
        raise ValueError("path contains unsupported characters")

    parts = p.split("/")
    out: list[str] = []
    for part in parts:
        if not part or part == ".":
            continue
        if part == "..":
            raise ValueError("path cannot contain '..'")
        out.append(part)
    normalized = "/" + "/".join(out)
    if normalized == "/":
        raise ValueError("path must point to a file")
    return normalized

