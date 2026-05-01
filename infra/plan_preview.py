from __future__ import annotations

import re


_PATH_RE = re.compile(r"(/[-_./A-Za-z0-9]+)")
_KV_PATH_RE = re.compile(r"\bpath=([/][-_./A-Za-z0-9]+)")


def extract_plan_file_intents(plan_text: str, existing_paths: set[str]) -> list[dict]:
    intents: dict[str, dict] = {}
    for raw in (plan_text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        paths = _KV_PATH_RE.findall(line)
        if not paths:
            paths = _PATH_RE.findall(line)
        if not paths:
            continue

        lower = line.lower()
        for path in paths:
            action = "edit" if path in existing_paths else "create"
            if any(tok in lower for tok in (" delete ", " remove ", " archive ", "drop ")):
                action = "delete"
            entry = intents.get(path)
            if entry is None:
                intents[path] = {"path": path, "action": action}
            else:
                # delete dominates; otherwise keep first-seen action.
                if action == "delete":
                    entry["action"] = "delete"
    return sorted(intents.values(), key=lambda x: x["path"])


