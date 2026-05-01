"""Per-language new-file templates. Pure stdlib."""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "untitled"


def template_for(path: str, name_hint: str = "") -> str:
    base = os.path.basename(path)
    stem, ext = os.path.splitext(base)
    if not ext:
        return ""
    if ext == ".tsx":
        if stem and stem[0].isupper() and re.match(r"^[A-Za-z0-9]+$", stem):
            comp = stem
        else:
            comp = "".join(p[:1].upper() + p[1:] for p in re.split(r"[^A-Za-z0-9]+", stem) if p) or "Component"
        return (
            f"import React from 'react';\n\n"
            f"export interface {comp}Props {{}}\n\n"
            f"export function {comp}({{}}: {comp}Props) {{\n"
            f"  return <div>{comp}</div>;\n"
            f"}}\n"
        )
    if ext in (".js", ".jsx", ".ts"):
        return f"// {base}\nexport {{}};\n"
    if ext == ".py":
        return f'"""{stem}."""\n\n\nif __name__ == "__main__":\n    pass\n'
    if ext == ".md":
        title = name_hint or stem.replace("-", " ").replace("_", " ").title()
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return f"---\ntitle: {title}\ndate: {date}\n---\n\n# {title}\n\n"
    if ext == ".json":
        return "{\n  \n}\n"
    if ext in (".yaml", ".yml"):
        return "version: 1\n"
    if ext == ".css":
        return "/* {} */\n".format(base)
    if ext == ".html":
        return "<!doctype html>\n<html>\n  <head>\n    <meta charset=\"utf-8\">\n  </head>\n  <body>\n  </body>\n</html>\n"
    return ""


def adr_template(number: int, title: str) -> str:
    return (
        f"# ADR-{number:03d}: {title}\n\n"
        "## Status\nProposed\n\n"
        "## Context\n_What is the problem we're solving and why now?_\n\n"
        "## Decision\n_The choice in one sentence, then a paragraph._\n\n"
        "## Consequences\n_What this enables / forecloses; trade-offs._\n"
    )


def adr_path(number: int, title: str) -> str:
    return f"/decisions/{number:03d}-{_slugify(title)}.md"


def conventions_template() -> str:
    return (
        "# Project conventions\n\n"
        "_Authoritative project rules. Capped at 4KB and always pinned._\n\n"
        "## Style\n- \n\n## Architecture\n- \n\n## Tests\n- \n"
    )
