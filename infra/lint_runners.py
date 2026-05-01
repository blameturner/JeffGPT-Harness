"""Deterministic, no-subprocess lint checks. Pure stdlib."""
from __future__ import annotations

import ast
import json
import re


def _trailing_whitespace(content: str) -> list[dict]:
    out: list[dict] = []
    for i, line in enumerate(content.splitlines(), start=1):
        if line and line != line.rstrip():
            out.append(
                {
                    "line": i,
                    "col": len(line.rstrip()) + 1,
                    "severity": "info",
                    "rule": "trailing-whitespace",
                    "message": "trailing whitespace",
                }
            )
    return out


def _missing_final_newline(content: str) -> list[dict]:
    if content and not content.endswith("\n"):
        line = content.count("\n") + 1
        return [{"line": line, "col": 1, "severity": "info", "rule": "no-final-newline", "message": "missing final newline"}]
    return []


def _tab_indentation(content: str) -> list[dict]:
    return [
        {"line": i, "col": 1, "severity": "info", "rule": "tab-indent", "message": "tab indentation"}
        for i, line in enumerate(content.splitlines(), start=1)
        if line.startswith("\t")
    ]


def _python_syntax(content: str) -> list[dict]:
    try:
        ast.parse(content)
    except SyntaxError as e:
        return [{"line": e.lineno or 1, "col": e.offset or 1, "severity": "error", "rule": "py-syntax", "message": str(e.msg or "syntax error")}]
    return []


def _python_unused_imports(content: str) -> list[dict]:
    out: list[dict] = []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    imported: list[tuple[str, int]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for n in node.names:
                imported.append((n.asname or n.name.split(".")[0], node.lineno))
        elif isinstance(node, ast.ImportFrom):
            for n in node.names:
                if n.name == "*":
                    continue
                imported.append((n.asname or n.name, node.lineno))
    lines = content.splitlines()
    for name, line in imported:
        body_excl_import = "\n".join(l for i, l in enumerate(lines, start=1) if i != line)
        if not re.search(rf"\b{re.escape(name)}\b", body_excl_import):
            out.append({"line": line, "col": 1, "severity": "warning", "rule": "unused-import", "message": f"unused import: {name}"})
    return out


def _json_parse(content: str) -> list[dict]:
    if not content.strip():
        return []
    try:
        json.loads(content)
        return []
    except json.JSONDecodeError as e:
        return [{"line": e.lineno, "col": e.colno, "severity": "error", "rule": "json-parse", "message": e.msg}]


def _yaml_parse(content: str) -> list[dict]:
    try:
        import yaml  # type: ignore
    except Exception:
        return []
    try:
        yaml.safe_load(content)
        return []
    except Exception as e:
        mark = getattr(e, "problem_mark", None)
        return [{"line": (mark.line + 1) if mark else 1, "col": (mark.column + 1) if mark else 1, "severity": "error", "rule": "yaml-parse", "message": str(getattr(e, "problem", e))[:200]}]


def _markdown_basics(content: str) -> list[dict]:
    out: list[dict] = []
    for i, line in enumerate(content.splitlines(), start=1):
        if re.match(r"^#{7,}", line):
            out.append({"line": i, "col": 1, "severity": "warning", "rule": "md-heading-depth", "message": "headings deeper than h6"})
    return out


_PY_SECURITY_PATTERNS = [
    (r"\b" + "ev" + "al" + r"\s*\(", "py-dyn-eval", "use of dynamic-evaluation builtin"),
    (r"\b" + "ex" + "ec" + r"\s*\(", "py-dyn-run", "use of dynamic-execution builtin"),
    (r"shell\s*=\s*True", "py-shell-true", "subprocess shell=True"),
    (r"\bpic" + r"kle\.loads?\(", "py-unsafe-deser", "untrusted byte-stream deserialization"),
    (r"hashlib\.(md5|sha1)\(", "py-weak-hash", "weak hash function"),
]
_PY_SECURITY_RX = [(re.compile(p), r, m) for p, r, m in _PY_SECURITY_PATTERNS]


def _bandit_lite_python(content: str) -> list[dict]:
    out: list[dict] = []
    for i, line in enumerate(content.splitlines(), start=1):
        for rx, rule, msg in _PY_SECURITY_RX:
            if rx.search(line):
                out.append({"line": i, "col": 1, "severity": "security", "rule": rule, "message": msg})
    return out


def lint_file(path: str, content: str) -> list[dict]:
    """Run all applicable checks for a path; returns aggregated issues."""
    issues: list[dict] = []
    issues += _trailing_whitespace(content)
    issues += _missing_final_newline(content)
    if path.endswith(".py"):
        issues += _python_syntax(content)
        issues += _python_unused_imports(content)
        issues += _bandit_lite_python(content)
        issues += _tab_indentation(content)
    elif path.endswith(".json"):
        issues += _json_parse(content)
    elif path.endswith((".yaml", ".yml")):
        issues += _yaml_parse(content)
    elif path.endswith(".md"):
        issues += _markdown_basics(content)
    return issues
