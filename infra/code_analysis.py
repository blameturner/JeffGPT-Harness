from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class Symbol:
    path: str
    name: str
    kind: str  # function | class | method | const | export | heading
    line: int
    signature: str = ""


_PY_DEF_RE = re.compile(r"^(\s*)(async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", re.MULTILINE)
_PY_CLASS_RE = re.compile(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_PY_IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))", re.MULTILINE)

_JS_FUNC_RE = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(([^)]*)\)", re.MULTILINE)
_JS_CONST_RE = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", re.MULTILINE)
_JS_CLASS_RE = re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)", re.MULTILINE)
_JS_IMPORT_RE = re.compile(r"^\s*import\s+(?:.+?\s+from\s+)?['\"]([^'\"]+)['\"]", re.MULTILINE)

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def extract_symbols(path: str, content: str) -> list[Symbol]:
    out: list[Symbol] = []
    if path.endswith(".py"):
        for m in _PY_CLASS_RE.finditer(content):
            line = content[: m.start()].count("\n") + 1
            out.append(Symbol(path=path, name=m.group(1), kind="class", line=line))
        for m in _PY_DEF_RE.finditer(content):
            indent = m.group(1)
            name = m.group(3)
            args = (m.group(4) or "").strip()
            line = content[: m.start()].count("\n") + 1
            kind = "method" if indent else "function"
            out.append(Symbol(path=path, name=name, kind=kind, line=line, signature=f"({args})"))
    elif path.endswith((".js", ".jsx", ".ts", ".tsx")):
        for m in _JS_CLASS_RE.finditer(content):
            line = content[: m.start()].count("\n") + 1
            out.append(Symbol(path=path, name=m.group(1), kind="class", line=line))
        for m in _JS_FUNC_RE.finditer(content):
            line = content[: m.start()].count("\n") + 1
            out.append(Symbol(path=path, name=m.group(1), kind="function", line=line, signature=f"({m.group(2)})"))
        for m in _JS_CONST_RE.finditer(content):
            line = content[: m.start()].count("\n") + 1
            out.append(Symbol(path=path, name=m.group(1), kind="const", line=line))
    elif path.endswith(".md"):
        for m in _MD_HEADING_RE.finditer(content):
            line = content[: m.start()].count("\n") + 1
            level = len(m.group(1))
            out.append(Symbol(path=path, name=m.group(2).strip(), kind=f"heading{level}", line=line))
    return out


def extract_imports(path: str, content: str) -> list[str]:
    out: list[str] = []
    if path.endswith(".py"):
        for m in _PY_IMPORT_RE.finditer(content):
            mod = m.group(1) or m.group(2)
            if mod:
                out.append(mod)
    elif path.endswith((".js", ".jsx", ".ts", ".tsx")):
        for m in _JS_IMPORT_RE.finditer(content):
            out.append(m.group(1))
    return out


def cyclomatic_complexity(path: str, content: str) -> int:
    """Cheap stand-in for radon: count branching keywords + 1 per function."""
    if not content:
        return 0
    score = 0
    if path.endswith(".py"):
        score += len(re.findall(r"\b(?:if|elif|for|while|except|with|and|or)\b", content))
        score += len(re.findall(r"(?:^|\n)(?:async )?def ", content))
    elif path.endswith((".js", ".jsx", ".ts", ".tsx")):
        score += len(re.findall(r"\b(?:if|else if|for|while|catch|case|&&|\|\|)\b", content))
        score += content.count("function ")
        score += content.count("=> ")
    return score


def doc_coverage(path: str, content: str) -> dict:
    """Count public symbols with vs without docstrings/JSDoc."""
    if path.endswith(".py"):
        total = 0
        documented = 0
        for m in re.finditer(r"^(?:class|def|async def)\s+(\w+)[^\n]*:\s*\n([ \t]+(\"\"\"|'''))?", content, re.MULTILINE):
            if m.group(1).startswith("_"):
                continue
            total += 1
            if m.group(2):
                documented += 1
        return {"total": total, "documented": documented, "coverage": (documented / total) if total else 1.0}
    if path.endswith((".js", ".jsx", ".ts", ".tsx")):
        # Look for /** ... */ immediately preceding export
        total = 0
        documented = 0
        for m in re.finditer(r"(/\*\*[\s\S]*?\*/\s*)?\n?\s*export\s+(?:default\s+)?(?:async\s+)?(?:function|class|const|let|var)\s+(\w+)", content):
            total += 1
            if m.group(1):
                documented += 1
        return {"total": total, "documented": documented, "coverage": (documented / total) if total else 1.0}
    return {"total": 0, "documented": 0, "coverage": 1.0}


_TEST_PATH_RE = re.compile(r"(^|/)(?:tests?/|__tests__/|test_)|(_test|\.test|\.spec)\.(?:py|js|jsx|ts|tsx)$")
_TEST_DEF_RE = re.compile(r"\bdef\s+test_\w+|\bit\s*\(|\btest\s*\(")


def detect_tests(files: list[dict]) -> dict:
    """Returns {test_files, total_tests, files: [{path, count}]}"""
    out_files: list[dict] = []
    total = 0
    for f in files:
        path = f.get("path") or ""
        if not _TEST_PATH_RE.search(path):
            continue
        content = f.get("content") or ""
        count = len(_TEST_DEF_RE.findall(content))
        out_files.append({"path": path, "count": count})
        total += count
    return {"test_files": len(out_files), "total_tests": total, "files": out_files}


def parse_dep_files(files: list[dict]) -> list[dict]:
    """Read package.json / requirements.txt / pyproject.toml."""
    deps: list[dict] = []
    for f in files:
        path = f.get("path") or ""
        content = f.get("content") or ""
        if path.endswith("/package.json") or path == "/package.json":
            try:
                data = json.loads(content)
                for section in ("dependencies", "devDependencies", "peerDependencies"):
                    for name, ver in (data.get(section) or {}).items():
                        deps.append({"manager": "npm", "name": name, "version": ver, "scope": section})
            except Exception:
                pass
        elif path.endswith("/requirements.txt") or path == "/requirements.txt":
            for line in content.splitlines():
                line = line.split("#", 1)[0].strip()
                if not line or line.startswith("-"):
                    continue
                m = re.match(r"^([A-Za-z0-9_.\-\[\]]+)\s*([<>=!~].+)?", line)
                if m:
                    deps.append({"manager": "pip", "name": m.group(1), "version": (m.group(2) or "").strip(), "scope": "runtime"})
        elif path.endswith("/pyproject.toml") or path == "/pyproject.toml":
            in_deps = False
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("dependencies"):
                    in_deps = True
                    continue
                if in_deps:
                    if stripped.startswith("]"):
                        in_deps = False
                        continue
                    m = re.search(r"\"([A-Za-z0-9_.\-\[\]]+)([<>=!~][^\"]*)?\"", stripped)
                    if m:
                        deps.append({"manager": "pip", "name": m.group(1), "version": (m.group(2) or "").strip(), "scope": "runtime"})
    return deps


_STOPWORDS = {"The", "This", "That", "These", "Those", "An", "And", "But", "Or", "So", "It", "Its", "He", "She", "We", "You", "I"}


def extract_glossary(files: list[dict], top_n: int = 50) -> list[dict]:
    """Frequency-rank capitalised multi-word terms across docs / pinned files.

    Strips a leading stopword article ("The", "An", "This", …) so "The Auth
    Provider" and "Auth Provider" land in the same bucket.
    """
    counts: dict[str, int] = {}
    for f in files:
        content = f.get("content") or ""
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", content):
            term = m.group(1)
            words = term.split()
            if words and words[0] in _STOPWORDS and len(words) > 1:
                term = " ".join(words[1:])
            counts[term] = counts.get(term, 0) + 1
    items = [{"term": t, "count": c} for t, c in counts.items() if c > 1]
    items.sort(key=lambda x: x["count"], reverse=True)
    return items[:top_n]
