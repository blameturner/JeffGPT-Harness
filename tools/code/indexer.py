"""
Hybrid repo indexer.

Generates:
1. Manifest — file tree with metadata (path, size, sha, extension)
2. Summaries — one-line description per file (via tool model, batched)
3. Structure — imports, exports, function signatures, class definitions
   (extracted via regex, no model call needed)

Full file content is NEVER stored in the index. Files are fetched
on demand from Gitea when the model needs them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time

import httpx
from pydantic import BaseModel, Field

from config import no_think_params
from workers.search.models import acquire_model

_log = logging.getLogger("tools.code.indexer")

# ------------------------------------------------------------------
# skip / include rules
# ------------------------------------------------------------------

_SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".next", ".nuxt", "dist", "build",
    ".turbo", ".cache", "coverage", ".venv", "venv", ".mypy_cache", ".ruff_cache",
    ".pytest_cache", ".tox",
}

_SKIP_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", ".DS_Store",
    "bun.lockb", "Cargo.lock", "poetry.lock", "Pipfile.lock",
}

_BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".woff", ".woff2",
    ".ttf", ".eot", ".mp3", ".mp4", ".wav", ".ogg", ".webm", ".webp",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".pdf", ".exe",
    ".dll", ".so", ".dylib", ".pyc", ".pyo", ".class", ".o", ".a",
}

_INDEXABLE_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".css", ".scss", ".html",
    ".svelte", ".vue", ".json", ".yaml", ".yml", ".toml", ".md",
    ".sql", ".sh", ".dockerfile", ".go", ".rs", ".rb", ".java",
    ".kt", ".swift", ".c", ".cpp", ".h", ".hpp", ".cs", ".lua",
}

_ALWAYS_INDEX = {
    "package.json", "tsconfig.json", "pyproject.toml", "docker-compose.yml",
    "docker-compose.yaml", "Dockerfile", ".env.example", "requirements.txt",
    "Cargo.toml", "go.mod", "Makefile", "CMakeLists.txt", "setup.py",
    "setup.cfg", ".gitignore", "README.md", "readme.md",
}

_MAX_FILE_SIZE = 100_000  # 100 KB


# ------------------------------------------------------------------
# data models
# ------------------------------------------------------------------

class RepoFile(BaseModel):
    path: str
    size: int
    sha: str
    extension: str
    summary: str | None = None
    structure: dict | None = None


class RepoManifest(BaseModel):
    repo_name: str
    branch: str
    files: dict[str, RepoFile] = Field(default_factory=dict)
    file_count: int = 0
    total_size: int = 0
    indexed_at: float = 0.0

    def to_context_string(self) -> str:
        """Format manifest as compact text for model system prompt injection."""
        lines = [f"Repository: {self.repo_name} ({self.branch})"]
        lines.append(f"Files: {self.file_count} | Total size: {self.total_size:,} bytes")
        lines.append("")

        # group by directory
        dirs: dict[str, list[RepoFile]] = {}
        for f in sorted(self.files.values(), key=lambda x: x.path):
            d = os.path.dirname(f.path) or "."
            dirs.setdefault(d, []).append(f)

        for d in sorted(dirs):
            if d != ".":
                lines.append(f"{d}/")
            for f in dirs[d]:
                name = os.path.basename(f.path)
                prefix = "  " if d != "." else ""
                if f.summary:
                    lines.append(f"{prefix}{name} — {f.summary}")
                else:
                    lines.append(f"{prefix}{name} ({f.size:,}b)")

        return "\n".join(lines)


# ------------------------------------------------------------------
# structure extraction (regex, no model call)
# ------------------------------------------------------------------

_PY_IMPORT = re.compile(r"^(?:from\s+\S+\s+)?import\s+.+", re.MULTILINE)
_PY_FUNC = re.compile(r"^(?:async\s+)?def\s+(\w+)\s*\(", re.MULTILINE)
_PY_CLASS = re.compile(r"^class\s+(\w+)\s*[\(:]", re.MULTILINE)

_JS_IMPORT = re.compile(r"^import\s+.+from\s+['\"]", re.MULTILINE)
_JS_EXPORT = re.compile(r"^export\s+(?:default\s+)?(?:const|function|class|let|var|type|interface|enum)\s+(\w+)", re.MULTILINE)
_JS_FUNC = re.compile(r"(?:^|\s)(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\()", re.MULTILINE)
_JS_CLASS = re.compile(r"^class\s+(\w+)", re.MULTILINE)


def _extract_structure(content: str, ext: str) -> dict:
    """Extract structural elements from source code via regex."""
    result: dict[str, list[str]] = {}

    if ext == ".py":
        result["imports"] = _PY_IMPORT.findall(content)[:20]
        result["functions"] = _PY_FUNC.findall(content)[:30]
        result["classes"] = _PY_CLASS.findall(content)[:20]
    elif ext in (".ts", ".tsx", ".js", ".jsx", ".svelte", ".vue"):
        result["imports"] = _JS_IMPORT.findall(content)[:20]
        exports = _JS_EXPORT.findall(content)[:20]
        result["exports"] = [e for e in exports if e]
        funcs = _JS_FUNC.findall(content)[:30]
        result["functions"] = [f[0] or f[1] for f in funcs if f[0] or f[1]]
        result["classes"] = _JS_CLASS.findall(content)[:20]

    # strip empty keys
    return {k: v for k, v in result.items() if v}


# ------------------------------------------------------------------
# filtering
# ------------------------------------------------------------------

def _should_index(path: str, size: int) -> bool:
    """Decide if a file should be included in the manifest."""
    parts = path.split("/")

    # skip if any path component is a skip dir
    for part in parts[:-1]:
        if part in _SKIP_DIRS:
            return False

    filename = parts[-1]
    if filename in _SKIP_FILES:
        return False

    if size > _MAX_FILE_SIZE:
        return False

    # always-index overrides extension check
    if filename.lower() in _ALWAYS_INDEX:
        return True

    ext = os.path.splitext(filename)[1].lower()
    if ext in _BINARY_EXTENSIONS:
        return False
    if ext in _INDEXABLE_EXTENSIONS:
        return True

    # skip unknown extensions
    return False


# ------------------------------------------------------------------
# summary generation (batched tool model call)
# ------------------------------------------------------------------

async def _summarise_batch(
    file_contents: dict[str, str],
) -> dict[str, str]:
    """
    Generate one-line summaries for a batch of files via the tool model.

    Returns {path: summary}. On failure, returns empty dict (caller skips).
    """
    if not file_contents:
        return {}

    prompt_parts = ["For each file below, output a JSON object mapping file path to a one-line summary (under 15 words). Output ONLY the JSON object.\n"]
    for path, content in file_contents.items():
        truncated = content[:2000]
        prompt_parts.append(f"--- {path} ---\n{truncated}\n")

    user_prompt = "\n".join(prompt_parts)

    try:
        with acquire_model("tool") as (tool_url, tool_model_id):
            if not tool_url:
                _log.debug("no tool model available for summaries — skipping batch")
                return {}

            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{tool_url}/v1/chat/completions",
                    json={
                        "model": tool_model_id,
                        "messages": [
                            {"role": "system", "content": "Output ONLY valid JSON. No markdown, no prose."},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 500,
                        **no_think_params(tool_model_id),
                    },
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]

        # strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            raw = raw.strip()

        # find JSON object
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            _log.debug("no JSON found in summary response")
            return {}

        summaries = json.loads(match.group())
        return {k: str(v)[:80] for k, v in summaries.items() if isinstance(v, str)}

    except Exception as e:
        _log.warning("summary batch failed: %s", e)
        return {}


# ------------------------------------------------------------------
# main indexing function
# ------------------------------------------------------------------

async def index_repo(
    repo_name: str,
    branch: str = "main",
    changed_files: list[str] | None = None,
    org_id: int = 0,
) -> RepoManifest | None:
    """
    Index a repository and return its manifest.

    If changed_files is provided, only reprocess those files (incremental).
    """
    from tools.code.gitea_client import GiteaClient

    gitea = GiteaClient()
    t0 = time.time()

    # 1. fetch file tree
    tree = await gitea.get_file_tree(repo_name, ref=branch)
    if not tree:
        _log.warning("index_repo  repo=%s branch=%s — empty tree", repo_name, branch)
        return None

    # 2. filter to indexable files
    blobs = [
        e for e in tree
        if e["type"] == "blob" and _should_index(e["path"], e.get("size", 0))
    ]

    _log.info(
        "index_repo  repo=%s branch=%s total_tree=%d indexable=%d",
        repo_name, branch, len(tree), len(blobs),
    )

    # 3. build base manifest
    files: dict[str, RepoFile] = {}
    for b in blobs:
        ext = os.path.splitext(b["path"])[1].lower()
        files[b["path"]] = RepoFile(
            path=b["path"],
            size=b.get("size", 0),
            sha=b.get("sha", ""),
            extension=ext,
        )

    # 4. determine which files need content fetching
    if changed_files is not None:
        fetch_paths = [p for p in changed_files if p in files]
    else:
        fetch_paths = list(files.keys())

    # 5. fetch content in batches for structure extraction + summaries
    batch_size = 20
    all_contents: dict[str, str] = {}

    for i in range(0, len(fetch_paths), batch_size):
        batch = fetch_paths[i : i + batch_size]
        tasks = [gitea.get_file_content(repo_name, p, ref=branch) for p in batch]
        results = await asyncio.gather(*tasks)
        for path, content in zip(batch, results):
            if content is not None:
                all_contents[path] = content

    _log.info("index_repo  fetched=%d files for analysis", len(all_contents))

    # 6. extract structure (regex, no model call)
    for path, content in all_contents.items():
        if path in files:
            structure = _extract_structure(content, files[path].extension)
            if structure:
                files[path].structure = structure

    # 7. generate summaries in batches via tool model
    summary_batch_size = 10
    summary_paths = list(all_contents.keys())

    for i in range(0, len(summary_paths), summary_batch_size):
        batch_paths = summary_paths[i : i + summary_batch_size]
        batch_contents = {p: all_contents[p] for p in batch_paths}
        summaries = await _summarise_batch(batch_contents)
        for path, summary in summaries.items():
            if path in files:
                files[path].summary = summary

    # 8. build manifest
    manifest = RepoManifest(
        repo_name=repo_name,
        branch=branch,
        files=files,
        file_count=len(files),
        total_size=sum(f.size for f in files.values()),
        indexed_at=time.time(),
    )

    # 9. store in ChromaDB for semantic search
    if org_id:
        try:
            from memory import remember
            context = manifest.to_context_string()
            await asyncio.to_thread(
                remember,
                context,
                {"repo": repo_name, "branch": branch, "type": "repo_manifest"},
                org_id,
                f"repo_{repo_name}",
            )
            _log.info("index_repo  stored manifest in ChromaDB collection=repo_%s", repo_name)
        except Exception as e:
            _log.warning("index_repo  ChromaDB store failed: %s", e)

    elapsed = round(time.time() - t0, 2)
    _log.info(
        "index_repo  repo=%s branch=%s files=%d size=%d elapsed=%.2fs",
        repo_name, branch, manifest.file_count, manifest.total_size, elapsed,
    )

    return manifest
