"""
Code repo executor — orient, read, write, search, and tree operations
against Gitea-hosted repositories via the hybrid indexer.

Operations dispatched by params["op"]:
  orient  — index repo if not cached, return manifest context string
  read    — fetch specific files from Gitea
  write   — create or update a file on a branch
  search  — semantic search across the repo index in ChromaDB
  tree    — return directory structure as formatted text
"""

from __future__ import annotations

import asyncio
import logging
import time

from tools.framework.contract import ToolName, ToolResult
from tools.framework.dispatcher import register_executor

_log = logging.getLogger("tools.code_repo")

# in-memory manifest cache: "{repo_name}:{branch}" -> RepoManifest
_manifest_cache: dict[str, object] = {}

_OUTPUT_CAP = 8000


def _cache_key(repo: str, branch: str) -> str:
    return f"{repo}:{branch}"


@register_executor(ToolName.CODE_REPO)
async def execute(params: dict, emit) -> ToolResult:
    op = str(params.get("op") or "").strip()
    repo = str(params.get("repo") or "").strip()

    if not op:
        return ToolResult(
            tool=ToolName.CODE_REPO, action_index=0, ok=False,
            data="code_repo: missing 'op' parameter",
        )
    if not repo:
        return ToolResult(
            tool=ToolName.CODE_REPO, action_index=0, ok=False,
            data="code_repo: missing 'repo' parameter",
        )

    branch = str(params.get("branch") or params.get("ref") or "main")
    org_id = params.get("_org_id") or 0
    t0 = time.time()

    try:
        if op == "orient":
            return await _orient(repo, branch, org_id, t0)
        elif op == "read":
            return await _read(repo, branch, params, t0)
        elif op == "write":
            return await _write(repo, branch, params, t0)
        elif op == "search":
            return await _search(repo, params, org_id, t0)
        elif op == "tree":
            return await _tree(repo, branch, t0)
        else:
            return ToolResult(
                tool=ToolName.CODE_REPO, action_index=0, ok=False,
                data=f"code_repo: unknown op '{op}' (expected orient|read|write|search|tree)",
                elapsed_s=round(time.time() - t0, 2),
            )
    except Exception as e:
        _log.error("code_repo op=%s repo=%s failed: %s", op, repo, e, exc_info=True)
        return ToolResult(
            tool=ToolName.CODE_REPO, action_index=0, ok=False,
            data=f"code_repo error: {e}",
            elapsed_s=round(time.time() - t0, 2),
        )


async def _orient(repo: str, branch: str, org_id: int, t0: float) -> ToolResult:
    from tools.code.indexer import index_repo, RepoManifest

    key = _cache_key(repo, branch)
    manifest = _manifest_cache.get(key)

    if manifest is None:
        manifest = await index_repo(repo, branch=branch, org_id=org_id)
        if manifest is None:
            return ToolResult(
                tool=ToolName.CODE_REPO, action_index=0, ok=False,
                data=f"Failed to index repository '{repo}' (branch={branch}). Is the repo accessible in Gitea?",
                elapsed_s=round(time.time() - t0, 2),
            )
        _manifest_cache[key] = manifest

    data = manifest.to_context_string()
    if len(data) > _OUTPUT_CAP:
        data = data[:_OUTPUT_CAP] + "\n\n…[truncated for model context]"

    return ToolResult(
        tool=ToolName.CODE_REPO, action_index=0, ok=True,
        data=data,
        elapsed_s=round(time.time() - t0, 2),
    )


async def _read(repo: str, branch: str, params: dict, t0: float) -> ToolResult:
    from tools.code.gitea_client import GiteaClient

    files_param = params.get("files") or params.get("file") or params.get("path")
    if not files_param:
        return ToolResult(
            tool=ToolName.CODE_REPO, action_index=0, ok=False,
            data="code_repo read: missing 'files' or 'path' parameter",
            elapsed_s=round(time.time() - t0, 2),
        )

    # normalise to list
    if isinstance(files_param, str):
        file_paths = [files_param]
    elif isinstance(files_param, list):
        file_paths = [str(f) for f in files_param]
    else:
        file_paths = [str(files_param)]

    gitea = GiteaClient()
    parts: list[str] = []

    tasks = [gitea.get_file_content(repo, p, ref=branch) for p in file_paths]
    results = await asyncio.gather(*tasks)

    for path, content in zip(file_paths, results):
        if content is None:
            parts.append(f"--- {path} ---\n[file not found or inaccessible]")
        else:
            parts.append(f"--- {path} ---\n{content}")

    data = "\n\n".join(parts)
    if len(data) > _OUTPUT_CAP:
        data = data[:_OUTPUT_CAP] + "\n\n…[truncated for model context]"

    return ToolResult(
        tool=ToolName.CODE_REPO, action_index=0, ok=True,
        data=data,
        elapsed_s=round(time.time() - t0, 2),
    )


async def _write(repo: str, branch: str, params: dict, t0: float) -> ToolResult:
    from tools.code.gitea_client import GiteaClient

    path = str(params.get("file") or params.get("path") or "").strip()
    content = str(params.get("content") or "")
    message = str(params.get("message") or f"Update {path}")
    sha = str(params.get("sha") or "")

    if not path:
        return ToolResult(
            tool=ToolName.CODE_REPO, action_index=0, ok=False,
            data="code_repo write: missing 'path' parameter",
            elapsed_s=round(time.time() - t0, 2),
        )
    if branch == "main":
        return ToolResult(
            tool=ToolName.CODE_REPO, action_index=0, ok=False,
            data="code_repo write: refusing to write directly to 'main' — use a feature branch",
            elapsed_s=round(time.time() - t0, 2),
        )

    gitea = GiteaClient()

    # if no sha, try to get existing file sha for update
    if not sha:
        existing = await gitea.get_file_content(repo, path, ref=branch)
        if existing is not None:
            # need to fetch the file metadata for sha
            resp = await gitea._get(
                f"/api/v1/repos/{gitea.owner}/{repo}/contents/{path}",
                params={"ref": branch},
            )
            if resp:
                try:
                    sha = resp.json().get("sha", "")
                except Exception:
                    pass

    result = await gitea.update_file(repo, path, content, message, branch, sha=sha)
    if result is None:
        return ToolResult(
            tool=ToolName.CODE_REPO, action_index=0, ok=False,
            data=f"Failed to write {path} to {repo}:{branch}",
            elapsed_s=round(time.time() - t0, 2),
        )

    # invalidate cache
    key = _cache_key(repo, branch)
    _manifest_cache.pop(key, None)

    return ToolResult(
        tool=ToolName.CODE_REPO, action_index=0, ok=True,
        data=f"Written {path} to {repo}:{branch} — {message}",
        elapsed_s=round(time.time() - t0, 2),
    )


async def _search(repo: str, params: dict, org_id: int, t0: float) -> ToolResult:
    query = str(params.get("query") or "").strip()
    if not query:
        return ToolResult(
            tool=ToolName.CODE_REPO, action_index=0, ok=False,
            data="code_repo search: missing 'query' parameter",
            elapsed_s=round(time.time() - t0, 2),
        )

    if not org_id:
        return ToolResult(
            tool=ToolName.CODE_REPO, action_index=0, ok=False,
            data="code_repo search: missing org context",
            elapsed_s=round(time.time() - t0, 2),
        )

    try:
        from rag import retrieve
        block = await asyncio.to_thread(
            retrieve, query, int(org_id), f"repo_{repo}", 10, 5,
        )
    except Exception as e:
        _log.warning("code_repo search failed: %s", e)
        block = ""

    if not block or not block.strip():
        return ToolResult(
            tool=ToolName.CODE_REPO, action_index=0, ok=True,
            data=f"No results found for '{query}' in repo '{repo}'. Try using 'orient' first to index the repo.",
            elapsed_s=round(time.time() - t0, 2),
        )

    return ToolResult(
        tool=ToolName.CODE_REPO, action_index=0, ok=True,
        data=block,
        elapsed_s=round(time.time() - t0, 2),
    )


async def _tree(repo: str, branch: str, t0: float) -> ToolResult:
    from tools.code.gitea_client import GiteaClient

    gitea = GiteaClient()
    tree = await gitea.get_file_tree(repo, ref=branch)
    if not tree:
        return ToolResult(
            tool=ToolName.CODE_REPO, action_index=0, ok=False,
            data=f"Could not fetch file tree for '{repo}' (branch={branch})",
            elapsed_s=round(time.time() - t0, 2),
        )

    # build nested structure as formatted text
    blobs = sorted([e for e in tree if e["type"] == "blob"], key=lambda x: x["path"])

    lines = [f"File tree: {repo} ({branch})", ""]
    prev_dir = ""
    for blob in blobs:
        d = "/".join(blob["path"].split("/")[:-1])
        name = blob["path"].split("/")[-1]
        if d != prev_dir:
            if d:
                lines.append(f"{d}/")
            prev_dir = d
        prefix = "  " if d else ""
        lines.append(f"{prefix}{name}")

    data = "\n".join(lines)
    if len(data) > _OUTPUT_CAP:
        data = data[:_OUTPUT_CAP] + "\n\n…[truncated]"

    return ToolResult(
        tool=ToolName.CODE_REPO, action_index=0, ok=True,
        data=data,
        elapsed_s=round(time.time() - t0, 2),
    )
