"""
Code agents API router — agent control, repo browsing, and webhook endpoints.
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from code_agents.config import (
    AgentConfig,
    Behaviour,
    BranchStrategy,
    Permissions,
    Schedule,
    Task,
)
from code_agents import runner

_log = logging.getLogger("main.code_agents")

router = APIRouter(prefix="/code-agents", tags=["code-agents"])


# ------------------------------------------------------------------
# request / response models
# ------------------------------------------------------------------

class AgentStartRequest(BaseModel):
    name: str
    org_id: int
    repo_name: str
    objective: str
    branch: str = "main"
    description: str = ""
    constraints: list[str] = Field(default_factory=list)
    context: str = ""
    focus_areas: list[str] = Field(default_factory=list)
    plan: str = ""
    permissions: dict = Field(default_factory=dict)
    behaviour: dict = Field(default_factory=dict)
    branch_strategy: dict = Field(default_factory=dict)
    schedule: dict = Field(default_factory=dict)


# ------------------------------------------------------------------
# agent control endpoints
# ------------------------------------------------------------------

@router.post("/start")
def start_agent(req: AgentStartRequest):
    """Start a code agent. Returns agent_id and job_id for streaming."""
    config = AgentConfig(
        name=req.name,
        description=req.description,
        org_id=req.org_id,
        repo_name=req.repo_name,
        branch=req.branch,
        permissions=Permissions(**req.permissions) if req.permissions else Permissions(),
        behaviour=Behaviour(**req.behaviour) if req.behaviour else Behaviour(),
        branch_strategy=BranchStrategy(**req.branch_strategy) if req.branch_strategy else BranchStrategy(),
        task=Task(
            objective=req.objective,
            constraints=req.constraints,
            context=req.context,
            focus_areas=req.focus_areas,
            plan=req.plan,
        ),
        schedule=Schedule(**req.schedule) if req.schedule else Schedule(),
    )

    agent_id, job_id = runner.start_agent(config)
    return {"agent_id": agent_id, "job_id": job_id}


@router.post("/{agent_id}/stop")
def stop_agent(agent_id: str):
    """Gracefully stop a running agent."""
    ok = runner.stop_agent(agent_id)
    if not ok:
        raise HTTPException(404, f"Agent {agent_id} not active")
    return {"ok": True, "agent_id": agent_id}


@router.get("/{agent_id}/state")
def get_agent_state(agent_id: str):
    """Get current state of an active agent."""
    state = runner.get_agent_state(agent_id)
    if state is None:
        # try NocoDB for completed agents
        try:
            from nocodb_client import NocodbClient
            from config import NOCODB_TABLE_CODE_AGENT_TASKS
            db = NocodbClient()
            result = db._get(
                NOCODB_TABLE_CODE_AGENT_TASKS,
                params={"where": f"(agent_id,eq,{agent_id})", "limit": 1},
            )
            rows = result.get("list", [])
            if rows:
                return rows[0]
        except Exception:
            pass
        raise HTTPException(404, f"Agent {agent_id} not found")
    return state


@router.get("/{agent_id}/thoughts")
def get_agent_thoughts(agent_id: str, type: str | None = None):
    """List thoughts for an agent, optionally filtered by type."""
    # check active first
    state = runner.get_agent_state(agent_id)
    if state is not None:
        entry = runner._active.get(agent_id)
        if entry:
            thoughts = entry["agent"].thoughts.active_thoughts
            if type:
                thoughts = [t for t in thoughts if t.type.value == type]
            return {
                "agent_id": agent_id,
                "thoughts": [t.model_dump() for t in thoughts],
                "count": len(thoughts),
            }

    # fall back to NocoDB
    try:
        from nocodb_client import NocodbClient
        from config import NOCODB_TABLE_CODE_AGENT_THOUGHTS
        db = NocodbClient()
        params = {
            "where": f"(agent_id,eq,{agent_id})",
            "sort": "created_at",
            "limit": 200,
        }
        if type:
            params["where"] += f"~and(type,eq,{type})"
        result = db._get(NOCODB_TABLE_CODE_AGENT_THOUGHTS, params=params)
        return {
            "agent_id": agent_id,
            "thoughts": result.get("list", []),
            "count": len(result.get("list", [])),
        }
    except Exception as e:
        _log.warning("get_thoughts failed: %s", e)
        raise HTTPException(404, f"Agent {agent_id} not found")


@router.get("/{agent_id}/proposal")
def get_agent_proposal(agent_id: str):
    """Get latest proposal for an agent."""
    try:
        from nocodb_client import NocodbClient
        from config import NOCODB_TABLE_CODE_AGENT_PROPOSALS
        db = NocodbClient()
        result = db._get(
            NOCODB_TABLE_CODE_AGENT_PROPOSALS,
            params={
                "where": f"(agent_id,eq,{agent_id})",
                "sort": "-CreatedAt",
                "limit": 1,
            },
        )
        rows = result.get("list", [])
        if not rows:
            raise HTTPException(404, f"No proposal found for agent {agent_id}")
        return rows[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("")
def list_agents(org_id: int | None = None):
    """List all agents — active in memory + completed from NocoDB."""
    active = runner.list_active_agents()

    completed = []
    try:
        from nocodb_client import NocodbClient
        from config import NOCODB_TABLE_CODE_AGENT_TASKS
        db = NocodbClient()
        params: dict = {"sort": "-CreatedAt", "limit": 50}
        if org_id:
            params["where"] = f"(org_id,eq,{org_id})"
        result = db._get(NOCODB_TABLE_CODE_AGENT_TASKS, params=params)
        completed = result.get("list", [])
    except Exception as e:
        _log.debug("list completed agents failed: %s", e)

    return {
        "active": active,
        "completed": completed,
    }


@router.post("/{agent_id}/approve-merge")
async def approve_merge(agent_id: str):
    """Approve and merge an agent's branch."""
    from tools.code.gitea_client import GiteaClient

    # find the agent's config from NocoDB
    try:
        from nocodb_client import NocodbClient
        from config import NOCODB_TABLE_CODE_AGENT_TASKS
        import json as _json
        db = NocodbClient()
        result = db._get(
            NOCODB_TABLE_CODE_AGENT_TASKS,
            params={"where": f"(agent_id,eq,{agent_id})", "limit": 1},
        )
        rows = result.get("list", [])
        if not rows:
            raise HTTPException(404, f"Agent {agent_id} not found")

        state_json = rows[0].get("state_json", "{}")
        state = _json.loads(state_json) if isinstance(state_json, str) else state_json
        branch = state.get("current_branch", "")
        config_json = rows[0].get("config_json", "{}")
        config = _json.loads(config_json) if isinstance(config_json, str) else config_json
        repo_name = config.get("repo_name") or rows[0].get("repo_name", "")
        base_branch = config.get("branch", "main")

        if not branch:
            raise HTTPException(400, "Agent has no branch to merge")

        gitea = GiteaClient()

        # create PR and merge
        pr = await gitea.create_pr(
            repo_name,
            title=f"[agent] {rows[0].get('description', 'Agent changes')[:60]}",
            head=branch,
            base=base_branch,
        )
        if not pr:
            raise HTTPException(500, "Failed to create PR")

        pr_number = pr.get("number")
        merge_ok = await gitea.merge_pr(repo_name, pr_number)
        if not merge_ok:
            raise HTTPException(500, f"Failed to merge PR #{pr_number}")

        return {"ok": True, "pr_number": pr_number, "merged": True}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/{agent_id}")
def delete_agent(agent_id: str):
    """Remove agent session from NocoDB."""
    # stop if active
    runner.stop_agent(agent_id)

    try:
        from nocodb_client import NocodbClient
        from config import NOCODB_TABLE_CODE_AGENT_TASKS
        db = NocodbClient()
        result = db._get(
            NOCODB_TABLE_CODE_AGENT_TASKS,
            params={"where": f"(agent_id,eq,{agent_id})", "limit": 1},
        )
        rows = result.get("list", [])
        if rows:
            row_id = rows[0].get("Id")
            if row_id:
                db._patch(NOCODB_TABLE_CODE_AGENT_TASKS, row_id, {"status": "deleted"})
        return {"ok": True}
    except Exception as e:
        _log.warning("delete_agent failed: %s", e)
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------------
# repo browsing endpoints
# ------------------------------------------------------------------

@router.get("/repos")
async def list_repos():
    """List all repos from Gitea."""
    from tools.code.gitea_client import GiteaClient
    gitea = GiteaClient()
    repos = await gitea.list_repos()
    return {
        "repos": [
            {
                "name": r.get("name", ""),
                "full_name": r.get("full_name", ""),
                "description": r.get("description", ""),
                "mirror": r.get("mirror", False),
                "default_branch": r.get("default_branch", "main"),
                "updated_at": r.get("updated_at", ""),
            }
            for r in repos
        ],
    }


@router.get("/repos/{repo_name}/tree")
async def get_repo_tree(repo_name: str, branch: str = "main"):
    """Return nested file tree for UI sidebar display."""
    from tools.code.gitea_client import GiteaClient
    gitea = GiteaClient()
    flat_tree = await gitea.get_file_tree(repo_name, ref=branch)
    if not flat_tree:
        raise HTTPException(404, f"Could not fetch tree for {repo_name}")

    nested = _build_nested_tree(flat_tree)
    return {"repo": repo_name, "branch": branch, "tree": nested}


@router.get("/repos/{repo_name}/file")
async def get_repo_file(repo_name: str, path: str, branch: str = "main"):
    """Fetch a single file's content for code viewer."""
    from tools.code.gitea_client import GiteaClient
    gitea = GiteaClient()
    content = await gitea.get_file_content(repo_name, path, ref=branch)
    if content is None:
        raise HTTPException(404, f"File not found: {path}")
    return {"path": path, "content": content, "branch": branch}


@router.get("/repos/{repo_name}/manifest")
async def get_repo_manifest(repo_name: str, branch: str = "main", org_id: int = 0):
    """Return the hybrid index manifest."""
    from tools.code.indexer import index_repo
    from tools.framework.executors.code_repo import _manifest_cache, _cache_key

    key = _cache_key(repo_name, branch)
    manifest = _manifest_cache.get(key)
    if manifest is None:
        manifest = await index_repo(repo_name, branch=branch, org_id=org_id)
        if manifest is None:
            raise HTTPException(404, f"Could not index {repo_name}")
        _manifest_cache[key] = manifest

    return manifest.model_dump()


@router.post("/repos/{repo_name}/index")
async def trigger_index(repo_name: str, branch: str = "main", org_id: int = 0):
    """Manually trigger re-indexing."""
    from tools.code.indexer import index_repo
    from tools.framework.executors.code_repo import _manifest_cache, _cache_key

    # invalidate cache
    key = _cache_key(repo_name, branch)
    _manifest_cache.pop(key, None)

    # run in background
    asyncio.create_task(index_repo(repo_name, branch=branch, org_id=org_id))
    return {"ok": True, "queued": True}


# ------------------------------------------------------------------
# webhook endpoint
# ------------------------------------------------------------------

@router.post("/hooks/gitea")
async def gitea_webhook(request: Request):
    """
    Receives push events from Gitea.
    Triggers incremental re-indexing of the affected repo.
    """
    from tools.code.indexer import index_repo
    from tools.framework.executors.code_repo import _manifest_cache, _cache_key

    payload = await request.json()
    repo_name = payload.get("repository", {}).get("name", "")
    ref = payload.get("ref", "")
    commits = payload.get("commits", [])

    if not repo_name:
        return {"ok": False, "error": "no repo name"}

    # collect changed files from all commits
    changed_files: set[str] = set()
    for commit in commits:
        changed_files.update(commit.get("added", []))
        changed_files.update(commit.get("modified", []))
        changed_files.update(commit.get("removed", []))

    _log.info("gitea webhook  repo=%s ref=%s changed=%d", repo_name, ref, len(changed_files))

    # determine branch from ref
    branch = ref.replace("refs/heads/", "") if ref.startswith("refs/heads/") else "main"

    # invalidate cache
    key = _cache_key(repo_name, branch)
    _manifest_cache.pop(key, None)

    # queue re-index in background
    asyncio.create_task(
        index_repo(repo_name, branch=branch, changed_files=list(changed_files))
    )

    return {"ok": True, "queued": True, "changed_files": len(changed_files)}


# ------------------------------------------------------------------
# conversation-repo binding
# ------------------------------------------------------------------

@router.post("/conversations/{conversation_id}/repo")
def bind_repo(conversation_id: int, repo_name: str, org_id: int):
    """Bind a repo to a conversation."""
    try:
        from nocodb_client import NocodbClient
        db = NocodbClient()
        db._patch("conversations", conversation_id, {"repo_name": repo_name})
        return {"ok": True, "conversation_id": conversation_id, "repo_name": repo_name}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/conversations/{conversation_id}/repo")
def unbind_repo(conversation_id: int):
    """Disconnect repo from a conversation."""
    try:
        from nocodb_client import NocodbClient
        db = NocodbClient()
        db._patch("conversations", conversation_id, {"repo_name": ""})
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, str(e))


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _build_nested_tree(flat_tree: list[dict]) -> list[dict]:
    """
    Convert flat file list to nested directory structure for UI.

    Input:  [{path: "src/auth/index.ts", size: 1234, type: "blob"}, ...]
    Output: [{name: "src", type: "dir", children: [...]}]
    """
    root: list[dict] = []
    dirs: dict[str, list[dict]] = {}

    for entry in sorted(flat_tree, key=lambda e: e.get("path", "")):
        if entry.get("type") != "blob":
            continue

        path = entry.get("path", "")
        parts = path.split("/")

        # ensure all parent directories exist
        current = root
        for i, part in enumerate(parts[:-1]):
            dir_path = "/".join(parts[: i + 1])
            if dir_path not in dirs:
                node = {"name": part, "type": "dir", "path": dir_path, "children": []}
                current.append(node)
                dirs[dir_path] = node["children"]
            current = dirs[dir_path]

        # add file
        current.append({
            "name": parts[-1],
            "type": "file",
            "path": path,
            "size": entry.get("size", 0),
        })

    return root
