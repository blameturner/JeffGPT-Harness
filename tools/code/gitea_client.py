"""
Gitea API client for MST-AG.

Single point of contact for all Gitea operations. Every other module
that needs Gitea goes through this client.

Methods are async, use httpx, and follow the same error handling pattern
as the rest of the harness — fail-open, return None/False/[] on error.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from config import SVC_GITEA_URL, GITEA_TOKEN, GITEA_OWNER

_log = logging.getLogger("tools.code.gitea")


class GiteaClient:
    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        owner: str | None = None,
    ):
        self.base_url = (base_url or SVC_GITEA_URL or "").rstrip("/")
        self.token = token or GITEA_TOKEN or ""
        self.owner = owner or GITEA_OWNER or ""

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        params: dict | None = None,
        timeout: float = 30.0,
    ) -> httpx.Response | None:
        if not self.base_url:
            _log.warning("gitea request skipped — SVC_GITEA_URL not configured")
            return None
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"token {self.token}"} if self.token else {}
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.request(
                    method, url, headers=headers, json=json, params=params,
                )
                resp.raise_for_status()
                return resp
        except httpx.HTTPStatusError as e:
            _log.warning(
                "gitea %s %s  status=%d body=%s",
                method, path, e.response.status_code, e.response.text[:300],
            )
        except Exception as e:
            _log.warning("gitea %s %s  error=%s", method, path, e)
        return None

    async def _get(self, path: str, **kwargs: Any) -> httpx.Response | None:
        return await self._request("GET", path, **kwargs)

    async def _post(self, path: str, **kwargs: Any) -> httpx.Response | None:
        return await self._request("POST", path, **kwargs)

    async def _put(self, path: str, **kwargs: Any) -> httpx.Response | None:
        return await self._request("PUT", path, **kwargs)

    async def _delete(self, path: str, **kwargs: Any) -> httpx.Response | None:
        return await self._request("DELETE", path, **kwargs)

    # ------------------------------------------------------------------
    # repository operations
    # ------------------------------------------------------------------

    async def list_repos(self) -> list[dict]:
        """List all repos for the configured owner."""
        resp = await self._get(
            "/api/v1/repos/search",
            params={"owner": self.owner, "limit": 50},
        )
        if resp is None:
            return []
        try:
            return resp.json().get("data", [])
        except Exception:
            return []

    async def get_repo(self, repo_name: str) -> dict | None:
        """Get repo metadata."""
        resp = await self._get(f"/api/v1/repos/{self.owner}/{repo_name}")
        if resp is None:
            return None
        try:
            return resp.json()
        except Exception:
            return None

    async def get_file_tree(
        self, repo_name: str, ref: str = "main",
    ) -> list[dict]:
        """
        Recursive file tree.

        Returns [{path, size, sha, type}, ...] where type is "blob" or "tree".
        """
        resp = await self._get(
            f"/api/v1/repos/{self.owner}/{repo_name}/git/trees/{ref}",
            params={"recursive": "true", "per_page": "0"},
        )
        if resp is None:
            return []
        try:
            tree = resp.json().get("tree", [])
            return [
                {
                    "path": e.get("path", ""),
                    "size": e.get("size", 0),
                    "sha": e.get("sha", ""),
                    "type": e.get("type", "blob"),
                }
                for e in tree
            ]
        except Exception:
            return []

    async def get_file_content(
        self, repo_name: str, path: str, ref: str = "main",
    ) -> str | None:
        """Fetch raw file content (base64-decoded)."""
        resp = await self._get(
            f"/api/v1/repos/{self.owner}/{repo_name}/contents/{path}",
            params={"ref": ref},
        )
        if resp is None:
            return None
        try:
            data = resp.json()
            content_b64 = data.get("content", "")
            if not content_b64:
                return ""
            return base64.b64decode(content_b64).decode("utf-8", errors="replace")
        except Exception as e:
            _log.warning("gitea decode failed  repo=%s path=%s: %s", repo_name, path, e)
            return None

    async def update_file(
        self,
        repo_name: str,
        path: str,
        content: str,
        message: str,
        branch: str,
        sha: str = "",
    ) -> dict | None:
        """
        Create or update a file. Returns commit info dict or None.

        If sha is provided, updates the existing file. If empty, creates new.
        """
        payload: dict[str, Any] = {
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "message": message,
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha

        method = "PUT"
        resp = await self._request(
            method,
            f"/api/v1/repos/{self.owner}/{repo_name}/contents/{path}",
            json=payload,
        )
        if resp is None:
            return None
        try:
            return resp.json().get("commit", resp.json())
        except Exception:
            return None

    async def delete_file(
        self,
        repo_name: str,
        path: str,
        message: str,
        branch: str,
        sha: str,
    ) -> bool:
        """Delete a file from the repo."""
        resp = await self._request(
            "DELETE",
            f"/api/v1/repos/{self.owner}/{repo_name}/contents/{path}",
            json={"message": message, "branch": branch, "sha": sha},
        )
        return resp is not None

    # ------------------------------------------------------------------
    # branch operations
    # ------------------------------------------------------------------

    async def list_branches(self, repo_name: str) -> list[dict]:
        """List all branches for a repo."""
        resp = await self._get(
            f"/api/v1/repos/{self.owner}/{repo_name}/branches",
        )
        if resp is None:
            return []
        try:
            return resp.json()
        except Exception:
            return []

    async def create_branch(
        self, repo_name: str, branch_name: str, from_branch: str = "main",
    ) -> bool:
        """Create a new branch from an existing one."""
        resp = await self._post(
            f"/api/v1/repos/{self.owner}/{repo_name}/branches",
            json={"new_branch_name": branch_name, "old_branch_name": from_branch},
        )
        return resp is not None

    async def delete_branch(self, repo_name: str, branch_name: str) -> bool:
        """Delete a branch."""
        resp = await self._delete(
            f"/api/v1/repos/{self.owner}/{repo_name}/branches/{branch_name}",
        )
        return resp is not None

    async def get_branch_diff(
        self, repo_name: str, base: str, head: str,
    ) -> dict | None:
        """
        Compare two branches.

        Returns {files_changed, additions, deletions, diff_files} or None.
        """
        resp = await self._get(
            f"/api/v1/repos/{self.owner}/{repo_name}/compare/{base}...{head}",
        )
        if resp is None:
            return None
        try:
            data = resp.json()
            files = data.get("files", [])
            return {
                "files_changed": len(files),
                "additions": sum(f.get("additions", 0) for f in files),
                "deletions": sum(f.get("deletions", 0) for f in files),
                "diff_files": [
                    {
                        "filename": f.get("filename", ""),
                        "status": f.get("status", ""),
                        "additions": f.get("additions", 0),
                        "deletions": f.get("deletions", 0),
                    }
                    for f in files
                ],
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # pull request operations
    # ------------------------------------------------------------------

    async def create_pr(
        self,
        repo_name: str,
        title: str,
        head: str,
        base: str = "main",
        body: str = "",
    ) -> dict | None:
        """Create a pull request. Returns PR dict or None."""
        resp = await self._post(
            f"/api/v1/repos/{self.owner}/{repo_name}/pulls",
            json={"title": title, "head": head, "base": base, "body": body},
        )
        if resp is None:
            return None
        try:
            return resp.json()
        except Exception:
            return None

    async def merge_pr(
        self,
        repo_name: str,
        pr_number: int,
        merge_type: str = "squash",
        delete_branch: bool = True,
    ) -> bool:
        """Merge a pull request."""
        resp = await self._post(
            f"/api/v1/repos/{self.owner}/{repo_name}/pulls/{pr_number}/merge",
            json={
                "Do": merge_type,
                "delete_branch_after_merge": delete_branch,
            },
        )
        return resp is not None

    async def list_prs(
        self, repo_name: str, state: str = "open",
    ) -> list[dict]:
        """List pull requests for a repo."""
        resp = await self._get(
            f"/api/v1/repos/{self.owner}/{repo_name}/pulls",
            params={"state": state},
        )
        if resp is None:
            return []
        try:
            return resp.json()
        except Exception:
            return []

    # ------------------------------------------------------------------
    # commit operations
    # ------------------------------------------------------------------

    async def get_commits(
        self,
        repo_name: str,
        branch: str = "main",
        limit: int = 20,
        path: str | None = None,
    ) -> list[dict]:
        """Get recent commits for a branch."""
        params: dict[str, Any] = {"sha": branch, "limit": limit}
        if path:
            params["path"] = path
        resp = await self._get(
            f"/api/v1/repos/{self.owner}/{repo_name}/commits",
            params=params,
        )
        if resp is None:
            return []
        try:
            return resp.json()
        except Exception:
            return []

    # ------------------------------------------------------------------
    # webhook operations
    # ------------------------------------------------------------------

    async def create_webhook(
        self,
        repo_name: str,
        target_url: str,
        events: list[str] | None = None,
    ) -> bool:
        """Create a webhook for push events."""
        resp = await self._post(
            f"/api/v1/repos/{self.owner}/{repo_name}/hooks",
            json={
                "type": "gitea",
                "active": True,
                "config": {
                    "url": target_url,
                    "content_type": "json",
                },
                "events": events or ["push"],
            },
        )
        return resp is not None
