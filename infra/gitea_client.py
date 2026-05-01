"""Minimal Gitea REST wrapper for the project sync flow.

Audited against Gitea API v1 (current as of 2026-05). Endpoints used:

  GET   /api/v1/version
  GET   /api/v1/user                            — token verify
  GET   /api/v1/user/repos                      — repos the auth user owns/can write
  GET   /api/v1/user/orgs                       — orgs the auth user belongs to
  GET   /api/v1/orgs/{org}/repos
  POST  /api/v1/user/repos                      — create repo under auth user
  POST  /api/v1/orgs/{org}/repos                — create repo under org
  GET   /api/v1/repos/{owner}/{repo}
  GET   /api/v1/repos/{owner}/{repo}/branches/{branch}
  GET   /api/v1/repos/{owner}/{repo}/contents/{path}?ref=
  GET   /api/v1/repos/{owner}/{repo}/git/trees/{sha}?recursive=true   — fast tree walk
  POST  /api/v1/repos/{owner}/{repo}/contents/{path}                  — create file
  PUT   /api/v1/repos/{owner}/{repo}/contents/{path}                  — update file (needs sha)
  GET   /api/v1/repos/{owner}/{repo}/archive/{ref}.zip                — repo snapshot
  GET   /api/v1/repos/{owner}/{repo}/commits?sha=&limit=

Auth uses `Authorization: token <PAT>` (Gitea Personal Access Token).
"""
from __future__ import annotations

import base64
import logging
from typing import Any
from urllib.parse import quote

import requests

_log = logging.getLogger("gitea")

REQUEST_TIMEOUT = 15


class GiteaError(Exception):
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


def _enc(segment: str) -> str:
    """Encode a single URL path segment, preserving slashes inside `path/to/file.py`
    when the segment is intentionally a sub-path."""
    return quote(segment, safe="/")


class GiteaClient:
    def __init__(self, base_url: str, token: str, username: str = ""):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.username = username

    # ---------- low-level ----------
    def _headers(self, json_body: bool = True) -> dict:
        h = {"Authorization": f"token {self.token}", "Accept": "application/json"}
        if json_body:
            h["Content-Type"] = "application/json"
        return h

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/v1{path}"

    def _check(self, response: requests.Response) -> Any:
        if response.status_code == 204 or not response.content:
            return None
        if response.status_code >= 400:
            method = getattr(response.request, "method", "?") if response.request else "?"
            url = getattr(response.request, "url", "?") if response.request else "?"
            raise GiteaError(
                f"gitea {method} {url} -> {response.status_code}: {response.text[:400]}",
                response.status_code,
            )
        try:
            return response.json()
        except Exception:
            return response.content

    def _get(self, path: str, params: dict | None = None, timeout: int = REQUEST_TIMEOUT) -> Any:
        r = requests.get(self._url(path), headers=self._headers(json_body=False), params=params, timeout=timeout)
        return self._check(r)

    def _get_raw(self, path: str, timeout: int = REQUEST_TIMEOUT) -> requests.Response:
        return requests.get(self._url(path), headers={"Authorization": f"token {self.token}"}, timeout=timeout)

    def _post(self, path: str, body: dict) -> Any:
        r = requests.post(self._url(path), headers=self._headers(), json=body, timeout=REQUEST_TIMEOUT)
        return self._check(r)

    def _put(self, path: str, body: dict) -> Any:
        r = requests.put(self._url(path), headers=self._headers(), json=body, timeout=REQUEST_TIMEOUT)
        return self._check(r)

    # ---------- discovery ----------
    def server_version(self) -> str:
        try:
            data = self._get("/version") or {}
            return str(data.get("version") or "")
        except GiteaError:
            return ""

    def whoami(self) -> dict:
        return self._get("/user") or {}

    def verify_credentials(self) -> dict:
        """Single call surface for the connection-setup UI."""
        me = self.whoami()
        return {
            "login": me.get("login"),
            "id": me.get("id"),
            "is_admin": bool(me.get("is_admin")),
            "server_version": self.server_version(),
        }

    def _user_id(self) -> int:
        return int(self.whoami().get("id") or 0)

    # ---------- repo listing ----------
    def list_repos(self, limit: int = 50) -> list[dict]:
        """Repos the authenticated user owns or has push access to.

        `/user/repos` is the canonical 'my repos' endpoint and is paginated;
        we fetch up to `limit` items across pages.
        """
        out: list[dict] = []
        page = 1
        while len(out) < limit:
            page_size = min(50, limit - len(out))
            chunk = self._get("/user/repos", params={"page": page, "limit": page_size}) or []
            if not isinstance(chunk, list) or not chunk:
                break
            out.extend(chunk)
            if len(chunk) < page_size:
                break
            page += 1
        return out[:limit]

    def list_user_orgs(self) -> list[dict]:
        return self._get("/user/orgs") or []

    def list_org_repos(self, org: str, limit: int = 50) -> list[dict]:
        return self._get(f"/orgs/{_enc(org)}/repos", params={"limit": limit}) or []

    # ---------- repo metadata ----------
    def get_repo(self, owner: str, repo: str) -> dict:
        return self._get(f"/repos/{_enc(owner)}/{_enc(repo)}") or {}

    def repo_exists(self, owner: str, repo: str) -> bool:
        try:
            self.get_repo(owner, repo)
            return True
        except GiteaError as e:
            if e.status_code == 404:
                return False
            raise

    def get_branch(self, owner: str, repo: str, branch: str) -> dict | None:
        try:
            return self._get(f"/repos/{_enc(owner)}/{_enc(repo)}/branches/{_enc(branch)}")
        except GiteaError as e:
            if e.status_code == 404:
                return None
            raise

    def branch_exists(self, owner: str, repo: str, branch: str) -> bool:
        return self.get_branch(owner, repo, branch) is not None

    # ---------- file contents ----------
    def list_repo_contents(self, owner: str, repo: str, path: str = "", ref: str = "") -> list[dict]:
        url = f"/repos/{_enc(owner)}/{_enc(repo)}/contents/{path.lstrip('/')}"
        params = {"ref": ref} if ref else None
        data = self._get(url, params=params)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return []

    def git_tree_recursive(self, owner: str, repo: str, sha: str) -> list[dict]:
        """Fetch the full file tree for a commit/branch in one call.

        Returns entries shaped like `{"path": "src/x.py", "type": "blob", "sha": "...", "size": 123}`.
        Truncated trees are flagged by the response's `truncated` field.
        """
        data = self._get(f"/repos/{_enc(owner)}/{_enc(repo)}/git/trees/{_enc(sha)}", params={"recursive": "true", "per_page": 1000}) or {}
        if isinstance(data, dict):
            return data.get("tree") or []
        return []

    def get_file(self, owner: str, repo: str, path: str, ref: str = "") -> dict:
        url = f"/repos/{_enc(owner)}/{_enc(repo)}/contents/{path.lstrip('/')}"
        params = {"ref": ref} if ref else None
        data = self._get(url, params=params)
        return data if isinstance(data, dict) else {}

    def get_file_content(self, owner: str, repo: str, path: str, ref: str = "") -> tuple[str, str]:
        """Returns (text_content, blob_sha).

        Falls back to `download_url` when the contents response omits inline
        content (Gitea does this for binaries and for files above
        `MAX_BLOB_SIZE`, configurable per-server).
        """
        info = self.get_file(owner, repo, path, ref)
        sha = info.get("sha") or ""
        encoded = info.get("content") or ""
        if encoded:
            try:
                return base64.b64decode(encoded.encode("utf-8")).decode("utf-8", errors="replace"), sha
            except Exception:
                pass
        download_url = info.get("download_url") or ""
        if download_url:
            try:
                r = requests.get(download_url, headers={"Authorization": f"token {self.token}"}, timeout=REQUEST_TIMEOUT * 2)
                if r.status_code < 400:
                    return r.text, sha
            except Exception:
                _log.warning("download_url fetch failed for %s/%s:%s", owner, repo, path, exc_info=True)
        return "", sha

    # ---------- archives & history ----------
    def archive_zip(self, owner: str, repo: str, ref: str = "main") -> bytes:
        r = self._get_raw(f"/repos/{_enc(owner)}/{_enc(repo)}/archive/{_enc(ref)}.zip", timeout=REQUEST_TIMEOUT * 4)
        if r.status_code >= 400:
            raise GiteaError(f"archive failed: {r.status_code}: {r.text[:200]}", r.status_code)
        return r.content

    def list_commits(self, owner: str, repo: str, sha: str = "", limit: int = 10) -> list[dict]:
        params: dict = {"limit": min(max(1, limit), 50)}
        if sha:
            params["sha"] = sha
        data = self._get(f"/repos/{_enc(owner)}/{_enc(repo)}/commits", params=params)
        return data if isinstance(data, list) else []

    # ---------- repo creation ----------
    def create_repo(
        self,
        owner: str,
        owner_kind: str,
        repo: str,
        description: str = "",
        private: bool = True,
        default_branch: str = "main",
        init_readme: bool = False,
    ) -> dict:
        if owner_kind not in ("user", "org"):
            raise ValueError("owner_kind must be 'user' or 'org'")
        path = "/user/repos" if owner_kind == "user" else f"/orgs/{_enc(owner)}/repos"
        body: dict = {
            "name": repo,
            "description": description,
            "private": bool(private),
            "default_branch": default_branch,
            "auto_init": bool(init_readme),
        }
        try:
            return self._post(path, body) or {}
        except GiteaError as e:
            if e.status_code == 409:
                # Already exists — return the existing repo metadata.
                return self.get_repo(owner, repo)
            raise

    # ---------- file mutations ----------
    def put_file(
        self,
        owner: str,
        repo: str,
        path: str,
        content_text: str,
        message: str,
        branch: str,
        sha: str = "",
        author_name: str = "",
        author_email: str = "",
    ) -> dict:
        url = f"/repos/{_enc(owner)}/{_enc(repo)}/contents/{path.lstrip('/')}"
        body: dict = {
            "content": base64.b64encode(content_text.encode("utf-8")).decode("ascii"),
            "message": message,
            "branch": branch,
        }
        if author_name and author_email:
            body["author"] = {"name": author_name, "email": author_email}
            body["committer"] = {"name": author_name, "email": author_email}
        if sha:
            body["sha"] = sha
            return self._put(url, body) or {}
        return self._post(url, body) or {}

    def delete_file(self, owner: str, repo: str, path: str, sha: str, branch: str, message: str) -> dict:
        url = self._url(f"/repos/{_enc(owner)}/{_enc(repo)}/contents/{path.lstrip('/')}")
        body = {"sha": sha, "branch": branch, "message": message}
        r = requests.delete(url, headers=self._headers(), json=body, timeout=REQUEST_TIMEOUT)
        return self._check(r) or {}
