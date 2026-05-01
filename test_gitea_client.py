import base64
import unittest
from unittest.mock import patch, Mock

from infra.gitea_client import GiteaClient, GiteaError


def _resp(status_code: int, json_payload=None, content: bytes | None = None, text: str = ""):
    r = Mock()
    r.status_code = status_code
    r.content = content if content is not None else (b"x" if json_payload is None and not text else (text.encode("utf-8") or b"{}"))
    r.text = text or ""
    r.json = Mock(return_value=json_payload if json_payload is not None else {})
    r.request = Mock(method="GET", url="http://gitea/x")
    return r


class WhoAmITests(unittest.TestCase):
    def test_whoami_returns_user(self):
        c = GiteaClient("http://gitea", "tok")
        with patch("infra.gitea_client.requests.get", return_value=_resp(200, {"login": "mike", "id": 7})):
            self.assertEqual(c.whoami()["login"], "mike")

    def test_verify_credentials_includes_server_version(self):
        c = GiteaClient("http://gitea", "tok")
        side = [
            _resp(200, {"login": "mike", "id": 7, "is_admin": True}),
            _resp(200, {"version": "1.22.3"}),
        ]
        with patch("infra.gitea_client.requests.get", side_effect=side):
            info = c.verify_credentials()
        self.assertEqual(info["login"], "mike")
        self.assertTrue(info["is_admin"])
        self.assertEqual(info["server_version"], "1.22.3")


class FileContentTests(unittest.TestCase):
    def test_get_file_content_decodes_base64(self):
        encoded = base64.b64encode(b"hello world").decode("ascii")
        c = GiteaClient("http://gitea", "tok")
        with patch("infra.gitea_client.requests.get", return_value=_resp(200, {"content": encoded, "sha": "abc"})):
            text, sha = c.get_file_content("o", "r", "p")
        self.assertEqual(text, "hello world")
        self.assertEqual(sha, "abc")

    def test_get_file_content_falls_back_to_download_url_when_inline_missing(self):
        c = GiteaClient("http://gitea", "tok")
        responses = [
            _resp(200, {"content": "", "sha": "abc", "download_url": "http://gitea/dl/x"}),
            _resp(200, text="binary-content-here"),
        ]
        with patch("infra.gitea_client.requests.get", side_effect=responses):
            text, sha = c.get_file_content("o", "r", "p")
        self.assertEqual(text, "binary-content-here")
        self.assertEqual(sha, "abc")


class PutFileTests(unittest.TestCase):
    def test_put_file_uses_put_when_sha_supplied(self):
        c = GiteaClient("http://gitea", "tok")
        captured = {}

        def fake_put(url, headers=None, json=None, timeout=None):
            captured["method"] = "put"
            captured["body"] = json
            return _resp(200, {"commit": {"sha": "deadbeef"}})

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["method"] = "post"
            return _resp(200, {})

        with patch("infra.gitea_client.requests.put", fake_put), patch("infra.gitea_client.requests.post", fake_post):
            res = c.put_file("o", "r", "p", "hi", "msg", "main", sha="oldsha")
        self.assertEqual(captured["method"], "put")
        self.assertEqual(captured["body"]["sha"], "oldsha")
        self.assertEqual(captured["body"]["branch"], "main")
        self.assertEqual(res["commit"]["sha"], "deadbeef")

    def test_put_file_uses_post_when_no_sha(self):
        c = GiteaClient("http://gitea", "tok")
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["method"] = "post"
            captured["body"] = json
            return _resp(201, {"commit": {"sha": "newsha"}})

        with patch("infra.gitea_client.requests.post", fake_post):
            c.put_file("o", "r", "p", "hi", "msg", "main")
        self.assertEqual(captured["method"], "post")
        self.assertNotIn("sha", captured["body"])

    def test_put_file_includes_author_when_supplied(self):
        c = GiteaClient("http://gitea", "tok")
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["body"] = json
            return _resp(201, {})

        with patch("infra.gitea_client.requests.post", fake_post):
            c.put_file("o", "r", "p", "hi", "m", "main", author_name="A", author_email="a@x")
        self.assertEqual(captured["body"]["author"], {"name": "A", "email": "a@x"})


class CreateRepoTests(unittest.TestCase):
    def test_user_repo_targets_user_endpoint(self):
        c = GiteaClient("http://gitea", "tok")
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["body"] = json
            return _resp(201, {"name": "x"})

        with patch("infra.gitea_client.requests.post", fake_post):
            c.create_repo("mike", "user", "x")
        self.assertTrue(captured["url"].endswith("/api/v1/user/repos"))
        self.assertEqual(captured["body"]["name"], "x")
        self.assertEqual(captured["body"]["auto_init"], False)

    def test_org_repo_targets_org_endpoint(self):
        c = GiteaClient("http://gitea", "tok")
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            return _resp(201, {})

        with patch("infra.gitea_client.requests.post", fake_post):
            c.create_repo("acme", "org", "x")
        self.assertTrue(captured["url"].endswith("/api/v1/orgs/acme/repos"))

    def test_create_repo_409_returns_existing(self):
        c = GiteaClient("http://gitea", "tok")
        with patch("infra.gitea_client.requests.post", return_value=_resp(409, {}, text="exists")):
            with patch("infra.gitea_client.requests.get", return_value=_resp(200, {"name": "x", "default_branch": "main"})):
                r = c.create_repo("mike", "user", "x")
        self.assertEqual(r["name"], "x")

    def test_invalid_owner_kind_raises(self):
        c = GiteaClient("http://gitea", "tok")
        with self.assertRaises(ValueError):
            c.create_repo("mike", "team", "x")


class BranchTests(unittest.TestCase):
    def test_branch_exists_handles_404(self):
        c = GiteaClient("http://gitea", "tok")
        with patch("infra.gitea_client.requests.get", return_value=_resp(404, {}, text="not found")):
            self.assertFalse(c.branch_exists("o", "r", "missing"))

    def test_branch_exists_returns_true_when_present(self):
        c = GiteaClient("http://gitea", "tok")
        with patch("infra.gitea_client.requests.get", return_value=_resp(200, {"name": "main"})):
            self.assertTrue(c.branch_exists("o", "r", "main"))


class TreeTests(unittest.TestCase):
    def test_git_tree_recursive_returns_blobs(self):
        c = GiteaClient("http://gitea", "tok")
        payload = {"sha": "abc", "tree": [
            {"path": "src/a.py", "type": "blob", "sha": "1", "size": 10},
            {"path": "src/b.py", "type": "blob", "sha": "2", "size": 20},
        ], "truncated": False}
        with patch("infra.gitea_client.requests.get", return_value=_resp(200, payload)):
            tree = c.git_tree_recursive("o", "r", "abc")
        self.assertEqual(len(tree), 2)
        self.assertEqual(tree[0]["path"], "src/a.py")


class ListReposTests(unittest.TestCase):
    def test_list_repos_paginates(self):
        c = GiteaClient("http://gitea", "tok")
        responses = [
            _resp(200, [{"name": f"r{i}"} for i in range(50)]),
            _resp(200, [{"name": f"r{i}"} for i in range(50, 75)]),
        ]
        with patch("infra.gitea_client.requests.get", side_effect=responses):
            repos = c.list_repos(limit=80)
        self.assertEqual(len(repos), 75)


class ErrorTests(unittest.TestCase):
    def test_error_raises_giteaerror(self):
        r = _resp(500, content=b"boom", text="boom")
        c = GiteaClient("http://gitea", "tok")
        with patch("infra.gitea_client.requests.get", return_value=r):
            with self.assertRaises(GiteaError):
                c.get_file("o", "r", "p")


if __name__ == "__main__":
    unittest.main()
