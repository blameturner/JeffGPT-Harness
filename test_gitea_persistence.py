"""Tests for the iteration loop: edit → push → status → edit → push.

These verify the contracts that `pushed_to_sha` carries between
`write_project_file_version` and the gitea status/push endpoints.
"""
import unittest

from infra.nocodb_client import NocodbClient


def _stub() -> NocodbClient:
    c = object.__new__(NocodbClient)
    c.tables = {
        "projects": 1, "project_files": 2, "project_file_versions": 3,
        "project_audit": 4,
    }
    return c


class WriteVersionDoesNotSetPushedSha(unittest.TestCase):
    """A new version must NOT carry a pushed_to_sha — it's by definition ahead."""

    def test_new_version_payload_omits_pushed_to_sha(self):
        c = _stub()
        c.get_project_file = lambda project_id, path: None
        c.list_project_files = lambda project_id, limit=5000, **kw: []
        c.add_project_audit_event = lambda *a, **kw: None
        captured: list[tuple[str, dict]] = []

        def fake_post(table, data):
            captured.append((table, data))
            return {"Id": len(captured), **data}

        def fake_patch(table, row_id, data):
            captured.append((f"PATCH:{table}", data))
            return {"Id": row_id, **data}

        c._post = fake_post
        c._patch = fake_patch

        c.write_project_file_version(
            project_id=1, path="/src/a.py", content="x\n",
            edit_summary="initial", created_by="user",
        )

        version_inserts = [d for t, d in captured if t == "project_file_versions"]
        self.assertEqual(len(version_inserts), 1)
        self.assertNotIn("pushed_to_sha", version_inserts[0])


class MarkVersionPushedSetsSha(unittest.TestCase):
    def test_mark_version_pushed_patches_only_pushed_to_sha(self):
        c = _stub()
        captured: list[dict] = []

        def fake_patch(table, row_id, data):
            captured.append({"table": table, "id": row_id, "data": data})
            return {"Id": row_id, **data}

        c._patch = fake_patch
        c.mark_version_pushed(99, "deadbeef")

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["table"], "project_file_versions")
        self.assertEqual(captured[0]["id"], 99)
        self.assertEqual(captured[0]["data"]["pushed_to_sha"], "deadbeef")

    def test_mark_version_pushed_with_empty_sha_clears_state(self):
        """Pull `ours` writes empty string to mark file as needing-push."""
        c = _stub()
        captured: list[dict] = []
        c._patch = lambda t, i, d: captured.append({"data": d}) or {"Id": i, **d}
        c.mark_version_pushed(99, "")
        self.assertEqual(captured[0]["data"]["pushed_to_sha"], "")

    def test_mark_version_pushed_silently_skips_when_table_missing(self):
        c = _stub()
        c.tables = {"projects": 1}  # no project_file_versions
        # Should not raise.
        c.mark_version_pushed(99, "x")


class UpdateProjectGiteaStateSetsBothShaAndTimestamp(unittest.TestCase):
    def test_state_update_sets_sha_and_iso_timestamp(self):
        c = _stub()
        captured: list[dict] = []
        c._patch = lambda t, i, d: captured.append({"table": t, "data": d}) or {}
        c.update_project_gitea_state(7, "abc123")

        self.assertEqual(captured[0]["table"], "projects")
        self.assertEqual(captured[0]["data"]["gitea_last_synced_sha"], "abc123")
        self.assertIn("gitea_last_synced_at", captured[0]["data"])
        # ISO 8601 with timezone offset.
        ts = captured[0]["data"]["gitea_last_synced_at"]
        self.assertRegex(ts, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

    def test_state_update_with_origin_includes_origin(self):
        c = _stub()
        captured: list[dict] = []
        c._patch = lambda t, i, d: captured.append({"data": d}) or {}
        c.update_project_gitea_state(7, "abc", origin="me/repo@main")
        self.assertEqual(captured[0]["data"]["gitea_origin"], "me/repo@main")


class IterationLoopStateMachine(unittest.TestCase):
    """Walk through edit → push → edit → push and check state at each step."""

    def setUp(self):
        # Simulated DB: file_id -> {current_version_id, ...}
        self.files = {}
        # version_id -> {file_id, version, content, content_hash, pushed_to_sha}
        self.versions = {}
        self.next_version_id = 100
        self.next_file_id = 10

        c = _stub()
        c.add_project_audit_event = lambda *a, **kw: None

        def get_project_file(project_id, path, include_archived=False):
            for fid, f in self.files.items():
                if f["path"] == path and (include_archived or not f.get("archived_at")):
                    return {"Id": fid, **f}
            return None

        def get_project_file_version(version_id):
            v = self.versions.get(version_id)
            return {"Id": version_id, **v} if v else None

        def list_project_files(project_id, **kw):
            return [{"Id": fid, **f} for fid, f in self.files.items() if not f.get("archived_at")]

        def post(table, data):
            if table == "project_files":
                fid = self.next_file_id
                self.next_file_id += 1
                self.files[fid] = data
                return {"Id": fid, **data}
            elif table == "project_file_versions":
                vid = self.next_version_id
                self.next_version_id += 1
                self.versions[vid] = {**data, "pushed_to_sha": data.get("pushed_to_sha")}
                return {"Id": vid, **data}
            return {"Id": 1}

        def patch(table, row_id, data):
            if table == "project_files":
                self.files[row_id].update({k: v for k, v in data.items() if k != "Id"})
                return {"Id": row_id, **self.files[row_id]}
            elif table == "project_file_versions":
                if row_id in self.versions:
                    self.versions[row_id].update({k: v for k, v in data.items() if k != "Id"})
                return {"Id": row_id, **(self.versions.get(row_id) or {})}
            return {"Id": row_id}

        c.get_project_file = get_project_file
        c.get_project_file_version = get_project_file_version
        c.list_project_files = list_project_files
        c._post = post
        c._patch = patch
        self.c = c

    def _ahead_paths(self):
        """Replicate the gitea_status `ahead` calc."""
        ahead = []
        for f in self.c.list_project_files(project_id=1):
            vid = f.get("current_version_id")
            if not vid:
                continue
            v = self.c.get_project_file_version(int(vid))
            if not v.get("pushed_to_sha"):
                ahead.append(f.get("path"))
        return ahead

    def test_full_iteration_cycle(self):
        # Step 1: agent writes a new file.
        self.c.write_project_file_version(
            project_id=1, path="/src/a.py", content="v1\n",
            edit_summary="initial", created_by="agent:1", conversation_id=1,
        )
        self.assertEqual(self._ahead_paths(), ["/src/a.py"], "new files start ahead")

        # Step 2: simulate push success — mark the current version as pushed.
        f = self.c.get_project_file(1, "/src/a.py")
        self.c.mark_version_pushed(f["current_version_id"], "sha-001")
        self.assertEqual(self._ahead_paths(), [], "after push, file no longer ahead")

        # Step 3: user manually edits the same file.
        self.c.write_project_file_version(
            project_id=1, path="/src/a.py", content="v2\n",
            edit_summary="manual edit", created_by="user",
        )
        self.assertEqual(self._ahead_paths(), ["/src/a.py"], "edit re-marks as ahead")

        # Step 4: push again.
        f = self.c.get_project_file(1, "/src/a.py")
        self.c.mark_version_pushed(f["current_version_id"], "sha-002")
        self.assertEqual(self._ahead_paths(), [])

        # Step 5: agent writes a second file in the same project.
        self.c.write_project_file_version(
            project_id=1, path="/src/b.py", content="b1\n",
            edit_summary="agent added b", created_by="agent:1", conversation_id=1,
        )
        ahead = self._ahead_paths()
        self.assertEqual(ahead, ["/src/b.py"], "only the new file shows ahead")

    def test_pull_ours_marks_file_ahead_for_next_push(self):
        # Initial state: file in sync.
        self.c.write_project_file_version(
            project_id=1, path="/x.py", content="local\n",
            edit_summary="initial", created_by="user",
        )
        f = self.c.get_project_file(1, "/x.py")
        self.c.mark_version_pushed(f["current_version_id"], "sha-A")
        self.assertEqual(self._ahead_paths(), [])

        # Pull `ours` semantics: clear pushed_to_sha so next push includes it.
        self.c.mark_version_pushed(f["current_version_id"], "")
        self.assertEqual(self._ahead_paths(), ["/x.py"], "ours-decision queues file for push")

    def test_pull_theirs_writes_pre_marked_version(self):
        # Initial state: file in sync.
        self.c.write_project_file_version(
            project_id=1, path="/y.py", content="local\n",
            edit_summary="initial", created_by="user",
        )
        f = self.c.get_project_file(1, "/y.py")
        self.c.mark_version_pushed(f["current_version_id"], "sha-A")

        # Pull `theirs`: write remote content as new local version, then mark it pushed.
        _, version_row, _ = self.c.write_project_file_version(
            project_id=1, path="/y.py", content="remote\n",
            edit_summary="gitea pull abc12345", created_by="gitea:pull@abc12345",
        )
        self.c.mark_version_pushed(int(version_row["Id"]), "sha-B")
        self.assertEqual(self._ahead_paths(), [], "theirs-decision should land in_sync")


if __name__ == "__main__":
    unittest.main()
