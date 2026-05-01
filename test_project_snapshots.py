import unittest

from infra.nocodb_client import ConflictError, NocodbClient


class _StubClient(NocodbClient):
    def __init__(self, tables=None):
        self.tables = tables or {
            "projects": 1,
            "project_files": 2,
            "project_file_versions": 3,
            "project_audit": 4,
            "project_snapshots": 5,
            "project_snapshot_files": 6,
        }


class IdempotentWriteTests(unittest.TestCase):
    def test_if_content_hash_mismatch_raises_conflict_error(self):
        c = _StubClient()
        c.get_project_file = lambda project_id, path: {"Id": 1, "current_version_id": 99, "locked": False}
        c.get_project_file_version = lambda version_id: {"Id": 99, "version": 4, "content_hash": "actual-hash"}

        with self.assertRaises(ConflictError) as ctx:
            c.write_project_file_version(
                project_id=1,
                path="/src/a.py",
                content="new content\n",
                if_content_hash="expected-hash",
            )

        self.assertEqual(ctx.exception.expected, "expected-hash")
        self.assertEqual(ctx.exception.actual, "actual-hash")

    def test_if_content_hash_match_proceeds_to_no_op_when_content_unchanged(self):
        c = _StubClient()
        same_hash = "0" * 64
        c.get_project_file = lambda project_id, path: {"Id": 1, "current_version_id": 99, "locked": False}
        # content is empty and matches; write should be a no-op.
        c.get_project_file_version = lambda version_id: {"Id": 99, "version": 4, "content_hash": same_hash}

        # Use empty content -> sha256 of "" is e3b0c44... so use an actual matching hash
        import hashlib

        empty_hash = hashlib.sha256(b"").hexdigest()
        c.get_project_file_version = lambda version_id: {"Id": 99, "version": 4, "content_hash": empty_hash}

        file_row, version_row, changed = c.write_project_file_version(
            project_id=1,
            path="/src/a.py",
            content="",
            if_content_hash=empty_hash,
        )
        self.assertFalse(changed)
        self.assertEqual(version_row["version"], 4)


class SnapshotTests(unittest.TestCase):
    def test_list_snapshots_returns_empty_when_table_missing(self):
        c = _StubClient(tables={"projects": 1})
        self.assertEqual(c.list_project_snapshots(project_id=1), [])

    def test_list_snapshot_files_returns_empty_when_table_missing(self):
        c = _StubClient(tables={"projects": 1})
        self.assertEqual(c.list_project_snapshot_files(snapshot_id=1), [])

    def test_create_snapshot_rejects_duplicate_label(self):
        c = _StubClient()
        c.get_project_snapshot = lambda project_id, label: {"Id": 1, "label": label}

        with self.assertRaises(ValueError):
            c.create_project_snapshot(project_id=1, label="v1", actor="user")

    def test_create_snapshot_captures_current_versions_of_active_files(self):
        c = _StubClient()
        c.get_project_snapshot = lambda project_id, label: None
        c.list_project_files = lambda project_id, **kwargs: [
            {"Id": 10, "path": "/a.py", "current_version_id": 100},
            {"Id": 11, "path": "/b.py", "current_version_id": None},  # skipped
            {"Id": 12, "path": "/c.py", "current_version_id": 102},
        ]

        posted: list[tuple[str, dict]] = []

        def fake_post(table, data):
            posted.append((table, data))
            return {"Id": len(posted), **data}

        c._post = fake_post
        c.add_project_audit_event = lambda *a, **kw: None

        snap = c.create_project_snapshot(project_id=1, label="v1", actor="user")

        snap_inserts = [d for t, d in posted if t == "project_snapshots"]
        snap_file_inserts = [d for t, d in posted if t == "project_snapshot_files"]
        self.assertEqual(len(snap_inserts), 1)
        self.assertEqual(snap_inserts[0]["label"], "v1")
        self.assertEqual(len(snap_file_inserts), 2)
        self.assertEqual({d["path"] for d in snap_file_inserts}, {"/a.py", "/c.py"})
        self.assertEqual(snap["file_count"], 2)


if __name__ == "__main__":
    unittest.main()
