import unittest

from infra.nocodb_client import NocodbClient, PROJECT_MAX_FILE_BYTES, PROJECT_MAX_FILES


class NocodbProjectFsTests(unittest.TestCase):
    def _client(self):
        c = object.__new__(NocodbClient)
        c.tables = {"project_files": 1, "project_file_versions": 2, "projects": 3}
        return c

    def test_list_project_files_filters_default_to_active(self):
        c = self._client()
        captured = {}

        def fake_get_paginated(table, params, page_size=50):
            captured["table"] = table
            captured["params"] = params
            return []

        c._get_paginated = fake_get_paginated
        c.list_project_files(project_id=7)

        self.assertEqual(captured["table"], "project_files")
        self.assertIn("(archived_at,is,null)", captured["params"]["where"])

    def test_list_project_files_include_archived_removes_archived_filter(self):
        c = self._client()
        captured = {}

        def fake_get_paginated(table, params, page_size=50):
            captured["params"] = params
            return []

        c._get_paginated = fake_get_paginated
        c.list_project_files(project_id=7, include_archived=True)

        self.assertNotIn("archived_at", captured["params"]["where"])

    def test_list_project_files_archived_only_filters_archived(self):
        c = self._client()
        captured = {}

        def fake_get_paginated(table, params, page_size=50):
            captured["params"] = params
            return []

        c._get_paginated = fake_get_paginated
        c.list_project_files(project_id=7, archived_only=True)

        self.assertIn("(archived_at,isnot,null)", captured["params"]["where"])

    def test_archive_project_file_is_idempotent_when_missing(self):
        c = self._client()
        c.get_project_file = lambda project_id, path, include_archived=False: None

        out = c.archive_project_file(4, "/src/missing.py")

        self.assertEqual(out.get("ok"), True)
        self.assertEqual(out.get("already_absent"), True)

    def test_write_project_file_version_rejects_oversized_content(self):
        c = self._client()
        c.get_project_file = lambda project_id, path: {"Id": 10, "current_version_id": 1}
        c.get_project_file_version = lambda version_id: {"Id": 1, "version": 1, "content_hash": "x"}

        with self.assertRaises(ValueError):
            c.write_project_file_version(
                project_id=1,
                path="/src/large.txt",
                content="a" * (PROJECT_MAX_FILE_BYTES + 1),
            )

    def test_write_project_file_version_rejects_project_file_limit(self):
        c = self._client()
        c.get_project_file = lambda project_id, path: None
        c.list_project_files = lambda project_id, limit=5000, **kwargs: [{} for _ in range(PROJECT_MAX_FILES)]

        with self.assertRaises(ValueError):
            c.write_project_file_version(
                project_id=1,
                path="/src/new.py",
                content="print('x')\n",
            )

    def test_move_project_file_rejects_existing_destination(self):
        c = self._client()
        c.get_project_file = lambda project_id, path: {"Id": 1} if path in {"/src/a.py", "/src/b.py"} else None

        with self.assertRaises(ValueError):
            c.move_project_file(1, "/src/a.py", "/src/b.py")

    def test_restore_project_file_version_uses_target_content(self):
        c = self._client()
        c.get_project_file = lambda project_id, path, include_archived=False: {"Id": 10, "kind": "code", "mime": "text/plain"}
        c.list_project_file_versions = lambda file_id, limit=500: [{"version": 2, "content": "restored\n"}]
        captured = {}

        def fake_write(**kwargs):
            captured.update(kwargs)
            return {"Id": 10}, {"Id": 101, "version": 3}, True

        c.write_project_file_version = fake_write
        c.restore_project_file_version(project_id=1, path="/src/a.py", version=2)

        self.assertEqual(captured["content"], "restored\n")
        self.assertEqual(captured["edit_summary"], "restore v2")

    def test_add_project_audit_event_is_noop_when_table_missing(self):
        c = self._client()
        c.tables = {"projects": 1}
        out = c.add_project_audit_event(1, "org:1", "file_write", {"path": "/a.py"})
        self.assertIsNone(out)

    def test_add_project_audit_event_writes_when_table_present(self):
        c = self._client()
        c.tables["project_audit"] = 9
        captured = {}

        def fake_post(table, data):
            captured["table"] = table
            captured["data"] = data
            return {"Id": 1}

        c._post = fake_post
        c.add_project_audit_event(1, "org:1", "file_write", {"path": "/a.py"})

        self.assertEqual(captured["table"], "project_audit")
        self.assertEqual(captured["data"]["project_id"], 1)

    def test_list_project_audit_events_returns_empty_when_table_missing(self):
        c = self._client()
        c.tables = {"projects": 1}
        out = c.list_project_audit_events(project_id=1)
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()


