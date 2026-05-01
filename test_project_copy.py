import unittest

from infra.project_copy import import_from_project


class _FakeDb:
    def __init__(self):
        self.files = {
            (1, "/src/a.py"): {"Id": 10, "current_version_id": 110, "kind": "code", "mime": "text/plain"},
            (1, "/src/b.py"): {"Id": 11, "current_version_id": 111, "kind": "code", "mime": "text/plain"},
        }
        self.versions = {
            110: {"Id": 110, "content": "A\n"},
            111: {"Id": 111, "content": "B\n"},
        }
        self.writes = []

    def get_project_file(self, project_id, path):
        return self.files.get((project_id, path))

    def get_project_file_version(self, version_id):
        return self.versions.get(version_id)

    def write_project_file_version(self, **kwargs):
        self.writes.append(kwargs)
        # Mark one path as no-op to exercise skipped path.
        changed = kwargs.get("path") != "/src/b.py"
        return {"Id": 1}, {"Id": 2, "version": 2}, changed


class ProjectCopyTests(unittest.TestCase):
    def test_import_from_project_counts_written_skipped_missing(self):
        db = _FakeDb()
        out = import_from_project(
            db,
            src_project_id=1,
            dst_project_id=2,
            paths=["/src/a.py", "/src/b.py", "/src/missing.py"],
            actor="org:9",
        )

        self.assertEqual(out["written"], 1)
        self.assertEqual(out["skipped"], 1)
        self.assertEqual(out["missing"], 1)
        self.assertEqual(db.writes[0]["audit_kind"], "file_import_from")


if __name__ == "__main__":
    unittest.main()

