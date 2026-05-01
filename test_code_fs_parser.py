import unittest

from infra.paths import normalize_project_path
from workers.code.fs_parser import _apply_unified_patch, apply_file_fences, parse_file_fences


class PathNormalizeTests(unittest.TestCase):
    def test_normalize_project_path_accepts_clean_posix_path(self):
        self.assertEqual(normalize_project_path("/src//app.py"), "/src/app.py")

    def test_normalize_project_path_rejects_parent_traversal(self):
        with self.assertRaises(ValueError):
            normalize_project_path("/src/../secrets.txt")

    def test_normalize_project_path_rejects_query_breaking_characters(self):
        with self.assertRaises(ValueError):
            normalize_project_path("/src/a,b.py")


class FileFenceParserTests(unittest.TestCase):
    def test_parse_file_fences_reads_metadata_and_content(self):
        text = (
            "```file path=/src/app.py mode=replace summary=\"init\"\n"
            "print('ok')\n"
            "```\n"
        )
        fences = parse_file_fences(text)
        self.assertEqual(len(fences), 1)
        self.assertEqual(fences[0].path, "/src/app.py")
        self.assertEqual(fences[0].mode, "replace")
        self.assertEqual(fences[0].summary, "init")
        self.assertIn("print", fences[0].content)

    def test_parse_file_fences_accepts_block_without_trailing_newline(self):
        text = "```file path=/src/a.py mode=replace summary=\"x\"\nprint('ok')```"
        fences = parse_file_fences(text)
        self.assertEqual(len(fences), 1)
        self.assertEqual(fences[0].path, "/src/a.py")

    def test_apply_unified_patch_updates_expected_line(self):
        base = "a\nb\nc\n"
        patch = (
            "@@ -1,3 +1,3 @@\n"
            " a\n"
            "-b\n"
            "+x\n"
            " c\n"
        )
        self.assertEqual(_apply_unified_patch(base, patch), "a\nx\nc\n")


class _FakeDb:
    def __init__(self):
        self.archived = []
        self.writes = []
        self.files = {
            "/src/app.py": {"Id": 1, "current_version_id": 11},
        }
        self.versions = {
            11: {"Id": 11, "version": 1, "content": "hello\n", "content_hash": "x"},
        }

    def archive_project_file(self, project_id, path, audit_actor=None):
        self.archived.append((project_id, path))

    def get_project_file(self, project_id, path):
        return self.files.get(path)

    def get_project_file_version(self, version_id):
        return self.versions.get(version_id)

    def write_project_file_version(self, **kwargs):
        self.writes.append(kwargs)
        file_row = {"Id": 1}
        version_row = {"version": len(self.writes) + 1}
        return file_row, version_row, True


class _FakeLockedDb(_FakeDb):
    def write_project_file_version(self, **kwargs):
        raise PermissionError("file is locked: " + kwargs.get("path", ""))


class ApplyFencesTests(unittest.TestCase):
    def test_apply_file_fences_writes_and_deletes(self):
        db = _FakeDb()
        text = (
            "```file path=/src/app.py mode=append summary=\"extend\"\n"
            "world\n"
            "```\n"
            "```file path=/src/old.py mode=delete summary=\"remove\"\n"
            "\n"
            "```\n"
        )
        changes = apply_file_fences(db=db, project_id=9, response_text=text, conversation_id=55, assistant_message_id=77)

        self.assertEqual(len(changes), 2)
        self.assertEqual(db.archived, [(9, "/src/old.py")])
        self.assertEqual(db.writes[0]["path"], "/src/app.py")
        self.assertEqual(db.writes[0]["created_by"], "agent:55")
        self.assertEqual(db.writes[0]["created_by_message_id"], 77)

    def test_apply_file_fences_respects_seen_keys_dedupe(self):
        db = _FakeDb()
        text = (
            "```file path=/src/app.py mode=replace summary=\"rewrite\"\n"
            "print('one')\n"
            "```\n"
        )
        seen = set()
        first = apply_file_fences(db=db, project_id=9, response_text=text, seen_keys=seen)
        second = apply_file_fences(db=db, project_id=9, response_text=text, seen_keys=seen)

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 0)
        self.assertEqual(len(db.writes), 1)

    def test_apply_file_fences_emits_permission_required_for_locked_file(self):
        db = _FakeLockedDb()
        text = (
            "```file path=/src/app.py mode=replace summary=\"rewrite\"\n"
            "print('one')\n"
            "```\n"
        )
        changes = apply_file_fences(db=db, project_id=9, response_text=text)

        self.assertEqual(len(changes), 1)
        self.assertTrue(changes[0].get("permission_required"))
        self.assertEqual(changes[0].get("path"), "/src/app.py")


if __name__ == "__main__":
    unittest.main()

