import unittest

from workers.code.fs_tools import apply_tool_directives, parse_tool_directives


class _FakeDb:
    def __init__(self):
        self.files = {"/src/a.py": {"Id": 1, "current_version_id": 11}}
        self.versions = {11: {"Id": 11, "version": 1, "content": "old\n"}}
        self.archived = []
        self.writes = []

    def list_project_files(self, project_id):
        return [{"path": k, **v} for k, v in self.files.items()]

    def get_project_file(self, project_id, path):
        return self.files.get(path)

    def get_project_file_version(self, version_id):
        return self.versions.get(version_id)

    def archive_project_file(self, project_id, path, audit_actor=None):
        self.archived.append((project_id, path, audit_actor))
        return {"ok": True}

    def write_project_file_version(self, **kwargs):
        self.writes.append(kwargs)
        return {"Id": 1}, {"version": 2}, True


class _FakeLockedDb(_FakeDb):
    def write_project_file_version(self, **kwargs):
        raise PermissionError("file is locked: " + kwargs.get("path", ""))


class FsToolTests(unittest.TestCase):
    def test_parse_tool_directives(self):
        text = "```tool name=fs_read path=/src/a.py\n\n```"
        directives = parse_tool_directives(text)
        self.assertEqual(len(directives), 1)
        self.assertEqual(directives[0].name, "fs_read")
        self.assertEqual(directives[0].path, "/src/a.py")

    def test_apply_tool_directives_write_and_delete(self):
        db = _FakeDb()
        text = (
            "```tool name=fs_write path=/src/a.py summary=\"rewrite\"\n"
            "print('x')\n"
            "```\n"
            "```tool name=fs_delete path=/src/a.py\n\n```\n"
        )
        changes, tool_events = apply_tool_directives(db, 7, text, conversation_id=55, assistant_message_id=77)
        self.assertEqual(len(changes), 2)
        self.assertEqual(len(tool_events), 2)
        self.assertEqual(db.writes[0]["path"], "/src/a.py")
        self.assertEqual(db.writes[0]["created_by"], "agent:55")
        self.assertEqual(db.archived[0][1], "/src/a.py")

    def test_apply_tool_directives_locked_write_emits_permission_required(self):
        db = _FakeLockedDb()
        text = (
            "```tool name=fs_write path=/src/a.py summary=\"rewrite\"\n"
            "print('x')\n"
            "```\n"
        )
        changes, tool_events = apply_tool_directives(db, 7, text, conversation_id=55, assistant_message_id=77)
        self.assertEqual(changes, [])
        self.assertEqual(len(tool_events), 1)
        self.assertTrue(tool_events[0].get("permission_required"))


if __name__ == "__main__":
    unittest.main()


