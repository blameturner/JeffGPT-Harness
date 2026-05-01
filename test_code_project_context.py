import unittest

from infra.prompts import assemble_code_system_prompt
from infra.project_context import (
    build_context_inspector_metadata,
    build_context_inspector_summary,
    build_project_context_pack,
    coerce_retrieval_scope,
    find_query_snippet,
)


class CodeProjectContextTests(unittest.TestCase):
    def test_coerce_retrieval_scope_from_json_string(self):
        scope = coerce_retrieval_scope('["a","b"," "]')
        self.assertEqual(scope, ["a", "b"])

    def test_coerce_retrieval_scope_from_csv_string(self):
        scope = coerce_retrieval_scope("a, b , c")
        self.assertEqual(scope, ["a", "b", "c"])

    def test_assemble_code_system_prompt_includes_manifest_and_notice(self):
        prompt = assemble_code_system_prompt(
            mode="apply",
            style_prompt="focus on tests",
            project_name="alpha",
            project_slug="alpha",
            system_note="ship safely",
            pinned_context="/spec/a.md: ...",
            path_manifest="/src/app.py (v=2, 120B)",
            context_notice="truncated",
            interactive_fs=False,
        )
        self.assertIn("Workspace manifest", prompt)
        self.assertIn("Context budget notes", prompt)
        self.assertIn("/src/app.py", prompt)

    def test_build_project_context_pack_applies_budget_caps(self):
        class _Db:
            def list_project_files(self, project_id):
                return [
                    {"path": "/a.py", "pinned": 1, "current_version_id": 1, "size_bytes": 20000, "UpdatedAt": "2026-05-01"},
                    {"path": "/b.py", "pinned": 0, "current_version_id": 2, "size_bytes": 12, "UpdatedAt": "2026-05-01"},
                ]

            def get_project_file_version(self, version_id):
                if version_id == 1:
                    return {"version": 1, "content": "x" * 10000}
                return {"version": 1, "content": "tiny"}

        pack = build_project_context_pack(_Db(), 1, pin_per_file_char_cap=100, pin_total_char_cap=100)
        self.assertEqual(pack["truncated_files"], 1)
        self.assertEqual(pack["total_pinned_chars"], 100)
        self.assertIn("/a.py", pack["pinned_context"])
        self.assertIn("/b.py", pack["path_manifest"])

    def test_build_context_inspector_metadata_has_estimates(self):
        data = build_context_inspector_metadata(
            project_id=1,
            conversation_id=2,
            message_id=3,
            mode="plan",
            style="none",
            interactive_fs=False,
            retrieval_collections=["codebase_alpha"],
            system_prompt="abc",
            history=[{"role": "user", "content": "hello"}],
            user_message="ship it",
            context_pack={"pinned_file_count": 1, "truncated_files": 0, "total_pinned_chars": 42, "manifest_count": 2},
        )
        self.assertEqual(data["project_id"], 1)
        self.assertEqual(data["sections"]["history_count"], 1)
        self.assertGreater(data["token_estimate"], 0)
        self.assertGreater(data["char_estimate"], 0)

    def test_build_context_inspector_summary_compacts_payload(self):
        full = build_context_inspector_metadata(
            project_id=1,
            conversation_id=2,
            message_id=3,
            mode="apply",
            style="tests",
            interactive_fs=True,
            retrieval_collections=["a", "b"],
            system_prompt="hello",
            history=[{"role": "user", "content": "x"}],
            user_message="go",
            context_pack={"pinned_file_count": 2, "truncated_files": 1, "total_pinned_chars": 120, "manifest_count": 9},
        )
        summary = build_context_inspector_summary(full)
        self.assertEqual(summary["retrieval_collection_count"], 2)
        self.assertEqual(summary["history_count"], 1)
        self.assertEqual(summary["pinned_file_count"], 2)
        self.assertTrue(summary["interactive_fs"])

    def test_find_query_snippet_returns_windowed_snippet(self):
        text = "aaaa bbbb cccc target-token dddd eeee"
        snip = find_query_snippet(text, "target-token", radius=6)
        self.assertIsNotNone(snip)
        self.assertIn("target-token", snip)


if __name__ == "__main__":
    unittest.main()





