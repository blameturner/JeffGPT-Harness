import unittest
from unittest.mock import patch

from infra import ai_flows


def _stub(text: str):
    return (text, 100)


class JsonExtractTests(unittest.TestCase):
    def test_extract_json_from_fenced_response(self):
        out = ai_flows._extract_json('```json\n{"a": 1}\n```')
        self.assertEqual(out, {"a": 1})

    def test_extract_json_handles_leading_prose(self):
        out = ai_flows._extract_json('Here you go:\n{"a": 1, "b": [2,3]}\nthanks')
        self.assertEqual(out, {"a": 1, "b": [2, 3]})

    def test_extract_json_returns_none_on_unparseable(self):
        self.assertIsNone(ai_flows._extract_json("not json at all"))


class CodeReviewTests(unittest.TestCase):
    def test_review_diff_parses_structured_response(self):
        payload = '{"summary": "ok", "concerns": [{"path": "/a.py", "line": 5, "severity": "warning", "comment": "x"}], "suggested_followups": ["test it"]}'
        with patch("infra.ai_flows.model_call", return_value=_stub(payload)):
            r = ai_flows.review_diff("--- a\n+++ b\n@@ x @@\n-foo\n+bar")
        self.assertEqual(r["summary"], "ok")
        self.assertEqual(len(r["concerns"]), 1)
        self.assertEqual(r["concerns"][0]["severity"], "warning")
        self.assertEqual(r["suggested_followups"], ["test it"])

    def test_review_diff_falls_back_when_model_returns_garbage(self):
        with patch("infra.ai_flows.model_call", return_value=_stub("???")):
            r = ai_flows.review_diff("diff")
        self.assertEqual(r["summary"], "")
        self.assertEqual(r["concerns"], [])


class FileSummaryTests(unittest.TestCase):
    def test_summarise_file_strips_fences_and_collapses_newlines(self):
        with patch("infra.ai_flows.model_call", return_value=_stub("This file\n  manages auth.\n\nIt does X.")):
            summary, _ = ai_flows.summarise_file("/auth.py", "code")
        self.assertEqual(summary, "This file manages auth. It does X.")


class SmartPasteTests(unittest.TestCase):
    def test_classify_paste_uses_defaults_when_model_fails(self):
        with patch("infra.ai_flows.model_call", return_value=_stub("nonsense")):
            r = ai_flows.classify_paste("hello")
        self.assertEqual(r["kind"], "note")
        self.assertEqual(r["suggested_path"], "/notes/paste.md")

    def test_classify_paste_returns_model_classification(self):
        payload = '{"kind": "code", "language": "python", "suggested_path": "/src/x.py", "reason": "looks like Python"}'
        with patch("infra.ai_flows.model_call", return_value=_stub(payload)):
            r = ai_flows.classify_paste("def foo(): pass")
        self.assertEqual(r["kind"], "code")
        self.assertEqual(r["language"], "python")
        self.assertEqual(r["suggested_path"], "/src/x.py")


class PlaybookTests(unittest.TestCase):
    def test_generate_playbook_normalises_missing_fields(self):
        payload = '{"steps": [{"title": "step 1", "description": "do it"}]}'
        with patch("infra.ai_flows.model_call", return_value=_stub(payload)):
            pb = ai_flows.generate_playbook("upgrade pydantic", [{"path": "/a.py"}])
        self.assertEqual(pb["goal"], "upgrade pydantic")
        self.assertEqual(len(pb["steps"]), 1)


class FAQTests(unittest.TestCase):
    def test_update_faq_strips_outer_fences(self):
        with patch("infra.ai_flows.model_call", return_value=_stub("```markdown\n# FAQ\n### Q: x\nA\n```")):
            body, _ = ai_flows.update_faq("(empty)", "x", "A")
        self.assertTrue(body.startswith("# FAQ"))
        self.assertNotIn("```", body)


class SpecRegenTests(unittest.TestCase):
    def test_regenerate_from_spec_strips_language_fences(self):
        with patch("infra.ai_flows.model_call", return_value=_stub("```python\nprint('x')\n```")):
            body, _ = ai_flows.regenerate_from_spec("/spec.yaml", "x: 1", "/a.py", "")
        self.assertEqual(body, "print('x')")


class GlossaryInjectionTests(unittest.TestCase):
    def test_assemble_includes_glossary_terms(self):
        from infra.prompts import assemble_code_system_prompt
        prompt = assemble_code_system_prompt(
            mode="chat", style_prompt="default",
            project_name="X", project_slug="x",
            glossary_terms=["Auth Provider", "Login Flow"],
        )
        self.assertIn("Project vocabulary", prompt)
        self.assertIn("Auth Provider", prompt)


if __name__ == "__main__":
    unittest.main()
