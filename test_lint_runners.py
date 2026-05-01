import unittest

from infra.lint_runners import lint_file


class LintRunnerTests(unittest.TestCase):
    def test_python_syntax_error_is_reported(self):
        issues = lint_file("/a.py", "def broken(:\n    pass\n")
        rules = {i["rule"] for i in issues}
        self.assertIn("py-syntax", rules)

    def test_python_unused_import_is_flagged(self):
        issues = lint_file("/a.py", "import os\nprint('hi')\n")
        self.assertTrue(any(i["rule"] == "unused-import" for i in issues))

    def test_python_used_import_is_not_flagged(self):
        issues = lint_file("/a.py", "import os\nprint(os.getcwd())\n")
        self.assertFalse(any(i["rule"] == "unused-import" for i in issues))

    def test_security_rules_catch_eval_and_shell_true(self):
        issues = lint_file("/a.py", "x = ev" + "al('1+1')\nimport subprocess\nsubprocess.run('ls', shell=True)\n")
        rules = {i["rule"] for i in issues}
        self.assertIn("py-dyn-eval", rules)
        self.assertIn("py-shell-true", rules)

    def test_json_parse_error(self):
        issues = lint_file("/a.json", "{not json}")
        self.assertTrue(any(i["rule"] == "json-parse" for i in issues))

    def test_trailing_whitespace_detected(self):
        issues = lint_file("/a.py", "x = 1   \n")
        self.assertTrue(any(i["rule"] == "trailing-whitespace" for i in issues))

    def test_missing_final_newline_detected(self):
        issues = lint_file("/a.py", "x = 1")
        self.assertTrue(any(i["rule"] == "no-final-newline" for i in issues))

    def test_markdown_excessive_heading_depth(self):
        issues = lint_file("/a.md", "####### too deep\n")
        self.assertTrue(any(i["rule"] == "md-heading-depth" for i in issues))


if __name__ == "__main__":
    unittest.main()
