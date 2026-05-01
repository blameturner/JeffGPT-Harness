import unittest

from workers.code.config import CODE_DEFAULT_MODE, CODE_DEFAULT_STYLE, code_mode_prompt, code_style_prompt, resolve_code_mode


class CodeConfigModeTests(unittest.TestCase):
    def test_resolve_code_mode_maps_execute_alias(self):
        self.assertEqual(resolve_code_mode("execute"), "apply")

    def test_resolve_code_mode_defaults(self):
        self.assertEqual(resolve_code_mode(None), CODE_DEFAULT_MODE)

    def test_code_mode_prompt_returns_mode_specific_prompt(self):
        key, prompt = code_mode_prompt("decide")
        self.assertEqual(key, "decide")
        self.assertIn("ADR", prompt)

    def test_code_style_prompt_maps_legacy_keys(self):
        key, _ = code_style_prompt("general")
        self.assertEqual(key, CODE_DEFAULT_STYLE)
        key2, _ = code_style_prompt("test")
        self.assertEqual(key2, "tests")


if __name__ == "__main__":
    unittest.main()

