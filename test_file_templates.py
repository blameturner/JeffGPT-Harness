import unittest

from infra.file_templates import adr_path, adr_template, conventions_template, template_for


class TemplateTests(unittest.TestCase):
    def test_tsx_template_emits_component_with_proper_name(self):
        out = template_for("/src/components/UserCard.tsx")
        self.assertIn("export function UserCard", out)
        self.assertIn("UserCardProps", out)

    def test_python_template_has_module_docstring(self):
        out = template_for("/src/utils/helpers.py")
        self.assertIn('"""helpers."""', out)

    def test_markdown_template_includes_frontmatter(self):
        out = template_for("/notes/intro.md", name_hint="Intro")
        self.assertIn("title: Intro", out)
        self.assertIn("# Intro", out)

    def test_unknown_extension_returns_empty(self):
        self.assertEqual(template_for("/path/to/binary.xyz"), "")

    def test_adr_template_renders_sections(self):
        body = adr_template(7, "Use Postgres")
        self.assertIn("# ADR-007: Use Postgres", body)
        self.assertIn("## Status", body)
        self.assertIn("## Decision", body)

    def test_adr_path_is_slugified_and_zero_padded(self):
        self.assertEqual(adr_path(3, "Use Postgres"), "/decisions/003-use-postgres.md")

    def test_conventions_template_has_required_sections(self):
        body = conventions_template()
        self.assertIn("# Project conventions", body)


if __name__ == "__main__":
    unittest.main()
