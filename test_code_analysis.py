import unittest

from infra.code_analysis import (
    cyclomatic_complexity,
    detect_tests,
    doc_coverage,
    extract_glossary,
    extract_imports,
    extract_symbols,
    parse_dep_files,
)


class SymbolExtractionTests(unittest.TestCase):
    def test_extract_python_class_and_def(self):
        content = "class Foo:\n    def bar(self, x):\n        return x\n"
        syms = extract_symbols("/a.py", content)
        kinds = {s.name: s.kind for s in syms}
        self.assertEqual(kinds.get("Foo"), "class")
        self.assertEqual(kinds.get("bar"), "method")

    def test_extract_typescript_export_function_and_const(self):
        content = "export function greet(name: string) {}\nexport const X = 1;\n"
        syms = extract_symbols("/a.ts", content)
        names = {s.name for s in syms}
        self.assertIn("greet", names)
        self.assertIn("X", names)

    def test_extract_markdown_headings(self):
        content = "# Title\n## Sub\n### Sub-sub\n"
        syms = extract_symbols("/doc.md", content)
        self.assertEqual([s.kind for s in syms], ["heading1", "heading2", "heading3"])


class ImportExtractionTests(unittest.TestCase):
    def test_extract_python_imports(self):
        self.assertEqual(extract_imports("/a.py", "import os\nfrom pathlib import Path\n"), ["os", "pathlib"])

    def test_extract_ts_imports(self):
        self.assertEqual(extract_imports("/a.ts", "import x from './foo';\nimport './bar';\n"), ["./foo", "./bar"])


class ComplexityTests(unittest.TestCase):
    def test_python_complexity_counts_branching(self):
        score = cyclomatic_complexity("/a.py", "def f(x):\n    if x and x > 0:\n        return x\n    return 0\n")
        self.assertGreater(score, 2)


class DocCoverageTests(unittest.TestCase):
    def test_python_doc_coverage(self):
        content = (
            'def documented():\n    """yes"""\n    return 1\n\n'
            'def undocumented():\n    return 1\n'
        )
        d = doc_coverage("/a.py", content)
        self.assertEqual(d["total"], 2)
        self.assertEqual(d["documented"], 1)


class TestDiscoveryTests(unittest.TestCase):
    def test_detects_python_test_files(self):
        files = [
            {"path": "/test_foo.py", "content": "def test_one():\n    pass\ndef test_two():\n    pass\n"},
            {"path": "/src/app.py", "content": "def main(): pass\n"},
        ]
        out = detect_tests(files)
        self.assertEqual(out["test_files"], 1)
        self.assertEqual(out["total_tests"], 2)


class DepParseTests(unittest.TestCase):
    def test_parse_package_json(self):
        files = [{"path": "/package.json", "content": '{"dependencies": {"react": "^18.0.0"}}'}]
        deps = parse_dep_files(files)
        self.assertEqual(deps[0]["name"], "react")
        self.assertEqual(deps[0]["manager"], "npm")

    def test_parse_requirements_txt(self):
        files = [{"path": "/requirements.txt", "content": "requests>=2.0\nnumpy==1.26.0\n# comment\n"}]
        deps = parse_dep_files(files)
        names = [d["name"] for d in deps]
        self.assertIn("requests", names)
        self.assertIn("numpy", names)


class GlossaryTests(unittest.TestCase):
    def test_extracts_repeated_capitalised_terms(self):
        files = [{"content": "The Auth Provider does X. The Auth Provider also does Y."}]
        terms = extract_glossary(files)
        self.assertTrue(any(t["term"] == "Auth Provider" for t in terms))


if __name__ == "__main__":
    unittest.main()
