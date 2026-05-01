import unittest

from infra.plan_preview import extract_plan_file_intents


class PlanPreviewTests(unittest.TestCase):
    def test_extract_plan_file_intents_detects_create_edit_delete(self):
        text = """
        Files:
        - path=/src/new.py create module
        - Edit /src/existing.py to refactor
        - delete /src/old.py
        """
        intents = extract_plan_file_intents(text, {"/src/existing.py", "/src/old.py"})
        by_path = {i["path"]: i["action"] for i in intents}
        self.assertEqual(by_path["/src/new.py"], "create")
        self.assertEqual(by_path["/src/existing.py"], "edit")
        self.assertEqual(by_path["/src/old.py"], "delete")


if __name__ == "__main__":
    unittest.main()

