import unittest

from infra.project_metrics import count_todo_markers, parse_period_days


class ProjectHelperTests(unittest.TestCase):
    def test_parse_period_days_supports_days_and_weeks(self):
        self.assertEqual(parse_period_days("14d"), 14)
        self.assertEqual(parse_period_days("2w"), 14)

    def test_parse_period_days_defaults_on_invalid(self):
        self.assertEqual(parse_period_days("abc"), 30)

    def test_count_todo_markers_counts_known_tokens(self):
        text = "TODO one\nFix later\nFIXME now\nxxx\nHACK this\nnote that"
        self.assertEqual(count_todo_markers(text), 5)


if __name__ == "__main__":
    unittest.main()


