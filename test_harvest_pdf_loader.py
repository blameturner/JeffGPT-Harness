import unittest

from tools.harvest.loaders import _collate_pdf_pages


class HarvestPdfLoaderTests(unittest.TestCase):
    def test_collate_pdf_pages_respects_page_and_char_caps(self):
        text, truncated, consumed = _collate_pdf_pages(
            ["a" * 5, "b" * 5, "c" * 5],
            max_pages=2,
            max_chars=8,
        )
        self.assertTrue(truncated)
        self.assertEqual(consumed, 2)
        self.assertEqual(text, "aaaaa\n\nbbb")

    def test_collate_pdf_pages_handles_empty_pages(self):
        text, truncated, consumed = _collate_pdf_pages(["", "ok"], max_pages=5, max_chars=10)
        self.assertFalse(truncated)
        self.assertEqual(consumed, 2)
        self.assertEqual(text, "ok")


if __name__ == "__main__":
    unittest.main()

