from __future__ import annotations

import logging

from tools.scraper.base import BaseScraper

_log = logging.getLogger("scraper.search")


class SearchScraper(BaseScraper):
    def scrape(self, url: str) -> dict:
        result = {
            "url": url,
            "final_url": url,
            "text": "",
            "links": [],
            "domain": "",
            "canonical": url,
            "status": "failed",
        }

        html, final_url = self.fetch_html(url)
        if not html:
            result["error"] = "empty_response"
            return result

        try:
            result["final_url"] = final_url
            result["canonical"] = self.canonical_url(html, final_url)
            result["text"] = self.extract_text(html)
            result["links"] = self.extract_links(html, final_url)
            result["domain"] = self.get_domain(final_url)
            result["status"] = "ok"
        except Exception as e:
            result["error"] = str(e)[:200]
            _log.warning("search scrape failed  url=%s  error=%s", url[:80], e)

        return result
