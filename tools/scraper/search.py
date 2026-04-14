import logging
from tools.scraper.base import BaseScraper

_log = logging.getLogger("scraper.search")


class SearchScraper(BaseScraper):
    def __init__(self, timeout: int = 30):
        super().__init__(timeout)

    def scrape(self, url: str) -> dict:
        result = {"url": url, "text": "", "links": [], "domain": "", "status": "failed"}

        try:
            html = self.fetch(url)
            if not html:
                result["error"] = "empty_response"
                return result

            result["text"] = self.extract_text(html)
            result["links"] = self.extract_links(html, url)
            result["domain"] = self.get_domain(url)
            result["status"] = "ok"
        except Exception as e:
            result["error"] = str(e)[:200]
            _log.warning("search scrape failed  url=%s  error=%s", url[:80], e)

        return result