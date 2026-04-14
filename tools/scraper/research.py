import logging
import re
from typing import Any

from tools.scraper.base import BaseScraper

_log = logging.getLogger("scraper.research")


class ResearchScraper(BaseScraper):
    def __init__(self, timeout: int = 30):
        super().__init__(timeout)

    def scrape(self, url: str, schema: dict[str, str] | None = None) -> dict:
        result = {
            "url": url,
            "data": {},
            "status": "failed"
        }

        try:
            html = self.fetch(url)
            if not html:
                result["error"] = "empty_response"
                return result

            text = self.extract_text(html)
            result["text"] = text[:5000]

            if schema:
                result["data"] = self._extract_schema_data(text, schema)
            else:
                result["data"] = {"raw": text[:2000]}

            result["status"] = "ok"
        except Exception as e:
            result["error"] = str(e)[:200]
            _log.warning("research scrape failed  url=%s  error=%s", url[:80], e)

        return result

    def _extract_schema_data(self, text: str, schema: dict[str, str]) -> dict[str, Any]:
        data = {}
        for field, field_type in schema.items():
            patterns = self._get_patterns(field, field_type)
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    if field_type == "numeric":
                        try:
                            data[field] = float(matches[0].replace(",", ""))
                        except ValueError:
                            data[field] = matches[0]
                    else:
                        data[field] = matches[0]
                    break
        return data

    def _get_patterns(self, field: str, field_type: str) -> list[str]:
        field_lower = field.lower().replace(" ", "_")
        if field_type == "numeric":
            return [
                rf"{field}[\s:]+(\d+(?:,\d+)*(?:\.\d+)?)",
                rf"(\d+(?:,\d+)*)\s*{field}",
            ]
        if field_type == "date":
            return [
                rf"{field}[\s:]+(\d{{4}}-\d{{2}}-\d{{2}})",
                rf"(\d{{1,2}}/\d{{1,2}}/\d{{4}})",
            ]
        if field_type == "percent":
            return [
                rf"{field}[\s:]+(\d+(?:\.\d+)?)%",
                rf"(\d+(?:\.\d+)?)\s*%",
            ]
        return [
            rf"{field}[\s:]+([^\.\n]+)",
        ]