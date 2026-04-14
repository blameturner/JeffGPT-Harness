from __future__ import annotations

import logging
import re
from typing import Any

from tools.scraper.base import BaseScraper

_log = logging.getLogger("scraper.research")


class ResearchScraper(BaseScraper):
    def scrape(self, url: str, schema: dict[str, str] | None = None) -> dict:
        result: dict[str, Any] = {
            "url": url,
            "data": {},
            "text": "",
            "status": "failed",
        }

        text = self.fetch_text(url)
        if not text:
            html, _ = self.fetch_html(url)
            text = self.extract_text(html)

        if not text:
            result["error"] = "empty_response"
            return result

        try:
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
        data: dict[str, Any] = {}
        for field, field_type in schema.items():
            for pattern in self._patterns_for(field, field_type):
                matches = re.findall(pattern, text, re.IGNORECASE)
                if not matches:
                    continue
                raw = matches[0]
                if field_type == "numeric":
                    try:
                        data[field] = float(str(raw).replace(",", ""))
                    except (ValueError, TypeError):
                        data[field] = raw
                elif field_type == "percent":
                    try:
                        data[field] = float(str(raw))
                    except (ValueError, TypeError):
                        data[field] = raw
                else:
                    data[field] = raw
                break
        return data

    @staticmethod
    def _patterns_for(field: str, field_type: str) -> list[str]:
        escaped = re.escape(field)
        if field_type == "numeric":
            return [
                rf"{escaped}[\s:]+(\d+(?:,\d+)*(?:\.\d+)?)",
                rf"(\d+(?:,\d+)*)\s*{escaped}",
            ]
        if field_type == "date":
            return [
                rf"{escaped}[\s:]+(\d{{4}}-\d{{2}}-\d{{2}})",
                rf"{escaped}[\s:]+(\d{{1,2}}/\d{{1,2}}/\d{{4}})",
            ]
        if field_type == "percent":
            return [
                rf"{escaped}[\s:]+(\d+(?:\.\d+)?)\s*%",
            ]
        return [rf"{escaped}[\s:]+([^\.\n]+)"]