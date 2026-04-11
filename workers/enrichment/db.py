from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from config import (
    NOCODB_BASE_ID,
    NOCODB_TABLE_AGENT_RUNS,
    NOCODB_TABLE_ENRICHMENT_AGENTS,
    NOCODB_TABLE_ENRICHMENT_LOG,
    NOCODB_TABLE_ORGANISATION,
    NOCODB_TABLE_SCRAPE_TARGETS,
    NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS,
    NOCODB_TOKEN,
    NOCODB_URL,
)
from workers.crawler import should_recrawl

_log = logging.getLogger("enrichment_agent.db")


class EnrichmentDB:

    def __init__(self) -> None:
        self.base = f"{NOCODB_URL}/api/v1/db/data/noco/{NOCODB_BASE_ID}"
        self.headers = {"xc-token": NOCODB_TOKEN, "Content-Type": "application/json"}
        self.tables = self._load_tables()

    def _load_tables(self) -> dict[str, str]:
        r = requests.get(
            f"{NOCODB_URL}/api/v1/db/meta/projects/{NOCODB_BASE_ID}/tables",
            headers={"xc-token": NOCODB_TOKEN},
            timeout=10,
        )
        r.raise_for_status()
        return {t["title"]: t["id"] for t in r.json()["list"]}

    def _get(self, table: str, params: dict | None = None) -> dict:
        r = requests.get(
            f"{self.base}/{self.tables[table]}",
            headers=self.headers,
            params=params,
            timeout=15,
        )
        r.raise_for_status()
        return r.json()

    def _post(self, table: str, data: dict) -> dict:
        r = requests.post(
            f"{self.base}/{self.tables[table]}",
            headers=self.headers,
            json=data,
            timeout=15,
        )
        r.raise_for_status()
        return r.json()

    def _patch(self, table: str, row_id: int, data: dict) -> dict:
        r = requests.patch(
            f"{self.base}/{self.tables[table]}/{row_id}",
            headers=self.headers,
            json=data,
            timeout=15,
        )
        r.raise_for_status()
        return r.json()

    def _delete(self, table: str, row_id: int) -> None:
        r = requests.delete(
            f"{self.base}/{self.tables[table]}/{row_id}",
            headers=self.headers,
            timeout=15,
        )
        r.raise_for_status()

    def list_orgs(self) -> list[dict]:
        if NOCODB_TABLE_ORGANISATION not in self.tables:
            return [{"Id": 1}]
        return self._get(NOCODB_TABLE_ORGANISATION, params={"limit": 500}).get("list", [])

    def has_running_inferences(self) -> bool:
        if NOCODB_TABLE_AGENT_RUNS not in self.tables:
            return False
        data = self._get(
            NOCODB_TABLE_AGENT_RUNS,
            params={"where": "(status,eq,running)", "limit": 1},
        )
        return bool(data.get("list"))

    def list_enrichment_agents(self, org_id: int | None = None) -> list[dict]:
        if NOCODB_TABLE_ENRICHMENT_AGENTS not in self.tables:
            return []
        where = "(active,eq,1)"
        if org_id is not None:
            where = f"(org_id,eq,{org_id})~and(active,eq,1)"
        return self._get(NOCODB_TABLE_ENRICHMENT_AGENTS, params={"where": where, "limit": 200}).get("list", [])

    def get_enrichment_agent(self, agent_id: int) -> dict | None:
        if NOCODB_TABLE_ENRICHMENT_AGENTS not in self.tables:
            return None
        try:
            r = requests.get(
                f"{self.base}/{self.tables[NOCODB_TABLE_ENRICHMENT_AGENTS]}/{agent_id}",
                headers=self.headers,
                timeout=15,
            )
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def create_enrichment_agent(self, data: dict) -> dict:
        return self._post(NOCODB_TABLE_ENRICHMENT_AGENTS, data)

    def update_enrichment_agent(self, agent_id: int, data: dict) -> dict:
        return self._patch(NOCODB_TABLE_ENRICHMENT_AGENTS, agent_id, data)

    def list_sources(self, org_id: int, enrichment_agent_id: int | None = None, active_only: bool = False) -> list[dict]:
        where = f"(org_id,eq,{org_id})"
        if enrichment_agent_id is not None:
            where += f"~and(enrichment_agent_id,eq,{enrichment_agent_id})"
        if active_only:
            where += "~and(active,eq,1)"
        return self._get(
            NOCODB_TABLE_SCRAPE_TARGETS,
            params={"where": where, "limit": 500, "sort": "-CreatedAt"},
        ).get("list", [])

    def get_source(self, source_id: int) -> dict | None:
        try:
            r = requests.get(
                f"{self.base}/{self.tables[NOCODB_TABLE_SCRAPE_TARGETS]}/{source_id}",
                headers=self.headers,
                timeout=15,
            )
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def create_source(self, data: dict) -> dict:
        return self._post(NOCODB_TABLE_SCRAPE_TARGETS, data)

    def delete_source(self, source_id: int) -> None:
        self._delete(NOCODB_TABLE_SCRAPE_TARGETS, source_id)

    def flush_source(self, source_id: int) -> dict:
        return self._patch(NOCODB_TABLE_SCRAPE_TARGETS, source_id, {
            "content_hash": None,
            "last_scraped_at": None,
            "status": None,
            "chunk_count": 0,
        })

    def list_log(
        self,
        org_id: int | None = None,
        scrape_target_id: int | None = None,
        limit: int = 100,
    ) -> list[dict]:
        where_parts: list[str] = []
        if org_id is not None:
            where_parts.append(f"(org_id,eq,{org_id})")
        if scrape_target_id is not None:
            where_parts.append(f"(scrape_target_id,eq,{scrape_target_id})")
        params: dict = {"sort": "-CreatedAt", "limit": limit}
        if where_parts:
            params["where"] = "~and".join(where_parts)
        return self._get(NOCODB_TABLE_ENRICHMENT_LOG, params=params).get("list", [])

    def list_suggestions(self, org_id: int, status: str | None = None) -> list[dict]:
        where = f"(org_id,eq,{org_id})"
        if status:
            where += f"~and(status,eq,{status})"
        return self._get(
            NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS,
            params={"where": where, "limit": 500, "sort": "-CreatedAt"},
        ).get("list", [])

    def get_suggestion(self, suggestion_id: int) -> dict | None:
        try:
            r = requests.get(
                f"{self.base}/{self.tables[NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS]}/{suggestion_id}",
                headers=self.headers,
                timeout=15,
            )
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def update_suggestion(self, suggestion_id: int, data: dict) -> dict:
        return self._patch(NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS, suggestion_id, data)

    def approve_suggestion(self, suggestion_id: int, org_id: int, enrichment_agent_id: int | None = None) -> dict:
        suggestion = self.get_suggestion(suggestion_id)
        if not suggestion:
            raise ValueError("suggestion not found")
        source = self.create_source({
            "org_id": org_id,
            "url": suggestion.get("url"),
            "name": suggestion.get("name"),
            "category": suggestion.get("category"),
            "parent_target": suggestion.get("parent_target"),
            "active": True,
            "frequency_hours": 24,
            "enrichment_agent_id": enrichment_agent_id,
        })
        self.update_suggestion(suggestion_id, {"status": "approved"})
        return source

    def list_tracked_urls(self, org_id: int) -> set[str]:
        tracked = self._get(
            NOCODB_TABLE_SCRAPE_TARGETS,
            params={"where": f"(org_id,eq,{org_id})", "limit": 1000},
        ).get("list", [])
        pending = self._get(
            NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS,
            params={
                "where": f"(org_id,eq,{org_id})~and(status,eq,pending)",
                "limit": 1000,
            },
        ).get("list", [])
        urls = {r.get("url") for r in tracked if r.get("url")}
        urls.update(r.get("url") for r in pending if r.get("url"))
        return urls

    def list_due_sources(self, org_id: int, enrichment_agent_id: int | None = None) -> list[dict]:
        where = f"(org_id,eq,{org_id})~and(active,eq,1)"
        if enrichment_agent_id is not None:
            where += f"~and(enrichment_agent_id,eq,{enrichment_agent_id})"
        data = self._get(
            NOCODB_TABLE_SCRAPE_TARGETS,
            params={
                "where": where,
                "limit": 500,
            },
        )
        rows = data.get("list", [])
        now = datetime.now(timezone.utc)

        # next_crawl_at authoritative; fall back to last_scraped_at+frequency_hours for legacy rows
        due = [r for r in rows if should_recrawl(r, now=now)]

        # priority bucket: 0 never-scraped, 1 normal due, 2 previously errored
        def priority(row: dict) -> tuple[int, float]:
            last = row.get("last_scraped_at")
            if not last:
                return (0, 0)
            try:
                last_dt = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
            except Exception:
                return (0, 0)
            overdue = (now - last_dt).total_seconds()
            status = (row.get("status") or "").lower()
            if status == "error":
                return (2, -overdue)
            return (1, -overdue)

        return sorted(due, key=priority)

    def update_scrape_target(self, row_id: int | None, **fields: Any) -> None:
        if row_id is None:
            return
        self._patch(NOCODB_TABLE_SCRAPE_TARGETS, row_id, fields)

    def log_event(
        self,
        cycle_id: str,
        event_type: str,
        org_id: int | None = None,
        scrape_target_id: int | None = None,
        source_url: str | None = None,
        message: str | None = None,
        chunks_stored: int = 0,
        tokens_used: int = 0,
        duration_seconds: float = 0.0,
        flags: list[str] | None = None,
    ) -> None:
        try:
            self._post(
                NOCODB_TABLE_ENRICHMENT_LOG,
                {
                    "cycle_id": cycle_id,
                    "event_type": event_type,
                    "org_id": org_id,
                    "scrape_target_id": scrape_target_id,
                    "source_url": source_url,
                    "message": message,
                    "chunks_stored": chunks_stored,
                    "tokens_used": tokens_used,
                    "duration_seconds": duration_seconds,
                    "flags": flags or [],
                },
            )
        except Exception as e:
            _log.error("log_event failed (%s)", event_type, exc_info=True)

    def record_suggestion(
        self,
        org_id: int,
        url: str,
        name: str,
        category: str,
        reason: str,
        confidence: str,
        confidence_score: int,
        suggested_by_url: str | None,
        suggested_by_cycle: str,
        parent_target: int | None = None,
    ) -> None:
        # dedupe client-side — URLs may contain chars that break Nocodb where filters
        pending = self._get(
            NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS,
            params={
                "where": f"(org_id,eq,{org_id})~and(status,eq,pending)",
                "limit": 1000,
            },
        ).get("list", [])
        for row in pending:
            if row.get("url") == url:
                self._patch(
                    NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS,
                    row["Id"],
                    {"times_suggested": int(row.get("times_suggested") or 1) + 1},
                )
                return
        self._post(
            NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS,
            {
                "org_id": org_id,
                "url": url,
                "name": name,
                "category": category,
                "reason": reason,
                "confidence": confidence,
                "confidence_score": confidence_score,
                "suggested_by_url": suggested_by_url,
                "suggested_by_cycle": suggested_by_cycle,
                "times_suggested": 1,
                "status": "pending",
                "parent_target": parent_target,
            },
        )

    def purge_old_logs(self, retention_days: int) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        data = self._get(
            NOCODB_TABLE_ENRICHMENT_LOG,
            params={
                "where": f"(CreatedAt,lt,{cutoff.isoformat()})",
                "limit": 1000,
            },
        )
        rows = data.get("list", [])
        for row in rows:
            try:
                self._delete(NOCODB_TABLE_ENRICHMENT_LOG, row["Id"])
            except Exception as e:
                _log.error("purge row %s failed", row.get("Id"), exc_info=True)
        return len(rows)

    def tokens_used_in_cycle(self, cycle_id: str) -> int:
        data = self._get(
            NOCODB_TABLE_ENRICHMENT_LOG,
            params={"where": f"(cycle_id,eq,{cycle_id})", "limit": 1000},
        )
        return sum(int(r.get("tokens_used") or 0) for r in data.get("list", []))
