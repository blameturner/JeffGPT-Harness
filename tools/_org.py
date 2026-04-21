from __future__ import annotations

from infra.config import NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS
from infra.nocodb_client import NocodbClient


def count_inflight(client: NocodbClient, job_type: str) -> int:
    """Count queued+running rows of a given tool_jobs type. Dispatchers use
    this to decide whether to enqueue another job on this tick."""
    total = 0
    for status in ("queued", "running"):
        try:
            data = client._get("tool_jobs", params={
                "where": f"(type,eq,{job_type})~and(status,eq,{status})",
                "limit": 50,
            })
            total += len(data.get("list", []))
        except Exception:
            pass
    return total


def default_org_id(client: NocodbClient) -> int | None:
    """Resolve a single org_id for single-tenant periodic dispatchers.

    Scans scrape_targets and suggested_scrape_targets in newest-first order and
    returns the first non-zero org_id found. Returns None when the system has
    no enrichment data yet; callers should treat None as "no_org_context" and
    skip enqueuing. This is the same heuristic used by every enrichment
    dispatcher — keep it centralised so we never drift.
    """
    for table in ("scrape_targets", NOCODB_TABLE_SUGGESTED_SCRAPE_TARGETS):
        try:
            rows = client._get(table, params={
                "limit": 1,
                "fields": "org_id",
                "sort": "-CreatedAt",
            }).get("list", [])
            if rows:
                org = int(rows[0].get("org_id") or 0)
                if org:
                    return org
        except Exception:
            continue
    return None
