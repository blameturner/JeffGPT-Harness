"""Harvest persisters — write extraction outputs to the right destination.

Four persist targets:
  - knowledge        — standard ingestion via workers.post_turn.ingest_output
  - knowledge_update — update an existing knowledge row in place (by URL)
  - graph_node       — write/update graph node properties
  - artifacts        — append to harvest_runs.artifacts_json (no separate table)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from infra.nocodb_client import NocodbClient

_log = logging.getLogger("harvest.persister")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


# ── knowledge (standard ingestion) ──────────────────────────────────────────

def persist_knowledge(*, url: str, summary: str, fields: dict | None,
                      org_id: int, run_id: int, policy_name: str,
                      topic: str = "") -> bool:
    """Standard knowledge ingestion. Reuses the existing post_turn pipeline so
    summary lands in RAG with consistent metadata.

    By default (``features.harvest.skip_graph_extract_on_persist=true``) we
    do NOT queue a graph_extract job per URL — a 200-URL bulk_upload would
    otherwise queue 200 LLM jobs that contend for the same model slot.
    Graph relationships can be backfilled in a single pass instead. Override
    by setting the feature flag to false.
    """
    if not summary:
        return False
    try:
        from workers.post_turn import ingest_output
    except Exception:
        _log.warning("workers.post_turn.ingest_output unavailable")
        return False
    try:
        from infra.config import get_feature
        skip_graph = bool(get_feature("harvest", "skip_graph_extract_on_persist", True))
    except Exception:
        skip_graph = True

    extra = {
        "url": url,
        "harvest_run_id": run_id,
        "harvest_policy": policy_name,
    }
    if fields:
        extra["fields"] = fields

    try:
        ingest_output(
            output=summary,
            user_text=topic or url,
            org_id=org_id,
            conversation_id=0,
            model="harvest",
            rag_collection="harvest",
            knowledge_collection="harvest_knowledge",
            source="harvest",
            extra_metadata=extra,
            queue_graph_extract=not skip_graph,
        )
        return True
    except Exception:
        _log.warning("ingest_output failed for url=%s", url[:120], exc_info=True)
        return False


# ── knowledge_update (in-place update) ──────────────────────────────────────

def persist_knowledge_update(*, table: str, row_id: int, fields: dict,
                             client: NocodbClient | None = None) -> bool:
    """Update fields on an existing row (URL-bearing table). Used by
    url_column_backfill, stale_refresher, broken_link_sweep."""
    if not fields:
        return False
    client = client or NocodbClient()
    try:
        client._patch(table, row_id, fields)
        return True
    except Exception:
        _log.warning("knowledge_update patch failed table=%s row=%d", table, row_id, exc_info=True)
        return False


# ── graph_node ──────────────────────────────────────────────────────────────

def persist_graph_node(*, table: str, where: str, fields: dict,
                       client: NocodbClient | None = None) -> bool:
    """Update properties on an existing graph node, or insert a new one."""
    if not fields:
        return False
    client = client or NocodbClient()
    try:
        rows = client._get(table, params={"where": where, "limit": 1}).get("list", [])
    except Exception:
        _log.warning("graph_node lookup failed where=%s", where, exc_info=True)
        return False
    if rows:
        row_id = rows[0].get("Id")
        try:
            client._patch(table, row_id, fields)
            return True
        except Exception:
            _log.warning("graph_node patch failed", exc_info=True)
            return False
    # Insert new
    try:
        client._post(table, fields)
        return True
    except Exception:
        _log.warning("graph_node insert failed", exc_info=True)
        return False


# ── artifacts (in harvest_runs.artifacts_json) ──────────────────────────────

def persist_artifact_batch(*, run_id: int, key: str, items: list[dict],
                           mode: str = "append",
                           client: NocodbClient | None = None) -> bool:
    """Flush a batch of artifact items to harvest_runs.artifacts_json in ONE
    read-modify-write. Avoids the quadratic cost (and write race) of
    persisting each item individually during a long harvest run.

    Each ``item`` is dict with at least ``url``. ``mode`` matches
    persist_artifact: ``insert`` (always append), ``upsert`` (replace by
    url), ``diff`` (record only changed fields against prior).
    """
    if not items:
        return True
    client = client or NocodbClient()
    try:
        rows = client._get(
            "harvest_runs",
            params={"where": f"(Id,eq,{run_id})", "limit": 1},
        ).get("list", [])
    except Exception:
        _log.warning("harvest_runs lookup failed run=%d", run_id, exc_info=True)
        return False
    if not rows:
        return False
    row = rows[0]
    raw = row.get("artifacts_json") or "{}"
    try:
        arts = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except (json.JSONDecodeError, TypeError):
        arts = {}
    if not isinstance(arts, dict):
        arts = {}

    bucket = arts.setdefault(key, {"items": [], "by_url": {}})
    if not isinstance(bucket, dict):
        bucket = {"items": [], "by_url": {}}
        arts[key] = bucket
    bucket.setdefault("items", [])
    bucket.setdefault("by_url", {})

    now = _now_iso()
    for item in items:
        item = dict(item)
        item.setdefault("_persisted_at", now)
        url = item.get("url") or item.get("source_url") or ""
        if mode == "upsert" and url:
            prior = bucket["by_url"].get(url)
            if prior:
                prior.update(item)
            else:
                bucket["by_url"][url] = item
                bucket["items"].append(item)
        elif mode == "diff" and url:
            prior = bucket["by_url"].get(url)
            if prior:
                changed = {k: v for k, v in item.items() if prior.get(k) != v}
                if changed:
                    changed["_diff_at"] = now
                    changed["url"] = url
                    bucket["items"].append(changed)
                    prior.update(item)
            else:
                bucket["by_url"][url] = item
                bucket["items"].append(item)
        else:
            bucket["items"].append(item)
            if url:
                bucket["by_url"][url] = item

    try:
        client._patch("harvest_runs", row.get("Id"), {"artifacts_json": json.dumps(arts)})
        return True
    except Exception:
        _log.warning("artifacts_json batch patch failed run=%d", run_id, exc_info=True)
        return False


def persist_artifact(*, run_id: int, key: str, item: dict,
                     mode: str = "append",
                     client: NocodbClient | None = None) -> bool:
    """Append (or upsert / diff) a structured row into the run's
    artifacts_json under ``key`` (typically the policy name).

    artifacts_json shape:
        {
          "<policy>": {
            "items": [<item>, <item>, ...],
            "by_url": {"<url>": <item>}      # for upsert mode
          }
        }
    """
    client = client or NocodbClient()
    try:
        rows = client._get(
            "harvest_runs",
            params={"where": f"(Id,eq,{run_id})", "limit": 1},
        ).get("list", [])
    except Exception:
        _log.warning("harvest_runs lookup failed run=%d", run_id, exc_info=True)
        return False
    if not rows:
        return False
    row = rows[0]
    raw = row.get("artifacts_json") or "{}"
    try:
        arts = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except (json.JSONDecodeError, TypeError):
        arts = {}
    if not isinstance(arts, dict):
        arts = {}

    bucket = arts.setdefault(key, {"items": [], "by_url": {}})
    if not isinstance(bucket, dict):
        bucket = {"items": [], "by_url": {}}
        arts[key] = bucket
    bucket.setdefault("items", [])
    bucket.setdefault("by_url", {})

    item = dict(item)
    item.setdefault("_persisted_at", _now_iso())
    url = item.get("url") or item.get("source_url") or ""

    if mode == "upsert" and url:
        prior = bucket["by_url"].get(url)
        if prior:
            prior.update(item)
        else:
            bucket["by_url"][url] = item
            bucket["items"].append(item)
    elif mode == "diff" and url:
        prior = bucket["by_url"].get(url)
        if prior:
            # Compute keys-changed diff (cheap, no LLM)
            changed = {k: v for k, v in item.items() if prior.get(k) != v}
            if changed:
                changed["_diff_at"] = _now_iso()
                changed["url"] = url
                bucket["items"].append(changed)
                prior.update(item)
        else:
            bucket["by_url"][url] = item
            bucket["items"].append(item)
    else:
        bucket["items"].append(item)
        if url:
            bucket["by_url"][url] = item

    try:
        client._patch("harvest_runs", row.get("Id"), {"artifacts_json": json.dumps(arts)})
        return True
    except Exception:
        _log.warning("artifacts_json patch failed run=%d", run_id, exc_info=True)
        return False
