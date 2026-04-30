"""stale_refresher — re-fetch the oldest knowledge / scrape_targets rows,
update the row in place. Designed for cron schedules.

Params (via params_json):
  table          — table to refresh (required, e.g. 'knowledge')
  column         — URL column (default 'url')
  age_days       — only rows with updated_at older than this (handled by SQL filter)
  limit          — cap rows scanned (default 100)
"""
from tools.harvest.policy import HarvestPolicy, register

POLICY = register(HarvestPolicy(
    name="stale_refresher",
    seed_strategy="table_column",
    walk_enabled=False,
    extract_schema=None,
    persist_target="knowledge_update",
    persist_mode="upsert",
    rate_limit_per_host_s=1.5,
    max_pages=100,
    max_depth=1,
    timeout_total_s=3600,
    respect_robots=True,
))
