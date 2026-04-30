"""url_column_backfill — fetch every URL in a table column, populate summary
and any structured fields onto the existing rows.

Params (passed through `params_json` on the harvest_runs row):
  table          — NocoDB table name (required)
  column         — column holding URLs (default: "url")
  missing_field  — only process rows where this column is blank (optional)
  limit          — cap rows scanned (default 500)
"""
from tools.harvest.policy import HarvestPolicy, register

POLICY = register(HarvestPolicy(
    name="url_column_backfill",
    seed_strategy="table_column",
    walk_enabled=False,
    extract_schema=None,        # caller can override per-run via params
    persist_target="knowledge_update",
    persist_mode="upsert",
    rate_limit_per_host_s=1.0,
    max_pages=500,
    max_depth=1,
    timeout_total_s=7200,
    respect_robots=True,
))
