"""entity_profile_backfill — for each graph node missing a profile, run a
shallow lookup + structured extract, write properties back onto the node.

Params (via params_json):
  table          — graph nodes table (required)
  column         — column carrying canonical name (default 'name')
  missing_field  — only fill nodes lacking this field (default 'summary')
"""
from tools.harvest.policy import HarvestPolicy, register

POLICY = register(HarvestPolicy(
    name="entity_profile_backfill",
    seed_strategy="table_column",
    walk_enabled=True,
    walk_max_depth=1,
    walk_max_pages=3,
    walk_same_host_only=True,
    walk_link_class="article",
    extract_schema={
        "role": "text",
        "employer": "text",
        "summary": "text",
        "homepage": "text",
        "social": "text",
    },
    persist_target="graph_node",
    persist_mode="upsert",
    rate_limit_per_host_s=2.0,
    max_pages=300,
    max_depth=1,
    timeout_total_s=3600,
    respect_robots=True,
))
