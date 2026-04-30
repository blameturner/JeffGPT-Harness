"""broken_link_sweep — HEAD-check every URL in a column, flag dead/moved
ones in place. No LLM cost.

The runner uses ``head_only=True`` to short-circuit fetch/extract and only
do a HEAD request, then patches ``dead_url`` and ``last_check_status`` flags
onto the source rows.
"""
from tools.harvest.policy import HarvestPolicy, register

POLICY = register(HarvestPolicy(
    name="broken_link_sweep",
    seed_strategy="table_column",
    walk_enabled=False,
    extract_schema=None,
    persist_target="knowledge_update",
    persist_mode="upsert",
    rate_limit_per_host_s=0.5,
    max_pages=2000,
    max_depth=1,
    head_only=True,
    timeout_total_s=7200,
    respect_robots=True,
))
