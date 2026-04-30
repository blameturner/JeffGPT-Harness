"""single_url — fetch one URL (or a list of URLs) and ingest into knowledge.
Used by /harvest/scrape-now and /harvest/bulk-upload."""
from tools.harvest.policy import HarvestPolicy, register

POLICY = register(HarvestPolicy(
    name="single_url",
    seed_strategy="url_list",
    walk_enabled=False,
    extract_schema=None,
    persist_target="knowledge",
    persist_mode="upsert",
    rate_limit_per_host_s=1.0,
    max_pages=200,
    max_depth=1,
    timeout_total_s=2400,
    respect_robots=True,
))
