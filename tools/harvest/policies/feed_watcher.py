"""feed_watcher — poll an RSS/Atom feed, ingest new entries into knowledge.

`persist_mode` is "upsert" so re-runs deduplicate by URL.
"""
from tools.harvest.policy import HarvestPolicy, register

POLICY = register(HarvestPolicy(
    name="feed_watcher",
    seed_strategy="rss_feed",
    walk_enabled=False,
    extract_schema=None,
    persist_target="knowledge",
    persist_mode="upsert",
    rate_limit_per_host_s=1.5,
    max_pages=50,
    max_depth=1,
    timeout_total_s=1800,
    respect_robots=True,
))
