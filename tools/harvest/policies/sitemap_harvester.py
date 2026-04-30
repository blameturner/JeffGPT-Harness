"""sitemap_harvester — given a sitemap URL (or a homepage; we'll find
/sitemap.xml), batch-ingest every <loc> entry into knowledge."""
from tools.harvest.policy import HarvestPolicy, register

POLICY = register(HarvestPolicy(
    name="sitemap_harvester",
    seed_strategy="sitemap_expand",
    walk_enabled=False,
    extract_schema=None,
    persist_target="knowledge",
    persist_mode="insert",
    rate_limit_per_host_s=1.5,
    max_pages=200,
    max_depth=1,
    timeout_total_s=7200,
    respect_robots=True,
))
