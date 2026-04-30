"""domain_crawler — crawl a domain to depth N, ingest each page to knowledge."""
from tools.harvest.policy import HarvestPolicy, register

POLICY = register(HarvestPolicy(
    name="domain_crawler",
    seed_strategy="literal_url",
    walk_enabled=True,
    walk_max_depth=2,
    walk_max_pages=80,
    walk_same_host_only=True,
    walk_link_class="article",
    extract_schema=None,
    persist_target="knowledge",
    persist_mode="insert",
    rate_limit_per_host_s=2.0,
    max_pages=80,
    max_depth=2,
    timeout_total_s=3600,
    respect_robots=True,
))
