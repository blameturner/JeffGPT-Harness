"""newsletter_archive — given a newsletter homepage, batch-ingest every
back issue. Walks same-host with article-class links."""
from tools.harvest.policy import HarvestPolicy, register

POLICY = register(HarvestPolicy(
    name="newsletter_archive",
    seed_strategy="literal_url",
    walk_enabled=True,
    walk_max_depth=2,
    walk_max_pages=300,
    walk_same_host_only=True,
    walk_link_class="article",
    walk_url_pattern=r"(issue|archive|posts?/|/p/|newsletter/)",
    extract_schema=None,
    persist_target="knowledge",
    persist_mode="upsert",
    rate_limit_per_host_s=2.0,
    max_pages=300,
    max_depth=2,
    timeout_total_s=7200,
    respect_robots=True,
))
