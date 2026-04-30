"""topic_seeder — given a topic, seed RAG with high-signal sources.

User-triggered counterpart of the existing `discover_agent` cron job.
"""
from tools.harvest.policy import HarvestPolicy, register

POLICY = register(HarvestPolicy(
    name="topic_seeder",
    seed_strategy="topic_search",
    seed_strategy_params={"max_seeds": 20},
    walk_enabled=False,
    extract_schema=None,
    persist_target="knowledge",
    persist_mode="insert",
    rate_limit_per_host_s=1.5,
    max_pages=20,
    max_depth=1,
    timeout_total_s=1800,
    respect_robots=True,
))
