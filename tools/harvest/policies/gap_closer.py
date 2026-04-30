"""gap_closer — given a topic representing a known coverage gap, run
targeted search + ingest to close the gap. Sister policy to
``topic_seeder`` but tuned for narrow, lower-cost top-ups."""
from tools.harvest.policy import HarvestPolicy, register

POLICY = register(HarvestPolicy(
    name="gap_closer",
    seed_strategy="topic_search",
    seed_strategy_params={"max_seeds": 8},
    walk_enabled=False,
    extract_schema=None,
    persist_target="knowledge",
    persist_mode="insert",
    rate_limit_per_host_s=1.5,
    max_pages=8,
    max_depth=1,
    timeout_total_s=900,
    respect_robots=True,
))
