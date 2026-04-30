"""press_monitor — track press / news mentions for a company or topic.
Schedule with cron for recurring runs (re-runs upsert by URL)."""
from tools.harvest.policy import HarvestPolicy, register

POLICY = register(HarvestPolicy(
    name="press_monitor",
    seed_strategy="criteria_search",
    seed_strategy_params={"max_seeds": 30},
    walk_enabled=False,
    extract_schema={
        "headline": "text",
        "publication": "text",
        "published": "date",
        "summary": "text",
        "sentiment": "text",
    },
    persist_target="knowledge",
    persist_mode="upsert",
    rate_limit_per_host_s=2.0,
    max_pages=30,
    max_depth=1,
    timeout_total_s=2400,
    respect_robots=True,
))
