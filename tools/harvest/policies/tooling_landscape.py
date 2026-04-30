"""tooling_landscape — given a market category, identify tools and pull
structured rows (name, vendor, pricing, use case) into artifacts."""
from tools.harvest.policy import HarvestPolicy, register

POLICY = register(HarvestPolicy(
    name="tooling_landscape",
    seed_strategy="criteria_search",
    seed_strategy_params={"max_seeds": 30},
    walk_enabled=True,
    walk_max_depth=1,
    walk_max_pages=3,
    walk_same_host_only=True,
    walk_url_pattern=r"(pricing|features|about|product|solutions)",
    extract_schema={
        "name": "text",
        "vendor": "text",
        "primary_use_case": "text",
        "pricing_model": "text",
        "license": "text",
        "key_strength": "text",
        "key_weakness": "text",
    },
    persist_target="artifacts",
    persist_mode="upsert",
    rate_limit_per_host_s=2.0,
    max_pages=90,
    max_depth=1,
    timeout_total_s=3600,
    respect_robots=True,
))
