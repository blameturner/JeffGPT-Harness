"""funding_tracker — search for recent fundraises in given sectors,
extract round amount / investors / valuation into artifacts."""
from tools.harvest.policy import HarvestPolicy, register

POLICY = register(HarvestPolicy(
    name="funding_tracker",
    seed_strategy="criteria_search",
    seed_strategy_params={
        "sites": ["techcrunch.com", "crunchbase.com", "axios.com",
                  "businesswire.com", "prnewswire.com"],
        "max_seeds": 40,
    },
    walk_enabled=True,
    walk_max_depth=1,
    walk_max_pages=80,
    walk_same_host_only=True,
    walk_link_class="article",
    walk_url_pattern=r"(funding|series|raise|valuation|invest)",
    extract_schema={
        "company": "text",
        "round_type": "text",
        "amount_usd": "numeric",
        "lead_investor": "text",
        "other_investors": "text",
        "valuation_usd": "numeric",
        "announced_date": "date",
        "sector": "text",
    },
    persist_target="artifacts",
    persist_mode="upsert",
    rate_limit_per_host_s=2.0,
    max_pages=80,
    max_depth=1,
    timeout_total_s=3600,
    respect_robots=True,
))
