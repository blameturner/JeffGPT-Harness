"""patent_watcher — pull recent patents / preprints matching keywords."""
from tools.harvest.policy import HarvestPolicy, register

POLICY = register(HarvestPolicy(
    name="patent_watcher",
    seed_strategy="criteria_search",
    seed_strategy_params={
        "sites": ["patents.google.com", "uspto.gov", "arxiv.org", "biorxiv.org"],
        "max_seeds": 40,
    },
    walk_enabled=False,
    extract_schema={
        "title": "text",
        "authors": "text",
        "abstract": "text",
        "filed_date": "date",
        "publication_number": "text",
        "claims_summary": "text",
    },
    persist_target="artifacts",
    persist_mode="upsert",
    rate_limit_per_host_s=2.0,
    max_pages=40,
    max_depth=1,
    timeout_total_s=3600,
    respect_robots=True,
))
