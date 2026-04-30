"""bookmark_importer — import a Netscape-style bookmarks export
(or a flat URL list), scrape, categorise, and persist."""
from tools.harvest.policy import HarvestPolicy, register

POLICY = register(HarvestPolicy(
    name="bookmark_importer",
    seed_strategy="url_list",
    seed_strategy_params={"source": "bookmarks_html"},
    walk_enabled=False,
    extract_schema={
        "category": "text",
        "summary": "text",
    },
    persist_target="knowledge",
    persist_mode="upsert",
    rate_limit_per_host_s=1.5,
    max_pages=2000,
    max_depth=1,
    timeout_total_s=14400,
    respect_robots=True,
))
