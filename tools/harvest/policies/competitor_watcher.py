"""competitor_watcher — periodically re-fetch competitor URLs and persist
*diff records* (only what changed) into harvest_runs.artifacts_json.

Seed: a list of competitor URLs (URL-list seed_strategy) or a single URL.
Walk: shallow same-host walk that biases toward pricing/product/blog pages.
Persist: artifacts mode 'diff' — second-and-later runs only record changes.
"""
from tools.harvest.policy import HarvestPolicy, register

POLICY = register(HarvestPolicy(
    name="competitor_watcher",
    seed_strategy="url_list",
    walk_enabled=True,
    walk_max_depth=1,
    walk_max_pages=10,
    walk_same_host_only=True,
    # walk_link_class="all" because most marketing sites use <section>/<div>
    # rather than <article>; the url_pattern below is the real filter.
    walk_link_class="all",
    walk_url_pattern=r"(pricing|product|features|blog|news|release|update)",
    extract_schema={
        "headline": "text",
        "summary": "text",
        "section": "text",
    },
    persist_target="artifacts",
    persist_mode="diff",
    rate_limit_per_host_s=2.0,
    max_pages=20,
    max_depth=1,
    timeout_total_s=1800,
    respect_robots=True,
))
