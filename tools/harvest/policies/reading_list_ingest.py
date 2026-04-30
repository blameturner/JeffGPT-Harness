"""reading_list_ingest — bulk-process a list of saved-for-later URLs
(Pocket / Readwise export, or a manual list) into knowledge with
reading-time + key-takeaway annotations."""
from tools.harvest.policy import HarvestPolicy, register

POLICY = register(HarvestPolicy(
    name="reading_list_ingest",
    seed_strategy="url_list",
    walk_enabled=False,
    extract_schema={
        "reading_time_min": "numeric",
        "key_takeaway": "text",
        "topic": "text",
    },
    persist_target="knowledge",
    persist_mode="upsert",
    rate_limit_per_host_s=1.5,
    max_pages=200,
    max_depth=1,
    timeout_total_s=5400,
    respect_robots=True,
))
