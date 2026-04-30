"""job_board_collector — search configured job boards for criteria,
extract structured posting fields into artifacts."""
from tools.harvest.policy import HarvestPolicy, register

POLICY = register(HarvestPolicy(
    name="job_board_collector",
    seed_strategy="criteria_search",
    seed_strategy_params={
        "sites": ["linkedin.com/jobs", "indeed.com", "wellfound.com",
                  "ycombinator.com/jobs", "remoteok.com"],
        "max_seeds": 50,
    },
    walk_enabled=True,
    walk_max_depth=2,
    walk_max_pages=200,
    walk_same_host_only=True,
    # Job boards rarely use <article> — the url_pattern below is the filter.
    walk_link_class="all",
    walk_url_pattern=r"(/jobs?/|/posting|/career)",
    extract_schema={
        "title": "text",
        "company": "text",
        "location": "text",
        "remote": "text",
        "salary": "text",
        "posted_date": "date",
        "requirements": "text",
        "url": "text",
    },
    persist_target="artifacts",
    persist_mode="upsert",
    rate_limit_per_host_s=3.0,
    max_pages=200,
    max_depth=2,
    timeout_total_s=5400,
    respect_robots=True,
))
