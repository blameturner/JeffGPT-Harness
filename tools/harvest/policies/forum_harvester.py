"""forum_harvester — search forums (HN, Reddit, SO etc.) for a topic,
extract Q→A structure into knowledge."""
from tools.harvest.policy import HarvestPolicy, register

POLICY = register(HarvestPolicy(
    name="forum_harvester",
    seed_strategy="criteria_search",
    seed_strategy_params={
        "sites": ["news.ycombinator.com", "reddit.com", "stackoverflow.com"],
        "max_seeds": 30,
    },
    walk_enabled=True,
    walk_max_depth=1,
    walk_max_pages=40,
    walk_same_host_only=True,
    # HN/Reddit/SO don't tag thread links inside <article>/<main>, so the
    # 'article' filter would drop everything. Pattern-match thread/comment
    # URLs directly instead.
    walk_link_class="all",
    walk_url_pattern=r"(/comments/|/item\?id=|/questions/|/r/[^/]+/comments/|/posts/)",
    extract_schema={
        "question": "text",
        "top_answer": "text",
        "score": "numeric",
        "thread_url": "text",
    },
    persist_target="knowledge",
    persist_mode="upsert",
    rate_limit_per_host_s=2.0,
    max_pages=40,
    max_depth=1,
    timeout_total_s=2400,
    respect_robots=True,
))
