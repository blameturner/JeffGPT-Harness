"""citation_crawler — given a paper / article URL, recursively follow
academic-style citation links to depth N, ingesting each into knowledge.
Citation edges land in the graph via the entity extractor."""
from tools.harvest.policy import HarvestPolicy, register

POLICY = register(HarvestPolicy(
    name="citation_crawler",
    seed_strategy="literal_url",
    walk_enabled=True,
    walk_max_depth=2,
    walk_max_pages=50,
    walk_same_host_only=False,
    walk_link_class="all",
    walk_url_pattern=r"(arxiv\.org|doi\.org|pubmed|jstor|semanticscholar|biorxiv|nature\.com|sciencedirect|wiley\.com)",
    extract_schema=None,
    persist_target="knowledge",
    persist_mode="insert",
    rate_limit_per_host_s=2.0,
    max_pages=50,
    max_depth=2,
    timeout_total_s=3600,
    respect_robots=True,
))
