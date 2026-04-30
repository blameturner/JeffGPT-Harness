"""HarvestPolicy dataclass + global registry.

A `HarvestPolicy` is a frozen config that drives the harvest runner.
Adding a new harvest pattern = create a policy in `tools.harvest.policies.*`
and call ``register(POLICY)`` at module level. Discovery happens via
imports in ``tools/harvest/__init__.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


SeedStrategy = Literal[
    "literal_url",      # seed = a single URL
    "url_list",         # seed = list of URLs
    "topic_search",     # seed = topic; planner generates queries → SearXNG → URLs
    "sitemap_expand",   # seed = sitemap URL (or homepage; we'll find /sitemap.xml)
    "rss_feed",         # seed = RSS URL; new entries since last run
    "table_column",     # seed = {table, column, filter?}; URLs from existing rows
    "criteria_search",  # seed = freeform criteria; planner builds queries
]

PersistTarget = Literal[
    "knowledge",         # standard ingestion (rag + summary + entities)
    "knowledge_update",  # update existing knowledge rows in place
    "graph_node",        # write/update graph node properties
    "artifacts",         # stash in harvest_runs.artifacts_json only
]

PersistMode = Literal["insert", "upsert", "diff"]

LinkClass = Literal["all", "article", "doc", "sitemap"]


@dataclass(frozen=True)
class HarvestPolicy:
    name: str

    # Seed strategy
    seed_strategy: SeedStrategy
    seed_strategy_params: dict = field(default_factory=dict)

    # Walk strategy (post-fetch link expansion)
    walk_enabled: bool = False
    walk_max_depth: int = 1
    walk_max_pages: int = 50
    walk_same_host_only: bool = True
    walk_link_class: LinkClass = "all"
    walk_url_pattern: str | None = None  # optional regex filter

    # Extraction
    extract_schema: dict | None = None
    # NOTE: extract_entities / max_cost_usd were removed in the
    # CPU/local-LLM hardening pass — neither was honoured by the runner
    # and they bred confusion. Graph extraction is now controlled
    # globally by `features.harvest.skip_graph_extract_on_persist`.
    # Cost capping (if needed) belongs in `shared.models.model_usage_scope`.

    # Persistence
    persist_target: PersistTarget = "knowledge"
    persist_mode: PersistMode = "insert"

    # Bounds — every harvest is hard-bounded
    rate_limit_per_host_s: float = 1.0
    max_pages: int = 100
    max_depth: int = 2
    # `head_only` short-circuits the LLM extract path entirely. Used by
    # broken_link_sweep to do a HEAD-only validation pass.
    head_only: bool = False
    timeout_total_s: int = 3600
    respect_robots: bool = True


REGISTRY: dict[str, HarvestPolicy] = {}


def register(policy: HarvestPolicy) -> HarvestPolicy:
    """Register a policy. Idempotent — re-registering with the same name
    overwrites the entry (handy for hot-reload)."""
    REGISTRY[policy.name] = policy
    return policy


def get_policy(name: str) -> HarvestPolicy | None:
    return REGISTRY.get(name)


def list_policies() -> list[dict]:
    """Catalog form for the API."""
    out = []
    for p in REGISTRY.values():
        out.append({
            "name": p.name,
            "seed_strategy": p.seed_strategy,
            "persist_target": p.persist_target,
            "persist_mode": p.persist_mode,
            "max_pages": p.max_pages,
            "max_cost_usd": p.max_cost_usd,
            "respect_robots": p.respect_robots,
            "walk_enabled": p.walk_enabled,
        })
    return sorted(out, key=lambda d: d["name"])
