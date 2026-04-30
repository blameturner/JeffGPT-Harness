"""Harvest system — generic scraper/pathfinder-driven harvest pipeline.

Public entry points:
  run_harvest(run_id)       — main tool-queue handler
  finalise_harvest(run_id)  — finaliser handler (runs after sub-jobs)
  REGISTRY                  — policy name → HarvestPolicy
"""
from tools.harvest.policy import HarvestPolicy, REGISTRY, register
from tools.harvest.runner import run_harvest, finalise_harvest

# Policy modules — import order doesn't matter; each calls register() at top level.
from tools.harvest.policies import (  # noqa: F401
    bookmark_importer,
    broken_link_sweep,
    citation_crawler,
    competitor_watcher,
    domain_crawler,
    entity_profile_backfill,
    feed_watcher,
    forum_harvester,
    funding_tracker,
    gap_closer,
    job_board_collector,
    newsletter_archive,
    patent_watcher,
    press_monitor,
    reading_list_ingest,
    sitemap_harvester,
    single_url,
    stale_refresher,
    tooling_landscape,
    topic_seeder,
    url_column_backfill,
)

__all__ = [
    "HarvestPolicy",
    "REGISTRY",
    "register",
    "run_harvest",
    "finalise_harvest",
]
