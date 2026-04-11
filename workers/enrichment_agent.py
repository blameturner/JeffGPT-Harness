from __future__ import annotations

from workers.enrichment.cycle import (
    _last_runs,
    get_last_run,
    run_enrichment_cycle,
    run_log_cleanup,
    sources_due_count,
)
from workers.enrichment.db import EnrichmentDB
from workers.enrichment.models import (
    FAST_TIMEOUT,
    _assert_not_reasoner,
    _fast_call,
    _model_call,
    _tool_call,
)
from workers.enrichment.proactive import _proactive_search
from workers.enrichment.processing import _process_source
from workers.enrichment.quality import (
    CONTENT_TYPE_ACCEPT,
    CONTENT_TYPE_ENUM,
    CONTENT_TYPE_REJECT,
    CONTENT_TYPE_SOFT_ACCEPT,
    VALIDATOR_CLASSIFIER_CHAR_BUDGET,
    VALIDATOR_MAX_TOP5_LINE_RATIO,
    VALIDATOR_MIN_LEN,
    VALIDATOR_MIN_UNIQUE_RATIO,
    _CLASSIFIER_PROMPT_TEMPLATE,
    _INJECTION_CHECK_PROMPT,
    _INJECTION_RESIDUE,
    _classify_content_type,
    _content_hash,
    _heuristic_quality_gate,
    _looks_like_injection,
    _validate_content,
)
from workers.enrichment.relationships import (
    _RELATIONSHIP_EXTRACTION_PROMPT,
    _extract_relationships,
)
from workers.enrichment.sources import _discover_sources, _verify_url_reachable
from workers.enrichment.summarise import _salvage_json_array, _summarise
