"""Model profile loader — maps model IDs to family-specific configuration.

Each JSON file in this directory defines a model family profile with:
  - match_patterns: substrings matched against the model_id (case-insensitive)
  - thinking.style: how the model signals thinking ("reasoning_content",
    "think_tags", or "none")
  - thinking.disable_params: extra payload params to suppress thinking

Profiles are loaded once at import time and cached.  Use `profile_for(model_id)`
to resolve a model_id to its profile dict.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

_log = logging.getLogger("model_profiles")

_PROFILES_DIR = Path(__file__).parent
_profiles: list[dict] = []
_default_profile: dict = {}


def _load_profiles() -> None:
    global _default_profile
    for path in sorted(_PROFILES_DIR.glob("*.json")):
        try:
            profile = json.loads(path.read_text())
        except Exception as e:
            _log.warning("failed to load model profile %s: %s", path.name, e)
            continue
        if profile.get("family") == "default":
            _default_profile = profile
        else:
            _profiles.append(profile)
    _log.info(
        "loaded %d model profiles (+default): %s",
        len(_profiles),
        [p.get("family") for p in _profiles],
    )


def profile_for(model_id: str) -> dict:
    """Return the profile dict for a model_id, falling back to default."""
    if not model_id:
        return _default_profile
    lower = model_id.lower()
    for profile in _profiles:
        for pattern in profile.get("match_patterns", []):
            if pattern.lower() in lower:
                return profile
    return _default_profile


def thinking_style(model_id: str) -> str:
    """Return the thinking style for a model: 'reasoning_content', 'think_tags', or 'none'."""
    return profile_for(model_id).get("thinking", {}).get("style", "none")


def thinking_tags(model_id: str) -> tuple[str, str]:
    """Return (open_tag, close_tag) for think_tags-style models."""
    t = profile_for(model_id).get("thinking", {})
    return t.get("open_tag", "<think>"), t.get("close_tag", "</think>")


def no_think_params_for(model_id: str) -> dict:
    """Return the payload params to disable thinking for a specific model."""
    return profile_for(model_id).get("thinking", {}).get("disable_params", {})


_load_profiles()
