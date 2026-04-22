"""Home conversation helpers.

Each org gets a single rolling "home" conversation. All digest chat replies
and answers to assistant-initiated questions append to it so the frontend
dashboard renders a single continuous feed.

Kept in a separate module (not `infra.nocodb_client`) so the table/kind
convention lives near the home dashboard code it serves.
"""
from __future__ import annotations

import logging
from typing import Any

from infra.config import NOCODB_TABLE_CONVERSATIONS, get_feature
from infra.nocodb_client import NocodbClient

_log = logging.getLogger("home.conversation")

HOME_KIND = "home"
HOME_TITLE = "Home — ongoing"


def _default_model() -> str:
    return str(get_feature("home", "default_chat_model", "chat"))


def _find_home(client: NocodbClient, org_id: int) -> dict | None:
    try:
        rows = client._get_paginated(NOCODB_TABLE_CONVERSATIONS, params={
            "where": f"(org_id,eq,{org_id})~and(kind,eq,{HOME_KIND})",
            "sort": "-CreatedAt",
            "limit": 1,
        })
    except Exception:
        # `kind` column may not exist yet — fall back to title match so the
        # helper stays functional while the NocoDB column is being added.
        _log.debug("home lookup by kind failed, trying title", exc_info=True)
        try:
            rows = client._get_paginated(NOCODB_TABLE_CONVERSATIONS, params={
                "where": f"(org_id,eq,{org_id})~and(title,eq,{HOME_TITLE})",
                "sort": "-CreatedAt",
                "limit": 1,
            })
        except Exception:
            _log.warning("home conversation lookup failed  org=%d", org_id, exc_info=True)
            return None
    return rows[0] if rows else None


def get_or_create_home_conversation(org_id: int, model: str | None = None) -> dict:
    """Return the org's home conversation, creating it on first use."""
    client = NocodbClient()
    existing = _find_home(client, org_id)
    if existing:
        return existing

    model_name = model or _default_model()
    _log.info("creating home conversation  org=%d model=%s", org_id, model_name)
    try:
        row = client._post(NOCODB_TABLE_CONVERSATIONS, {
            "org_id": org_id,
            "model": model_name,
            "title": HOME_TITLE,
            "kind": HOME_KIND,
            "rag_enabled": 0,
            "rag_collection": "",
            "knowledge_enabled": 0,
        })
        return row
    except Exception:
        # NocoDB silently drops unknown columns, but if `kind` is *required-not-null*
        # the write could fail differently. Fall back to standard create without kind.
        _log.warning("home create with kind failed, retrying without", exc_info=True)
        return client.create_conversation(
            org_id=org_id,
            model=model_name,
            title=HOME_TITLE,
        )


def home_conversation_summary(org_id: int) -> dict[str, Any] | None:
    """Compact summary for the overview payload."""
    client = NocodbClient()
    convo = _find_home(client, org_id)
    if not convo:
        return None
    last_at: str | None = None
    try:
        msgs = client._get_paginated("messages", params={
            "where": f"(conversation_id,eq,{convo['Id']})",
            "sort": "-CreatedAt",
            "limit": 1,
        })
        if msgs:
            last_at = msgs[0].get("CreatedAt")
    except Exception:
        _log.debug("home last_message lookup failed  conv=%s", convo.get("Id"), exc_info=True)
    return {
        "id": convo.get("Id"),
        "title": convo.get("title") or HOME_TITLE,
        "model": convo.get("model"),
        "last_message_at": last_at,
    }
