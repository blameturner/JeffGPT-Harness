"""Post-turn graph extraction.

Runs after every chat turn, off the critical path. Extracts entity
relationships from the combined user+assistant text, writes them to
FalkorDB with full provenance (conversation_id + source chunk id), and
mirrors a single annotated chunk into a dedicated Chroma collection
``chat_entity_mentions`` so the insight producer can walk entity →
source text in one hop.
"""
from __future__ import annotations

import json
import logging
import re
import uuid

_log = logging.getLogger("chat.graph")

_ENTITY_MENTIONS_COLLECTION = "chat_entity_mentions"


def _combined_text(user_text: str, assistant_text: str) -> str:
    return (
        f"USER TURN:\n{user_text.strip()}\n\n"
        f"ASSISTANT REPLY:\n{assistant_text.strip()}"
    )


def _extract_entity_names_from_triples_raw(raw_output: str) -> list[str]:
    """Peek inside the relationship extractor's JSON to recover the entity
    names so we can tag the Chroma chunk without re-invoking the LLM."""
    if not raw_output:
        return []
    try:
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw_output.strip()).strip()
        data = json.loads(cleaned)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out: set[str] = set()
    for t in data:
        if not isinstance(t, dict):
            continue
        for k in ("from_name", "to_name"):
            v = t.get(k)
            if isinstance(v, str) and v.strip():
                out.add(v.strip())
    return sorted(out)[:40]


def extract_and_write_graph(
    user_text: str,
    assistant_text: str,
    conversation_id: int,
    org_id: int,
) -> None:
    _log.info("graph extraction starting  conv=%d", conversation_id)
    try:
        from shared.relationships import _extract_relationships
    except Exception as e:
        _log.warning("graph extraction skipped — import failed: %s", e)
        return

    combined = _combined_text(user_text, assistant_text)

    # Reserve a chunk id *before* extraction so the edges written by
    # _extract_relationships can point back to the same chunk we embed
    # below. Nothing references it until both writes succeed.
    source_chunk_id = uuid.uuid4().hex

    try:
        written, tokens = _extract_relationships(
            combined,
            org_id,
            conversation_id=conversation_id,
            source_chunk_id=source_chunk_id,
        )
        _log.info(
            "graph extraction done  conv=%d written=%d tokens=%d chunk=%s",
            conversation_id, written, tokens, source_chunk_id,
        )
    except Exception:
        _log.warning("graph extraction failed", exc_info=True)
        return

    if written <= 0:
        # Nothing to annotate — don't spend the embed budget.
        return

    # Mirror the text into the entity-mentions collection so the insight
    # producer and graph-expanded recall can traverse chunk → entities
    # (via metadata) or entity → chunk (via the source_chunks list on the
    # edge).
    try:
        _write_entity_mention_chunk(
            org_id=org_id,
            conversation_id=conversation_id,
            chunk_id=source_chunk_id,
            text=combined,
        )
    except Exception:
        _log.warning(
            "entity-mentions mirror failed  conv=%d chunk=%s",
            conversation_id, source_chunk_id, exc_info=True,
        )


def _write_entity_mention_chunk(
    org_id: int,
    conversation_id: int,
    chunk_id: str,
    text: str,
) -> None:
    """Write a single chunk to Chroma tagged with the conversation + chunk
    id. The entity names themselves live on the graph edges
    (``r.source_chunks`` contains this chunk id) so we don't need to
    duplicate them here — this collection is a *reverse index* the insight
    producer uses to pull source text for any edge it cites."""
    from infra.config import scoped_collection
    from infra.embedder import embed
    from infra.memory import client as chroma_client

    scoped = scoped_collection(org_id, _ENTITY_MENTIONS_COLLECTION)
    coll = chroma_client.get_or_create_collection(scoped)

    snippet = (text or "").strip()[:4000]
    if not snippet:
        return

    try:
        vec = embed(snippet)
    except Exception:
        _log.warning("entity-mentions embed failed  chunk=%s", chunk_id, exc_info=True)
        return

    try:
        coll.add(
            ids=[chunk_id],
            embeddings=[vec],
            documents=[snippet],
            metadatas=[{
                "conversation_id": int(conversation_id),
                "org_id": int(org_id),
                "chunk_id": chunk_id,
                "kind": "chat_turn",
            }],
        )
    except Exception:
        _log.warning("entity-mentions add failed  chunk=%s", chunk_id, exc_info=True)
