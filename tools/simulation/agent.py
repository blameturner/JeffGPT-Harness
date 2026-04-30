"""Simulation Lab — multi-agent rehearsal runs.

One row in ``simulations`` = one rehearsal. The handler walks ``max_turns``
of round-robin dialogue between ``participants`` grounded in ``scenario``,
appends each turn to ``transcript_json``, then writes a final ``debrief``.

Required ``simulations`` columns (NocoDB):
  Id              auto
  org_id          Number
  title           SingleLineText
  scenario        LongText
  participants_json  LongText   # JSON list of {"name": str, "persona": str}
  max_turns       Number       # default 8
  status          SingleLineText  # queued | running | completed | failed
  transcript_json LongText      # JSON list of {"turn": int, "speaker": str, "text": str}
  debrief         LongText
  error           SingleLineText
  started_at      SingleLineText
  completed_at    SingleLineText
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from infra.nocodb_client import NocodbClient
from shared.models import model_call
from tools._org import resolve_org_id

_log = logging.getLogger("simulation.agent")

TABLE = "simulations"

DEFAULT_MAX_TURNS = 8
TURN_MAX_TOKENS = 700
DEBRIEF_MAX_TOKENS = 2000


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _safe_json(raw, default):
    if raw in (None, ""):
        return default
    if not isinstance(raw, str):
        return raw if raw is not None else default
    try:
        return json.loads(raw)
    except Exception:
        return default


def _format_transcript(transcript: list[dict]) -> str:
    if not transcript:
        return "(no turns yet)"
    return "\n\n".join(
        f"[Turn {t.get('turn')}] {t.get('speaker')}: {t.get('text','').strip()}"
        for t in transcript
    )


def _turn_prompt(scenario: str, participant: dict, others: list[dict],
                 transcript: list[dict], turn_num: int, max_turns: int) -> str:
    name = participant.get("name", "Participant")
    persona = participant.get("persona", "").strip() or "(no persona supplied)"
    other_lines = "\n".join(
        f"- {p.get('name')}: {p.get('persona','').strip()[:300]}" for p in others
    ) or "(none)"
    return f"""You are role-playing as {name} in a multi-party simulation. Stay strictly in character.

YOUR PERSONA:
{persona}

OTHER PARTICIPANTS:
{other_lines}

SCENARIO:
{scenario}

TRANSCRIPT SO FAR:
{_format_transcript(transcript)}

This is turn {turn_num} of {max_turns} total. Produce ONLY {name}'s next contribution — 2-6 sentences, in first person, no stage directions, no quoting your own name as a prefix. Move the conversation forward; do not repeat what others have already said."""


def _debrief_prompt(scenario: str, participants: list[dict],
                    transcript: list[dict]) -> str:
    parts = "\n".join(f"- {p.get('name')}: {p.get('persona','').strip()[:200]}"
                      for p in participants)
    return f"""You are debriefing a completed multi-agent rehearsal.

SCENARIO:
{scenario}

PARTICIPANTS:
{parts}

FULL TRANSCRIPT:
{_format_transcript(transcript)}

Write a Markdown debrief with these sections (omit any that don't apply):

# Debrief

## Headline
2-3 sentences: what happened and how it ended.

## Key tensions
Bullet the points where participants disagreed sharply or where assumptions clashed.

## Surprises
Anything a thoughtful observer would not have predicted from the persona briefs alone.

## Open questions
Bullet questions raised by the conversation that nobody resolved.

## What to prepare for
If this rehearsal previewed a real upcoming event, the 3-5 things to be ready for, ranked by likelihood.

Output raw Markdown only, no preamble."""


def _load(plan_id: int):
    client = NocodbClient()
    if TABLE not in client.tables:
        return client, None
    rows = client._get(TABLE, params={"where": f"(Id,eq,{plan_id})", "limit": 1}).get("list", [])
    return client, (rows[0] if rows else None)


def _patch(client: NocodbClient, sim_id: int, fields: dict):
    fields["Id"] = sim_id
    client._patch(TABLE, sim_id, fields)


def run_simulation_job(payload: dict) -> dict:
    sim_id = payload.get("sim_id") or payload.get("simulation_id")
    if not sim_id:
        return {"status": "failed", "error": "missing sim_id"}
    sim_id = int(sim_id)

    client, row = _load(sim_id)
    if row is None:
        return {"status": "failed", "error": f"simulation {sim_id} not found", "sim_id": sim_id}

    org_id = resolve_org_id(row.get("org_id"))
    scenario = (row.get("scenario") or "").strip()
    participants = _safe_json(row.get("participants_json"), [])
    if not isinstance(participants, list):
        participants = []
    participants = [p for p in participants if isinstance(p, dict) and p.get("name")]
    if not scenario or len(participants) < 2:
        msg = "scenario required and at least 2 participants"
        _patch(client, sim_id, {"status": "failed", "error": msg, "completed_at": _now_iso()})
        return {"status": "failed", "error": msg, "sim_id": sim_id}

    try:
        max_turns = int(row.get("max_turns") or DEFAULT_MAX_TURNS)
    except Exception:
        max_turns = DEFAULT_MAX_TURNS
    max_turns = max(2, min(max_turns, 30))

    transcript = _safe_json(row.get("transcript_json"), [])
    if not isinstance(transcript, list):
        transcript = []
    start_turn = len(transcript)

    _patch(client, sim_id, {
        "status": "running",
        "started_at": row.get("started_at") or _now_iso(),
        "error": "",
    })
    _log.info("sim run START  sim_id=%d  org=%d  participants=%d  max_turns=%d  resume_from=%d",
              sim_id, org_id, len(participants), max_turns, start_turn)

    for turn_index in range(start_turn, max_turns):
        speaker = participants[turn_index % len(participants)]
        others = [p for p in participants if p is not speaker]
        prompt = _turn_prompt(scenario, speaker, others, transcript,
                              turn_index + 1, max_turns)
        text, _ = model_call("research_section_writer", prompt,
                             temperature=0.6, max_tokens=TURN_MAX_TOKENS)
        text = (text or "").strip()
        if not text:
            err = f"empty model response at turn {turn_index + 1} (speaker={speaker.get('name')})"
            _log.warning("sim run  sim_id=%d  %s", sim_id, err)
            _patch(client, sim_id, {
                "status": "failed",
                "error": err[:500],
                "transcript_json": json.dumps(transcript),
                "completed_at": _now_iso(),
            })
            return {"status": "failed", "error": err, "sim_id": sim_id, "turns_completed": len(transcript)}

        transcript.append({
            "turn": turn_index + 1,
            "speaker": speaker.get("name"),
            "text": text,
        })
        # Persist after every turn so a crash mid-run doesn't lose progress
        # and the user sees the conversation streaming into the row.
        _patch(client, sim_id, {"transcript_json": json.dumps(transcript)})
        _log.info("sim turn  sim_id=%d  turn=%d/%d  speaker=%s  chars=%d",
                  sim_id, turn_index + 1, max_turns, speaker.get("name"), len(text))

    debrief_text, _ = model_call(
        "research_reviewer",
        _debrief_prompt(scenario, participants, transcript),
        temperature=0.3,
        max_tokens=DEBRIEF_MAX_TOKENS,
    )
    debrief_text = (debrief_text or "").strip() or "# Debrief\n\n(debrief generation returned empty)"

    _patch(client, sim_id, {
        "status": "completed",
        "transcript_json": json.dumps(transcript),
        "debrief": debrief_text,
        "completed_at": _now_iso(),
    })
    _log.info("sim run DONE  sim_id=%d  turns=%d  debrief_chars=%d",
              sim_id, len(transcript), len(debrief_text))
    return {
        "status": "completed",
        "sim_id": sim_id,
        "turns": len(transcript),
        "debrief_chars": len(debrief_text),
    }
