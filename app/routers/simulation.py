"""Simulation Lab API.

Endpoints:
  POST   /simulations                 body: {title, scenario, participants, max_turns?, org_id?}
  GET    /simulations                 ?status=&limit=
  GET    /simulations/{sim_id}
  POST   /simulations/{sim_id}/cancel

Backed by the ``simulations`` NocoDB table — see tools.simulation.agent for
the column list. Run is dispatched as a ``simulation_run`` tool-queue job.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from infra.nocodb_client import NocodbClient
from tools._org import resolve_org_id
from tools.simulation.agent import TABLE, DEFAULT_MAX_TURNS
from workers.tool_queue import get_tool_queue

_log = logging.getLogger("main.simulation")

router = APIRouter(prefix="/simulations", tags=["simulation"])


class Participant(BaseModel):
    name: str
    persona: str = ""


class CreateSimulationRequest(BaseModel):
    title: str = ""
    scenario: str
    participants: list[Participant]
    max_turns: int | None = None
    org_id: int | None = None


def _row_to_api(row: dict, include_transcript: bool = False) -> dict:
    participants = row.get("participants_json")
    if isinstance(participants, str):
        try:
            participants = json.loads(participants or "[]")
        except Exception:
            participants = []
    out = {
        "sim_id": row.get("Id"),
        "title": row.get("title") or "",
        "status": row.get("status") or "",
        "scenario": row.get("scenario") or "",
        "participants": participants,
        "max_turns": row.get("max_turns"),
        "started_at": row.get("started_at") or "",
        "completed_at": row.get("completed_at") or "",
        "error": row.get("error") or "",
        "org_id": row.get("org_id"),
    }
    if include_transcript:
        transcript = row.get("transcript_json")
        if isinstance(transcript, str):
            try:
                transcript = json.loads(transcript or "[]")
            except Exception:
                transcript = []
        out["transcript"] = transcript or []
        out["debrief"] = row.get("debrief") or ""
    return out


@router.post("")
def create_simulation(body: CreateSimulationRequest):
    if not body.scenario.strip():
        raise HTTPException(status_code=400, detail="scenario is required")
    parts = [p for p in (body.participants or []) if p.name.strip()]
    if len(parts) < 2:
        raise HTTPException(status_code=400, detail="at least 2 named participants required")

    org_id = int(body.org_id or resolve_org_id(0) or 1)
    max_turns = max(2, min(int(body.max_turns or DEFAULT_MAX_TURNS), 30))

    client = NocodbClient()
    if TABLE not in client.tables:
        raise HTTPException(status_code=500,
                            detail=f"{TABLE} table missing — create it in NocoDB first")

    payload = {
        "org_id": org_id,
        "title": (body.title or body.scenario.splitlines()[0])[:200],
        "scenario": body.scenario.strip(),
        "participants_json": json.dumps([p.model_dump() for p in parts]),
        "max_turns": max_turns,
        "status": "queued",
        "transcript_json": "[]",
        "debrief": "",
        "error": "",
        "started_at": "",
        "completed_at": "",
    }
    try:
        row = client._post(TABLE, payload)
        sim_id = row.get("Id")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"insert failed: {str(e)[:300]}")
    if not sim_id:
        raise HTTPException(status_code=500, detail="insert returned no Id")

    tq = get_tool_queue()
    if tq is None:
        client._patch(TABLE, sim_id, {"Id": sim_id, "status": "failed",
                                       "error": "tool_queue_unavailable"})
        raise HTTPException(status_code=503, detail="tool_queue_unavailable")

    job_id = tq.submit(
        "simulation_run",
        {"sim_id": sim_id, "org_id": org_id},
        source="simulation",
        priority=4,
        org_id=org_id,
    )
    _log.info("simulation queued  sim_id=%s  job=%s  participants=%d  turns=%d",
              sim_id, job_id, len(parts), max_turns)
    return {"status": "queued", "sim_id": sim_id, "job_id": job_id}


@router.get("")
def list_simulations(status: str | None = None, limit: int = 50,
                     org_id: int | None = None):
    limit = max(1, min(int(limit), 500))
    client = NocodbClient()
    if TABLE not in client.tables:
        return {"simulations": []}
    where_parts: list[str] = []
    if status:
        where_parts.append(f"(status,eq,{status})")
    if org_id is not None:
        where_parts.append(f"(org_id,eq,{int(org_id)})")
    params = {"sort": "-CreatedAt", "limit": limit}
    if where_parts:
        params["where"] = "~and".join(where_parts)
    rows = client._get(TABLE, params=params).get("list", [])
    return {"simulations": [_row_to_api(r) for r in rows]}


@router.get("/{sim_id}")
def get_simulation(sim_id: int):
    client = NocodbClient()
    if TABLE not in client.tables:
        raise HTTPException(status_code=404, detail="table missing")
    rows = client._get(TABLE, params={"where": f"(Id,eq,{sim_id})", "limit": 1}).get("list", [])
    if not rows:
        raise HTTPException(status_code=404, detail="not found")
    return _row_to_api(rows[0], include_transcript=True)


@router.post("/{sim_id}/cancel")
def cancel_simulation(sim_id: int):
    client = NocodbClient()
    if TABLE not in client.tables:
        raise HTTPException(status_code=404, detail="table missing")
    rows = client._get(TABLE, params={"where": f"(Id,eq,{sim_id})", "limit": 1}).get("list", [])
    if not rows:
        raise HTTPException(status_code=404, detail="not found")
    if rows[0].get("status") in ("completed", "failed", "cancelled"):
        return {"status": rows[0].get("status"), "sim_id": sim_id}
    client._patch(TABLE, sim_id, {"Id": sim_id, "status": "cancelled"})
    return {"status": "cancelled", "sim_id": sim_id}
