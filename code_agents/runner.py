"""
Agent runner — starts, stops, and monitors agent instances.

Uses the existing workers/jobs.py pattern for background execution
and SSE streaming.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from workers.jobs import STORE, Job, run_in_background

from code_agents.agent import CodeAgent
from code_agents.config import AgentConfig

_log = logging.getLogger("code_agents.runner")

# active agents: agent_id -> {agent, job, started_at}
_active: dict[str, dict[str, Any]] = {}


def start_agent(config: AgentConfig) -> tuple[str, str]:
    """
    Start an agent as a background job.

    Returns (agent_id, job_id).
    """
    job = STORE.create()
    agent = CodeAgent(
        config=config,
        emit=lambda event: STORE.append(job, event),
    )

    _active[config.agent_id] = {
        "agent": agent,
        "job": job,
        "started_at": time.time(),
        "config": config,
    }

    def worker(j: Job) -> None:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(agent.run())
        except Exception as e:
            _log.error("agent %s crashed: %s", config.agent_id, e, exc_info=True)
            STORE.append(j, {"type": "error", "message": str(e)})
        finally:
            loop.close()
            _active.pop(config.agent_id, None)

    run_in_background(job, worker)

    _log.info(
        "agent started  id=%s name=%s repo=%s job=%s",
        config.agent_id, config.name, config.repo_name, job.id,
    )
    return config.agent_id, job.id


def stop_agent(agent_id: str) -> bool:
    """Request graceful stop for an active agent."""
    entry = _active.get(agent_id)
    if entry is None:
        _log.warning("stop_agent: agent %s not active", agent_id)
        return False

    agent: CodeAgent = entry["agent"]
    agent.stop()
    _log.info("agent %s stop requested", agent_id)
    return True


def get_agent_state(agent_id: str) -> dict | None:
    """Get current state of an active agent."""
    entry = _active.get(agent_id)
    if entry is None:
        return None

    agent: CodeAgent = entry["agent"]
    return {
        "agent_id": agent_id,
        "job_id": entry["job"].id,
        "state": agent.state.model_dump(),
        "thought_count": len(agent.thoughts.active_thoughts),
        "readiness": agent.thoughts.calculate_readiness(),
        "started_at": entry["started_at"],
        "uptime_seconds": round(time.time() - entry["started_at"], 1),
    }


def list_active_agents() -> list[dict]:
    """List all active agents with summary info."""
    agents = []
    for agent_id, entry in _active.items():
        agent: CodeAgent = entry["agent"]
        agents.append({
            "agent_id": agent_id,
            "name": entry["config"].name,
            "repo_name": entry["config"].repo_name,
            "status": agent.state.status,
            "cycle": agent.state.cycle_count,
            "thought_count": len(agent.thoughts.active_thoughts),
            "job_id": entry["job"].id,
            "uptime_seconds": round(time.time() - entry["started_at"], 1),
        })
    return agents
