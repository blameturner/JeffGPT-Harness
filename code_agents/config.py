"""
Unified code agent configuration.

One agent, configurable behaviour. Not three rigid modes — one adaptive
agent that shifts between thinking, researching, planning, coding, and
reviewing based on its configuration and its own judgment.
"""

from __future__ import annotations

import time
import uuid

from pydantic import BaseModel, Field


class Permissions(BaseModel):
    """Granular permission toggles — defaults are conservative."""

    can_read: bool = True
    can_write: bool = False
    can_create_branch: bool = False
    can_commit: bool = False
    can_create_pr: bool = False
    can_merge_pr: bool = False
    can_delete_branch: bool = False
    can_execute_code: bool = False
    can_search_web: bool = False
    can_lint: bool = False
    allowed_paths: list[str] = Field(default_factory=list)
    blocked_paths: list[str] = Field(default_factory=list)


class Behaviour(BaseModel):
    """How the agent thinks and acts."""

    orientation: str = "explore"  # explore|analyze|plan|build|review|maintain|research
    adaptive_orientation: bool = True
    think_before_acting: bool = True
    accumulate_thoughts: bool = True
    confidence_threshold: float = 0.7
    min_thinking_cycles: int = 2
    max_thinking_cycles: int = 10
    batch_size: int = 3
    self_review: bool = True

    # model pool keys — resolved via acquire_model()
    thinking_model: str = "reasoner"
    coding_model: str = "t2_coder"
    tool_model: str = "t3_tool"

    verbose_logging: bool = True
    explain_decisions: bool = True
    max_commits: int = 10
    max_files_changed: int = 20
    stop_on_error: bool = True
    dry_run: bool = False


class BranchStrategy(BaseModel):
    """How the agent manages branches."""

    branch_prefix: str = "agent/"
    create_branch_per_task: bool = True
    merge_strategy: str = "squash"  # squash|merge|rebase
    auto_pr: bool = False
    auto_pr_description: bool = True


class Task(BaseModel):
    """What the agent should do."""

    objective: str
    constraints: list[str] = Field(default_factory=list)
    context: str = ""
    focus_areas: list[str] = Field(default_factory=list)
    plan: str = ""  # optional pre-written plan for review mode
    watch_for: list[str] = Field(default_factory=list)


class Schedule(BaseModel):
    """When / how often the agent runs."""

    run_mode: str = "once"  # once|continuous|scheduled|triggered
    cron: str = ""
    max_runtime_minutes: int = 30
    max_total_cycles: int = 50
    cycle_cooldown_seconds: int = 5
    stop_conditions: list[str] = Field(default_factory=list)


class AgentConfig(BaseModel):
    """Top-level agent configuration."""

    agent_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    description: str = ""
    org_id: int
    repo_name: str
    branch: str = "main"
    repos: list[str] = Field(default_factory=list)  # additional read-only repos
    permissions: Permissions = Field(default_factory=Permissions)
    behaviour: Behaviour = Field(default_factory=Behaviour)
    branch_strategy: BranchStrategy = Field(default_factory=BranchStrategy)
    task: Task
    schedule: Schedule = Field(default_factory=Schedule)
    system_prompt_override: str = ""


class AgentState(BaseModel):
    """Runtime state — persisted in NocoDB after each step."""

    agent_id: str
    status: str = "idle"
    # idle|thinking|researching|coding|reviewing|proposing|
    # waiting_approval|error|complete
    current_phase: str = ""
    cycle_count: int = 0
    thinking_cycles: int = 0
    action_cycles: int = 0
    current_branch: str = ""
    commits: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    thought_count: int = 0
    readiness_score: float = 0.0
    started_at: float = Field(default_factory=time.time)
    last_activity_at: float = Field(default_factory=time.time)
