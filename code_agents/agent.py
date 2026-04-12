"""
Unified code agent.

Core loop:
1. Orient — understand current state of repo and own thoughts
2. Think — analyze, research, observe, form new thoughts
3. Decide — assess readiness, choose action (or continue thinking)
4. Act — execute the chosen action
5. Reflect — review what happened, update thoughts
6. Repeat or stop

The agent's behaviour is driven entirely by its AgentConfig.
The same code handles exploring, analyzing, planning, building,
reviewing, maintaining, and researching — the orientation config
determines which thinking prompts are used and what decision
thresholds apply.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Callable

import httpx

from config import no_think_params
from workers.search.models import acquire_model

from code_agents.config import AgentConfig, AgentState
from code_agents.thoughts import Thought, ThoughtBody, ThoughtType
from tools.code.gitea_client import GiteaClient
from tools.code.indexer import RepoManifest, index_repo

_log = logging.getLogger("code_agents.agent")

Emit = Callable[[dict], None]


# ------------------------------------------------------------------
# orientation-specific thinking prompts
# ------------------------------------------------------------------

_THINKING_PROMPTS = {
    "explore": "What patterns do you notice? What questions arise? What is the overall architecture?",
    "analyze": "What issues exist? What metrics are concerning? What could be improved?",
    "plan": "What approach would you take? What are the risks? What are the dependencies?",
    "build": "What should be built next? What files need changing? What is the implementation approach?",
    "review": "Is this correct? What are the problems? Is the approach sound?",
    "maintain": "What has degraded? What needs attention? What is fragile?",
    "research": "What did you learn? How does this compare to alternatives? What patterns apply?",
}


class CodeAgent:
    def __init__(self, config: AgentConfig, emit: Emit | None = None):
        self.config = config
        self.emit: Emit = emit or (lambda _: None)
        self.gitea = GiteaClient()
        self.state = AgentState(
            agent_id=config.agent_id,
            started_at=time.time(),
            last_activity_at=time.time(),
        )
        self.thoughts = ThoughtBody(config.agent_id)
        self.manifest: RepoManifest | None = None
        self._stop_requested = False
        self._db = None

    def stop(self) -> None:
        """Request graceful stop. Agent finishes current cycle then stops."""
        self._stop_requested = True

    def has_permission(self, permission: str) -> bool:
        return getattr(self.config.permissions, permission, False)

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    async def run(self) -> AgentState:
        """Main agent loop. Returns final state."""
        try:
            self._db = self._get_db()
        except Exception as e:
            _log.warning("NocoDB unavailable for agent state: %s", e)

        self._update_state(status="thinking", phase="orienting")

        # orient
        await self._orient()

        # main cycle loop
        deadline = time.time() + (self.config.schedule.max_runtime_minutes * 60)
        max_cycles = self.config.schedule.max_total_cycles

        for cycle in range(1, max_cycles + 1):
            if self._stop_requested:
                _log.info("agent %s stop requested at cycle %d", self.config.agent_id, cycle)
                break
            if time.time() > deadline:
                _log.info("agent %s deadline reached at cycle %d", self.config.agent_id, cycle)
                break

            self.state.cycle_count = cycle

            # think
            await self._think(cycle)

            # decide
            action = await self._decide(cycle)

            if action == "continue_thinking":
                # cooldown between cycles
                if self.config.schedule.cycle_cooldown_seconds > 0:
                    await asyncio.sleep(self.config.schedule.cycle_cooldown_seconds)
                continue

            # act
            await self._act(action)

            # if we acted, we're done (for "once" mode)
            if self.config.schedule.run_mode == "once":
                break

        # finalise
        self._update_state(status="complete", phase="finished")
        self.emit({
            "type": "agent_done",
            "agent_id": self.config.agent_id,
            "summary": self.thoughts.summary(),
            "commits": self.state.commits,
            "thought_count": self.state.thought_count,
        })

        return self.state

    # ------------------------------------------------------------------
    # orient
    # ------------------------------------------------------------------

    async def _orient(self) -> None:
        self.emit({"type": "agent_status", "status": "orienting", "cycle": 0})

        # load previous thoughts if resuming
        if self._db:
            loaded = ThoughtBody.load_thoughts(self.config.agent_id, self._db)
            if loaded.thoughts:
                self.thoughts = loaded
                _log.info("agent %s resumed with %d thoughts", self.config.agent_id, len(loaded.thoughts))

        # index repo
        self.manifest = await index_repo(
            self.config.repo_name,
            branch=self.config.branch,
            org_id=self.config.org_id,
        )

        if self.manifest:
            self._add_thought(Thought(
                type=ThoughtType.OBSERVATION,
                content=f"Repository {self.config.repo_name} has {self.manifest.file_count} files ({self.manifest.total_size:,} bytes)",
                confidence=1.0,
                cycle=0,
            ))
        else:
            self._add_thought(Thought(
                type=ThoughtType.CONCERN,
                content=f"Could not index repository {self.config.repo_name} — may not be accessible",
                confidence=1.0,
                cycle=0,
            ))

        # review recent commits
        commits = await self.gitea.get_commits(self.config.repo_name, branch=self.config.branch, limit=5)
        if commits:
            summaries = [c.get("commit", {}).get("message", "")[:80] for c in commits[:5]]
            self._add_thought(Thought(
                type=ThoughtType.OBSERVATION,
                content=f"Recent commits: {'; '.join(summaries)}",
                confidence=0.9,
                cycle=0,
            ))

    # ------------------------------------------------------------------
    # think
    # ------------------------------------------------------------------

    async def _think(self, cycle: int) -> None:
        self._update_state(status="thinking", phase=f"thinking (cycle {cycle})")
        self.emit({"type": "agent_status", "status": "thinking", "cycle": cycle})

        orientation = self.config.behaviour.orientation
        thinking_hint = _THINKING_PROMPTS.get(orientation, _THINKING_PROMPTS["explore"])

        # build context for the thinking model
        manifest_ctx = ""
        if self.manifest:
            manifest_ctx = self.manifest.to_context_string()
            if len(manifest_ctx) > 4000:
                manifest_ctx = manifest_ctx[:4000] + "\n…[truncated]"

        thoughts_ctx = self.thoughts.to_context_string(max_chars=6000)

        system_prompt = f"""You are a code analysis agent in '{orientation}' mode.
Repository: {self.config.repo_name} (branch: {self.config.branch})
Task: {self.config.task.objective}

{manifest_ctx}

{thoughts_ctx}

Permissions: read={self.has_permission('can_read')}, write={self.has_permission('can_write')}, branch={self.has_permission('can_create_branch')}, commit={self.has_permission('can_commit')}, PR={self.has_permission('can_create_pr')}

Output a JSON array of 3-8 thoughts. Each thought:
{{"type": "observation|question|concern|idea|pattern|research|comparison|priority|dependency|risk|opportunity|decision",
 "content": "specific observation referencing actual files/functions",
 "confidence": 0.0-1.0,
 "related_files": ["path/to/file"],
 "tags": ["topic"]}}

Rules:
- Be specific — reference actual files and functions from the manifest
- Build on previous thoughts — don't repeat
- Only mark as "decision" when confidence > 0.8
- {thinking_hint}"""

        user_prompt = f"Cycle {cycle}. {self.config.task.objective}"
        if self.config.task.constraints:
            user_prompt += f"\nConstraints: {', '.join(self.config.task.constraints)}"
        if self.config.task.focus_areas:
            user_prompt += f"\nFocus areas: {', '.join(self.config.task.focus_areas)}"

        # call thinking model
        raw = await self._call_model(
            self.config.behaviour.thinking_model,
            system_prompt,
            user_prompt,
            max_tokens=1500,
            temperature=0.7,
        )

        if not raw:
            self._add_thought(Thought(
                type=ThoughtType.CONCERN,
                content="Thinking model unavailable or returned empty response",
                confidence=0.5,
                cycle=cycle,
            ))
            return

        # parse thoughts from model response
        new_thoughts = self._parse_thoughts(raw, cycle)
        for t in new_thoughts:
            self._add_thought(t)
            self.emit({
                "type": "agent_thought",
                "thought_type": t.type.value,
                "cycle": cycle,
                "content": t.content,
                "confidence": t.confidence,
            })

        self.state.thinking_cycles += 1
        _log.info(
            "agent %s cycle=%d new_thoughts=%d total=%d readiness=%.2f",
            self.config.agent_id, cycle, len(new_thoughts),
            len(self.thoughts.active_thoughts), self.thoughts.calculate_readiness(),
        )

    # ------------------------------------------------------------------
    # decide
    # ------------------------------------------------------------------

    async def _decide(self, cycle: int) -> str:
        readiness = self.thoughts.calculate_readiness(
            min_cycles=self.config.behaviour.min_thinking_cycles,
        )
        self.state.readiness_score = readiness

        # must meet min thinking cycles before any action
        if self.state.thinking_cycles < self.config.behaviour.min_thinking_cycles:
            return "continue_thinking"

        # must meet confidence threshold
        if readiness < self.config.behaviour.confidence_threshold:
            # but don't exceed max thinking cycles
            if self.state.thinking_cycles >= self.config.behaviour.max_thinking_cycles:
                _log.info("agent %s max thinking cycles reached — forcing action", self.config.agent_id)
            else:
                return "continue_thinking"

        # choose action based on orientation
        orientation = self.config.behaviour.orientation

        if orientation in ("build",):
            if self.has_permission("can_commit"):
                return "build"
            return "propose"
        elif orientation in ("review",):
            return "review"
        elif orientation in ("explore", "analyze", "research", "maintain"):
            return "report"
        elif orientation in ("plan",):
            return "propose"
        else:
            return "report"

    # ------------------------------------------------------------------
    # act
    # ------------------------------------------------------------------

    async def _act(self, action: str) -> None:
        self._update_state(status=action, phase=f"executing: {action}")
        self.state.action_cycles += 1

        self.emit({
            "type": "agent_status",
            "status": action,
            "cycle": self.state.cycle_count,
        })

        if action == "propose":
            await self._generate_proposal()
        elif action == "build":
            await self._execute_build()
        elif action == "report":
            await self._generate_report()
        elif action == "review":
            await self._execute_review()

    async def _generate_proposal(self) -> None:
        """Generate a structured proposal from accumulated thoughts."""
        self._update_state(status="proposing")

        themes = self.thoughts.get_actionable_themes(0.5)
        decisions = self.thoughts.get_by_type(ThoughtType.DECISION)
        ideas = self.thoughts.get_by_type(ThoughtType.IDEA)

        system_prompt = """Generate a structured implementation proposal as JSON:
{"title": "short title",
 "summary": "one paragraph",
 "approach": "detailed approach",
 "files_affected": ["path/to/file"],
 "implementation_steps": ["step 1", "step 2"],
 "risks": ["risk 1"],
 "estimated_effort": "small|medium|large"}
Output ONLY the JSON."""

        user_prompt = f"""Task: {self.config.task.objective}
Repository: {self.config.repo_name}

Accumulated analysis:
{self.thoughts.to_context_string(max_chars=8000)}

Generate a concrete implementation proposal."""

        raw = await self._call_model(
            self.config.behaviour.thinking_model,
            system_prompt,
            user_prompt,
            max_tokens=2000,
            temperature=0.3,
        )

        if raw:
            self.emit({
                "type": "agent_proposal",
                "agent_id": self.config.agent_id,
                "proposal": raw,
            })
            self._persist_proposal(raw)

    async def _execute_build(self) -> None:
        """Write code based on accumulated decisions."""
        if self.config.behaviour.dry_run:
            _log.info("agent %s dry_run — skipping build", self.config.agent_id)
            self._add_thought(Thought(
                type=ThoughtType.OBSERVATION,
                content="Dry run mode — build skipped",
                confidence=1.0,
                cycle=self.state.cycle_count,
            ))
            return

        self._update_state(status="coding")

        # create feature branch
        branch_name = f"{self.config.branch_strategy.branch_prefix}{self.config.agent_id}"
        if self.has_permission("can_create_branch"):
            created = await self.gitea.create_branch(
                self.config.repo_name, branch_name, from_branch=self.config.branch,
            )
            if created:
                self.state.current_branch = branch_name
                _log.info("agent %s created branch %s", self.config.agent_id, branch_name)
            else:
                self.state.errors.append(f"Failed to create branch {branch_name}")
                if self.config.behaviour.stop_on_error:
                    return
        else:
            self.state.errors.append("No permission to create branch")
            return

        # gather decisions/ideas for the build plan
        decisions = self.thoughts.get_high_confidence(0.7)
        build_context = "\n".join(
            f"- [{t.type.value}] {t.content}" for t in decisions
        )

        # generate code changes
        manifest_ctx = ""
        if self.manifest:
            manifest_ctx = self.manifest.to_context_string()
            if len(manifest_ctx) > 4000:
                manifest_ctx = manifest_ctx[:4000] + "\n…[truncated]"

        system_prompt = f"""You are a code agent. Generate file changes as:
FILE: path/to/file
```
file content here
```

Rules:
- Output complete file contents, not diffs
- Each file is preceded by FILE: path
- Only output files that need changing
- Max {self.config.behaviour.batch_size} files per batch"""

        user_prompt = f"""Task: {self.config.task.objective}
Repository: {self.config.repo_name} (branch: {branch_name})

Codebase:
{manifest_ctx}

Analysis and decisions:
{build_context}

Generate the code changes."""

        raw = await self._call_model(
            self.config.behaviour.coding_model,
            system_prompt,
            user_prompt,
            max_tokens=4000,
            temperature=0.2,
        )

        if not raw:
            self.state.errors.append("Coding model returned empty response")
            return

        # parse FILE: blocks
        file_changes = self._parse_file_blocks(raw)
        if not file_changes:
            self.state.errors.append("No file changes parsed from model output")
            return

        # apply changes
        commits = 0
        for path, content in file_changes.items():
            if commits >= self.config.behaviour.max_commits:
                _log.info("agent %s max commits reached", self.config.agent_id)
                break

            if not self.has_permission("can_commit"):
                self.state.errors.append(f"No permission to commit {path}")
                continue

            self.emit({"type": "agent_action", "action": "write_file", "path": path})

            result = await self.gitea.update_file(
                self.config.repo_name,
                path,
                content,
                message=f"agent: update {path}",
                branch=branch_name,
            )
            if result:
                self.state.commits.append(path)
                commits += 1
                _log.info("agent %s committed %s", self.config.agent_id, path)
            else:
                self.state.errors.append(f"Failed to write {path}")
                if self.config.behaviour.stop_on_error:
                    return

        # create PR if enabled
        if self.config.branch_strategy.auto_pr and self.has_permission("can_create_pr"):
            pr = await self.gitea.create_pr(
                self.config.repo_name,
                title=f"[agent] {self.config.task.objective[:60]}",
                head=branch_name,
                base=self.config.branch,
                body=f"Auto-generated by agent {self.config.name}\n\n{self.thoughts.summary()}",
            )
            if pr:
                _log.info("agent %s created PR #%s", self.config.agent_id, pr.get("number"))

    async def _generate_report(self) -> None:
        """Compile thoughts into a structured report."""
        themes = self.thoughts.get_actionable_themes(0.4)
        high_conf = self.thoughts.get_high_confidence(0.6)

        report_parts = [
            f"# Agent Report: {self.config.task.objective}",
            f"Repository: {self.config.repo_name} ({self.config.branch})",
            f"Cycles: {self.state.cycle_count} | Thoughts: {len(self.thoughts.active_thoughts)}",
            "",
        ]

        if themes:
            report_parts.append("## Themes")
            for theme in themes:
                report_parts.append(f"\n### {theme['name']} (confidence: {theme['confidence']:.0%})")
                related_thoughts = [
                    t for t in self.thoughts.active_thoughts
                    if t.id in theme["thoughts"]
                ]
                for t in related_thoughts:
                    report_parts.append(f"- [{t.type.value}] {t.content}")
                if theme["files"]:
                    report_parts.append(f"  Files: {', '.join(theme['files'])}")

        if high_conf:
            report_parts.append("\n## Key Findings")
            for t in high_conf:
                report_parts.append(f"- [{t.type.value}] {t.content} ({t.confidence:.0%})")

        report = "\n".join(report_parts)
        self.emit({
            "type": "agent_report",
            "agent_id": self.config.agent_id,
            "report": report,
        })

    async def _execute_review(self) -> None:
        """Review code or a plan."""
        self._update_state(status="reviewing")

        # if reviewing a plan
        if self.config.task.plan:
            system_prompt = """Review this implementation plan. Assess:
- Feasibility
- Risks
- Missing considerations
- Suggested improvements
Output a structured review."""

            user_prompt = f"""Plan to review:
{self.config.task.plan}

Codebase context:
{self.thoughts.to_context_string(max_chars=6000)}"""

            raw = await self._call_model(
                self.config.behaviour.thinking_model,
                system_prompt,
                user_prompt,
                max_tokens=2000,
                temperature=0.3,
            )

            if raw:
                self.emit({
                    "type": "agent_review",
                    "agent_id": self.config.agent_id,
                    "review": raw,
                })
            return

        # if reviewing own work (diff-based)
        if self.state.current_branch:
            diff = await self.gitea.get_branch_diff(
                self.config.repo_name, self.config.branch, self.state.current_branch,
            )
            if diff:
                self.emit({
                    "type": "agent_review",
                    "agent_id": self.config.agent_id,
                    "review": f"Branch diff: {diff['files_changed']} files, +{diff['additions']}/-{diff['deletions']}",
                })

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    async def _call_model(
        self,
        model_pool: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1500,
        temperature: float = 0.7,
    ) -> str | None:
        """Call a model via acquire_model. Returns raw content or None."""
        try:
            with acquire_model(model_pool, priority=False) as (url, model_id):
                if not url:
                    _log.warning("agent %s no model available pool=%s", self.config.agent_id, model_pool)
                    return None

                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(
                        f"{url}/v1/chat/completions",
                        json={
                            "model": model_id,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            **no_think_params(model_id),
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
        except Exception as e:
            _log.warning("agent %s model call failed pool=%s: %s", self.config.agent_id, model_pool, e)
            return None

    def _parse_thoughts(self, raw: str, cycle: int) -> list[Thought]:
        """Parse model output into Thought objects."""
        raw = raw.strip()
        # strip markdown fences
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            raw = raw.strip()

        # find JSON array
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            _log.debug("no JSON array in thinking response")
            return []

        try:
            items = json.loads(match.group())
        except json.JSONDecodeError:
            _log.debug("invalid JSON in thinking response")
            return []

        thoughts = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                t = Thought(
                    type=ThoughtType(item.get("type", "observation")),
                    content=str(item.get("content", "")),
                    confidence=float(item.get("confidence", 0.5)),
                    related_files=item.get("related_files", []),
                    tags=item.get("tags", []),
                    cycle=cycle,
                )
                thoughts.append(t)
            except (ValueError, KeyError) as e:
                _log.debug("skipping invalid thought: %s", e)
                continue

        return thoughts[:8]  # cap at 8

    def _parse_file_blocks(self, raw: str) -> dict[str, str]:
        """Parse FILE: path + fenced code blocks from model output."""
        files: dict[str, str] = {}
        pattern = re.compile(
            r"FILE:\s*(.+?)\s*\n```\w*\n(.*?)\n```",
            re.DOTALL,
        )
        for match in pattern.finditer(raw):
            path = match.group(1).strip()
            content = match.group(2)
            if path:
                files[path] = content

        return files

    def _add_thought(self, thought: Thought) -> None:
        """Add a thought and persist to NocoDB."""
        self.thoughts.add_thought(thought)
        self.state.thought_count = len(self.thoughts.active_thoughts)
        if self._db:
            self.thoughts.persist_thought(thought, self._db)

    def _update_state(self, **kwargs) -> None:
        """Update state fields and persist."""
        for k, v in kwargs.items():
            if hasattr(self.state, k):
                setattr(self.state, k, v)
        self.state.last_activity_at = time.time()
        self._persist_state()

    def _persist_state(self) -> None:
        """Write current state to NocoDB."""
        if not self._db:
            return
        from config import NOCODB_TABLE_CODE_AGENT_TASKS
        try:
            # upsert by agent_id — use a simple approach of updating if exists
            state_json = self.state.model_dump_json()
            config_json = self.config.model_dump_json()
            self._db._post(NOCODB_TABLE_CODE_AGENT_TASKS, {
                "agent_id": self.state.agent_id,
                "name": self.config.name,
                "description": self.config.task.objective,
                "repo_name": self.config.repo_name,
                "status": self.state.status,
                "state_json": state_json,
                "config_json": config_json,
                "thought_count": self.state.thought_count,
            })
        except Exception as e:
            _log.debug("persist_state failed: %s", e)

    def _persist_proposal(self, proposal_raw: str) -> None:
        """Write proposal to NocoDB."""
        if not self._db:
            return
        from config import NOCODB_TABLE_CODE_AGENT_PROPOSALS
        try:
            self._db._post(NOCODB_TABLE_CODE_AGENT_PROPOSALS, {
                "agent_id": self.state.agent_id,
                "title": self.config.task.objective[:100],
                "proposal_json": proposal_raw,
                "status": "pending",
            })
        except Exception as e:
            _log.debug("persist_proposal failed: %s", e)

    def _get_db(self):
        from nocodb_client import NocodbClient
        return NocodbClient()
