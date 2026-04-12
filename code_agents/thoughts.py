"""
Thought accumulation system.

The agent doesn't just execute — it thinks. Thoughts are structured
observations that build on each other over time.

A thought has a type, content, confidence, evidence, and relationships
to files and other thoughts. Thoughts are persisted in NocoDB and can
be reloaded when an agent resumes.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from enum import Enum

from pydantic import BaseModel, Field

_log = logging.getLogger("code_agents.thoughts")


class ThoughtType(str, Enum):
    OBSERVATION = "observation"
    QUESTION = "question"
    CONCERN = "concern"
    IDEA = "idea"
    PATTERN = "pattern"
    RESEARCH = "research"
    COMPARISON = "comparison"
    PRIORITY = "priority"
    DEPENDENCY = "dependency"
    RISK = "risk"
    OPPORTUNITY = "opportunity"
    DECISION = "decision"


class Thought(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    type: ThoughtType
    content: str
    confidence: float = 0.5  # 0-1
    evidence: list[str] = Field(default_factory=list)
    related_files: list[str] = Field(default_factory=list)
    related_thoughts: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    cycle: int = 0
    superseded_by: str | None = None
    tags: list[str] = Field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.superseded_by is None


class ThoughtBody:
    """
    Accumulated body of thoughts for an agent.

    Provides methods to add thoughts, query by type, cluster into themes,
    calculate readiness, and serialise for model prompt injection.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.thoughts: list[Thought] = []

    def add_thought(self, thought: Thought) -> None:
        self.thoughts.append(thought)

    @property
    def active_thoughts(self) -> list[Thought]:
        return [t for t in self.thoughts if t.is_active]

    def get_by_type(self, t: ThoughtType) -> list[Thought]:
        return [th for th in self.active_thoughts if th.type == t]

    def get_high_confidence(self, threshold: float = 0.7) -> list[Thought]:
        return [t for t in self.active_thoughts if t.confidence >= threshold]

    def get_actionable_themes(self, threshold: float = 0.6) -> list[dict]:
        """
        Cluster thoughts that share related_files or related_thoughts.

        Returns a list of theme dicts: {name, thoughts, confidence, files}.
        Uses a simple union-find approach.
        """
        active = self.active_thoughts
        if not active:
            return []

        # build adjacency via shared files and explicit related_thoughts
        n = len(active)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # map files to thought indices
        file_to_idx: dict[str, list[int]] = {}
        id_to_idx: dict[str, int] = {}
        for i, t in enumerate(active):
            id_to_idx[t.id] = i
            for f in t.related_files:
                file_to_idx.setdefault(f, []).append(i)

        # union thoughts sharing files
        for indices in file_to_idx.values():
            for j in range(1, len(indices)):
                union(indices[0], indices[j])

        # union thoughts with explicit related_thoughts links
        for i, t in enumerate(active):
            for rt_id in t.related_thoughts:
                if rt_id in id_to_idx:
                    union(i, id_to_idx[rt_id])

        # collect clusters
        clusters: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            clusters.setdefault(root, []).append(i)

        themes = []
        for indices in clusters.values():
            cluster_thoughts = [active[i] for i in indices]
            avg_conf = sum(t.confidence for t in cluster_thoughts) / len(cluster_thoughts)
            if avg_conf < threshold:
                continue

            # name: highest-confidence thought's content (first sentence)
            best = max(cluster_thoughts, key=lambda t: t.confidence)
            name = best.content.split(".")[0][:80]

            all_files = set()
            for t in cluster_thoughts:
                all_files.update(t.related_files)

            themes.append({
                "name": name,
                "thoughts": [t.id for t in cluster_thoughts],
                "confidence": round(avg_conf, 2),
                "files": sorted(all_files),
            })

        return sorted(themes, key=lambda t: t["confidence"], reverse=True)

    def calculate_readiness(self, min_cycles: int = 2) -> float:
        """
        Composite readiness score (0-1).

        Considers:
        - Number of decisions formed (high-confidence ideas/decisions)
        - Number of actionable themes
        - Average confidence across active thoughts
        - Whether minimum thinking cycles have been met
        """
        active = self.active_thoughts
        if not active:
            return 0.0

        # check min cycles
        max_cycle = max((t.cycle for t in active), default=0)
        cycle_factor = min(1.0, max_cycle / max(min_cycles, 1))

        # decisions
        decisions = [
            t for t in active
            if t.type in (ThoughtType.DECISION, ThoughtType.IDEA)
            and t.confidence >= 0.7
        ]
        decision_factor = min(1.0, len(decisions) / 3)

        # themes
        themes = self.get_actionable_themes(0.6)
        theme_factor = min(1.0, len(themes) / 2)

        # average confidence
        avg_conf = sum(t.confidence for t in active) / len(active)

        # weighted composite
        readiness = (
            0.3 * cycle_factor
            + 0.3 * decision_factor
            + 0.2 * theme_factor
            + 0.2 * avg_conf
        )

        return round(min(1.0, readiness), 3)

    def to_context_string(self, max_chars: int = 12000) -> str:
        """Format thoughts for model prompt injection."""
        active = self.active_thoughts
        if not active:
            return "[No thoughts accumulated yet]"

        lines = [f"[Agent thoughts — {len(active)} active]"]

        # group by cycle
        by_cycle: dict[int, list[Thought]] = {}
        for t in active:
            by_cycle.setdefault(t.cycle, []).append(t)

        for cycle in sorted(by_cycle):
            lines.append(f"\n— Cycle {cycle} —")
            for t in by_cycle[cycle]:
                conf = f"({t.confidence:.0%})"
                files_str = ""
                if t.related_files:
                    files_str = f" [{', '.join(t.related_files[:3])}]"
                lines.append(f"  [{t.type.value}] {t.content} {conf}{files_str}")

        lines.append(f"\n[Readiness: {self.calculate_readiness():.0%}]")

        text = "\n".join(lines)
        if len(text) > max_chars:
            # keep most recent cycles, truncate oldest
            text = text[-max_chars:]
            text = "…[earlier thoughts truncated]\n" + text

        return text

    def summary(self) -> str:
        """One-paragraph summary of accumulated thoughts."""
        active = self.active_thoughts
        if not active:
            return "No thoughts yet."

        decisions = self.get_by_type(ThoughtType.DECISION)
        concerns = self.get_by_type(ThoughtType.CONCERN)
        ideas = self.get_by_type(ThoughtType.IDEA)

        parts = [f"{len(active)} thoughts across {max((t.cycle for t in active), default=0)} cycles."]
        if decisions:
            parts.append(f"{len(decisions)} decisions formed.")
        if ideas:
            parts.append(f"{len(ideas)} ideas proposed.")
        if concerns:
            parts.append(f"{len(concerns)} concerns raised.")
        parts.append(f"Readiness: {self.calculate_readiness():.0%}.")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # NocoDB persistence
    # ------------------------------------------------------------------

    def persist_thought(self, thought: Thought, db) -> None:
        """Write a single thought to NocoDB."""
        from config import NOCODB_TABLE_CODE_AGENT_THOUGHTS

        try:
            db._post(NOCODB_TABLE_CODE_AGENT_THOUGHTS, {
                "id": thought.id,
                "agent_id": self.agent_id,
                "type": thought.type.value,
                "content": thought.content,
                "confidence": thought.confidence,
                "evidence": json.dumps(thought.evidence),
                "related_files": json.dumps(thought.related_files),
                "related_thoughts": json.dumps(thought.related_thoughts),
                "tags": ", ".join(thought.tags),
                "cycle": thought.cycle,
                "superseded_by": thought.superseded_by or "",
                "created_at": thought.created_at,
            })
        except Exception as e:
            _log.warning("persist_thought failed id=%s: %s", thought.id, e)

    @classmethod
    def load_thoughts(cls, agent_id: str, db) -> ThoughtBody:
        """Load existing thoughts from NocoDB for this agent."""
        from config import NOCODB_TABLE_CODE_AGENT_THOUGHTS

        body = cls(agent_id)
        try:
            result = db._get(
                NOCODB_TABLE_CODE_AGENT_THOUGHTS,
                params={
                    "where": f"(agent_id,eq,{agent_id})",
                    "sort": "created_at",
                    "limit": 500,
                },
            )
            for row in result.get("list", []):
                thought = Thought(
                    id=row.get("id", uuid.uuid4().hex[:12]),
                    type=ThoughtType(row.get("type", "observation")),
                    content=row.get("content", ""),
                    confidence=float(row.get("confidence", 0.5)),
                    evidence=_safe_json_list(row.get("evidence", "[]")),
                    related_files=_safe_json_list(row.get("related_files", "[]")),
                    related_thoughts=_safe_json_list(row.get("related_thoughts", "[]")),
                    tags=[t.strip() for t in (row.get("tags") or "").split(",") if t.strip()],
                    cycle=int(row.get("cycle", 0)),
                    superseded_by=row.get("superseded_by") or None,
                    created_at=float(row.get("created_at", 0) or time.time()),
                )
                body.thoughts.append(thought)
            _log.info("loaded %d thoughts for agent=%s", len(body.thoughts), agent_id)
        except Exception as e:
            _log.warning("load_thoughts failed agent=%s: %s", agent_id, e)

        return body


def _safe_json_list(val: str) -> list:
    try:
        result = json.loads(val)
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError):
        return []
