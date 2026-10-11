"""Persistent per-family specialists (m13 G-14; audit 06; the Prime Intellect essay's persistent sub-agents as PATTERN).

A family's hacker remembers what worked. ``FamilyMemory`` keeps, per family, the *strategies* of confirmed A16 hacks
(``memory.jsonl``), not their raw actions (actions are task-specific; the strategy re-applies to a new task's solver
trajectories). ``SpecialistHacker`` replays the family's known strategies first (``MutationHacker`` restricted to them:
no hacker episodes) and calls the expensive hacker (an LLM / sandboxed hacker) only when the replay finds nothing to try.
So the second forge run of a family re-finds its known hacks at zero extra episodes; ``remember`` stores new ones after
A16 confirms them.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from crucible.forge.hacker import HackAttempt, MutationHacker


class FamilyMemory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.rows: list[dict] = [json.loads(x) for x in self.path.read_text().splitlines()] if self.path.exists() else []

    def strategies(self, family: str) -> list[str]:
        out: list[str] = []
        for r in self.rows:
            if r["family"] == family and r["strategy"] not in out:
                out.append(r["strategy"])
        return out

    def remember(self, family: str, task_id: str, strategies: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            for s in strategies:
                row = {"family": family, "task": task_id, "strategy": s}
                self.rows.append(row)
                f.write(json.dumps(row) + "\n")


class SpecialistHacker:
    def __init__(self, family: str, memory: FamilyMemory, *, fallback: Any = None, max_attempts: int = 32):
        self.family, self.memory, self.fallback, self.max_attempts = family, memory, fallback, max_attempts
        self.episodes = 0  # expensive-hacker calls
        self.source = ""

    async def propose(self, view: dict[str, Any]) -> list[HackAttempt]:
        known = set(self.memory.strategies(self.family))
        if known:
            replay = [a for a in MutationHacker(max_attempts=10 * self.max_attempts).propose(view) if a.strategy in known][: self.max_attempts]
            if replay:
                self.source = "memory"
                return replay
        if self.fallback is None:
            self.source = "none"
            return []
        self.episodes += 1
        self.source = "fallback"
        out = self.fallback.propose(view)
        return list(await out if inspect.isawaitable(out) else out)
