"""Persistent per-family specialists (m13 G-14; audit 06; the Prime Intellect essay's persistent sub-agents as PATTERN).

A family's hacker remembers what worked. ``FamilyMemory`` keeps, per family, the *strategies* of confirmed A16 hacks
(``memory.jsonl``), not their raw actions (actions are task-specific; the strategy re-applies to a new task's solver
trajectories). ``SpecialistHacker`` replays the family's known strategies first (``MutationHacker`` restricted to them:
no hacker episodes) and calls the expensive hacker (an LLM / sandboxed hacker) when the replay finds nothing to try, or
(``propose_more``, M8 audit V7) when A16 confirmed no hack from the replay. So a known hack costs zero extra episodes; ``remember`` stores new ones after
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
        """Phase 1: the family's remembered strategies (no hacker episode); the fallback only when memory has nothing."""
        self._proposed: set[tuple[str, str]] = set()
        known = set(self.memory.strategies(self.family))
        if known:
            replay = [a for a in MutationHacker(max_attempts=10 * self.max_attempts).propose(view) if a.strategy in known][: self.max_attempts]
            if replay:
                self.source = "memory"
                self._proposed = {_key(a) for a in replay}
                return replay
        return await self._fallback(view, "fallback")

    async def propose_more(self, view: dict[str, Any]) -> list[HackAttempt]:
        """Phase 2, called by A16 when phase 1 confirmed no hack: the fallback search too (M8 audit V7: a clean replay
        used to end the search, so a hack a plain search finds was missed whenever memory held an unrelated strategy)."""
        if self.source != "memory":
            return []
        return await self._fallback(view, "memory+fallback")

    async def _fallback(self, view: dict[str, Any], source: str) -> list[HackAttempt]:
        if self.fallback is None:
            self.source = self.source if source == "memory+fallback" else "none"
            return []
        self.episodes += 1
        self.source = source
        out = self.fallback.propose(view)
        seen = getattr(self, "_proposed", set())
        return [a for a in list(await out if inspect.isawaitable(out) else out) if _key(a) not in seen][: self.max_attempts]


def _key(a: HackAttempt) -> tuple[str, str]:
    return (a.strategy, json.dumps(a.actions, sort_keys=True, default=str))
