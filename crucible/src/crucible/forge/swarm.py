"""The swarm stage runner (m13 G-8; audit 06 §4.2; GitSwarm §2 as PATTERN, no licence).

One stage = one role's homogeneous workers writing into the zoned forge repo (``forge/repo.py``):

- **Scheduling:** at most ``C`` workers in flight (GitSwarm's 5), with replacement on exit, until ``B`` *charged*
  episodes are spent. Charged: a contribution (committed, rejected by validation, or a zone breach) or a nomination.
  A voluntary **abstention is free** in B (its tokens are still charged to the ledger by the role's client).
  ``max_episodes`` (default 4·B) bounds a stage whose workers keep abstaining.
- **Announce before editing:** a worker announces ``Gap | Deliverable | Success | Abort``. An announcement whose gap
  matches one in flight, or one already contributed (same ``key``, or gap-token Jaccard ≥ ``j_min``), is a
  **collision**: the worker is cancelled before it edits. Collisions are counted, not charged (they end before any work);
  this is GitSwarm's 1-in-6 duplicate waste, caught at announce time.
- **Episode end:** exactly one of ``Contribute`` (files, one parent, ``informed_by``), ``Nominate`` (a commit) or
  ``Abstain``. A contribution is validated (publication requirements: ``validate(files) -> findings``, not
  correctness) and committed through the repo's zone rule.
- **Dashboard:** budget used / left, in-flight announcements, contributions, collisions, and the nomination tally,
  labelled as *not correctness evidence*. It assigns no work: workers read it (and their zoned repo view) and choose.
- **Readout** (``winner``): the most nominated commit; ties go to the lexicographically smallest full sha; with no valid
  nomination, the root.

Acceptance is still the deterministic DAG plus M9: a swarm only proposes.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from crucible.forge.repo import ForgeRepo, ZoneBreach


@dataclass
class Announcement:
    worker: str
    gap: str
    deliverable: str = ""
    success: str = ""
    abort: str = ""
    key: str = ""  # a canonical key when the gap has one (e.g. "chain:t1+t3")


@dataclass
class Contribute:
    files: dict[str, str]
    type: str
    summary: str
    informed_by: list[str] = field(default_factory=list)
    parent: str | None = None


@dataclass
class Nominate:
    sha: str


@dataclass
class Abstain:
    reason: str = ""


class Cancelled(Exception):
    """The worker's announcement collided; it must stop before editing."""


@dataclass
class Dashboard:
    budget: int
    used: int = 0
    in_flight: list[Announcement] = field(default_factory=list)
    contributions: list[str] = field(default_factory=list)
    contributed_gaps: list[Announcement] = field(default_factory=list)
    collisions: int = 0
    announcements: int = 0
    nominations: dict[str, int] = field(default_factory=dict)

    def render(self) -> dict[str, Any]:
        return {"budget": {"used": self.used, "left": self.budget - self.used},
                "in_flight": [f"{a.gap} | {a.deliverable} | {a.success} | {a.abort}" for a in self.in_flight],
                "contributions": len(self.contributions), "collisions": self.collisions,
                "nominations (not correctness evidence)": dict(self.nominations)}


class Episode:
    """What a worker gets: its zoned view of the repo, the dashboard, and ``announce``."""

    def __init__(self, stage: SwarmStage, worker: str):
        self.stage, self.worker = stage, worker
        self.announced: Announcement | None = None

    def view(self, sha: str | None = None) -> dict[str, str]:
        return self.stage.repo.view(self.stage.role, sha)

    @property
    def dashboard(self) -> dict[str, Any]:
        return self.stage.board.render()

    def heads(self) -> list[str]:
        return list(self.stage.board.contributions)

    def announce(self, gap: str, deliverable: str = "", success: str = "", abort: str = "", *, key: str = "") -> None:
        a = Announcement(self.worker, gap, deliverable, success, abort, key)
        b = self.stage.board
        b.announcements += 1
        hit = next((o for o in [*b.in_flight, *b.contributed_gaps] if self.stage.collides(a, o)), None)
        if hit is not None:
            b.collisions += 1
            raise Cancelled(f"gap matches {hit.worker}'s: {hit.gap}")
        b.in_flight.append(a)
        self.announced = a


class Worker(Protocol):
    async def __call__(self, ep: Episode) -> Contribute | Nominate | Abstain: ...


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", s.lower()))


@dataclass
class StageResult:
    commits: list[str]
    rejected: list[dict]
    abstentions: int
    collisions: int
    announcements: int
    charged: int
    episodes: int
    nominations: dict[str, int]
    dashboard: dict[str, Any]

    @property
    def collision_rate(self) -> float:
        return self.collisions / self.announcements if self.announcements else 0.0


class SwarmStage:
    def __init__(self, repo: ForgeRepo, role: str, worker: Callable[[str], Worker], *, C: int = 5, B: int = 30, max_episodes: int | None = None,
                 validate: Callable[[dict[str, str]], list[str]] | None = None, j_min: float = 0.8):
        self.repo, self.role, self.make_worker, self.C, self.B = repo, role, worker, C, B
        self.max_episodes = max_episodes or 4 * B
        self.validate = validate or (lambda files: [])
        self.j_min = j_min
        self.board = Dashboard(B)

    def collides(self, a: Announcement, b: Announcement) -> bool:
        if a.key and b.key:
            return a.key == b.key
        ta, tb = _tokens(a.gap), _tokens(b.gap)
        return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= self.j_min

    async def run(self) -> StageResult:
        b = self.board
        commits: list[str] = []
        rejected: list[dict] = []
        abst = episodes = n = 0
        lock = asyncio.Lock()

        async def episode(wid: str) -> None:
            nonlocal abst
            ep = Episode(self, wid)
            charged = False
            try:
                out = await self.make_worker(wid)(ep)
            except Cancelled:
                return
            finally:
                if ep.announced in b.in_flight:
                    b.in_flight.remove(ep.announced)
            async with lock:
                if isinstance(out, Abstain):
                    abst += 1
                elif isinstance(out, Nominate):
                    charged = True
                    if out.sha in self.repo._index["commits"]:
                        b.nominations[out.sha] = b.nominations.get(out.sha, 0) + 1
                    else:
                        rejected.append({"worker": wid, "kind": "nominate", "reasons": [f"unknown commit {out.sha[:10]}"]})
                elif isinstance(out, Contribute):
                    charged = True
                    findings = self.validate(out.files)
                    if findings:
                        rejected.append({"worker": wid, "kind": "invalid", "reasons": findings})
                    else:
                        try:
                            c = self.repo.commit(self.role, out.files, type_=out.type, summary=out.summary, informed_by=out.informed_by, parent=out.parent)
                            commits.append(c.sha)
                            b.contributions.append(c.sha)
                            if ep.announced is not None:
                                b.contributed_gaps.append(ep.announced)
                        except ZoneBreach as ex:
                            rejected.append({"worker": wid, "kind": "breach", "reasons": ex.reasons})
                if charged:
                    b.used += 1

        tasks: set[asyncio.Task] = set()
        while True:
            # every episode in flight may still be charged: never start one the budget cannot pay for
            while len(tasks) < self.C and b.used + len(tasks) < self.B and episodes < self.max_episodes:
                episodes += 1
                n += 1
                tasks.add(asyncio.create_task(episode(f"{self.role}-{n}")))
            if not tasks:
                break
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if t.exception() is not None:
                    raise t.exception()
        return StageResult(commits, rejected, abst, b.collisions, b.announcements, b.used, episodes, dict(b.nominations), b.render())


def winner(repo: ForgeRepo, nominations: dict[str, int]) -> str:
    valid = {s: k for s, k in nominations.items() if s in repo._index["commits"] and k > 0}
    if not valid:
        return repo.root
    top = max(valid.values())
    return min(s for s, k in valid.items() if k == top)

