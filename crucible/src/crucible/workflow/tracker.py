"""The Workflow State Tracker (DESIGN §5.3.5): "where am I?" for an episode in progress, or for every step of a
recorded trace (the labelling engine that turns existing agent logs into step-credited data).

``observe(step, view)`` after each step returns a ``TrackerState``: the current stage, Φ (C1), the milestones gained and
lost at this step, the next expected milestones (in the current stage, prerequisites met, not yet satisfied) and
deviations (answer-key probes the step made, regressions). Cost policy: cheap milestones every step; medium ones when
the step is a checkpoint (a build/test/run/submit step, or every ``every`` steps); expensive ones at ``finalize``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from crucible.workflow.graph import WorkflowGraph
from crucible.workflow.milestones.evaluate import StateView, evaluate
from crucible.workflow.potential import credited, potential
from crucible.workflow.stages import stage_of
from crucible.workflow.steps import Step

CHECKPOINT_KINDS = frozenset({"build", "test", "run", "submit", "commit", "ci"})


@dataclass
class TrackerState:
    step: int
    stage: int
    stage_name: str
    phi: float
    delta_phi: float
    gained: list[str] = field(default_factory=list)
    lost: list[str] = field(default_factory=list)
    next_expected: list[str] = field(default_factory=list)
    deviations: list[str] = field(default_factory=list)
    evaluated: list[str] = field(default_factory=list)
    partial: bool = False

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class WorkflowTracker:
    def __init__(self, graph: WorkflowGraph, *, every: int = 0, done_share: float = 0.8):
        self.graph, self.every, self.done_share = graph, every, done_share
        self.model = graph.stage_model()
        self.phis: dict[str, float | None] = {m.id: None for m in graph.milestones}
        self.prev_phi, self.prev_set = 0.0, set()
        self.history: list[TrackerState] = []

    def _due(self, step: Step | None, final: bool) -> set[str]:
        checkpoint = final or (step is not None and step.kind in CHECKPOINT_KINDS) or (self.every and len(self.history) % self.every == 0)
        return {m.id for m in self.graph.milestones if m.cost == "cheap" or (m.cost == "medium" and checkpoint) or (m.cost == "expensive" and final)}

    async def start(self, view: StateView) -> float:
        """Evaluate the state before any step (s0), so a milestone already true there is not credited to the first
        step (M5 audit: Φ(s0) was taken as 0)."""
        fresh = await evaluate(self.graph.milestones, view)
        self.phis.update({k: v for k, v in fresh.items() if v is not None})
        self.prev_phi, _ = potential(self.graph.milestones, self.phis)
        self.prev_set = credited(self.graph.milestones, self.phis)
        return self.prev_phi

    async def observe(self, step: Step | None, view: StateView, *, final: bool = False) -> TrackerState:
        due = self._due(step, final)
        fresh = await evaluate([m for m in self.graph.milestones if m.id in due], view)
        self.phis.update({k: v for k, v in fresh.items() if v is not None or self.phis.get(k) is None})
        phi, partial = potential(self.graph.milestones, self.phis)
        now = credited(self.graph.milestones, self.phis)
        stage = stage_of(self.model, self.graph.milestones, self.phis, done_share=self.done_share)
        names = {s.index: s.name for s in self.graph.stages}
        cur = next((s for s in self.graph.stages if s.index == stage), None)
        req = {x.id: x.requires for x in self.graph.milestones}
        nxt = [m for m in (cur.milestones if cur else []) if m not in now and all((self.phis.get(p) or 0) >= 1 for p in req.get(m, []))]
        dev = [f"regressed:{m}" for m in sorted(self.prev_set - now)]
        if step is not None and step.data.get("probes"):
            dev += [f"probe:{p['kind']}" for p in step.data["probes"]]
        st = TrackerState(step=step.index if step else len(self.history), stage=stage, stage_name=names.get(stage, "done"), phi=round(phi, 6),
                          delta_phi=round(phi - self.prev_phi, 6), gained=sorted(now - self.prev_set), lost=sorted(self.prev_set - now),
                          next_expected=sorted(nxt), deviations=dev, evaluated=sorted(fresh), partial=partial)
        self.prev_phi, self.prev_set = phi, now
        self.history.append(st)
        return st

    async def run(self, steps: Sequence[Step], views: Sequence[StateView], *, base: StateView | None = None) -> list[TrackerState]:
        """Offline: track a recorded trace, one state view per step (the state after it); ``base``: the state before
        the first step."""
        if base is not None:
            await self.start(base)
        for i, (s, v) in enumerate(zip(steps, views, strict=True)):
            await self.observe(s, v, final=i == len(steps) - 1)
        return self.history
