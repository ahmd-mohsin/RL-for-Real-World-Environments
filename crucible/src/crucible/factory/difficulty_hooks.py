"""The interface between the swarm's generators and the difficulty subsystem (swarm-pipeline/13 §5).

The difficulty subsystem (built separately: online p̂ filtering, a controller that targets p ≈ 0.5, a cheap difficulty
predictor, checkpoint-conditioned starts) decides *what* to change about a task; the swarm decides *how* and the same
executed gates decide *whether* the change stands. This module is only the contract both sides code against:

- ``KnobAction``: a typed request: ``ease`` or ``harden``, one knob (``EASE_KNOBS`` K1-K6 of the easing doc, ``HARDEN_KNOBS``),
  an optional milestone, the reason and the evidence (p̂, its interval, Φ by milestone) that motivated it.
- ``Controller.propose(task, stats) -> KnobAction | None``: the difficulty side.
- ``Reviser.revise(task, action) -> Revision``: the swarm side (a generator: it rebuilds the task's World, prompt or
  milestones for the knob; a model may draft it, the gates decide).
- ``Gates.run(revision) -> GateResult``: the lane's executed gates on the revised task (the same S3/S5/S6/S8/S10, S7 lint).
- ``revise_and_gate``: the one call a generator makes: action in, a gated revised task (or the reason it was refused)
  out, every step journaled. A refused revision leaves the original task untouched.

Held-out tasks are never revised (``heldout=True`` refuses, V9): a benchmark task's difficulty is measured, not tuned.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

EASE_KNOBS: dict[str, str] = {
    "K1": "drop a fault (one milestone fewer)",
    "K2": "announce the structure (how many separate problems there are)",
    "K3": "point to the area (the subsystem, never the function)",
    "K4": "a visible reproduction per requirement (gate S7r)",
    "K5": "order the requirements (the dependency order)",
    "K6": "sharpen the stuck requirement (more observable detail)",
}
HARDEN_KNOBS: dict[str, str] = {
    "H1": "add a fault (one more real fix reverted)",
    "H2": "withhold the structure",
    "H3": "a distractor symptom the tests do not check",
    "H4": "a longer horizon (a dependent fault behind another)",
}


@dataclass(frozen=True)
class KnobAction:
    direction: Literal["ease", "harden"]
    knob: str
    reason: str
    milestone: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)  # p̂ and its interval, Φ per milestone, the stats window
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        table = EASE_KNOBS if self.direction == "ease" else HARDEN_KNOBS if self.direction == "harden" else None
        if table is None:
            raise ValueError("direction is 'ease' or 'harden'")
        if self.knob not in table:
            raise ValueError(f"unknown {self.direction} knob {self.knob!r} (known: {sorted(table)})")

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "evidence": dict(self.evidence), "params": dict(self.params)}


@dataclass
class Revision:
    task_id: str
    parent_id: str
    action: KnobAction
    task: Any  # the revised task (the lane's record or LongTask)
    notes: str = ""


@dataclass
class GateResult:
    ok: bool
    gates: Mapping[str, Any] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)


@runtime_checkable
class Controller(Protocol):
    def propose(self, task: Any, stats: Mapping[str, Any]) -> KnobAction | None: ...


@runtime_checkable
class Reviser(Protocol):
    async def revise(self, task: Any, action: KnobAction) -> Revision: ...


@runtime_checkable
class Gates(Protocol):
    async def run(self, revision: Revision) -> GateResult: ...


class HeldOutRefused(ValueError):
    """A held-out (benchmark) task is never revised: its difficulty is measured, not tuned (V9)."""


async def revise_and_gate(task: Any, action: KnobAction, *, reviser: Reviser, gates: Gates, heldout: bool = False, journal: Any = None,
                          run: str = "difficulty") -> dict[str, Any]:
    """The generator's one call: ``action`` -> ``reviser.revise`` -> ``gates.run``. Returns ``{ok, revision, gates,
    findings, action}``; ``ok`` False keeps the original task (the caller never replaces it). ``journal``: a
    ``SwarmJournal`` (the verdict row blames the reviser's role when the gates refuse)."""
    if heldout:
        raise HeldOutRefused(f"task {getattr(task, 'id', task)!r} is held out: no revision")
    rev = await reviser.revise(task, action)
    res = await gates.run(rev)
    if journal is not None:
        journal.gate(run=run, state=f"revise:{action.knob}", entry=1, attempt=1, role="reviser", ok=res.ok, findings=res.findings,
                     owner="" if res.ok else "reviser")
    return {"ok": res.ok, "revision": rev if res.ok else None, "gates": dict(res.gates), "findings": list(res.findings), "action": action.to_json()}
