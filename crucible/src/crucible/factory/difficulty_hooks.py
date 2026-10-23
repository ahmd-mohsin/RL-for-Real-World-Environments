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

from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from crucible.factory.calibration import KNOBS as _CAL_KNOBS
from crucible.factory.difficulty import CHECKPOINT as _C1
from crucible.factory.difficulty import HARDEN as _HARDEN
from crucible.factory.difficulty import HeldOutRefused  # one exception for both sides (V9)
from crucible.factory.difficulty import KnobAction as ControlAction

# One knob vocabulary for both sides (the controller, ``factory.difficulty``, and the swarm's revisers): the easing knobs
# K1-K6 of ``factory.calibration.KNOBS``, the checkpoint start C1 and the hardening knobs H1-H3 of ``factory.difficulty``.
# (The first draft of this table had its own H2-H4, one of them a distractor symptom the tests do not check; that one asks
# the agent for behaviour no check grades, which V5 forbids, so it is gone.)
EASE_KNOBS: dict[str, str] = {**{k: f"{name.lower()}: {what}" for k, name, what in _CAL_KNOBS}, _C1[0]: f"{_C1[1].lower()}: {_C1[2]}"}
HARDEN_KNOBS: dict[str, str] = {k: f"{name.lower()}: {what}" for k, name, what in _HARDEN}
PROMPT_KNOBS = frozenset({"K2", "K3", "K4", "K5", "K6"})  # the report changes; the World, the oracle and the hidden tests do not
WORLD_KNOBS = frozenset({"K1", "C1", "H1", "H3"})  # the start tree or the milestone set changes: every executed gate runs again


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


def from_control(a: ControlAction, *, supported: Collection[str] | None = None, milestone: str | None = None,
                 params: Mapping[str, Any] | None = None) -> KnobAction | None:
    """The controller's decision (``factory.difficulty.KnobAction``: an action and its knobs, best first) as the one typed
    request a reviser takes: the first of its knobs the reviser ``supported``. ``budget_bound`` is an ease restricted to the
    horizon-shortening knobs (C1, K1); ``keep``, ``sample_more`` and ``fix_task`` are not difficulty moves: None (the caller
    records why)."""
    if a.action not in ("ease", "harden", "budget_bound"):
        return None
    direction = "harden" if a.action == "harden" else "ease"  # budget_bound: an ease that shortens the horizon (C1, K1)
    for k in a.knobs:
        if supported is not None and k not in supported:
            continue
        table = EASE_KNOBS if direction == "ease" else HARDEN_KNOBS
        if k in table:
            return KnobAction(direction, k, a.reason, milestone=milestone, params=dict(params or {}),  # type: ignore[arg-type]
                              evidence={"p": a.p, "lo": a.lo, "hi": a.hi, "n": a.n, "knobs": list(a.knobs), "generator_reward": a.generator_reward})
    return None


async def revise_and_gate(task: Any, action: KnobAction, *, reviser: Reviser, gates: Gates, heldout: bool = False, journal: Any = None,
                          run: str = "difficulty") -> dict[str, Any]:
    """The generator's one call: ``action`` -> ``reviser.revise`` -> ``gates.run``. Returns ``{ok, revision, gates,
    findings, action}``; ``ok`` False keeps the original task (the caller never replaces it). ``journal``: a
    ``SwarmJournal`` (the verdict row blames the reviser's role when the gates refuse)."""
    if heldout:
        raise HeldOutRefused(f"task {getattr(task, 'id', task)!r} is held out: no revision")
    rev = await reviser.revise(task, action)
    res = await gates.run(rev)
    infra = bool((res.gates or {}).get("infra"))  # e.g. the report writer's session hit a meter cap: no verdict on the reviser's work (V8)
    if journal is not None:
        journal.gate(run=run, state=f"revise:{action.knob}", entry=1, attempt=1, role="reviser", ok=None if infra else res.ok, findings=res.findings,
                     owner="" if res.ok or infra else "reviser", kind="infra" if infra else "verdict")
    return {"ok": res.ok, "revision": rev if res.ok else None, "gates": dict(res.gates), "findings": list(res.findings), "action": action.to_json()}
