"""Dense progress for long horizons (DESIGN §10.4): potential-based shaping over the grader's *own* subgoals, computed
online while the episode runs.

On an episode of thousands of tool calls, one binary outcome at the end is almost no signal, and replaying a
million-token trajectory afterwards to compute milestones is too slow. Here a task declares its grade as a set of
**subgoals** (per-entity predicates the final grade is made of: "ticket 412 is in its correct final state"), and:

- ``Φ(s) = Σ wᵢ·[subgoal i holds in s] / Σ wᵢ`` is the same quantity the final grade reports, so the shaping
  rewards ``rₜ = Φ(sₜ₊₁) − Φ(sₜ)`` **telescope to the final reward** (Φ at the start is 0 for a task whose start
  state satisfies nothing). The decomposition is exact, adds no bias, and cannot be farmed: undoing and redoing work
  nets zero (Ng, Harada & Russell 1999, with γ = 1 on an episodic task).
- **Incremental evaluation.** After each step only the subgoals whose entity keys the step touched are re-evaluated
  (keys from the action's arguments and from the World events that fired during the step), so the cost per step is
  O(touched), not O(task). Every ``full_every`` steps and at the end everything is re-evaluated; a subgoal that
  changed without being touched is reported as **drift** (a missing key dependency in the task's declaration) and
  corrected, so a bad declaration is visible, never silent.
- **Streaming.** Each step's row (Φ, ΔΦ, subgoals gained and lost, the tool call that caused it) is appended to a
  JSONL file as it happens, so a crash loses nothing and a 1M-token episode never holds its rows in memory twice.

A subgoal set is any object with ``subgoals`` (``Subgoal`` list) and ``async evaluate(world, ids) -> {id: bool}``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class Subgoal:
    id: str
    keys: frozenset[str]  # entity keys whose change can change this subgoal ("ticket:412", "order:O-5001", "event:followup_412")
    weight: float = 1.0


class SubgoalSet(Protocol):
    subgoals: list[Subgoal]

    async def evaluate(self, world: Any, ids: Iterable[str]) -> dict[str, bool]: ...


ARG_KEYS = {"ticket_id": "ticket", "order_id": "order", "customer_id": "customer", "invoice_id": "invoice", "vendor_id": "vendor", "po_id": "po",
            "path": "doc", "id": "id"}


def keys_of_action(action: Any) -> set[str]:
    """Entity keys an action touches, from its arguments (``ticket_id=412`` -> ``ticket:412``)."""
    args = getattr(action, "arguments", None) or (action.get("arguments") if isinstance(action, dict) else None) or {}
    return {f"{ARG_KEYS[k]}:{v}" for k, v in args.items() if k in ARG_KEYS and v is not None}


def keys_of_events(events: Iterable[Any]) -> set[str]:
    out = set()
    for e in events:
        ev = (e.data or {}).get("event")
        if e.type.startswith("event:") and ev:
            out.add(f"event:{ev}")
    return out


@dataclass
class ProgressRow:
    step: int
    phi: float
    delta_phi: float
    gained: list[str]
    lost: list[str]
    tool_call_id: str | None = None
    actor: str | None = None
    drift: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, separators=(",", ":"))


class ProgressTracker:
    """An episode observer (``run_episode(..., observers=[tracker])``)."""

    def __init__(self, sset: SubgoalSet, *, out: Path | None = None, full_every: int = 250,
                 keys_of: Callable[[Any], set[str]] = keys_of_action):
        self.sset, self.out, self.full_every, self.keys_of = sset, Path(out) if out else None, full_every, keys_of
        self.by_key: dict[str, list[str]] = {}
        for g in sset.subgoals:
            for k in g.keys:
                self.by_key.setdefault(k, []).append(g.id)
        self.weight = {g.id: g.weight for g in sset.subgoals}
        self.total = sum(self.weight.values()) or 1.0
        self.state: dict[str, bool] = {}
        self.phi = 0.0
        self.rows = 0
        self.sum_delta = 0.0
        self.drift_total: list[str] = []
        self._log_pos = 0

    def _phi(self) -> float:
        return sum(self.weight[i] for i, v in self.state.items() if v) / self.total

    def _emit(self, row: ProgressRow) -> None:
        self.rows += 1
        self.sum_delta += row.delta_phi
        if self.out is not None:
            with self.out.open("a") as f:
                f.write(row.to_json() + "\n")

    async def on_start(self, world: Any) -> None:
        if getattr(self, "_resumed", False):  # continuing from a checkpoint: the state was loaded
            return
        self.state = await self.sset.evaluate(world, list(self.weight))
        self.phi = self._phi()
        self.phi0 = self.phi
        self._log_pos = len(world.log)
        if self.out is not None:
            self.out.parent.mkdir(parents=True, exist_ok=True)
            self.out.write_text("")
        # the start state as row 0 (M13 audit H1 V2): the subgoals already held and Φ_0, so a reader sees milestones that
        # were never "gained" (a fault-start task begins at an expert state) and losing one counts as a regression.
        # ΔΦ = 0, so the stream still telescopes to Φ_T − Φ_0; per-step consumers skip it (step 0 is no action).
        self._emit(ProgressRow(step=0, phi=round(self.phi, 9), delta_phi=0.0, gained=sorted(i for i, v in self.state.items() if v), lost=[]))

    async def on_step(self, world: Any, step: int, actor: str, action: Any, result: Any) -> ProgressRow:
        new_events = world.log[self._log_pos:]
        self._log_pos = len(world.log)
        keys = self.keys_of(action) | keys_of_events(new_events)
        ids = sorted({i for k in keys for i in self.by_key.get(k, [])})
        full = self.full_every and step % self.full_every == 0
        if full:
            ids = list(self.weight)
        upd = await self.sset.evaluate(world, ids) if ids else {}
        drift: list[str] = []
        if full:
            touched = {i for k in keys for i in self.by_key.get(k, [])}
            drift = sorted(i for i, v in upd.items() if v != self.state.get(i) and i not in touched)
            self.drift_total += drift
        gained = sorted(i for i, v in upd.items() if v and not self.state.get(i))
        lost = sorted(i for i, v in upd.items() if not v and self.state.get(i))
        self.state.update(upd)
        phi = self._phi()
        row = ProgressRow(step=step, phi=round(phi, 9), delta_phi=round(phi - self.phi, 9), gained=gained, lost=lost,
                          tool_call_id=getattr(action, "tool_call_id", None), actor=actor, drift=drift)
        self.phi = phi
        self._emit(row)
        return row

    def state_dict(self) -> dict:
        return {"state": self.state, "phi": self.phi, "phi0": getattr(self, "phi0", 0.0), "rows": self.rows, "sum_delta": self.sum_delta,
                "drift": self.drift_total, "log_pos": self._log_pos, "out": str(self.out) if self.out else None}

    def load_state_dict(self, d: dict, *, prev_dir: Path | None = None, log_len: int | None = None) -> None:
        """Continue from a checkpoint: carry the previous attempt's stream over up to the checkpoint's row count."""
        self.state, self.phi, self.phi0, self.rows, self.sum_delta = d["state"], d["phi"], d["phi0"], d["rows"], d["sum_delta"]
        self.drift_total, self._log_pos = list(d["drift"]), d["log_pos"]
        self._resumed = True
        old = Path(d["out"]) if d.get("out") else None
        if self.out is not None:
            self.out.parent.mkdir(parents=True, exist_ok=True)
            lines = old.read_text().splitlines()[: self.rows] if old and old.exists() else []
            self.out.write_text("".join(x + "\n" for x in lines))

    async def on_end(self, world: Any) -> dict:
        """A final full evaluation: Φ_T must equal the grade the task reports, and the streamed rewards must sum to Φ_T − Φ_0."""
        upd = await self.sset.evaluate(world, list(self.weight))
        drift = sorted(i for i, v in upd.items() if v != self.state.get(i))
        self.drift_total += drift
        if drift:  # correct the stream so it still telescopes
            self.state.update(upd)
            phi = self._phi()
            self._emit(ProgressRow(step=-1, phi=round(phi, 9), delta_phi=round(phi - self.phi, 9), gained=[i for i in drift if upd[i]],
                                   lost=[i for i in drift if not upd[i]], drift=drift))
            self.phi = phi
        return {"phi0": self.phi0, "phi_final": round(self.phi, 9), "sum_delta": round(self.sum_delta, 9), "rows": self.rows,
                "drift": sorted(set(self.drift_total)), "satisfied": sum(1 for v in self.state.values() if v), "subgoals": len(self.state)}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
