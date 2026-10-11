"""The Curator (m13 G-12; audit 06 §4.2, §4.5): dedup, split placement on the solver ladder, lineage sealing, and the
retirement → re-composition loop.

- **Dedup** at curation (``forge/dedup.py`` layers L0–L3) against every task the Curator kept: a duplicate is rejected.
- **Placement** from a ``LadderReport`` (``forge/ladder.py``):
  - an inversion rejects (``fix_grader``);
  - ``bench`` when the frontier's upper bound is ≤ 0.2;
  - ``train`` when L2 lands in the band [0.1, 0.7];
  - otherwise ``hold``, with the steer the ladder gives (``compose_one_more`` / ``drop_outermost``).
- **Lineage sealing:** every task carries its lineage (the seeds it derives from; a composite carries all of its
  components'). A lineage's first placement seals it to that split, and a task whose lineage is sealed to the other
  split is held (``lineage_sealed``). Bench material must never reach training through a sibling or a composite
  (contamination, DESIGN §11.2).
- **Retirement → re-composition** (``recompose``): once training lifts L2 above 0.7 (``ladder.retire``), the task is
  retired from the train pool and becomes a seed. ``compose(task_id, round)`` builds one more composition and measures
  its ladder; the Curator places it, up to ``max_rounds``, until a composite lands back in the band. Instruction
  obscurity is never a knob: only composition depth moves difficulty.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from crucible.forge.dedup import Candidate, duplicate_of
from crucible.forge.ladder import LadderReport, accept, retire, steer


@dataclass
class CurationDecision:
    task_id: str
    action: str  # train | bench | hold | reject | retired
    reasons: list[str] = field(default_factory=list)
    steer: str | None = None
    lineage: list[str] = field(default_factory=list)
    rates: dict[str, float] = field(default_factory=dict)
    parent: str | None = None  # re-composition: the retired task it was composed from
    round: int | None = None


class Curator:
    def __init__(self, *, j_min: float = 0.8, max_rounds: int = 3):
        self.j_min, self.max_rounds = j_min, max_rounds
        self.kept: list[Candidate] = []
        self.decisions: dict[str, CurationDecision] = {}
        self.sealed: dict[str, str] = {}  # lineage -> split
        self.history: list[CurationDecision] = []

    def _record(self, d: CurationDecision) -> CurationDecision:
        self.decisions[d.task_id] = d
        self.history.append(d)
        return d

    def pool(self, split: str) -> list[str]:
        return sorted(t for t, d in self.decisions.items() if d.action == split)

    def place(self, task_id: str, rep: LadderReport, *, lineage: Sequence[str] = (), candidate: Candidate | None = None, prefer: str = "train") -> CurationDecision:
        rates = {k: v.p for k, v in rep.rates.items()}
        lin = sorted(set(lineage)) or [task_id]
        if candidate is not None:
            hit = duplicate_of(candidate, self.kept, j_min=self.j_min)
            if hit:
                return self._record(CurationDecision(task_id, "reject", [f"duplicate of {hit[0]} ({hit[1]})"], lineage=lin, rates=rates))
        if rep.inversions:
            return self._record(CurationDecision(task_id, "reject", list(rep.inversions), "fix_grader", lin, rates))
        order = ["bench", "train"] if prefer == "bench" else ["train", "bench"]
        verdicts = {s: accept(rep, split=s) for s in order if (s != "bench" or "L3" in rep.rates) and (s != "train" or "L2" in rep.rates)}
        for split in order:
            ok, _ = verdicts.get(split, (False, []))
            if not ok:
                continue
            clash = sorted(x for x in lin if self.sealed.get(x, split) != split)
            if clash:
                return self._record(CurationDecision(task_id, "hold", [f"lineage {clash} is sealed to {self.sealed[clash[0]]}: no {split} task may derive from it"],
                                                     "lineage_sealed", lin, rates))
            for x in lin:
                self.sealed.setdefault(x, split)
            if candidate is not None:
                self.kept.append(candidate)
            return self._record(CurationDecision(task_id, split, [], None, lin, rates))
        why = [w for _, (ok, ws) in verdicts.items() if not ok for w in ws]
        return self._record(CurationDecision(task_id, "hold", why, steer(rep, split=prefer), lin, rates))

    async def recompose(self, task_id: str, l2_training_rate: float,
                        compose: Callable[[str, int], Awaitable[tuple[str, LadderReport, Sequence[str], Candidate | None]]]) -> CurationDecision:
        """``compose(task_id, round) -> (new_task_id, ladder report, lineage, dedup candidate)``."""
        cur = self.decisions.get(task_id)
        if not retire(l2_training_rate):
            return cur or CurationDecision(task_id, "hold", [f"L2 {l2_training_rate:.2f} has not left the band"])
        lineage = list(cur.lineage) if cur else [task_id]
        self._record(CurationDecision(task_id, "retired", [f"training lifted L2 to {l2_training_rate:.2f} (> 0.7): a seed for one more composition"],
                                      lineage=lineage, rates={"L2_training": l2_training_rate}))
        last: CurationDecision | None = None
        for rnd in range(self.max_rounds):
            new_id, rep, lin, cand = await compose(task_id, rnd)
            last = self.place(new_id, rep, lineage=[*lineage, *lin], candidate=cand)
            last.parent, last.round = task_id, rnd
            if last.action == "train":
                return last
        assert last is not None
        return last

    def report(self) -> dict[str, Any]:
        out: dict[str, Any] = {s: self.pool(s) for s in ("train", "bench", "hold", "reject", "retired")}
        out["sealed_lineages"] = dict(self.sealed)
        out["recomposed"] = [{"task": d.task_id, "from": d.parent, "round": d.round, "action": d.action, "rates": d.rates} for d in self.history if d.parent]
        return out
