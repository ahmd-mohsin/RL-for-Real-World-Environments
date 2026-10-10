"""Mapping milestones to the steps that achieved them, respecting prerequisites (PATTERN: ToolSandbox's milestone DAG
matcher in NeMo-Gym, ``tool_sandbox/common/evaluation.py``, which maps milestones to snapshot indices monotonically along
a topological order of the DAG).

A milestone counts as *achieved* at the first state where it holds **and** every prerequisite has already been
achieved (at the same or an earlier state); a milestone that holds before its prerequisites is out-of-order "progress"
(DESIGN §6.2: passing a test before the module builds only happens by stubbing) and is reported, not credited. The
achieving step of each milestone is what M5 stage induction orders and what M6 attributes ΔΦ to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from crucible.core.schema.milestone import Milestone
from crucible.workflow.milestones.evaluate import PhiSeries


@dataclass
class Match:
    achieved: dict[str, int] = field(default_factory=dict)  # milestone -> state index where it was (validly) achieved
    out_of_order: dict[str, int] = field(default_factory=dict)  # held before its prerequisites (first such index)
    regressed: dict[str, int] = field(default_factory=dict)  # achieved, later lost (first loss index)

    def order(self) -> list[str]:
        return sorted(self.achieved, key=lambda m: (self.achieved[m], m))


def match(milestones: list[Milestone], series: PhiSeries) -> Match:
    pre = {m.id: list(m.requires) for m in milestones}
    out = Match()
    for t, row in enumerate(series.values):
        progressed = True
        while progressed:  # a milestone and its prerequisite may both first hold at t
            progressed = False
            for m in milestones:
                if m.id in out.achieved or (row.get(m.id) or 0) < 1.0:
                    continue
                if all(p in out.achieved and out.achieved[p] <= t for p in pre[m.id]):
                    out.achieved[m.id] = t
                    progressed = True
                else:
                    out.out_of_order.setdefault(m.id, t)
        for mid, t0 in out.achieved.items():
            if t > t0 and mid not in out.regressed and row.get(mid) is not None and (row.get(mid) or 0) < 1.0:
                out.regressed[mid] = t
    return out
