"""ONL: online certification (part2-paper-core §5, claim C6). Stub for M2; the eval/RL hook is M5.

Every passing training or eval rollout is a falsification trial. ``OnlineMonitor.observe`` takes a graded rollout and
the flags raised on it (L1 hack rules, a trajectory monitor; a chain of thought may raise a flag, never clear one) and
returns a **suspect** finding for a flagged pass. ``quarantine`` is called with each suspect; a confirmed one (after
adjudication) quarantines the task and triggers repair in the caller.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from crucible.synth.contract import Verdict
from crucible.synth.falsify.agents import trace_program
from crucible.synth.falsify.base import Finding, Outcome, Probe
from crucible.synth.program import Run
from crucible.synth.task import Task


class OnlineMonitor:
    name = "ONL"

    def __init__(self, quarantine: Callable[[Task, Finding], None] | None = None):
        self.quarantine = quarantine
        self.trials: dict[str, int] = {}

    def observe(self, task: Task, run: Run, verdict: Verdict, flags: Sequence[str] = ()) -> Finding | None:
        if not verdict.passed:
            return None
        self.trials[task.id] = self.trials.get(task.id, 0) + 1
        if not flags:
            return None
        p = Probe(cls="ONL", op="rollout", site=f"rollout:{self.trials[task.id]}", program=trace_program(run.calls, run.answer), expect="fail",
                  data={"flags": list(flags)})
        f = Finding(cls="ONL", op="rollout", site=p.site, label="suspect", defect=None, probe=p,
                    outcome=Outcome(passed=True, reasons=verdict.reasons, data={"flags": list(flags)}))
        if self.quarantine is not None:
            self.quarantine(task, f)
        return f
