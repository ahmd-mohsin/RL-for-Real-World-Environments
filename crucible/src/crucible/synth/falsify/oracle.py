"""ORACLE: is the reference decision right? (lead, 2026-10-30; the yield finding of DESIGN §7.7 item 17.)

Six admitted finance_ap tasks pinned a decision the World's own policy does not reach: the oracle programmer and the ALT
solvers made the same mistake, so every falsifier passed and the certificate was clean. On such a task an "intended
fail" of the student is the student being right. Two checks, both executed:

1. **Policy truth** (``ORACLE``, a falsifier class): a World that has executable rules (a policy engine, a reference
   implementation, a pack's expected cases) registers a *truth* function ``truth(task) -> "right" | "wrong" | None``.
   ``wrong`` confirms a counterexample: the contract rejects the decision the World's rules require (a valid solution
   fails C: **D6**, op ``policy``). ``None`` (the task pins no decision the rules cover) is not a trial. A World with no
   registered truth makes the class not applicable, with the reason.
2. **Solver disagreement** (``decision_disagreement``): the decision fields the contract pins (required ``eq`` values)
   are read from the end states of independent runs (student rollouts, ALT traces). When at least ``min_agree`` runs
   agree on one value other than the oracle's, the reference decision is **disputed**: a trigger for panel review, never
   a verdict by itself (agents can share a mistake too).

The per-task probe loop (``meta.rsi.TaskLoop``) runs both before any difficulty move: a task is never DESCENDed while
its reference decision is wrong or disputed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import cjson
from crucible.synth.contract import PEq, is_glob, unglob
from crucible.synth.falsify.base import BaseFalsifier, FalsifyContext, Label, Outcome, Probe
from crucible.synth.program import Program
from crucible.synth.task import Task

Truth = Callable[[Task], Literal["right", "wrong"] | None]


class ORACLE(BaseFalsifier):
    """Policy truth for the reference decision (module docstring)."""

    name: ClassVar[str] = "ORACLE"

    def __init__(self, truths: Sequence[Truth] = ()):
        self.truths = list(truths)

    async def applies(self, task: Task, ctx: FalsifyContext) -> str | None:
        if not self.truths:
            return "no executable policy truth is registered for any World"
        if all(t(task) is None for t in self.truths):
            return "no registered truth covers this task's pinned decisions"
        return None

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Probe]:
        return [Probe(cls=self.name, op="policy", site="reference_decision", expect="property")]

    async def execute(self, probe: Probe, task: Task, ctx: FalsifyContext) -> Outcome:
        verdicts = [v for v in (t(task) for t in self.truths) if v is not None]
        if not verdicts:
            return Outcome(valid=False, data={"verdict": None})
        return Outcome(passed=None, data={"verdict": "wrong" if "wrong" in verdicts else "right", "truths": verdicts})

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        return ("confirmed", "D6") if outcome.data.get("verdict") == "wrong" else ("clean", None)

    async def replay(self, cx: Any, task: Task, ctx: FalsifyContext) -> bool:
        out = await self.execute(cx.probe, task, ctx)
        return out.data.get("verdict") == "wrong"


def pinned(task: Task) -> list[tuple[str, str, str, str, Any]]:
    """The decision fields the contract pins: required ``eq`` values on concrete addresses (app, coll, key, field, v)."""
    out = []
    for p in task.contract.required:
        if isinstance(p.value, PEq) and not any(is_glob(x) for x in (p.app, p.coll, p.key, p.field)):
            out.append((unglob(p.app), unglob(p.coll), unglob(p.key), unglob(p.field), p.value.v))
    return out


class Dispute(CrucibleModel):
    disputed: bool
    fields: list[dict[str, Any]] = Field(default_factory=list)  # address, the oracle's value, the agreeing alternative, how many runs
    runs: int = 0


async def decision_disagreement(task: Task, programs: Sequence[dict[str, Any] | Program], ctx: FalsifyContext, *, min_agree: int = 2) -> Dispute:
    """Replay independent runs and compare the pinned decision fields of their end states with the oracle's value."""
    pins = pinned(task)
    if not pins or not programs:
        return Dispute(disputed=False, runs=len(programs))
    seen: dict[tuple[str, str, str, str], Counter[str]] = {}
    n = 0
    for prog in programs:
        p = prog if isinstance(prog, Program) else Program.model_validate(prog)
        r = await ctx.run(task, p)
        end = r.end
        if end is None:
            continue
        n += 1
        for app, coll, key, fld, _v in pins:
            row = end.collection(app, coll).get(key)
            before = (r.start.collection(app, coll).get(key) or {}) if r.start is not None else {}
            if row is not None and fld in row and (fld not in before or cjson(before[fld]) != cjson(row[fld])):  # a decision the run MADE
                seen.setdefault((app, coll, key, fld), Counter())[cjson(row[fld])] += 1
    out = []
    for app, coll, key, fld, v in pins:
        for alt, k in (seen.get((app, coll, key, fld)) or Counter()).most_common():
            if alt != cjson(v) and k >= min_agree:
                out.append({"address": f"{app}.{coll}[{key}].{fld}", "oracle": v, "alternative": alt, "runs": k})
                break
    return Dispute(disputed=bool(out), fields=out, runs=n)
