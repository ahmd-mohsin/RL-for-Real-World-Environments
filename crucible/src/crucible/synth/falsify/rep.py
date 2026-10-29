"""REP: determinism (part2-paper-core §5; G2). π* again from ``reset(seed)``, and again after a snapshot restore.

Each probe compares one replay with the cached oracle run on four things: the start state (reset fidelity), Δ under the
contract's normalizers, the verdict, and the answer. Any difference is confirmed **D11**: the reward depends on run
noise (a timestamp or random id the contract reads without a normalizer, a nondeterministic tool, an inexact restore).
A World without snapshots gets only the reset replays.
"""

from __future__ import annotations

from typing import ClassVar

from crucible.synth.canon import same_diff
from crucible.synth.contract import answer_equal
from crucible.synth.falsify.base import BaseFalsifier, FalsifyContext, Label, Outcome, Probe, delta_of
from crucible.synth.program import Run, run
from crucible.synth.task import Task


class REP(BaseFalsifier):
    name: ClassVar[str] = "REP"

    def __init__(self, replays: int = 2):
        self.replays = replays

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Probe]:
        out = [Probe(cls="REP", op="reset_replay", site=f"reset_replay:{i}", expect="property") for i in range(self.replays)]
        out.append(Probe(cls="REP", op="restore_replay", site="restore_replay", expect="property"))
        return out

    async def execute(self, probe: Probe, task: Task, ctx: FalsifyContext) -> Outcome:
        oracle = await ctx.oracle(task)
        w = await ctx.world(task)
        if probe.op == "restore_replay":
            await w.reset(task.seed)
            try:
                ref = await w.snapshot()
            except (NotImplementedError, AttributeError):
                return Outcome(valid=False, error="no snapshot")
            await run(w, task.oracle, task.instruction, seed=task.seed, given=task.intent.given, reset=False, taint=False)
            await w.restore(ref)
            r = await run(w, task.oracle, task.instruction, seed=task.seed, given=task.intent.given, reset=False, taint=False)
        else:
            r = await ctx.run(task, task.oracle)
        diffs = compare(r, oracle, task, ctx)
        return Outcome(passed=None, data={"differs": diffs})

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        return ("confirmed", "D11") if outcome.data.get("differs") else ("clean", None)


def compare(r: Run, oracle: Run, task: Task, ctx: FalsifyContext) -> list[str]:
    out = []
    if (r.start is None) != (oracle.start is None) or (r.start is not None and oracle.start is not None and r.start.fingerprint() != oracle.start.fingerprint()):
        out.append("start")
    if not same_diff(delta_of(r, task.contract), delta_of(oracle, task.contract)):
        out.append("delta")
    v1, v2 = ctx.grade(r, task.contract), ctx.grade(oracle, task.contract)
    if v1.passed != v2.passed or v1.reasons != v2.reasons:
        out.append("verdict")
    if not ((r.answer is None and oracle.answer is None) or answer_equal(r.answer, oracle.answer)):
        out.append("answer")
    return out
