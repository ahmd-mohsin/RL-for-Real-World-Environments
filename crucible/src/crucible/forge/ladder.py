"""The solver ladder (m13 G-4; audit 06 §4.5): difficulty measured by policies of increasing strength, under the same
harness as training, and used to accept, steer and retire tasks.

Rungs (any subset; a rung is ``Rung(name, run, k)`` where ``run(n)`` returns n binary outcomes, e.g. M10 ``Runner``
samples of a policy):

- **L0** no-op / naive scripted (= A2), **L1** the cheapest model, **L2** the policy being trained (its checkpoint),
  **L3** the frontier, single attempt, **L4** the frontier, pass@k (k = 8), **L5** a frontier swarm (optional,
  "beyond frontier" bench tier).

Each rung is sampled with ``learnability.measure`` (4 → 8 → 16, Wilson, stop when the band is decided). pass@k uses the
unbiased estimator ``1 - C(n - c, k) / C(n, k)`` over the rung's samples.

- **Monotonicity**: L1 ≤ L2 ≤ L3 ≤ L4 must not invert beyond their intervals (a weaker rung's lower bound above a
  stronger rung's upper bound). An inversion means a flaky grader or a check that grades style; it blocks acceptance.
- **Acceptance**: train, ``L2.p`` in ``BANDS["train"]``; bench, ``L3`` upper bound ≤ 0.2.
- **Steering** (``steer``): harder, compose one more component; easier, drop the outermost component. Instruction
  obscurity is never a knob.
- **Retirement** (``retire``): once training lifts L2 above 0.7, the task goes back to the Proposer as a seed for one
  more composition.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from math import comb

from crucible.forge.learnability import BANDS, Band, PassRate, estimate, measure

ORDER = ("L0", "L1", "L2", "L3", "L4", "L5")


@dataclass
class Rung:
    name: str
    run: Callable[[int], Awaitable[list[bool]]]
    k: int = 1  # pass@k (L4)
    model: str = ""


@dataclass
class LadderReport:
    rates: dict[str, PassRate] = field(default_factory=dict)
    pass_at_k: dict[str, float] = field(default_factory=dict)
    outcomes: dict[str, list[bool]] = field(default_factory=dict)
    inversions: list[str] = field(default_factory=list)

    def ok(self) -> bool:
        return not self.inversions


def pass_at_k(n: int, c: int, k: int) -> float:
    """The unbiased pass@k estimator over n samples with c successes (Chen et al. 2021)."""
    if k <= 0 or n <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


async def run_ladder(rungs: Sequence[Rung], *, band: Band = BANDS["train"], steps: Sequence[int] = (4, 8, 16), task_hash: str | None = None) -> LadderReport:
    rep = LadderReport()
    for r in sorted(rungs, key=lambda x: ORDER.index(x.name) if x.name in ORDER else 99):
        got: list[bool] = []

        async def run(n: int, r: Rung = r, got: list[bool] = got) -> list[bool]:
            out = list(await r.run(n))
            got += out
            return out

        rate = await measure(run, band, steps=steps, model=r.model or r.name, harness="ladder", task_hash=task_hash)
        rep.rates[r.name], rep.outcomes[r.name] = rate, got
        if r.k > 1:
            rep.pass_at_k[r.name] = pass_at_k(len(got), sum(got), r.k)
    rep.inversions = monotonicity(rep.rates)
    return rep


def monotonicity(rates: dict[str, PassRate]) -> list[str]:
    """Inversions beyond the intervals among the rated rungs (a weaker rung clearly above a stronger one)."""
    names = [n for n in ORDER if n in rates and n != "L0"]
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if rates[a].lo > rates[b].hi:
                out.append(f"{a} (p={rates[a].p:.2f}, [{rates[a].lo:.2f}, {rates[a].hi:.2f}]) above {b} (p={rates[b].p:.2f}, "
                           f"[{rates[b].lo:.2f}, {rates[b].hi:.2f}]): a flaky grader or a check that grades style")
    return out


def accept(rep: LadderReport, *, split: str = "train") -> tuple[bool, list[str]]:
    reasons = list(rep.inversions)
    if split == "train":
        r = rep.rates.get("L2")
        if r is None:
            reasons.append("no L2 (training policy) rate")
        elif not (BANDS["train"].lo <= r.p <= BANDS["train"].hi):
            reasons.append(f"L2 p={r.p:.2f} outside the train band [{BANDS['train'].lo}, {BANDS['train'].hi}]")
    else:
        r = rep.rates.get("L3")
        if r is None:
            reasons.append("no L3 (frontier) rate")
        elif r.hi > BANDS["bench"].hi:
            reasons.append(f"L3 upper bound {r.hi:.2f} > {BANDS['bench'].hi}: not a bench task")
    return not reasons, reasons


def steer(rep: LadderReport, *, split: str = "train") -> str:
    """"compose_one_more" (too easy), "drop_outermost" (too hard), "keep" or "fix_grader" (inversions)."""
    if rep.inversions:
        return "fix_grader"
    rung = rep.rates.get("L2" if split == "train" else "L3")
    if rung is None:
        return "keep"
    if split == "train":
        if rung.p > BANDS["train"].hi:
            return "compose_one_more"
        if rung.p < BANDS["train"].lo:
            return "drop_outermost"
        return "keep"
    return "compose_one_more" if rung.hi > BANDS["bench"].hi else "keep"


def retire(l2_training_rate: float) -> bool:
    """A task the policy has learned (L2 above the band) goes back to the Proposer as a seed for one more composition."""
    return l2_training_rate > (BANDS["train"].hi or 0.7)


def rate_of(outcomes: Sequence[bool], name: str) -> PassRate:
    return estimate(list(outcomes), model=name, harness="ladder")
