"""Learnability steering (DESIGN §7.4, P29; pattern: verifiers ``environments/proposer_solver``, MIT).

A training task teaches most when the policy being trained sometimes solves it: the proposer reward ``4p(1-p)`` peaks at
p = 0.5. Upstream estimates p from 4 rollouts with no interval; here p is the **binary outcome** rate per (policy,
harness) with a Wilson interval, sampled 4 -> 8 -> 16 until the interval is clearly inside or outside the band.

Bands: train ``p in [0.1, 0.7]`` for the policy being trained; bench ``upper bound of frontier pass@1 <= 0.2``.
``advise`` says which way to move the generator's knobs (span length, traps, distractors, tighten), never "rewrite the
instruction to be vaguer" (difficulty through knobs only). For score-gated deep tasks, p is the share of rollouts that
beat the baseline; the record frontier is tracked separately (it does not saturate).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from crucible.credit.operators.branch_value import wilson


def learnability(p: float) -> float:
    return 4.0 * p * (1.0 - p)


@dataclass(frozen=True)
class PassRate:
    model: str
    harness: str
    n: int
    k: int
    p: float
    lo: float
    hi: float
    task_hash: str | None = None
    masked: int = 0  # draws with no verdict (infrastructure): outside n, never failures


@dataclass(frozen=True)
class Band:
    lo: float | None = None
    hi: float | None = None
    on: str = "p"  # "p": the point estimate must be in the band; "hi": the upper bound must be below ``hi``


@dataclass(frozen=True)
class KnobMove:
    direction: str  # "harder" | "easier"
    reason: str
    knobs: tuple[str, ...] = ("span_length", "traps", "distractors", "tighten")


BANDS = {"train": Band(lo=0.1, hi=0.7, on="p"), "bench": Band(hi=0.2, on="hi")}


def estimate(outcomes: Sequence[bool | None], *, model: str, harness: str, task_hash: str | None = None) -> PassRate:
    """``None`` is a masked draw: excluded from n and counted apart (M8 audit, sweeps G2-G4: a masked episode counted as
    a failure, so infrastructure noise lowered p and the advice said "easier"; MiMo's DAPO filter drops them)."""
    measured = [bool(x) for x in outcomes if x is not None]
    n, k = len(measured), sum(measured)
    lo, hi = wilson(k, n)
    return PassRate(model, harness, n, k, k / n if n else 0.0, round(lo, 4), round(hi, 4), task_hash, masked=len(outcomes) - n)


def decided(rate: PassRate, band: Band) -> bool:
    """Whether the interval already says inside or outside the band (stop sampling). Nothing measured decides nothing."""
    if rate.n == 0:
        return False
    if band.on == "hi":
        return rate.hi <= (band.hi or 1.0) or rate.lo > (band.hi or 1.0)
    inside = (band.lo is None or rate.lo >= band.lo) and (band.hi is None or rate.hi <= band.hi)
    outside = (band.lo is not None and rate.hi < band.lo) or (band.hi is not None and rate.lo > band.hi)
    return inside or outside


async def measure(run: Callable[[int], Awaitable[list[bool | None]]], band: Band, *, steps: Sequence[int] = (4, 8, 16), model: str, harness: str,
                  task_hash: str | None = None) -> PassRate:
    """``run(n)`` returns n more outcomes (``None`` = masked). Samples in steps of *measured* outcomes until the band is
    decided or the steps run out; masked draws are re-drawn, up to twice the last step in all."""
    outcomes: list[bool | None] = []
    rate = estimate([], model=model, harness=harness, task_hash=task_hash)
    cap = 2 * (steps[-1] if steps else 0)
    for target in steps:
        while len([x for x in outcomes if x is not None]) < target and len(outcomes) < cap:
            need = min(target - len([x for x in outcomes if x is not None]), cap - len(outcomes))
            got = list(await run(need))
            if not got:
                break
            outcomes += got
        rate = estimate(outcomes, model=model, harness=harness, task_hash=task_hash)
        if decided(rate, band) or len(outcomes) >= cap:
            break
    return rate


def advise(rate: PassRate, band: Band, *, final: bool = True) -> list[KnobMove]:
    """``final=False`` (more samples are possible): while the interval does not decide the band, the advice is to
    sample more, never to move a knob on noise. ``final=True``: the sampling budget is spent; act on the point estimate."""
    if rate.n == 0:  # every draw masked: no evidence to move a knob on
        return [KnobMove("sample_more", f"no measured outcome ({rate.masked} masked draws): re-run when the infrastructure is healthy", ())]
    if not final and not decided(rate, band):
        return [KnobMove("sample_more", f"p = {rate.p:.2f}, interval [{rate.lo:.2f}, {rate.hi:.2f}] does not decide the band yet", ())]
    if band.on == "hi":
        return [KnobMove("harder", f"frontier pass@1 upper bound {rate.hi:.2f} > {band.hi}")] if rate.hi > (band.hi or 1.0) else []
    if band.hi is not None and rate.p > band.hi:
        return [KnobMove("harder", f"p = {rate.p:.2f} above {band.hi}: little signal left (learnability {learnability(rate.p):.2f})")]
    if band.lo is not None and rate.p < band.lo:
        return [KnobMove("easier", f"p = {rate.p:.2f} below {band.lo}: all-fail groups carry no signal", ("span_length", "traps", "distractors"))]
    return []
