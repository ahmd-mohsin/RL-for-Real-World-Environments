"""G-tighten (DESIGN §7.4; m13 audit 06 §4.3): raise the verification strictness of a solved task, the way task04 went
from v1 to v3 (perf bars, concurrency bars, sanitizer tools).

A tightening is admissible only when it is *both* still fair and actually separating:

- **Fair:** the oracle keeps A1's headroom under every perf bar (``accept.policy.a1_headroom``, 15%; the old 10% was
  inconsistent with A1). For a new categorical gate (concurrency, a sanitizer tool), the oracle passes it on every
  measured run.
- **Separating:** the solution that solved the task before (the frontier's, recorded) fails the tightened bar.
  Otherwise the tightening makes the task stricter without making it harder for the solver that matters.

``tightest_bar`` gives the strictest admissible perf bar for a metric (``oracle / (1 - headroom)``, the oracle exactly at
the headroom); ``tighten_perf`` and ``add_gate`` decide, with reasons. The task is then re-accepted under the new
bars (A1 headroom is re-measured there; this module never edits a grader).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from crucible.verify.calibrate import perf_headroom


@dataclass
class TightenDecision:
    ok: bool
    bars: dict[str, float]
    reasons: list[str] = field(default_factory=list)
    headroom: dict = field(default_factory=dict)
    separates: dict[str, bool] = field(default_factory=dict)


def tightest_bar(oracle_ratio: float, *, headroom: float = 0.15, decimals: int = 2) -> float:
    """The strictest bar the oracle still meets with ``headroom`` (rounded up, never below the exact value)."""
    import math

    exact = oracle_ratio / (1 - headroom)
    q = 10 ** decimals
    return math.ceil(exact * q - 1e-9) / q


def tighten_perf(bars: dict[str, float], oracle_ratios: dict[str, float], frontier_ratios: dict[str, float], *, new: dict[str, float] | None = None,
                 headroom: float = 0.15) -> TightenDecision:
    """Lower perf bars (ratios to the original's time; lower is stricter). ``new`` defaults to the tightest admissible bar
    per metric the frontier solution meets today."""
    proposal = dict(new) if new is not None else {m: tightest_bar(oracle_ratios[m], headroom=headroom) for m in frontier_ratios if m in oracle_ratios}
    reasons: list[str] = []
    merged = dict(bars)
    for m, b in proposal.items():
        if m in bars and b >= bars[m]:
            reasons.append(f"{m}: {b} is not stricter than the current bar {bars[m]}")
        merged[m] = min(b, bars.get(m, b))
    h = perf_headroom(oracle_ratios, merged, required=headroom)
    if not h["ok"]:
        reasons.append(f"the oracle keeps only {h['min']} headroom (A1 requires {headroom}): not fair")
    sep = {m: frontier_ratios.get(m, 0.0) > merged[m] for m in proposal}
    if not any(sep.values()):
        reasons.append("the frontier solution meets every tightened bar: stricter, but not separating")
    return TightenDecision(not reasons, merged, reasons, h, sep)


def add_gate(gate: str, oracle_runs: Sequence[bool], frontier_runs: Sequence[bool], *, min_oracle_runs: int = 3) -> TightenDecision:
    """A new categorical gate (concurrency, a sanitizer tool): the oracle passes every run, the frontier solution fails one."""
    reasons = []
    if len(oracle_runs) < min_oracle_runs:
        reasons.append(f"{gate}: {len(oracle_runs)} oracle runs, need >= {min_oracle_runs}")
    elif not all(oracle_runs):
        reasons.append(f"{gate}: the oracle fails {sum(not x for x in oracle_runs)}/{len(oracle_runs)} runs: not fair")
    sep = not all(frontier_runs) if frontier_runs else False
    if not sep:
        reasons.append(f"{gate}: the frontier solution passes it: not separating")
    return TightenDecision(not reasons, {}, reasons, {}, {gate: sep})
