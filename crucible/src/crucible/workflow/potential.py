"""C1, the milestone potential (DESIGN §6.2):

    Φ(s) = Σ_i w_i · φ_i(s) · Π_{j ∈ pre(i)} φ_j(s)  /  Σ_i w_i   ∈ [0, 1]

The prerequisite product stops credit for out-of-order "progress" (a test passing before the module builds only
happens by stubbing). Regressions lower Φ by themselves. An unmeasured φ (None) keeps its last measured value; if a
milestone was never measured, it counts as 0 *and* the step is flagged ``partial`` so callers can tell "no progress"
from "not measured".

Shaping (P13 type 1, policy-invariant): with Φ(terminal) := 0 and γ = 1, the per-step term is
r_t = Φ(s_t) − Φ(s_{t−1}) for t < T and the terminal step adds −Φ(s_{T−1}); the terms telescope to −Φ(s_0), so they
cannot change which policy is optimal (Ng, Harada & Russell 1999). They only move credit earlier.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from crucible.core.schema.milestone import Milestone


@dataclass
class PotentialStep:
    phi: float
    delta: float
    new: list[str] = field(default_factory=list)  # milestones that became credited at this step
    lost: list[str] = field(default_factory=list)
    partial: bool = False  # some milestone had never been measured


def potential(milestones: Sequence[Milestone], phis: Mapping[str, float | None]) -> tuple[float, bool]:
    """Φ for one state; ``phis`` maps milestone id -> φ (None = unmeasured). Returns (Φ, partial).

    M6 audit: callers pass raw lists (filtered subsets of a ``MilestoneSet``), so the set's rules are checked here:
    weights > 0, φ finite in [0, 1], prerequisites known (weights [2, -1] gave Φ = 2.0; φ = NaN gave Φ = NaN; an
    unknown prerequisite zeroed its milestone forever)."""
    ids = {m.id for m in milestones}
    for m in milestones:
        if not (m.weight > 0 and math.isfinite(m.weight)):
            raise ValueError(f"milestone {m.id}: weight {m.weight} must be finite and > 0")
        if bad := [r for r in m.requires if r not in ids]:
            raise ValueError(f"milestone {m.id} requires {bad}, which are not in this milestone list")
    for k, v in phis.items():
        if v is not None and k in ids and not (math.isfinite(v) and 0.0 <= v <= 1.0):
            raise ValueError(f"φ({k}) = {v} is not in [0, 1]")
    total = sum(m.weight for m in milestones)
    if total <= 0:
        return 0.0, False
    val = {m.id: phis.get(m.id) for m in milestones}
    partial = any(v is None for v in val.values())
    acc = 0.0
    for m in milestones:
        f = val[m.id] or 0.0
        for p in m.requires:
            f *= val.get(p) or 0.0
        acc += m.weight * f
    return acc / total, partial


def credited(milestones: Sequence[Milestone], phis: Mapping[str, float | None]) -> set[str]:
    """Milestones that contribute at this state (satisfied and all prerequisites satisfied)."""
    return {m.id for m in milestones if (phis.get(m.id) or 0) >= 1.0 and all((phis.get(p) or 0) >= 1.0 for p in m.requires)}


def trajectory(milestones: Sequence[Milestone], series: Sequence[Mapping[str, float | None]],
               initial: Mapping[str, float | None] | None = None) -> list[PotentialStep]:
    """Φ and ΔΦ along a series of states (``series[t]`` = φ per milestone after step t, with carried values).
    ``initial``: φ at the state before the first step (s0), so a milestone already true there is not credited to
    step 0 (M6 audit: Φ(s0) was taken as 0)."""
    out: list[PotentialStep] = []
    last: dict[str, float | None] = {k: v for k, v in (initial or {}).items() if v is not None}
    prev_phi, prev_set = (potential(milestones, last)[0], credited(milestones, last)) if initial else (0.0, set())
    for row in series:
        cur = {**last, **{k: v for k, v in row.items() if v is not None}}
        for k, v in row.items():
            cur.setdefault(k, v)
        phi, partial = potential(milestones, cur)
        now = credited(milestones, cur)
        out.append(PotentialStep(phi=phi, delta=phi - prev_phi, new=sorted(now - prev_set), lost=sorted(prev_set - now), partial=partial))
        last, prev_phi, prev_set = cur, phi, now
    return out


def shaping(phis: Sequence[float], *, initial: float | None = None) -> list[float]:
    """Potential-based shaping rewards for T steps given Φ after each step. ``initial`` = Φ(s_0) before the first
    step (default: 0). The terminal step's Φ is defined as 0, so the sum telescopes to −Φ(s_0)."""
    if not phis:
        return []
    prev = 0.0 if initial is None else initial
    out = []
    for t, p in enumerate(phis):
        if t == len(phis) - 1:
            out.append(0.0 - prev)  # Φ(terminal) := 0
        else:
            out.append(p - prev)
        prev = p
    return out


def fraction_index(n: int, frac: float) -> int:
    """The one definition of "the state at fraction f" of n step states (M6 audit: three midpoints coexisted)."""
    return min(n - 1, max(0, round(frac * (n - 1)))) if n else 0


def at_fraction(steps: Sequence[PotentialStep], frac: float) -> float:
    """Φ at a fraction of the trajectory (0.5 = midpoint state), used by the predictive-validity gate (§6.5)."""
    if not steps:
        return 0.0
    return steps[fraction_index(len(steps), frac)].phi
