"""The four milestone filters (DESIGN §5.3.3): a candidate becomes a milestone only if it is

(a) **executable** on any state: its check measures (φ is not None) on the base state, the expert end state and the
    sampled intermediate states;
(b) **monotone on successful expert traces**: once satisfied it stays satisfied (regressions in good traces mean the
    predicate tracks scaffolding, not progress); measured as the share of good traces with no 1 -> 0 transition;
(c) **non-trivial**: false at the base state (true at the expert end state, else it is not progress);
(d) **predictive**: its value predicts the recorded outcome on later, held-out traces (``credit.validity``, §6.5).

Each filter returns a verdict with its evidence, so the family graph records why a candidate was kept or dropped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from crucible.workflow.milestones.evaluate import PhiSeries


@dataclass
class Verdict:
    milestone: str
    kept: bool
    checks: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


def executable(series: Sequence[PhiSeries], mid: str, *, min_share: float = 0.95) -> tuple[bool, float]:
    vals = [v for s in series for v in s.column(mid)]
    if not vals:
        return False, 0.0
    share = sum(v is not None for v in vals) / len(vals)
    return share >= min_share, share


def non_trivial(base_phi: float | None, end_phi: float | None) -> bool:
    return base_phi is not None and end_phi is not None and base_phi < 1.0 <= end_phi


def monotone(good: Sequence[PhiSeries], mid: str, *, min_share: float = 0.9) -> tuple[bool, float, int]:
    """Share of good traces in which the milestone, once satisfied, never falls back (only measured values count)."""
    relevant = ok = 0
    for s in good:
        col = [v for v in s.column(mid) if v is not None]
        if not any(v >= 1.0 for v in col):
            continue
        relevant += 1
        first = next(i for i, v in enumerate(col) if v >= 1.0)
        ok += all(v >= 1.0 for v in col[first:])
    share = ok / relevant if relevant else 0.0
    return (relevant > 0 and share >= min_share), share, relevant


def screen(mid: str, *, base_phi: float | None, end_phi: float | None, series: Sequence[PhiSeries], good: Sequence[PhiSeries],
           min_exec: float = 0.95, min_monotone: float = 0.9) -> Verdict:
    """Filters (a)-(c); (d) runs later on held-out traces with ``credit.validity.validate``."""
    ex, ex_share = executable(series, mid, min_share=min_exec)
    nt = non_trivial(base_phi, end_phi)
    mo, mo_share, n_rel = monotone(good, mid, min_share=min_monotone)
    checks = {"executable": round(ex_share, 3), "base_phi": base_phi, "end_phi": end_phi, "monotone": round(mo_share, 3), "good_traces_reaching": n_rel}
    if not ex:
        return Verdict(mid, False, checks, f"measured on only {ex_share:.0%} of states")
    if not nt:
        return Verdict(mid, False, checks, "already true at base or not reached by the expert")
    if not mo:
        return Verdict(mid, False, checks, f"regresses in good traces ({mo_share:.0%} monotone over {n_rel})" if n_rel else "no good trace reaches it")
    return Verdict(mid, True, checks, "executable, non-trivial, monotone")
