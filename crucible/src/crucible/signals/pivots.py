"""Decision points (pivots) in expert traces -- NeMo-Gym ``swe_pivot`` / PivotRL's idea, applied to signals.

PivotRL trains on the turns where the action taken decides success or failure. In a recorded workflow the clearest
such turns are steps that were graded wrong *and then corrected*: the trace contains both the wrong action and
the expert's fix, i.e. a (rejected, chosen) pair at a known state. ``mark_pivots`` tags those steps
(``outcome.details["pivots"]``) and links each one to the correction that resolved it.

A revert marks a pivot whose right step is unknown: ``studio.datasets`` pairs only corrections and applied suggestions
(a revert gives a negative-only row). Signals the trace's later evidence contradicts are not pivots (M4 audit).

M4 audit R-1 (2026-10-27): ``graded_wrong`` is every step a negative step signal grades (a rejection, escalation,
violation, rework, interrupt or failing check as well as the corrective ones) that the trace does not contradict. Such a
step is never a positive target; ``pivots`` still decides the (rejected, chosen) pairs. Only corrective signals counted,
so a reply the customer rejected was an SFT row of a good trace.
"""

from __future__ import annotations

from crucible.core.schema.signal import Signal
from crucible.core.schema.trace import Trace

CORRECTIVE = {"correction", "suggestion_accepted", "revert"}


def _against(trace: Trace, s: Signal) -> bool:
    """A negative signal that grades a known step and that the trace's later evidence does not contradict."""
    from crucible.signals.reliability import verdict

    # a revert that touched no PR line graded every commit: no decision point is known
    return s.polarity < 0 and bool(s.grades) and s.meta.get("attribution") != "all_commits" and verdict(trace, s) is not False  # M4 audit A22


def pivots(trace: Trace) -> dict[int, list[str]]:
    """Graded step seq -> ids of corrective signals that resolved it (only steps with a known fix)."""
    out: dict[int, list[str]] = {}
    for s in trace.signals:
        if s.type in CORRECTIVE and _against(trace, s):
            for g in s.grades:
                out.setdefault(g, []).append(s.id)
    return out


def graded_wrong(trace: Trace) -> dict[int, list[str]]:
    """Graded step seq -> ids of every step-level signal that graded it wrong, fix known or not (M4 audit R-1; PATTERN:
    MiMo ``signed_rebalance.py:14-22``, a marked span of a positive sample is never reinforced). A superset of ``pivots``."""
    from crucible.signals.taxonomy import TAXONOMY

    out: dict[int, list[str]] = {}
    for s in trace.signals:
        if TAXONOMY[s.type].grades == "step" and _against(trace, s):
            for g in s.grades:
                out.setdefault(g, []).append(s.id)
    return out


def mark_pivots(trace: Trace) -> Trace:
    pv = pivots(trace)
    details = {**trace.outcome.details, "pivots": {str(k): v for k, v in sorted(pv.items())}}
    return trace.model_copy(update={"outcome": trace.outcome.model_copy(update={"details": details})})
