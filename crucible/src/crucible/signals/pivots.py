"""Decision points (pivots) in expert traces -- NeMo-Gym ``swe_pivot`` / PivotRL's idea, applied to signals.

PivotRL trains on the turns where the action taken decides success or failure. In a recorded workflow the clearest
such turns are steps that were graded wrong *and then corrected*: the trace contains both the wrong action and
the expert's fix, i.e. a (rejected, chosen) pair at a known state. ``mark_pivots`` tags those steps
(``outcome.details["pivots"]``) and links each one to the correction that resolved it.
"""

from __future__ import annotations

from crucible.core.schema.trace import Trace

CORRECTIVE = {"correction", "suggestion_accepted", "revert"}


def pivots(trace: Trace) -> dict[int, list[str]]:
    """Graded step seq -> ids of corrective signals that resolved it (only steps with a known fix)."""
    out: dict[int, list[str]] = {}
    for s in trace.signals:
        if s.type in CORRECTIVE and s.polarity < 0:
            for g in s.grades:
                out.setdefault(g, []).append(s.id)
    return out


def mark_pivots(trace: Trace) -> Trace:
    pv = pivots(trace)
    details = {**trace.outcome.details, "pivots": {str(k): v for k, v in sorted(pv.items())}}
    return trace.model_copy(update={"outcome": trace.outcome.model_copy(update={"details": details})})
