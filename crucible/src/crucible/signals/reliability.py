"""Signal reliability: per type, how often a signal agrees with what happened later (ARCHITECTURE M4).

A signal is *confirmed* or *contradicted* by later evidence about the step it grades:

- a negative step signal (rejection, rework, escalation, a failing check, ...) is confirmed when the graded object
  was changed afterwards (the actor or someone else acted on it: ``meta["acted_on"]``) or the trace ended badly,
  and contradicted when nothing changed and the trace ended well (the work was accepted as it was);
- a positive step signal (approval, a passing check) is confirmed when the trace ended well and no later negative
  signal graded the same step, and contradicted otherwise;
- signals that are themselves the later evidence (correction, suggestion_accepted, revert) are confirmed by
  construction unless the trace outcome contradicts them, and outcome / target signals are not scored.

Reliability is the posterior mean of Beta(prior + confirmed, prior + contradicted) per type (per family),
written into each signal (``Signal.reliability``) so credit assignment (M6) can weigh evidence.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from crucible.core.schema.signal import Signal
from crucible.core.schema.trace import Trace
from crucible.signals.taxonomy import TAXONOMY

LATER_EVIDENCE = {"correction", "suggestion_accepted", "revert"}


@dataclass
class Reliability:
    type: str
    confirmed: int = 0
    contradicted: int = 0
    unscored: int = 0

    @property
    def n(self) -> int:
        return self.confirmed + self.contradicted

    @property
    def mean(self) -> float:
        a, b = TAXONOMY[self.type].prior
        return (a + self.confirmed) / (a + b + self.n)


def verdict(trace: Trace, s: Signal) -> bool | None:
    """True confirmed, False contradicted, None no evidence either way."""
    kind = TAXONOMY[s.type]
    if kind.grades == "trace" or s.polarity == 0 or not s.grades:
        return None
    outcome = trace.outcome.label
    if s.type in LATER_EVIDENCE:
        return not (outcome == "good" and s.type == "revert")
    if s.polarity < 0:
        acted = s.meta.get("acted_on")
        if acted or outcome == "bad":
            return True
        if acted is False and outcome == "good":
            return False
        return None
    later_neg = any(o.polarity < 0 and set(o.grades) & set(s.grades) and (o.source_event or -1) > (s.source_event or -1) for o in trace.signals)
    if outcome == "good" and not later_neg:
        return True
    if outcome == "bad" or later_neg:
        return False
    return None


def estimate(traces: Iterable[Trace]) -> dict[str, Reliability]:
    table: dict[str, Reliability] = {}
    for t in traces:
        for s in t.signals:
            r = table.setdefault(s.type, Reliability(s.type))
            v = verdict(t, s)
            if v is True:
                r.confirmed += 1
            elif v is False:
                r.contradicted += 1
            else:
                r.unscored += 1
    return table


def apply(trace: Trace, table: dict[str, Reliability]) -> Trace:
    sigs = [s.model_copy(update={"reliability": round(table[s.type].mean, 4)}) if s.type in table else s for s in trace.signals]
    return trace.model_copy(update={"signals": sigs})
