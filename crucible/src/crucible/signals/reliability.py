"""Signal reliability: per source of evidence, how often a signal agrees with what happened later (ARCHITECTURE M4).

A signal is *confirmed* or *contradicted* only by later evidence about the step it grades that its own mapping rule
did not use:

- a revert or a hotfix of the graded step (post-merge history, a refund reversal) confirms a negative signal on that
  step and contradicts a positive one;
- a correction or applied suggestion is contradicted when its fix was itself reverted, and confirmed when the trace
  ended well with the fix in place;
- a negative signal whose request was not acted on is contradicted when the trace ended well without it;
- a later rejection or correction of the same step contradicts an approval; a check is judged against the same check
  only (another CI job failing on that commit says nothing about this one);
- reverts and hotfixes define the coding outcome, so nothing independent judges them: they keep their prior.

Reliability is the posterior mean of Beta(prior + confirmed, prior + contradicted), kept apart per (family, type,
actor class: ``human`` or ``automated``) and written into each signal it can judge (``Signal.reliability``); other
signals keep ``None``.

M4 audit (2026-10-12): verdicts came from the trace outcome and the mapper's own rule (corrections could never be
contradicted, a comment rejection simply equalled the outcome, any bad trace confirmed every negative on any step, and
one CI job failing "contradicted" every other job's pass on that commit); estimates pooled families and AI reviewers
with people; ``apply`` gave the type mean to flaky and ungraded signals it could not judge; seq 0 read as "no seq".
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from crucible.core.schema.signal import Signal
from crucible.core.schema.trace import Trace
from crucible.signals.taxonomy import TAXONOMY

Key = tuple[str, str, str]  # (family, type, actor class)
UNJUDGED = frozenset({"revert"})  # they define the outcome: no evidence independent of them
REVISIONS = frozenset({"rejection", "correction", "suggestion_accepted", "revert"})  # later verdicts against an approval


def actor_class(s: Signal) -> str:
    return "automated" if s.meta.get("automated") else "human"


def key_of(trace: Trace, s: Signal) -> Key:
    return (trace.family or "", s.type, actor_class(s))


@dataclass
class Reliability:
    type: str
    confirmed: int = 0
    contradicted: int = 0
    unscored: int = 0
    family: str = ""
    actor_class: str = "human"

    @property
    def n(self) -> int:
        return self.confirmed + self.contradicted

    @property
    def mean(self) -> float:
        a, b = TAXONOMY[self.type].prior
        return (a + self.confirmed) / (a + b + self.n)

    def interval(self, z: float = 1.645) -> tuple[float, float]:
        """Approximate 90% credible interval (normal approximation to the Beta posterior)."""
        a, b = TAXONOMY[self.type].prior
        a, b = a + self.confirmed, b + self.contradicted
        m, sd = a / (a + b), math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        return max(0.0, m - z * sd), min(1.0, m + z * sd)


def judgeable(s: Signal) -> bool:
    return TAXONOMY[s.type].grades == "step" and s.polarity != 0 and bool(s.grades) and s.type not in UNJUDGED


def _after(o: Signal, s: Signal) -> bool:
    return o.source_event is not None and s.source_event is not None and o.source_event > s.source_event


def _trouble(trace: Trace, steps: set[int]) -> bool:
    """A revert or post-merge hotfix graded one of these steps."""
    return any((o.type == "revert" or (o.type == "correction" and o.meta.get("hotfix"))) and steps & set(o.grades) for o in trace.signals)


def verdict(trace: Trace, s: Signal) -> bool | None:
    """True confirmed, False contradicted, None no independent evidence either way."""
    if not judgeable(s):
        return None
    outcome = trace.outcome.label
    steps = set(s.grades)
    if s.type in ("correction", "suggestion_accepted"):
        if s.meta.get("hotfix"):
            return None  # a hotfix defines the bad outcome, like a revert
        fixed = s.meta.get("fixed_by_seq")
        if fixed is not None and _trouble(trace, {fixed}):
            return False  # the fix itself was rolled back
        return True if outcome == "good" else None
    if s.polarity < 0:
        if _trouble(trace, steps):
            return True
        if s.type == "check_result":
            same = [o for o in trace.signals if o.type == "check_result" and o.meta.get("check") == s.meta.get("check") and _after(o, s) and steps & set(o.grades)]
            return False if any(o.polarity > 0 for o in same) else None  # the same check passed on the same step later: it was noise
        acted = s.meta.get("acted_on")
        if acted:
            return True
        if acted is False and outcome == "good":
            return False
        return None
    # positive: approval, a passing check
    if _trouble(trace, steps):
        return False
    if s.type == "check_result":
        later = [o for o in trace.signals if o.type == "check_result" and o.meta.get("check") == s.meta.get("check") and _after(o, s) and steps & set(o.grades)]
    else:
        later = [o for o in trace.signals if o.type in REVISIONS and o.polarity < 0 and _after(o, s) and steps & set(o.grades)]
    if later:
        return False
    return True if outcome == "good" else None


def estimate(traces: Iterable[Trace]) -> dict[Key, Reliability]:
    table: dict[Key, Reliability] = {}
    for t in traces:
        for s in t.signals:
            k = key_of(t, s)
            r = table.setdefault(k, Reliability(s.type, family=k[0], actor_class=k[2]))
            v = verdict(t, s)
            if v is True:
                r.confirmed += 1
            elif v is False:
                r.contradicted += 1
            else:
                r.unscored += 1
    return table


def apply(trace: Trace, table: dict[Key, Reliability]) -> Trace:
    """Write the estimate onto the signals it can judge, from the row of their own family, type and actor class; a
    signal with no such row, or one the estimator cannot judge (flaky check, trace-level, ungraded), keeps ``None``."""
    sigs = []
    for s in trace.signals:
        r = table.get(key_of(trace, s)) if judgeable(s) else None
        sigs.append(s.model_copy(update={"reliability": round(r.mean, 4) if r is not None else None}))
    return trace.model_copy(update={"signals": sigs})
