"""Deep tasks: graders that reward doing better instead of clearing a bar, and an honest way to give up (DESIGN P20,
§7.5). Standard library only.

**Record scoring.** For tasks without a known ceiling (the highest-rank elliptic curve, the fastest kernel, the
smallest counterexample), a check measures a *score*, and the grader maps it onto the scale of what is known:

    normalized = (score - baseline) / (best_known - baseline)        (direction-aware)

0 is the baseline (the unmodified original, the naive solution), 1 matches the best known result, and **above 1 is a
new record**: the reward keeps growing, with a log tail (``1 + log(normalized)``) so a record pays more without
dominating the batch. A score counts only after the validity checks pass (an invalid "record" scores 0), and the
validity check is what the grader can verify, even when nobody knows how to produce the answer.

**Abstention.** A task may allow the agent to give up honestly (``ABSTAIN`` artifact or tool call with a reason).
What it earns depends on what the grader knows about feasibility:

- ``infeasible`` (constructed so: EnterpriseOps-Gym-style impossible requests, an in-transit refund): abstaining is
  the correct answer and earns full credit; any claimed solution earns 0;
- ``feasible``: abstaining earns 0 (it is not a solve), but never less than a wrong solution;
- ``unknown`` (open problems): abstaining earns a small fixed credit, below the learnability floor, so it can never
  beat a real attempt in expectation on a task a policy solves at least that often.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from crucible.verify.runtime.plan import Verdict


@dataclass
class RecordSpec:
    baseline: float  # the score of the unmodified original / naive solution
    best_known: float  # the best verified result so far (the record the task is anchored on)
    direction: str = "max"  # "max": higher is better; "min": lower is better (time, size, error)
    log_tail: bool = True

    def normalize(self, score: float) -> float:
        span = (self.best_known - self.baseline) if self.direction == "max" else (self.baseline - self.best_known)
        if span <= 0:
            raise ValueError("best_known must improve on the baseline")
        gain = (score - self.baseline) if self.direction == "max" else (self.baseline - score)
        return gain / span

    def reward(self, score: float) -> float:
        n = self.normalize(score)
        if n <= 0:
            return 0.0
        if n <= 1 or not self.log_tail:
            return n
        return 1.0 + math.log(n)


def record(score: float | None, spec: RecordSpec, *, valid: bool, detail: dict | None = None) -> Verdict:
    """A score check. ``passed`` means the result is valid and beats the baseline; the reward contribution is in
    ``detail["normalized_reward"]`` (read by ``Plan`` for gates of kind ``score``)."""
    d = dict(detail or {})
    # a "min" metric is a time, size or ratio: positive by nature (M7 audit V5: a forged ratio of -1000 scored 9.38)
    if score is not None and spec.direction == "min" and math.isfinite(float(score)) and float(score) <= 0:
        return Verdict(False, {**d, "score": score, "valid": False, "normalized_reward": 0.0, "reason": "a min-metric must be positive"})
    if not valid or score is None or not math.isfinite(float(score)):
        return Verdict(False, {**d, "score": score, "valid": valid, "normalized_reward": 0.0})
    r = spec.reward(float(score))
    n = spec.normalize(float(score))
    return Verdict(r > 0, {**d, "score": score, "valid": True, "normalized": round(n, 6), "normalized_reward": round(r, 6),
                           "new_record": n > 1.0, "best_known": spec.best_known, "baseline": spec.baseline})


@dataclass
class AbstainPolicy:
    feasibility: str = "feasible"  # "feasible" | "infeasible" | "unknown"
    unknown_credit: float = 0.05  # keep below the training band's floor (DESIGN §7.4: p >= 0.1)

    def credit(self) -> float:
        return {"infeasible": 1.0, "feasible": 0.0, "unknown": self.unknown_credit}[self.feasibility]
