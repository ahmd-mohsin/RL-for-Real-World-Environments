"""Online difficulty: per-task and per-milestone pass rates p̂ that follow the policy as it trains (swarm-pipeline/12 §1).

Pattern: Online Difficulty Filtering for reasoning-oriented RL (Bae et al., 2025; arXiv 2504.03380): the policy's own
rollouts estimate each prompt's pass rate *during* training, and only prompts of intermediate difficulty are trained on,
because an all-pass or all-fail group carries no GRPO signal. Rewritten for long-horizon tasks:

- **One estimate, decayed.** Each task keeps discounted counts: before a new observation at optimizer step ``s``, the
  old counts are multiplied by ``decay ** (s - last_step)``, so evidence from an older policy fades as the policy
  improves (``decay=1`` keeps every draw). p̂ = k_eff / n_eff with a Wilson interval on the effective counts
  (``forge.learnability`` uses the same interval for the factory's fixed-k measurement).
- **What is not evidence.** A masked draw (infrastructure, ``None`` outcome) is never counted (M8 audit: masked draws as
  failures pushed the advice to "easier"). A **budget stop** that failed is not a skill observation either (the
  targeting keep rule counts it as ``budget``): it is left out of the pass rate and counted apart; its measured Φ still
  counts, because that progress is real.
- **Φ spread.** The rate also tracks the final Φ of each draw (the checkpoint sidecar's ``phi_final``). A task whose
  draws all fail but whose Φ differs still teaches through step-wise credit (``train.quarantine`` never quarantines it
  either), so ``signal`` is ``max(4p(1-p), 4 Var(Φ))``: the learnability of the outcome, or of the progress.
- **Per milestone, two rates.** ``p(m)`` is the share of draws where m is *credited* (it passes and its prerequisites
  pass: ``factory.checkpoint.credited``'s C1 rule, re-implemented here on plain mappings because forge sits below the
  factory); ``p(m | prerequisites)`` counts only the draws where every prerequisite of m was credited. The second is
  checkpoint-conditioned difficulty: how hard m is once the agent got there.

Imports the stdlib, ``forge.learnability`` and the Wilson interval only, so every trainer can use it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from crucible.credit.operators.branch_value import wilson
from crucible.forge.learnability import BANDS, Band, learnability

BUDGET_STOPS = frozenset({"max_steps", "budget", "tool_limit", "timeout", "agent_timeout", "rollout_timeout"})


@dataclass(frozen=True)
class Draw:
    """One rollout as difficulty evidence. ``passed=None``: masked (infrastructure). ``stop``: why the episode ended."""

    passed: bool | None
    phi: float | None = None
    stop: str | None = None
    milestones: Mapping[str, bool] | None = None  # the final milestone vector, if graded

    @property
    def masked(self) -> bool:
        return self.passed is None

    @property
    def budget(self) -> bool:
        return self.passed is False and str(self.stop or "") in BUDGET_STOPS


def credited(status: Mapping[str, bool], requires: Mapping[str, Sequence[str]]) -> set[str]:
    """C1 on plain mappings: m counts when it passes and every direct prerequisite passes (same rule as
    ``factory.checkpoint.credited``, which takes milestone objects)."""
    return {m for m, ok in status.items() if ok and all(status.get(r) for r in requires.get(m, ()))}


@dataclass
class Counts:
    k: float = 0.0  # discounted successes
    n: float = 0.0  # discounted measured draws

    def decay(self, f: float) -> None:
        self.k *= f
        self.n *= f

    def add(self, ok: bool) -> None:
        self.n += 1.0
        self.k += 1.0 if ok else 0.0

    @property
    def p(self) -> float | None:
        return self.k / self.n if self.n > 0 else None

    def interval(self) -> tuple[float, float]:
        """Wilson on the effective counts (n rounded, k scaled to keep p): with no evidence, [0, 1]."""
        if self.n <= 0:
            return 0.0, 1.0
        n = max(1, round(self.n))
        lo, hi = wilson(self.k * n / self.n, n)
        return round(lo, 4), round(hi, 4)


@dataclass
class TaskRate:
    task: str
    full: Counts = field(default_factory=Counts)
    phi_n: float = 0.0  # discounted Φ moments (measured Φ only)
    phi_sum: float = 0.0
    phi_sq: float = 0.0
    milestone: dict[str, Counts] = field(default_factory=dict)  # p(m): credited
    conditional: dict[str, Counts] = field(default_factory=dict)  # p(m | every prerequisite credited)
    last_step: int | None = None
    masked: int = 0
    budget: int = 0
    draws: int = 0  # raw draws seen (every kind)

    def _decay(self, step: int, decay: float) -> None:
        if self.last_step is None or decay >= 1.0 or step <= self.last_step:
            self.last_step = step if self.last_step is None else max(self.last_step, step)
            return
        f = decay ** (step - self.last_step)
        for c in (self.full, *self.milestone.values(), *self.conditional.values()):
            c.decay(f)
        self.phi_n, self.phi_sum, self.phi_sq = self.phi_n * f, self.phi_sum * f, self.phi_sq * f
        self.last_step = step

    def add(self, d: Draw, requires: Mapping[str, Sequence[str]]) -> None:
        self.draws += 1
        if d.masked:
            self.masked += 1
            return
        if d.phi is not None and math.isfinite(d.phi):
            self.phi_n += 1.0
            self.phi_sum += float(d.phi)
            self.phi_sq += float(d.phi) ** 2
        if d.budget:
            self.budget += 1  # not a skill observation: out of the pass rate (its Φ above still counts)
        else:
            self.full.add(bool(d.passed))
        if d.milestones and not d.budget:
            got = credited(d.milestones, requires)
            for m in d.milestones:
                self.milestone.setdefault(m, Counts()).add(m in got)
                pre = list(requires.get(m, ()))
                if all(r in got for r in pre):
                    self.conditional.setdefault(m, Counts()).add(m in got)

    @property
    def p(self) -> float | None:
        return self.full.p

    def interval(self) -> tuple[float, float]:
        return self.full.interval()

    @property
    def phi_mean(self) -> float | None:
        return self.phi_sum / self.phi_n if self.phi_n > 0 else None

    @property
    def phi_var(self) -> float:
        if self.phi_n <= 1.0:
            return 0.0
        m = self.phi_sum / self.phi_n
        return max(0.0, self.phi_sq / self.phi_n - m * m)

    def signal(self) -> float:
        """How much a group of this task would teach now: the outcome's learnability 4p(1-p) or, when the outcome is
        flat, the progress spread 4 Var(Φ) (both in [0, 1]). Unmeasured: 1 (worth measuring)."""
        p = self.p
        if p is None and self.phi_n <= 0:
            return 1.0
        return max(learnability(p) if p is not None else 0.0, min(1.0, 4.0 * self.phi_var))

    def in_band(self, band: Band) -> bool | None:
        """Whether the interval overlaps the band (None: nothing measured). Overlap, not containment: a task is dropped
        only when the evidence says it is outside."""
        if self.full.n <= 0:
            return None
        lo, hi = self.interval()
        if band.on == "hi":
            return lo <= (band.hi if band.hi is not None else 1.0)
        return not ((band.lo is not None and hi < band.lo) or (band.hi is not None and lo > band.hi))

    def row(self, step: int | None = None) -> dict[str, Any]:
        lo, hi = self.interval()
        return {"task": self.task, "step": step if step is not None else self.last_step, "p": self.p, "lo": lo, "hi": hi, "n_eff": round(self.full.n, 4),
                "phi_mean": self.phi_mean, "phi_var": round(self.phi_var, 6), "signal": round(self.signal(), 6), "masked": self.masked,
                "budget": self.budget, "draws": self.draws,
                "milestones": {m: c.p for m, c in sorted(self.milestone.items())},
                "conditional": {m: c.p for m, c in sorted(self.conditional.items())}}


@dataclass
class DifficultyTracker:
    """p̂ per task over a run. ``requires``: task -> {milestone: its prerequisites} (for the per-milestone rates)."""

    decay: float = 0.9
    requires: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    rates: dict[str, TaskRate] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.decay <= 1.0:
            raise ValueError("decay is in (0, 1]")

    def rate(self, task: str) -> TaskRate:
        return self.rates.setdefault(str(task), TaskRate(str(task)))

    def observe(self, task: str, step: int, draws: Iterable[Draw]) -> TaskRate:
        r = self.rate(task)
        r._decay(int(step), self.decay)
        req = self.requires.get(str(task), {})
        for d in draws:
            r.add(d, req)
        return r

    def rows(self, step: int | None = None) -> list[dict[str, Any]]:
        return [r.row(step) for _, r in sorted(self.rates.items())]

    def to_json(self) -> dict[str, Any]:
        return {"decay": self.decay, "rates": {t: asdict(r) for t, r in self.rates.items()}}

    def load(self, doc: Mapping[str, Any] | None) -> DifficultyTracker:
        for t, r in ((doc or {}).get("rates") or {}).items():
            d = dict(r)
            self.rates[t] = TaskRate(task=t, full=Counts(**d.get("full") or {}), phi_n=d.get("phi_n", 0.0), phi_sum=d.get("phi_sum", 0.0),
                                     phi_sq=d.get("phi_sq", 0.0), milestone={m: Counts(**c) for m, c in (d.get("milestone") or {}).items()},
                                     conditional={m: Counts(**c) for m, c in (d.get("conditional") or {}).items()}, last_step=d.get("last_step"),
                                     masked=d.get("masked", 0), budget=d.get("budget", 0), draws=d.get("draws", 0))
        return self


TRAIN_BAND = BANDS["train"]
