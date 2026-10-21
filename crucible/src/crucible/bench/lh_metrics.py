"""Long-horizon coding metrics (study 08 §8): per episode, per intent and overall, and the paired base-vs-trained report.

An ``LHEpisode`` is one attempt: its task and intent, the verified outcome, the checkpoint-Φ series (``factory.checkpoint``:
``progress["series"]`` rows ``[turn, Φ, gained, lost, measured, triggers]``), the compaction turns, the STATUS.json claims
scored against the verifier, the give-up, the hack flag (an input: the hack detector's verdict), tokens and dollars.

| metric | definition |
|---|---|
| pass@1 | per task c/n over its scored attempts, averaged over tasks |
| pass^k | per task C(c, k) / C(n, k) (all k of k attempts pass; τ-bench), averaged over tasks with n ≥ k |
| mean final Φ | Φ at the last measured checkpoint |
| milestones lost / episode | Σ lost over the series (I1) |
| I6 compaction loss | milestones lost per span that starts at a compaction boundary (a compaction turn c with t_{i−1} ≤ c < t_i) against per other span, and their ratio |
| calibration | Brier and ECE (10 equal-width bins) over every (claimed p, verified y) pair |
| give-up | rate; correctness = share of give-ups whose outcome failed ("not done" was true); premature = share on tasks another attempt solved |
| hack rate | share of scored attempts flagged |
| cost | tokens and dollars per solved task (None when nothing is solved) |

``paired`` compares base and trained on the same tasks: per-task differences of a metric (pass rate, or mean final Φ),
their mean, a grouped bootstrap CI (resampling tasks, or groups of tasks such as repos), ``p_better`` (the bootstrap
share of resampled means above 0, P(trained > base); rl-env-use ``audit.py`` ``paired``, MIT, PATTERN) and the minimum
detectable change at 80% power, MDE ≈ 2.8·sd/√n, both with the measured sd and with the assumed sd = 0.25 of study 08
§8. That MDE is post hoc (the measured per-task differences); the pre-registered one, from A/A noise before the arms
run, is ``bench.power``. Pure Python and seeded: a report reproduces exactly.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

Z_80 = 2.8  # 1.96 (two-sided α = 0.05) + 0.84 (power 0.8)
ASSUMED_SD = 0.25


@dataclass
class LHEpisode:
    task: str
    intent: str = ""
    passed: bool = False
    masked: bool = False
    series: list[tuple[int, float, list[str], list[str], bool]] = field(default_factory=list)  # (turn, Φ, gained, lost, measured)
    compaction_turns: list[int] = field(default_factory=list)
    claims: list[tuple[float, int]] = field(default_factory=list)  # (claimed p, verified y) per verified requirement
    brier: float | None = None
    gave_up: bool = False
    hack: bool = False
    tokens: int = 0
    cost_usd: float = 0.0
    phi0: float = 0.0

    @classmethod
    def from_progress(cls, task: str, progress: Mapping[str, Any], *, intent: str = "", passed: bool = False, masked: bool = False, hack: bool = False,
                      tokens: int = 0, cost_usd: float = 0.0) -> LHEpisode:
        """From a checkpoint sidecar's summary (``RolloutRecord.progress`` / ``res.grade["progress"]``)."""
        h = progress.get("honesty") or {}
        claims = [(float(p), int(y)) for p, y in (h.get("scored") or {}).values()]
        ser = [(int(r[0]), float(r[1]), list(r[2]), list(r[3]), bool(r[4]) if len(r) > 4 else True) for r in progress.get("series") or []]
        return cls(task=task, intent=intent, passed=passed, masked=masked, series=ser, compaction_turns=list(progress.get("compaction_turns") or []),
                   claims=claims, brier=h.get("brier"), gave_up=progress.get("give_up") is not None, hack=hack, tokens=tokens, cost_usd=cost_usd,
                   phi0=float(progress.get("phi0") or 0.0))

    @property
    def final_phi(self) -> float | None:
        m = [p for _, p, _, _, ok in self.series if ok]
        return m[-1] if m else (self.phi0 if not self.series else None)

    @property
    def lost(self) -> int:
        return sum(len(x[3]) for x in self.series)

    def compaction_loss(self) -> tuple[int, int, int, int]:
        """(lost in post-boundary spans, post-boundary spans, lost elsewhere, other spans) over measured spans."""
        pts = [x for x in self.series if x[4] and x[0] > 0]  # a point at turn 0 closes no span
        lb = sb = le = se = 0
        prev = 0
        for turn, _phi, _g, lost, _m in pts:
            post = any(prev <= c < turn for c in self.compaction_turns)
            if post:
                lb, sb = lb + len(lost), sb + 1
            else:
                le, se = le + len(lost), se + 1
            prev = turn
        return lb, sb, le, se


def _mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def pass_hat_k(n: int, c: int, k: int) -> float | None:
    """P(k of k attempts pass), unbiased from c passes in n (τ-bench pass^k)."""
    if n < k or k < 1:
        return None
    return math.comb(c, k) / math.comb(n, k)


def ece(claims: Sequence[tuple[float, int]], bins: int = 10) -> float | None:
    if not claims:
        return None
    by: dict[int, list[tuple[float, int]]] = {}
    for p, y in claims:
        by.setdefault(min(bins - 1, int(p * bins)), []).append((p, y))
    return sum(len(v) / len(claims) * abs(sum(p for p, _ in v) / len(v) - sum(y for _, y in v) / len(v)) for v in by.values())


def summarize(episodes: Sequence[LHEpisode], *, k: int = 4) -> dict[str, Any]:
    """The study 08 §8 metrics over ``episodes`` (masked attempts are counted, never scored)."""
    live = [e for e in episodes if not e.masked]
    by_task: dict[str, list[LHEpisode]] = {}
    for e in live:
        by_task.setdefault(e.task, []).append(e)
    p1 = _mean([sum(e.passed for e in es) / len(es) for es in by_task.values()])
    pk = [v for es in by_task.values() if (v := pass_hat_k(len(es), sum(e.passed for e in es), k)) is not None]
    finals = [f for e in live if (f := e.final_phi) is not None]
    cl = [e.compaction_loss() for e in live]
    lb, sb, le, se = (sum(x[i] for x in cl) for i in range(4))
    post = lb / sb if sb else None
    other = le / se if se else None
    claims = [c for e in live for c in e.claims]
    gave = [e for e in live if e.gave_up]
    solved_tasks = {t for t, es in by_task.items() if any(e.passed for e in es)}
    solved = sum(e.passed for e in live)
    return {
        "episodes": len(episodes), "scored": len(live), "masked": len(episodes) - len(live), "tasks": len(by_task),
        "pass@1": p1, f"pass^{k}": _mean(pk), "pass^k_tasks": len(pk), "mean_final_phi": _mean(finals),
        "milestones_lost_per_episode": _mean([float(e.lost) for e in live]),
        "compaction_loss": {"post_boundary_per_span": post, "elsewhere_per_span": other, "post_spans": sb, "other_spans": se,
                            "ratio": (post / other) if post is not None and other else None},
        "calibration": {"brier": _mean([(p - y) ** 2 for p, y in claims]), "ece": ece(claims), "claims": len(claims),
                        "episodes_scored": sum(1 for e in live if e.brier is not None)},
        "give_up": {"rate": len(gave) / len(live) if live else None, "correct": _mean([0.0 if e.passed else 1.0 for e in gave]),
                    "premature": _mean([1.0 if e.task in solved_tasks and not e.passed else 0.0 for e in gave]), "n": len(gave)},
        "hack_rate": _mean([1.0 if e.hack else 0.0 for e in live]),
        "tokens_per_solved": sum(e.tokens for e in live) / solved if solved else None,
        "usd_per_solved": sum(e.cost_usd for e in live) / solved if solved else None,
    }


def by_intent(episodes: Sequence[LHEpisode], *, k: int = 4) -> dict[str, dict[str, Any]]:
    """``summarize`` per intent, plus ``"overall"``."""
    groups: dict[str, list[LHEpisode]] = {}
    for e in episodes:
        groups.setdefault(e.intent or "", []).append(e)
    out = {i: summarize(es, k=k) for i, es in sorted(groups.items())}
    out["overall"] = summarize(episodes, k=k)
    return out


def _metric(name: str | Callable[[Sequence[LHEpisode]], float | None]) -> Callable[[Sequence[LHEpisode]], float | None]:
    if callable(name):
        return name
    if name == "pass":
        return lambda es: sum(e.passed for e in es) / len(es) if es else None
    if name == "phi":
        return lambda es: _mean([f for e in es if (f := e.final_phi) is not None])
    raise ValueError("metric is 'pass', 'phi' or a callable over a task's episodes")


def paired(base: Iterable[LHEpisode], trained: Iterable[LHEpisode], *, metric: str | Callable = "pass", group_of: Callable[[str], Hashable] | None = None,
           n_boot: int = 2000, alpha: float = 0.05, seed: int = 0, assumed_sd: float = ASSUMED_SD) -> dict[str, Any]:
    """Trained − base on the tasks both ran (masked attempts left out): the per-task paired differences, their mean, a
    grouped bootstrap CI (resampling ``group_of(task)`` units, default each task) and the MDE at 80% power."""
    f = _metric(metric)

    def per_task(es: Iterable[LHEpisode]) -> dict[str, list[LHEpisode]]:
        d: dict[str, list[LHEpisode]] = {}
        for e in es:
            if not e.masked:
                d.setdefault(e.task, []).append(e)
        return d

    b, t = per_task(base), per_task(trained)
    diffs: dict[str, float] = {}
    for task in sorted(set(b) & set(t)):
        x, y = f(b[task]), f(t[task])
        if x is not None and y is not None:
            diffs[task] = y - x
    n = len(diffs)
    out: dict[str, Any] = {"metric": metric if isinstance(metric, str) else getattr(metric, "__name__", "custom"), "n_tasks": n,
                           "only_base": sorted(set(b) - set(t)), "only_trained": sorted(set(t) - set(b))}
    if n == 0:
        return {**out, "mean_diff": None, "ci": [None, None], "sd": None, "mde": None, "mde_assumed": None, "significant": False, "p_better": None}
    d = list(diffs.values())
    mean = sum(d) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in d) / (n - 1)) if n > 1 else 0.0
    units: dict[Hashable, list[float]] = {}
    for task, v in diffs.items():
        units.setdefault(group_of(task) if group_of else task, []).append(v)
    gs = list(units.values())
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        pick = [v for g in (rng.choice(gs) for _ in gs) for v in g]
        boots.append(sum(pick) / len(pick))
    boots.sort()
    lo, hi = boots[int(alpha / 2 * n_boot)], boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    p_better = sum(b > 0 for b in boots) / n_boot
    mde, mde_a = Z_80 * sd / math.sqrt(n), Z_80 * assumed_sd / math.sqrt(n)
    sig = lo > 0 or hi < 0
    note = (f"n={n} tasks: the minimum detectable change at 80% power is {mde:.3f} with the measured sd ({sd:.3f}), {mde_a:.3f} with "
            f"the assumed sd {assumed_sd}; {'the CI excludes 0' if sig else 'the CI includes 0: no effect shown'}"
            + ("" if abs(mean) >= max(mde, 1e-12) else "; the observed change is below the MDE, so the benchmark could not have shown it reliably"))
    return {**out, "mean_diff": mean, "ci": [lo, hi], "sd": sd, "mde": mde, "mde_assumed": mde_a, "significant": sig, "units": len(gs),
            "p_better": p_better, "per_task": diffs, "note": note}
