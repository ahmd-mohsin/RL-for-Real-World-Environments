"""Benchmark power: A/A noise from the attempts an arm already has, the minimum detectable effect (MDE) of a paired
trained − control comparison, the pre-registration gate and the planner (how many tasks N and attempts k).

Adapted from browser-use/rl-env-use (MIT; ``rl_env_use/climb.py`` 177-199: the A/A noise of even vs odd reps and the
"underpowered" stop before any spend; ``audit.py`` 46-69: the paired, case-clustered bootstrap). PATTERN, rewritten.

**Model.** Task i of a benchmark has per-attempt outcomes y_ij (pass in {0, 1}, reward or final Φ in [0, 1]) with mean
μ_i and within-task variance σ_i². An arm's score is the task-clustered mean Σ_i ȳ_i / N (each task weighs the same, as
pass@1 does). A paired comparison runs both arms on the same N tasks with k attempts each:

    Var(trained − control) = (σ̄_c² + σ̄_t²) / (k·N) + τ² / N

with σ̄² the mean within-task variance of an arm and τ² the spread of the true per-task effect across tasks (0 for the
claim "the trained model beats the control on these tasks"; > 0 for "on tasks like these").

**A/A noise** (``aa_noise``). Split each task's attempts into two halves A and B; under A/A the halves have the same
mean, so d_i = mean(A) − mean(B) has E[d_i²] = σ_i² (1/|A| + 1/|B|) and d_i² / (1/|A| + 1/|B|) is an unbiased estimate
of σ_i². Two splits: even vs odd attempt index (rl-env-use's), and ``splits`` seeded random halvings, averaged (lower
variance; the average converges to the per-task sample variance, reported as ``direct``). The task-clustered A/A
difference D = Σ d_i / N (even/odd) and its SE √(Σ d_i²) / N are reported: |D| well beyond 2 SE means the attempts are
not exchangeable (drift across attempt order).

**MDE** = (z_{1−α/2} + z_{power}) · SE, 2.80 · SE at α = 0.05 two-sided and power 0.8. rl-env-use writes
``mde = 1.43 · noise`` with ``noise`` the A/A interval's half-width over √2, i.e. 1.96 · SE_diff (halves have k/2 reps,
so SE_AA = √2 · SE_diff); 1.43 · 1.96 = 2.80 is the same constant. Two differences, both deliberate:

1. **The trained arm's variance.** rl-env-use sets σ̄_t = σ̄_c (A/A). On a pre-selected benchmark most control tasks
   are 0 of k (σ_i = 0), so the A/A noise is tiny while the trained arm, moved toward 0.5, is not. ``variance="bounded"``
   (the gate's default) uses σ̄_t² ≤ μ_t (1 − μ_t), an upper bound for any outcome in [0, 1] with mean μ_t (σ_i² ≤
   μ_i (1 − μ_i), and the mean of μ_i (1 − μ_i) is at most μ̄ (1 − μ̄) by concavity), at μ_t = μ_c + target (or
   ``trained_mean``). ``"aa"`` is rl-env-use's assumption, reported beside it.
2. **No 1/N floor.** rl-env-use floors the noise at one test task flipping. The bounded trained variance answers the
   same failure (zero measured noise) from the outcome's range; instead the gate refuses fewer than ``min_tasks`` = 10
   tasks (rl-env-use's own floor for a case bootstrap, ``climb.py`` 169; ``bench.metrics`` calls a task-mean interval
   approximate below 10 tasks).

``power_gate`` stamps a comparison ``powered`` or ``underpowered`` (MDE above the targeted effect, or too few tasks); it
never blocks a run. ``plan`` gives the N (per k) for MDE ≤ target. ``p_better`` of a paired task-clustered bootstrap is
in ``bench.lh_metrics.paired``. Pure Python, seeded.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Iterable, Mapping, Sequence
from statistics import NormalDist
from typing import Any

ALPHA = 0.05
POWER = 0.8
TARGET = 0.20  # the user's goal (2026-10-20): the student under 30% before RL, 50-60% after: a +0.20 effect at least
MIN_TASKS = 10
METRICS = ("pass", "reward", "phi")


def z_sum(alpha: float = ALPHA, power: float = POWER) -> float:
    """z_{1−α/2} + z_{power} (2.8016 at the defaults)."""
    nd = NormalDist()
    return nd.inv_cdf(1 - alpha / 2) + nd.inv_cdf(power)


def _value(row: Any, metric: str) -> float | None:
    get = (lambda k: row.get(k)) if isinstance(row, Mapping) else (lambda k: getattr(row, k, None))
    if get("masked"):
        return None
    if metric == "pass":
        return 1.0 if get("passed") else 0.0
    if metric == "reward":
        v = get("reward")
    elif metric == "phi":
        v = get("final_phi")
    else:
        raise ValueError(f"metric is one of {METRICS}")
    return None if v is None else float(v)


def by_task(rows: Iterable[Any], metric: str = "pass") -> dict[str, list[float]]:
    """task -> its unmasked attempts' values in attempt order (rows: ``episodes.jsonl`` mappings or ``LHEpisode``s; an
    attempt with no value for the metric, e.g. no measured Φ, is left out)."""
    out: dict[str, list[float]] = {}
    for r in rows:
        v = _value(r, metric)
        if v is not None:
            t = str(r.get("task") if isinstance(r, Mapping) else r.task)
            out.setdefault(t, []).append(v)
    return out


def _half_var(vals: Sequence[float], a: Sequence[int]) -> float:
    """d² / (1/|A| + 1/|B|) for the halving ``a`` (indices of A): an unbiased estimate of the task's per-attempt variance."""
    sa = set(a)
    xa = [v for i, v in enumerate(vals) if i in sa]
    xb = [v for i, v in enumerate(vals) if i not in sa]
    d = statistics.fmean(xa) - statistics.fmean(xb)
    return d * d / (1 / len(xa) + 1 / len(xb))


def aa_noise(values: Mapping[str, Sequence[float]], *, metric: str = "pass", splits: int = 200, seed: int = 0) -> dict[str, Any]:
    """The A/A noise of one arm (module doc): per-task within variance by even/odd halves and by ``splits`` seeded
    random halvings, the pooled σ̄² (the random-split average), the even/odd A/A difference with its SE, the arm's
    task-clustered mean μ_c. Tasks with fewer than 2 attempts are left out (no split exists)."""
    rng = random.Random(seed)
    per: dict[str, dict[str, Any]] = {}
    d_eo: list[float] = []
    for t in sorted(values):
        vals = [float(v) for v in values[t]]
        n = len(vals)
        if n < 2:
            continue
        eo = list(range(0, n, 2))
        d_eo.append(statistics.fmean(vals[0::2]) - statistics.fmean(vals[1::2]))
        rnd = [_half_var(vals, rng.sample(range(n), n // 2)) for _ in range(max(1, splits))]
        per[t] = {"n": n, "mean": statistics.fmean(vals), "var_even_odd": _half_var(vals, eo), "var_random": statistics.fmean(rnd),
                  "var_direct": statistics.variance(vals)}
    nt = len(per)
    if nt == 0:
        return {"metric": metric, "n_tasks": 0, "k": None, "sigma2": None, "control_mean": None, "per_task": {}}
    s2 = {m: statistics.fmean(p[f"var_{m}"] for p in per.values()) for m in ("even_odd", "random", "direct")}
    D = statistics.fmean(d_eo)
    se_aa = math.sqrt(sum(d * d for d in d_eo)) / nt
    ks = [p["n"] for p in per.values()]
    return {"metric": metric, "n_tasks": nt, "k": int(statistics.median(ks)), "k_range": [min(ks), max(ks)], "sigma2": s2["random"],
            "sigma2_by_split": s2, "splits": splits, "seed": seed, "control_mean": statistics.fmean(p["mean"] for p in per.values()),
            "aa": {"diff": D, "se": se_aa, "lo": D - 1.959964 * se_aa, "hi": D + 1.959964 * se_aa, "split": "even_odd"},
            "per_task": per}


def trained_var(control_mean: float, *, target: float = TARGET, trained_mean: float | None = None) -> float:
    """The bound μ_t (1 − μ_t) on the trained arm's mean within-task variance, μ_t = ``trained_mean`` or μ_c + target."""
    mu = min(1.0, max(0.0, trained_mean if trained_mean is not None else control_mean + target))
    return mu * (1 - mu)


def se_diff(sigma2_c: float, sigma2_t: float, n: int, k: int, *, tau: float = 0.0) -> float:
    """SE of the paired trained − control task-clustered mean difference: √((σ̄_c² + σ̄_t²)/(k·N) + τ²/N)."""
    if n < 1 or k < 1:
        return math.inf
    return math.sqrt((sigma2_c + sigma2_t) / (k * n) + tau * tau / n)


def mde(se: float, *, alpha: float = ALPHA, power: float = POWER) -> float:
    return z_sum(alpha, power) * se


def n_needed(target: float, sigma2_c: float, sigma2_t: float, k: int, *, tau: float = 0.0, alpha: float = ALPHA, power: float = POWER) -> int:
    """The smallest N with MDE ≤ ``target`` at k attempts per arm (no floor; ``plan`` applies ``min_tasks``)."""
    if target <= 0:
        raise ValueError("target must be > 0")
    z = z_sum(alpha, power)
    return max(1, math.ceil(z * z * ((sigma2_c + sigma2_t) / k + tau * tau) / (target * target) - 1e-9))


def _variances(noise: Mapping[str, Any], variance: str, target: float, trained_mean: float | None) -> tuple[float, float]:
    s2c = float(noise["sigma2"])
    if variance == "aa":
        return s2c, s2c
    if variance == "bounded":
        return s2c, trained_var(float(noise["control_mean"]), target=target, trained_mean=trained_mean)
    raise ValueError("variance is 'bounded' or 'aa'")


def power_gate(noise: Mapping[str, Any], n: int, k: int, *, target: float = TARGET, variance: str = "bounded", tau: float = 0.0,
               trained_mean: float | None = None, alpha: float = ALPHA, power: float = POWER, min_tasks: int = MIN_TASKS,
               source: str = "") -> dict[str, Any]:
    """The pre-registration stamp of a comparison of ``n`` tasks × ``k`` attempts per arm: its MDE (``variance``
    ``bounded``, the default, and ``aa`` beside it) against the targeted effect. ``verdict`` is ``powered`` when MDE ≤
    ``target`` and n ≥ ``min_tasks``, else ``underpowered`` (``why`` says which); ``unmeasured`` without a noise estimate.
    It never blocks a run: the report carries it and the headline says so."""
    base = {"metric": noise.get("metric"), "n_tasks": n, "k": k, "target": target, "alpha": alpha, "power": power, "variance": variance, "tau": tau,
            "min_tasks": min_tasks, "source": source, "noise_tasks": noise.get("n_tasks"), "noise_k": noise.get("k")}
    if not noise.get("n_tasks") or noise.get("sigma2") is None:
        return {**base, "verdict": "unmeasured", "mde": None, "mde_aa": None, "why": "no A/A noise estimate (an arm with >= 2 attempts per task)",
                "headline": "POWER UNMEASURED"}
    s2c, s2t = _variances(noise, variance, target, trained_mean)
    se = se_diff(s2c, s2t, n, k, tau=tau)
    m = mde(se, alpha=alpha, power=power)
    m_aa = mde(se_diff(float(noise["sigma2"]), float(noise["sigma2"]), n, k, tau=tau), alpha=alpha, power=power)
    why = []
    if m > target + 1e-12:
        why.append(f"MDE {m:.3f} > target {target:.3f}")
    if n < min_tasks:
        why.append(f"{n} tasks < {min_tasks}")
    v = "underpowered" if why else "powered"
    head = (f"{v.upper()}: {n} tasks x k={k}, MDE {m:.3f} ({noise.get('metric')}, alpha {alpha}, power {power}) vs target {target:.2f}"
            + (f" ({'; '.join(why)})" if why else ""))
    return {**base, "verdict": v, "mde": m, "mde_aa": m_aa, "se": se, "sigma2_control": s2c, "sigma2_trained": s2t,
            "control_mean": noise.get("control_mean"), "why": "; ".join(why), "headline": head,
            "n_needed": max(min_tasks, n_needed(target, s2c, s2t, k, tau=tau, alpha=alpha, power=power))}


def plan(noise: Mapping[str, Any], *, targets: Sequence[float] = (0.20, 0.15), ks: Sequence[int] = (4, 8, 16), variance: str = "bounded",
         taus: Sequence[float] = (0.0,), trained_mean: float | None = None, alpha: float = ALPHA, power: float = POWER,
         min_tasks: int = MIN_TASKS) -> dict[str, Any]:
    """Tasks needed for MDE ≤ each target, per k and τ: ``raw`` (the formula) and ``n`` (at least ``min_tasks``)."""
    out: dict[str, Any] = {"metric": noise.get("metric"), "variance": variance, "alpha": alpha, "power": power, "min_tasks": min_tasks, "rows": []}
    if noise.get("sigma2") is None:
        return out
    for tgt in targets:
        s2c, s2t = _variances(noise, variance, tgt, trained_mean)
        for k in ks:
            for tau in taus:
                raw = n_needed(tgt, s2c, s2t, k, tau=tau, alpha=alpha, power=power)
                out["rows"].append({"target": tgt, "k": k, "tau": tau, "raw": raw, "n": max(min_tasks, raw), "sigma2_control": s2c,
                                    "sigma2_trained": s2t})
    return out


def study(datasets: Mapping[str, Sequence[Any]], *, n: int = 10, k: int = 8, targets: Sequence[float] = (0.20, 0.15), ks: Sequence[int] = (4, 8, 16),
          taus: Sequence[float] = (0.0, 0.1, 0.2), trained_means: Sequence[float | None] = (None, 0.55), metrics: Sequence[str] = ("pass", "reward"),
          splits: int = 200, seed: int = 0) -> dict[str, Any]:
    """The offline power study ($0): per dataset (label -> episode rows of one arm) and pooled (tasks prefixed by their
    label, so a task measured in two runs counts as two units), per metric: the A/A noise, the gate for ``n`` × ``k``
    under each variance model and trained mean (None: μ_c + target; 0.55: the goal's 50-60% midpoint), and the planner."""
    pooled = [{**(r if isinstance(r, Mapping) else vars(r)), "task": f"{lab}/{r.get('task') if isinstance(r, Mapping) else r.task}"}
              for lab, rows in datasets.items() for r in rows]
    out: dict[str, Any] = {"n": n, "k": k, "targets": list(targets), "alpha": ALPHA, "power": POWER, "z_sum": z_sum(), "min_tasks": MIN_TASKS, "sets": {}}
    for lab, rows in [*datasets.items(), ("pooled", pooled)]:
        per: dict[str, Any] = {}
        for m in metrics:
            nz = aa_noise(by_task(rows, m), metric=m, splits=splits, seed=seed)
            gates = {}
            for tgt in targets:
                for var in ("bounded", "aa"):
                    for tm in (trained_means if var == "bounded" else (None,)):
                        for tau in taus:
                            key = f"target={tgt:.2f}|{var}" + (f"|mu_t={tm}" if tm is not None else "") + f"|tau={tau}"
                            g = power_gate(nz, n, k, target=tgt, variance=var, tau=tau, trained_mean=tm)
                            gates[key] = {x: g.get(x) for x in ("verdict", "mde", "mde_aa", "se", "sigma2_control", "sigma2_trained", "n_needed", "why")}
            plans = {f"{var}" + (f"|mu_t={tm}" if tm is not None else ""): plan(nz, targets=targets, ks=ks, variance=var, taus=taus, trained_mean=tm)["rows"]
                     for var in ("bounded", "aa") for tm in (trained_means if var == "bounded" else (None,))}
            per[m] = {"noise": {x: v for x, v in nz.items() if x != "per_task"},
                      "per_task": {t: {"n": p["n"], "mean": round(p["mean"], 4), "var": round(p["var_random"], 4)} for t, p in nz.get("per_task", {}).items()},
                      "gates": gates, "plan": plans}
        out["sets"][lab] = per
    return out


def power_report(rows: Sequence[Any], *, n: int | None = None, k: int | None = None, target: float = TARGET, metrics: Sequence[str] = ("pass", "reward"),
                 variance: str = "bounded", splits: int = 200, seed: int = 0, source: str = "") -> dict[str, Any]:
    """Per metric: the A/A noise of these attempts and the gate for a benchmark of ``n`` tasks × ``k`` attempts (default:
    this arm's own N and k). Compact: per-task detail is left out."""
    out: dict[str, Any] = {}
    for m in metrics:
        nz = aa_noise(by_task(rows, m), metric=m, splits=splits, seed=seed)
        gate = power_gate(nz, n if n is not None else int(nz["n_tasks"] or 0), k if k is not None else int(nz["k"] or 0), target=target,
                          variance=variance, source=source)
        out[m] = {"noise": {x: v for x, v in nz.items() if x != "per_task"}, "gate": gate}
    return out
