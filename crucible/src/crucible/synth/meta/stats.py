"""The statistics of the regularised loop (part2-paper-core §9.4 R1/R2; ``runs/rsi-stats.md``): paired A/A noise floors,
anytime-valid paired tests by betting, error control over a whole loop run, a reusable holdout, and power.

**What a decision tests.** An update u: Θ → Θ′ is evaluated on n decision units (planted tasks). Unit i gives the paired
difference D_i = Y_i(Θ′, s′) − Y_i(Θ, s) ∈ [−1, 1], s and s′ two independent evaluator seeds. The null of test t is

    H_t:  μ_t = E[D] ≤ δ_AA      (the update is not better than an unchanged Θ is against itself)

with δ_AA the paired noise floor from A/A replicates (``aa_floor``), measured before the loop and only raised after.

**The test** (``betting_test``) is the one-sided test by betting of Waudby-Smith & Ramdas (JRSS-B 2024, "Estimating means
of bounded random variables by betting"): on X = (D + 1)/2 ∈ [0, 1] and m = (δ_AA + 1)/2 the wealth
K_n = Π_{i≤n} (1 + λ_i (X_i − m)), with λ_i ∈ [0, c/m] chosen from X_1..X_{i−1} only (aGRAPA), is a non-negative
supermartingale under H_t when the units are i.i.d. (or exchangeable draws without replacement from a population whose
mean is ≤ m: the plug-in only ever bets on the past). Ville's inequality gives P_H(sup_n K_n ≥ 1/α_t) ≤ α_t at every
stopping time, so the decision may stop as soon as the wealth crosses (anytime-valid) or run to the end of its view.

**Error control over a run** (``AlphaSpender``). Test t gets α_t, fixed before its data are drawn, with Σ_t α_t ≤ α. Then

    P(some test of the run rejects a true null) ≤ Σ_t P(test t rejects | H_t true) ≤ Σ_t α_t ≤ α

by the union bound and the tower property, under ANY dependence between tests and any adaptive choice of candidates
(the candidate of test t may depend on everything before it). This is online Bonferroni (alpha-spending for the
family-wise error rate). Alpha-investing and online e-BH control the false discovery rate instead, which is the wrong
guarantee for "no non-improving update is ever accepted", so they are not offered.

**Conditional validity needs fresh units.** The union bound needs each test to be valid given the past. That holds when
the decision units of test t were never shown to the proposer or the birth gate and never used by an earlier decision
(R1-strict: ``rsi.Sampler(strict=True)`` draws decision units without replacement from a reserved decision split).

**The reusable holdout** (``Thresholdout``, Dwork et al., Science 2015): a decision that must read held-out data asks a
query; the answer is the training value unless the holdout disagrees by more than a noisy threshold, in which case a
noisy holdout value is returned and the budget drops by one. Every access is appended to a log.

**Power** (``mcnemar_power``, ``plants_needed``, ``mde``): exact power of the one-sided conditional McNemar test for paired
recall on the same plants, by enumeration over the discordant counts.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel

# ---------------------------------------------------------------------------------------------------------- basics


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def var(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def paired(a: Sequence[float], b: Sequence[float]) -> list[float]:
    if len(a) != len(b):
        raise ValueError(f"paired differences need equal lengths ({len(a)} vs {len(b)})")
    return [x - y for x, y in zip(a, b, strict=True)]


# ---------------------------------------------------------------------------------------------------------- betting


class BetResult(CrucibleModel):
    reject: bool
    alpha: float
    m0: float  # the null bound on E[D]
    n: int
    wealth: float  # K_n at the end of the sequence
    max_wealth: float  # sup_{k≤n} K_k (the e-process crossed 1/α when this is ≥ 1/α)
    stop: int | None = None  # the first k with K_k ≥ 1/α (an anytime-valid test may stop there)


def _bets(xs: Sequence[float], m: float, c: float) -> list[float]:
    """The aGRAPA bets λ_i on [0, 1]-valued data against the null mean m (predictable: λ_i uses x_1..x_{i−1})."""
    lam_max = c / m if m > 0 else 1e9
    out: list[float] = []
    s, s2, k = 0.5, 0.25, 1  # a half-observation prior at 1/2 (mean 1/2, variance 1/4): no bet is forced by the first unit
    for x in xs:
        mu = s / k
        v = max(s2 / k - mu * mu, 1e-6)
        lam = (mu - m) / (v + (mu - m) ** 2) if mu > m else 0.0
        out.append(min(max(lam, 0.0), lam_max))
        s, s2, k = s + x, s2 + x * x, k + 1
    return out


def betting_test(diffs: Sequence[float], m0: float, alpha: float, *, c: float = 0.5, lo: float = -1.0, hi: float = 1.0) -> BetResult:
    """Test H: E[D] ≤ m0 for D ∈ [lo, hi] by betting (module docstring). ``alpha`` = 0 never rejects (no budget)."""
    n = len(diffs)
    if alpha <= 0 or n == 0:
        return BetResult(reject=False, alpha=alpha, m0=m0, n=n, wealth=1.0, max_wealth=1.0)
    span = hi - lo
    xs = [(d - lo) / span for d in diffs]
    m = (m0 - lo) / span
    if m >= 1.0:  # the null covers every distribution on the range: never rejected
        return BetResult(reject=False, alpha=alpha, m0=m0, n=n, wealth=1.0, max_wealth=1.0)
    if m <= 0.0:  # the null mean is at or below the floor of the range: any unit above it is evidence; bet the house
        m = 1e-9
    lams = _bets(xs, m, c)
    w, top, stop = 1.0, 1.0, None
    for i, (x, lam) in enumerate(zip(xs, lams, strict=True), 1):
        w *= 1.0 + lam * (x - m)
        top = max(top, w)
        if stop is None and w >= 1.0 / alpha:
            stop = i
    return BetResult(reject=top >= 1.0 / alpha, alpha=alpha, m0=m0, n=n, wealth=w, max_wealth=top, stop=stop)


def betting_lcb(diffs: Sequence[float], alpha: float, *, c: float = 0.5, lo: float = -1.0, hi: float = 1.0, tol: float = 1e-4) -> float:
    """The anytime-valid lower confidence bound on E[D] at level α: the largest m0 the betting test still rejects
    (bisection; the rejection region is an interval [lo, LCB) for these predictable bets). −inf without data or budget."""
    if not diffs or alpha <= 0 or not betting_test(diffs, lo + 1e-9, alpha, c=c, lo=lo, hi=hi).reject:
        return -math.inf
    a, b = lo + 1e-9, hi
    while b - a > tol:
        mid = (a + b) / 2
        if betting_test(diffs, mid, alpha, c=c, lo=lo, hi=hi).reject:
            a = mid
        else:
            b = mid
    return a


# ---------------------------------------------------------------------------------------------------------- bootstrap


def bootstrap_ci(diffs: Sequence[float], *, level: float = 0.95, draws: int = 4000, seed: int = 0, clusters: Sequence[str] | None = None) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean paired difference; ``clusters`` resamples whole clusters (e.g. all the
    plants of one source task), the sensitivity check for correlated units."""
    if not diffs:
        return (-math.inf, math.inf)
    rng = random.Random(f"boot/{seed}")
    if clusters is None:
        groups = [[d] for d in diffs]
    else:
        by: dict[str, list[float]] = {}
        for d, c in zip(diffs, clusters, strict=True):
            by.setdefault(c, []).append(d)
        groups = list(by.values())
    k = len(groups)
    stats = []
    for _ in range(draws):
        tot, cnt = 0.0, 0
        for _ in range(k):
            g = groups[rng.randrange(k)]
            tot += sum(g)
            cnt += len(g)
        stats.append(tot / cnt)
    stats.sort()
    a = (1 - level) / 2
    return stats[int(a * (draws - 1))], stats[int((1 - a) * (draws - 1))]


class Effect(CrucibleModel):
    """A paired comparison's effect size with its intervals (reported with every evaluated decision)."""

    n: int
    mean: float
    gains: int  # units with D > 0
    losses: int  # units with D < 0
    ci95: tuple[float, float]  # paired percentile bootstrap, two-sided
    ci95_cluster: tuple[float, float] | None = None  # the same, resampling clusters (source tasks)
    lcb_anytime: float  # the betting LCB at the decision's α_t (what R2 compares with δ_AA)
    alpha_t: float


def effect(diffs: Sequence[float], alpha_t: float, *, clusters: Sequence[str] | None = None, seed: int = 0) -> Effect:
    return Effect(n=len(diffs), mean=mean(diffs), gains=sum(1 for d in diffs if d > 0), losses=sum(1 for d in diffs if d < 0),
                  ci95=bootstrap_ci(diffs, seed=seed), ci95_cluster=bootstrap_ci(diffs, seed=seed, clusters=clusters) if clusters else None,
                  lcb_anytime=betting_lcb(diffs, alpha_t) if alpha_t > 0 else -math.inf, alpha_t=alpha_t)


# ---------------------------------------------------------------------------------------------------------- A/A


class AAFloor(CrucibleModel):
    """The paired noise floor from A/A replicates: the unchanged Θ against itself on ONE view, seeds (a_r, b_r)."""

    delta_aa: float  # max(2·sd, max |d̄_r|) of the replicate mean differences
    replicate_means: list[float]
    sd: float
    max_abs: float
    n_units: int
    discord_rate: float  # share of (unit, replicate) pairs where the two seeds disagree
    seeds: list[tuple[int, int]] = Field(default_factory=list)
    view: str = ""

    @property
    def replicates(self) -> int:
        return len(self.replicate_means)


def aa_floor(replicates: Sequence[tuple[Sequence[float], Sequence[float]]], *, seeds: Sequence[tuple[int, int]] = (), view: str = "", min_replicates: int = 5) -> AAFloor:
    """δ_AA from ≥ ``min_replicates`` A/A pairs (j_units at seed a, at seed b) on one view."""
    if len(replicates) < min_replicates:
        raise ValueError(f"the A/A floor needs at least {min_replicates} replicates (got {len(replicates)})")
    ms, disc, tot = [], 0, 0
    n = len(replicates[0][0])
    for a, b in replicates:
        d = paired(b, a)
        ms.append(mean(d))
        disc += sum(1 for x in d if x != 0)
        tot += len(d)
    sd = math.sqrt(var(ms))
    mx = max(abs(x) for x in ms)
    return AAFloor(delta_aa=max(2 * sd, mx), replicate_means=ms, sd=sd, max_abs=mx, n_units=n, discord_rate=disc / tot if tot else 0.0,
                   seeds=[tuple(s) for s in seeds], view=view)


# ---------------------------------------------------------------------------------------------------------- alpha spending


class Spend(CrucibleModel):
    t: int
    alpha_t: float
    test: str  # what was tested (update id and rule)
    round: int = 0
    rejected: bool | None = None


class AlphaSpender(CrucibleModel):
    """Online Bonferroni over one loop run (module docstring). ``schedule``:
    - ``bonferroni``: α_t = α / t_max for t ≤ t_max, 0 after (the preregistered number of tests);
    - ``series``: α_t = α · 6 / (π² t²) (an open-ended run; Σ = α)."""

    alpha: float = 0.05
    schedule: Literal["bonferroni", "series"] = "bonferroni"
    t_max: int = 40
    spent: list[Spend] = Field(default_factory=list)

    def level(self, t: int) -> float:
        if t < 1:
            raise ValueError("tests are numbered from 1")
        if self.schedule == "bonferroni":
            return self.alpha / self.t_max if t <= self.t_max else 0.0
        return self.alpha * 6.0 / (math.pi**2 * t * t)

    def next(self, test: str, round: int = 0) -> Spend:
        t = len(self.spent) + 1
        s = Spend(t=t, alpha_t=self.level(t), test=test, round=round)
        self.spent.append(s)
        return s

    def record(self, t: int, rejected: bool) -> None:
        self.spent[t - 1].rejected = rejected

    @property
    def total(self) -> float:
        return sum(s.alpha_t for s in self.spent)

    @property
    def exhausted(self) -> bool:
        return self.level(len(self.spent) + 1) <= 0.0


# ---------------------------------------------------------------------------------------------------------- exact tests


def fisher_one_sided(k1: int, n1: int, k2: int, n2: int) -> float:
    """P-value of the one-sided Fisher exact test of p1 > p2 (k successes of n per group): P(X ≥ k1) under the
    hypergeometric with the margins fixed."""
    if n1 <= 0 or n2 <= 0:
        return 1.0
    tot, k = n1 + n2, k1 + k2
    denom = math.comb(tot, k)
    return sum(math.comb(n1, x) * math.comb(n2, k - x) for x in range(k1, min(n1, k) + 1) if 0 <= k - x <= n2) / denom


def binom_sf(k: int, n: int, p: float) -> float:
    """P(Bin(n, p) ≥ k)."""
    if k <= 0:
        return 1.0
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


# ---------------------------------------------------------------------------------------------------------- power


def _mcnemar_crit(nd: int, alpha: float) -> int | None:
    """The smallest b with P(Bin(nd, ½) ≥ b) ≤ α (one-sided exact conditional McNemar), None when no b is enough."""
    tail = 0.0
    for b in range(nd, -1, -1):
        t2 = tail + math.comb(nd, b) / 2**nd
        if t2 > alpha:
            return b + 1 if b + 1 <= nd else None
        tail = t2
    return 0


def mcnemar_power(n: int, p10: float, p01: float = 0.0, alpha: float = 0.05) -> float:
    """Exact power of the one-sided conditional McNemar test on n paired plants: p10 = P(Θ′ catches, Θ misses), p01 =
    P(Θ catches, Θ′ misses) (seed noise or a real loss). The recall gain is p10 − p01."""
    pd = p10 + p01
    if pd <= 0:
        return 0.0
    q = p10 / pd
    power = 0.0
    for nd in range(n + 1):
        w = math.comb(n, nd) * pd**nd * (1 - pd) ** (n - nd)
        if w < 1e-15:
            continue
        b = _mcnemar_crit(nd, alpha)
        if b is not None:
            power += w * binom_sf(b, nd, q)
    return power


def plants_needed(gain: float, *, p01: float = 0.0, alpha: float = 0.05, power: float = 0.8, n_max: int = 2000) -> int | None:
    """The smallest n with power ≥ ``power`` for a recall gain ``gain`` (= p10 − p01) and stays there for 5 more n (the
    exact power saw-tooths)."""
    p10 = gain + p01
    if p10 > 1:
        return None
    for n in range(1, n_max + 1):
        if all(mcnemar_power(m, p10, p01, alpha) >= power for m in range(n, n + 6)):
            return n
    return None


def mde(n: int, *, p01: float = 0.0, alpha: float = 0.05, power: float = 0.8, step: float = 0.005) -> float | None:
    """The minimum detectable recall gain on n paired plants (grid search), None when even a gain of 1 − p01 is not."""
    g = step
    while g + p01 <= 1.0 + 1e-12:
        if mcnemar_power(n, g + p01, p01, alpha) >= power:
            return round(g, 4)
        g += step
    return None


def power_table(counts: dict[str, int], *, gains: Sequence[float] = (0.1, 0.2, 0.3), p01: float = 0.0, alpha: float = 0.05,
                classes_tested: int | None = None) -> list[dict[str, Any]]:
    """Per defect class: n plants, the power at each stated gain, the MDE at power 0.8, and the MDE when the class
    tests are Bonferroni-corrected over ``classes_tested`` classes."""
    rows = []
    k = classes_tested or len(counts)
    for d, n in counts.items():
        row: dict[str, Any] = {"class": d, "n": n, "mde": mde(n, p01=p01, alpha=alpha) if n else None,
                               "mde_bonferroni": mde(n, p01=p01, alpha=alpha / max(1, k)) if n else None}
        for g in gains:
            row[f"power@{g}"] = round(mcnemar_power(n, g + p01, p01, alpha), 3) if n else 0.0
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------------------- reusable holdout


class HoldoutAnswer(CrucibleModel):
    query: str
    answer: float
    over_threshold: bool
    budget_left: int
    refused: bool = False


class Thresholdout:
    """Dwork et al.'s Thresholdout over a held-out set (module docstring). ``holdout_fn(query)`` computes the held-out
    value (only this object ever calls it); ``log`` gets one JSON line per access (the query, the purpose, whether the
    holdout was revealed, the budget left; never the exact held-out value). Budget exhausted: the query is refused and a
    caller must fail closed."""

    def __init__(self, holdout_fn: Callable[[str], float], *, threshold: float = 0.04, sigma: float = 0.01, budget: int = 10, seed: int = 0,
                 log: str | Path | None = None):
        self.holdout_fn = holdout_fn
        self.threshold, self.sigma, self.budget = threshold, sigma, budget
        self.rng = random.Random(f"thresholdout/{seed}")
        self.gamma = self._lap(2 * sigma)
        self.log = Path(log) if log is not None else None
        self.accesses = 0

    def _lap(self, b: float) -> float:
        u = self.rng.random() - 0.5
        return -b * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))

    def _write(self, rec: dict[str, Any]) -> None:
        if self.log is None:
            return
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    def ask(self, query: str, train_value: float, *, purpose: str = "decide") -> HoldoutAnswer:
        self.accesses += 1
        base = {"at": time.time(), "query": query, "purpose": purpose, "access": self.accesses}
        if self.budget <= 0:
            self._write({**base, "refused": True, "budget_left": 0})
            return HoldoutAnswer(query=query, answer=train_value, over_threshold=False, budget_left=0, refused=True)
        h = float(self.holdout_fn(query))
        eta = self._lap(4 * self.sigma)
        if abs(h - train_value) > self.threshold + self.gamma + eta:
            self.budget -= 1
            self.gamma = self._lap(2 * self.sigma)
            ans = HoldoutAnswer(query=query, answer=h + self._lap(self.sigma), over_threshold=True, budget_left=self.budget)
        else:
            ans = HoldoutAnswer(query=query, answer=train_value, over_threshold=False, budget_left=self.budget)
        self._write({**base, "over_threshold": ans.over_threshold, "budget_left": ans.budget_left, "train": round(train_value, 6)})
        return ans
