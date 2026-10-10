"""Calibrating checks from data (DESIGN §8.2, A1, A4, P15).

- ``flake_rate``: false negatives of the oracle over fresh-seed and induced-perturbation draws, with an exact
  (Clopper-Pearson) upper confidence bound; A4 requires 0 failures, and the bound goes on the task card;
- ``tolerance_tiers``: strict and relaxed tolerances from the natural spread (reference vs oracle over >= 40 draws) and
  induced bifurcations (semantics-preserving perturbations), each with a margin; the relaxed tier must still kill
  every negative (Appendix B.2: KinFu-LS shift threshold 0.79 vs 0.8);
- ``bifurcations``: split one-dimensional deviations into clusters at the largest gap (is the spread bimodal?);
- ``perf_headroom``: the oracle's margin under each perf bar; A1 requires >= 15%, G-tighten requires >= 10%;
- ``self_consistency``: identical reference fingerprints across repeated runs (Appendix B.2, the Haar truncation).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass


def _binom_cdf(x: int, n: int, p: float) -> float:
    return sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(x + 1))


def clopper_pearson_upper(failures: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1 - alpha) upper bound on a binomial rate."""
    if n <= 0:
        return 1.0
    if failures >= n:
        return 1.0
    if failures == 0:
        return 1.0 - alpha ** (1.0 / n)
    lo, hi = failures / n, 1.0
    for _ in range(80):  # P(X <= failures; p) is decreasing in p
        mid = (lo + hi) / 2
        if _binom_cdf(failures, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


@dataclass
class FlakeEstimate:
    draws: int
    false_negatives: int
    induced: int
    rate: float
    upper_95: float
    ok: bool  # A4: zero false negatives

    def to_dict(self) -> dict:
        return asdict(self)


def flake_rate(natural: Sequence[bool], induced: Sequence[bool] = ()) -> FlakeEstimate:
    """``natural``/``induced``: did the oracle pass each draw?"""
    draws = list(natural) + list(induced)
    fn = sum(1 for x in draws if not x)
    n = len(draws)
    return FlakeEstimate(n, fn, len(induced), fn / n if n else 1.0, round(clopper_pearson_upper(fn, n), 6), n > 0 and fn == 0)


def bifurcations(values: Sequence[float], *, min_gap_ratio: float = 4.0) -> list[list[float]]:
    """Split sorted values at the largest gap if it is ``min_gap_ratio`` times the median gap (else one cluster)."""
    v = sorted(values)
    if len(v) < 3:
        return [v]
    gaps = [b - a for a, b in itertools.pairwise(v)]
    k = max(range(len(gaps)), key=gaps.__getitem__)
    med = sorted(gaps)[len(gaps) // 2] or 1e-12
    return [v[: k + 1], v[k + 1:]] if gaps[k] / med >= min_gap_ratio else [v]


@dataclass
class TierCalibration:
    strict: float
    relaxed: float
    natural_max: float
    induced_max: float | None
    negatives_min: float | None
    bimodal: bool
    kills_negatives: bool
    ok: bool
    draws: int

    def to_dict(self) -> dict:
        return asdict(self)


def tolerance_tiers(natural: Sequence[float], induced: Sequence[float] = (), negatives: Sequence[float] = (), *, margin: float = 1.5,
                    min_draws: int = 40) -> TierCalibration:
    """Deviations are absolute errors of a correct solution vs the reference (natural / induced) and of each negative."""
    nat = max(natural) if natural else 0.0
    ind = max(induced) if induced else None
    strict = nat * margin
    relaxed = max(strict, (ind or 0.0) * margin)
    neg = min(negatives) if negatives else None
    kills = neg is None or relaxed < neg
    return TierCalibration(strict, relaxed, nat, ind, neg, len(bifurcations(list(natural) + list(induced))) > 1, kills,
                           kills and len(natural) + len(induced) >= min_draws, len(natural) + len(induced))


def perf_headroom(oracle_ratios: dict[str, float], bars: dict[str, float], *, required: float = 0.15) -> dict:
    """Headroom = 1 - ratio / bar (the oracle's margin under each bar)."""
    rows = {m: round(1 - oracle_ratios[m] / bars[m], 4) for m in bars if m in oracle_ratios}
    missing = sorted(set(bars) - set(oracle_ratios))
    worst = min(rows.values()) if rows else None
    return {"headroom": rows, "min": worst, "required": required, "missing": missing, "ok": not missing and worst is not None and worst >= required}


def self_consistency(fingerprints: Sequence[dict]) -> dict:
    """Reference runs must agree exactly (>= 3 runs, A4)."""
    if len(fingerprints) < 3:
        return {"ok": False, "reason": f"{len(fingerprints)} reference runs, need >= 3"}
    keys = set().union(*fingerprints)
    differing = sorted(k for k in keys if len({repr(f.get(k)) for f in fingerprints}) > 1)
    return {"ok": not differing, "runs": len(fingerprints), "differing": differing[:20]}
