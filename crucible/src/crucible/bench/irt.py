"""2PL item response theory for benchmark anchoring (DESIGN §11.4). Pure Python, deterministic.

P(respondent r solves item i) = σ(a_i (θ_r − b_i)). A respondent is a model × harness (× checkpoint); data are binomial
counts (s, n) per (respondent, item), so the ladder's and learnability's repeated samples are used as they are.

- **MAP with priors**: θ ~ N(0, 1), b ~ N(0, 2²), log a ~ N(0, 0.5²). With the 10–30 respondents a benchmark has, plain
  joint maximum likelihood is biased and ``a`` is weakly identified; the priors keep both finite (an item every
  respondent solves does not run b to −∞).
- **Fit**: alternating damped Newton (θ per respondent, (b, log a) per item); with no anchors the scale is fixed by
  standardizing θ after every sweep.
- **Fixed-anchor calibration** (the linking method): anchors keep their (a, b); only fresh items and abilities are
  estimated, so fresh items land on the anchors' scale. This is NeMo-Gym gdpval's anchored Bradley–Terry idea (reference
  ratings held fixed, ``comparison.py:2046-2119``), item-level.
- **Rasch** (a = 1) is chosen automatically below ``min_2pl`` respondents.
- **Ability** of a new respondent against fixed items: the MAP gradient is monotone in θ, so bisection (as gdpval
  does); SE = 1/√(Σ n a² p(1−p) + 1).
- **Drift**: refit the anchors freely and quarantine any whose |Δb| > 2·SE (drift usually means contamination; anchors
  are quarantined, never silently rotated).
- **Adaptive testing**: the next item maximizes Fisher information a² p(1−p) at the current θ.
- **Item QA**: discrimination below 0.2 flags a broken item (the IRT form of the ladder's inversion check).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

Counts = Mapping[tuple[str, str], tuple[float, int]]  # (respondent, item) -> (successes, trials)

SD_THETA, SD_B, SD_LOGA = 1.0, 2.0, 0.5


def _sig(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class Item:
    b: float
    a: float = 1.0
    se_b: float | None = None


@dataclass
class Fit:
    model: str
    theta: dict[str, float]
    theta_se: dict[str, float]
    items: dict[str, Item]
    anchors: list[str] = field(default_factory=list)
    loglik: float = 0.0
    sweeps: int = 0

    def to_json(self) -> dict[str, Any]:
        return {"model": self.model, "theta": {k: round(v, 6) for k, v in sorted(self.theta.items())},
                "theta_se": {k: round(v, 6) for k, v in sorted(self.theta_se.items())},
                "items": {k: {"a": round(v.a, 6), "b": round(v.b, 6), "se_b": None if v.se_b is None else round(v.se_b, 6)} for k, v in sorted(self.items.items())},
                "anchors": sorted(self.anchors), "loglik": round(self.loglik, 6), "sweeps": self.sweeps}


def _theta_step(theta: float, rows: Iterable[tuple[Item, float, int]], iters: int = 20) -> tuple[float, float]:
    rows = list(rows)
    h = -1.0
    for _ in range(iters):
        g, h = -theta / SD_THETA**2, -1.0 / SD_THETA**2
        for it, s, n in rows:
            p = _sig(it.a * (theta - it.b))
            g += it.a * (s - n * p)
            h -= n * it.a * it.a * p * (1 - p)
        step = g / h
        theta -= max(-1.0, min(1.0, step))
        if abs(step) < 1e-8:
            break
    return theta, 1.0 / math.sqrt(-h)


def _item_step(it: Item, rows: list[tuple[float, float, int]], rasch: bool, iters: int = 20) -> Item:
    b, la = it.b, (0.0 if rasch else math.log(it.a))
    hbb = -1.0
    for _ in range(iters):
        a = math.exp(la)
        gb, gl = -b / SD_B**2, -la / SD_LOGA**2
        hbb, hll, hbl = -1.0 / SD_B**2, -1.0 / SD_LOGA**2, 0.0
        for theta, s, n in rows:
            z = a * (theta - b)
            p = _sig(z)
            r, w = s - n * p, n * p * (1 - p)
            gb += -a * r
            hbb += -w * a * a
            gl += r * z
            hll += -w * z * z + r * z
            hbl += w * z * a - r * a
        if rasch:
            step_b, step_l = gb / hbb, 0.0
        else:
            det = hbb * hll - hbl * hbl
            if hbb < 0 and det > 0:  # negative definite: a Newton step
                step_b = (hll * gb - hbl * gl) / det
                step_l = (hbb * gl - hbl * gb) / det
            else:  # otherwise a damped gradient step
                step_b, step_l = -0.1 * gb, -0.1 * gl
        b -= max(-1.0, min(1.0, step_b))
        la -= max(-0.5, min(0.5, step_l))
        if abs(step_b) < 1e-8 and abs(step_l) < 1e-8:
            break
    return Item(b=b, a=1.0 if rasch else math.exp(la), se_b=1.0 / math.sqrt(-hbb) if hbb < 0 else None)


def fit(counts: Counts, *, anchors: Mapping[str, Item] | None = None, model: str = "auto", min_2pl: int = 15, sweeps: int = 200,
        tol: float = 1e-6) -> Fit:
    anchors = dict(anchors or {})
    resp = sorted({r for r, _ in counts})
    items = sorted({i for _, i in counts} | set(anchors))
    rasch = model == "1pl" or (model == "auto" and len(resp) < min_2pl)
    theta = {r: 0.0 for r in resp}
    se = {r: 1.0 for r in resp}
    par = {i: Item(anchors[i].b, anchors[i].a, anchors[i].se_b) if i in anchors else Item(0.0, 1.0) for i in items}
    by_r: dict[str, list[tuple[str, float, int]]] = {}
    by_i: dict[str, list[tuple[str, float, int]]] = {}
    for (r, i), (s, n) in sorted(counts.items()):
        if n > 0:
            by_r.setdefault(r, []).append((i, float(s), int(n)))
            by_i.setdefault(i, []).append((r, float(s), int(n)))
    done = 0
    for done in range(1, sweeps + 1):
        old = dict(theta), {i: (p.b, p.a) for i, p in par.items()}
        for r in resp:
            theta[r], se[r] = _theta_step(theta[r], ((par[i], s, n) for i, s, n in by_r.get(r, [])))
        for i in items:
            if i not in anchors and by_i.get(i):
                par[i] = _item_step(par[i], [(theta[r], s, n) for r, s, n in by_i[i]], rasch)
        if not anchors and len(resp) > 1:  # identify the scale: θ standardized, items moved with it
            mu = sum(theta.values()) / len(resp)
            sd = math.sqrt(sum((t - mu) ** 2 for t in theta.values()) / len(resp)) or 1.0
            if rasch:
                sd = 1.0
            theta = {r: (t - mu) / sd for r, t in theta.items()}
            par = {i: Item((p.b - mu) / sd, p.a * sd, None if p.se_b is None else p.se_b / sd) for i, p in par.items()}
        delta = max([abs(theta[r] - old[0][r]) for r in resp] + [abs(par[i].b - old[1][i][0]) for i in items] + [0.0])
        if delta < tol:
            break
    ll = sum(s * math.log(max(1e-12, _sig(par[i].a * (theta[r] - par[i].b)))) + (n - s) * math.log(max(1e-12, 1 - _sig(par[i].a * (theta[r] - par[i].b))))
             for (r, i), (s, n) in counts.items() if n > 0)
    return Fit("1pl" if rasch else "2pl", theta, se, par, sorted(anchors), ll, done)


def ability(items: Mapping[str, Item], responses: Mapping[str, tuple[float, int]], *, lo: float = -8.0, hi: float = 8.0) -> tuple[float, float]:
    """θ and SE of one respondent against fixed items, by bisection on the (monotone) MAP gradient."""
    def grad(t: float) -> float:
        return -t / SD_THETA**2 + sum(items[i].a * (s - n * _sig(items[i].a * (t - items[i].b))) for i, (s, n) in responses.items() if i in items)

    for _ in range(100):
        mid = (lo + hi) / 2
        if grad(mid) > 0:
            lo = mid
        else:
            hi = mid
    t = (lo + hi) / 2
    info = 1.0 / SD_THETA**2 + sum(n * items[i].a ** 2 * _sig(items[i].a * (t - items[i].b)) * (1 - _sig(items[i].a * (t - items[i].b)))
                                   for i, (_, n) in responses.items() if i in items)
    return round(t, 6), round(1.0 / math.sqrt(info), 6)


def information(it: Item, theta: float) -> float:
    p = _sig(it.a * (theta - it.b))
    return it.a * it.a * p * (1 - p)


def next_item(items: Mapping[str, Item], theta: float, *, exclude: Iterable[str] = ()) -> str | None:
    ex = set(exclude)
    pool = [(information(it, theta), i) for i, it in sorted(items.items()) if i not in ex]
    return max(pool)[1] if pool else None


def drift(anchors: Mapping[str, Item], counts: Counts, *, z: float = 2.0) -> dict[str, dict[str, float]]:
    """Refit the anchors freely (abilities from the same data) and return those whose b moved by more than z·SE."""
    free = fit(counts, model="2pl" if any(abs(a.a - 1.0) > 1e-9 for a in anchors.values()) else "1pl")
    # put the free fit on the anchors' scale by the anchors' own location (mean b) before comparing
    shift = sum(anchors[i].b - free.items[i].b for i in anchors if i in free.items) / max(1, sum(i in free.items for i in anchors))
    out = {}
    for i, a in sorted(anchors.items()):
        if i not in free.items:
            continue
        nb = free.items[i].b + shift
        se_ = free.items[i].se_b or a.se_b or 1.0
        if abs(nb - a.b) > z * se_:
            out[i] = {"anchor_b": round(a.b, 6), "refit_b": round(nb, 6), "se": round(se_, 6)}
    return out


def item_qa(f: Fit, *, min_a: float = 0.2) -> list[str]:
    return sorted(i for i, it in f.items.items() if f.model == "2pl" and it.a < min_a)
