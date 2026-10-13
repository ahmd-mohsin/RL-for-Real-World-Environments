"""2PL item response theory for benchmark anchoring (DESIGN §11.4). Pure Python (stdlib only), deterministic.

P(respondent r solves item i) = σ(a_i (θ_r − b_i)). A respondent is a model × harness (× checkpoint); data are binomial
counts (s, n) per (respondent, item), so the ladder's and learnability's repeated samples are used as they are, and a
fractional s (partial credit) is accepted.

- **Joint MAP** (M13 audit V5/V24). One objective, the log posterior
  ``Σ [s log p + (n−s) log(1−p)] − Σ (θ−μθ)²/2σθ² − Σ (b−μb)²/2σb² − Σ (log a−μa)²/2σa²``,
  maximized jointly by Levenberg–Marquardt: the negative Hessian is shifted by λI (so an indefinite (b, log a) block
  never takes a wild step), a step is accepted only if the objective increases, and convergence is a gradient-norm
  test (``Fit.converged``, ``Fit.iterations``, ``Fit.grad_norm``). The linear system is solved by eliminating the
  per-item 1×1 / 2×2 blocks (a Schur complement over respondents: bundle-adjustment structure), so a fit costs
  O(Σ_items respondents²) per iteration. The previous alternating scheme fell into a Newton/gradient 2-cycle and returned
  non-stationary items (28 nats off the MAP; b off by 1.0); the post-hoc standardization of θ was not the MAP either.
- **Priors.** Unanchored: θ ~ N(0, 1) (it fixes the scale), b ~ N(0, 2²), log a ~ N(0, 0.5²). With anchors (G2-2): the
  scale is the anchors', so θ gets a weak N(0, 3²) prior (a N(0, 1) prior shrank stronger new models toward the
  calibration cohort: −0.56 logits at θ = 2), and fresh items' b and log a priors are centred on the anchors' mean b
  and mean log a.
- **Standard errors** are marginal: the inverse of the joint observed information at the MAP (θ SEs include the
  fresh items' uncertainty; item SEs include θ's). The old θ SE treated items as known and under-covered (G2-2).
- **Fixed-anchor calibration** (the linking method): anchors keep their (a, b); fresh items and abilities are estimated,
  so fresh items land on the anchors' scale (NeMo-Gym gdpval's anchored Bradley–Terry, ``comparison.py:2046-2119``,
  item-level). With 2PL anchors the fit is 2PL whatever the respondent count (V14: a forced a = 1 put fresh items on
  another metric); the log-a prior centred on the anchors keeps a finite with few respondents.
- **Rasch** (a = 1) is chosen automatically below ``min_2pl`` respondents when there are no 2PL anchors.
- **Ability** of a new respondent against fixed items: bisection on the monotone MAP gradient (as gdpval does). Its SE
  adds the items' calibration uncertainty by the delta method when the items carry ``se_b`` (G2-2).
- **Drift** (V6, G2-3): refit with each anchor's slope fixed at its calibrated value (so the refit is on the anchors'
  unit) and its b free; link by a robust location constant (median, then leave-one-out precision-weighted means over
  the unflagged anchors, iterated: purification); test each anchor two-sided with the variance of the *linked contrast*
  from the joint covariance plus every anchor's own calibration SE; control the family-wise error over the anchors by
  Holm. Anchors are quarantined, never rotated, so a false positive is permanent: the old test (mean shift, refit SE
  only, no multiplicity) quarantined 17–89% of clean anchors.
- **Adaptive testing**: the next item maximizes Fisher information a² p(1−p) at the current θ.
- **Item QA**: discrimination below 0.2 flags a broken item (the IRT form of the ladder's inversion check).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Counts = Mapping[tuple[str, str], tuple[float, int]]  # (respondent, item) -> (successes, trials)

SD_THETA, SD_B, SD_LOGA = 1.0, 2.0, 0.5
SD_THETA_ANCHORED = 3.0  # G2-2: on a fixed anchor scale the θ prior is weak, so stronger models are not shrunk to 0


def _sig(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _logsig(z: float) -> float:
    return -math.log1p(math.exp(-z)) if z >= 0 else z - math.log1p(math.exp(z))


@dataclass
class Item:
    b: float
    a: float = 1.0
    se_b: float | None = None
    se_a: float | None = None


@dataclass
class Fit:
    model: str
    theta: dict[str, float]
    theta_se: dict[str, float]
    items: dict[str, Item]
    anchors: list[str] = field(default_factory=list)
    loglik: float = 0.0
    sweeps: int = 0  # = iterations (kept for readers of the old field)
    converged: bool = False
    iterations: int = 0
    grad_norm: float = float("nan")
    logpost: float = float("nan")
    priors: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        def r(x: float | None) -> float | None:
            return None if x is None else round(x, 6)

        return {"model": self.model, "theta": {k: round(v, 6) for k, v in sorted(self.theta.items())},
                "theta_se": {k: round(v, 6) for k, v in sorted(self.theta_se.items())},
                "items": {k: {"a": round(v.a, 6), "b": round(v.b, 6), "se_b": r(v.se_b), "se_a": r(v.se_a)} for k, v in sorted(self.items.items())},
                "anchors": sorted(self.anchors), "loglik": round(self.loglik, 6), "logpost": r(self.logpost) if math.isfinite(self.logpost) else None,
                "sweeps": self.sweeps,
                "iterations": self.iterations, "converged": self.converged,
                "grad_norm": float(f"{self.grad_norm:.3g}") if math.isfinite(self.grad_norm) else None,
                "priors": {k: round(v, 6) for k, v in sorted(self.priors.items())}}


# ---- input validation (V24) ----------------------------------------------------------------------------------------------
def _clean_counts(counts: Counts) -> dict[tuple[str, str], tuple[float, float]]:
    out = {}
    for key, v in counts.items():
        try:
            s, n = float(v[0]), float(v[1])
        except (TypeError, ValueError, IndexError) as e:
            raise ValueError(f"counts[{key!r}] = {v!r} is not (successes, trials)") from e
        if not (math.isfinite(s) and math.isfinite(n)) or n < 0 or s < 0 or s > n:
            raise ValueError(f"counts[{key!r}] = {v!r}: need finite 0 <= successes <= trials")
        if n > 0:
            out[(str(key[0]), str(key[1]))] = (s, n)
    return out


def _check_item(name: str, it: Item) -> None:
    if not (math.isfinite(it.b) and math.isfinite(it.a) and it.a > 0):
        raise ValueError(f"item {name!r} has a non-finite b or a non-positive / non-finite a: {it}")


# ---- the joint MAP solver --------------------------------------------------------------------------------------------------
@dataclass
class _Spec:
    """One free item: ``mode`` "ab" (b and log a free) or "b" (a fixed at ``a``); priors on b and log a."""

    mode: str
    b: float
    la: float
    mb: float
    sb: float
    mla: float = 0.0
    sla: float = SD_LOGA


def _chol(m: list[list[float]]) -> list[list[float]] | None:
    n = len(m)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Li = L[i]
        for j in range(i + 1):
            Lj = L[j]
            s = m[i][j] - sum(Li[k] * Lj[k] for k in range(j))
            if i == j:
                if not s > 1e-300:
                    return None
                Li[i] = math.sqrt(s)
            else:
                Li[j] = s / Lj[j]
    return L


def _chol_solve(L: list[list[float]], b: list[float]) -> list[float]:
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
    return x


def _chol_inv(L: list[list[float]]) -> list[list[float]]:
    n = len(L)
    cols = [_chol_solve(L, [1.0 if k == j else 0.0 for k in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def _inv_small(d: list[list[float]]) -> list[list[float]] | None:
    if len(d) == 1:
        return [[1.0 / d[0][0]]] if d[0][0] > 1e-300 else None
    a, b, c = d[0][0], d[0][1], d[1][1]
    det = a * c - b * b
    if not (a > 1e-300 and det > 1e-300 * max(1.0, a * c)):
        return None
    return [[c / det, -b / det], [-b / det, a / det]]


class _Problem:
    def __init__(self, counts: dict[tuple[str, str], tuple[float, float]], resp: list[str], fixed: Mapping[str, Item], free: Mapping[str, _Spec],
                 mt: float, st: float, hyper: tuple[float, float] | None = None):
        """``hyper=(μ0, σμ)``: the free items' b-prior mean is itself a parameter μ ~ N(μ0, σμ²), the last entry of the θ
        vector (a fixed prior mean is shared information the posterior would count once per item: G2-2)."""
        self.hyper = hyper
        self.resp = resp
        self.ri = {r: k for k, r in enumerate(resp)}
        self.fixed = dict(fixed)
        self.free = dict(free)
        self.names = sorted(self.free)
        self.mt, self.st = mt, st
        self.by_item: dict[str, list[tuple[int, float, float]]] = {}
        for (r, i), (s, n) in sorted(counts.items()):
            if i in self.fixed or i in self.free:
                self.by_item.setdefault(i, []).append((self.ri[r], s, n))
        self.fixed_rows = [(i, rows) for i, rows in sorted(self.by_item.items()) if i in self.fixed]

    # parameters: theta list, per free item [b] or [b, la]
    def logpost(self, th: list[float], par: Mapping[str, list[float]]) -> tuple[float, float]:
        ll = 0.0
        for i, rows in self.by_item.items():
            if i in self.fixed:
                b, a = self.fixed[i].b, self.fixed[i].a
            else:
                p = par[i]
                b, a = p[0], math.exp(p[1])
            for r, s, n in rows:
                z = a * (th[r] - b)
                ll += s * _logsig(z) + (n - s) * _logsig(-z)
        R = len(self.resp)
        lp = -sum((t - self.mt) ** 2 for t in th[:R]) / (2 * self.st**2)
        if self.hyper:
            lp -= (th[R] - self.hyper[0]) ** 2 / (2 * self.hyper[1] ** 2)
        for i, sp in self.free.items():
            p = par[i]
            lp -= (p[0] - (th[R] if self.hyper else sp.mb)) ** 2 / (2 * sp.sb**2)
            if sp.mode == "ab":
                lp -= (p[1] - sp.mla) ** 2 / (2 * sp.sla**2)
        return ll + lp, ll

    def derivs(self, th: list[float], par: Mapping[str, list[float]]):
        R = len(self.resp)
        gt = [-(t - self.mt) / self.st**2 for t in th[:R]]
        At = [1.0 / self.st**2] * R
        if self.hyper:
            gt.append(-(th[R] - self.hyper[0]) / self.hyper[1] ** 2)
            At.append(1.0 / self.hyper[1] ** 2)
        items = {}
        for i in self.names:
            sp, p = self.free[i], par[i]
            b, la = p[0], p[1]
            a = math.exp(la)
            k = 2 if sp.mode == "ab" else 1
            mb = th[R] if self.hyper else sp.mb
            gb, gl = -(b - mb) / sp.sb**2, -(la - sp.mla) / sp.sla**2
            nbb, nll, nbl = 1.0 / sp.sb**2, 1.0 / sp.sla**2, 0.0
            cross = []
            if self.hyper:
                gt[R] += (b - mb) / sp.sb**2
                At[R] += 1.0 / sp.sb**2
                cross.append((R, (-1.0 / sp.sb**2, 0.0) if k == 2 else (-1.0 / sp.sb**2,)))
            for r, s, n in self.by_item.get(i, ()):
                z = a * (th[r] - b)
                q = _sig(z)
                res, w = s - n * q, n * q * (1 - q)
                gt[r] += a * res
                At[r] += w * a * a
                gb -= a * res
                nbb += w * a * a
                if k == 2:
                    gl += res * z
                    nll += w * z * z - res * z
                    nbl += a * res - a * w * z
                    cross.append((r, (-w * a * a, a * w * z - a * res)))
                else:
                    cross.append((r, (-w * a * a,)))
            g = [gb, gl] if k == 2 else [gb]
            D = [[nbb, nbl], [nbl, nll]] if k == 2 else [[nbb]]
            items[i] = (g, D, cross)
        for _i, rows in self.fixed_rows:
            it = self.fixed[_i]
            for r, s, n in rows:
                z = it.a * (th[r] - it.b)
                q = _sig(z)
                gt[r] += it.a * (s - n * q)
                At[r] += n * q * (1 - q) * it.a * it.a
        return gt, At, items

    def solve(self, gt, At, items, lam: float):
        """Solve (−H + λI) δ = g through the Schur complement over respondents; None if not positive definite."""
        R = len(gt)
        S = [[0.0] * R for _ in range(R)]
        for r in range(R):
            S[r][r] = At[r] + lam
        rhs = list(gt)
        dinv = {}
        for i, (g, D, cross) in items.items():
            k = len(g)
            Dl = [[D[x][y] + (lam if x == y else 0.0) for y in range(k)] for x in range(k)]
            Di = _inv_small(Dl)
            if Di is None:
                return None
            dinv[i] = Di
            us = [(r, [sum(c[x] * Di[x][y] for x in range(k)) for y in range(k)], c) for r, c in cross]  # u_r = C_r D⁻¹
            for r, u, _ in us:
                rhs[r] -= sum(u[y] * g[y] for y in range(k))
                Sr = S[r]
                for r2, _, c2 in us:
                    Sr[r2] -= sum(u[y] * c2[y] for y in range(k))
        L = _chol(S)
        if L is None:
            return None
        dt = _chol_solve(L, rhs)
        di = {}
        for i, (g, _D, cross) in items.items():
            k = len(g)
            v = list(g)
            for r, c in cross:
                for y in range(k):
                    v[y] -= c[y] * dt[r]
            Di = dinv[i]
            di[i] = [sum(Di[x][y] * v[y] for y in range(k)) for x in range(k)]
        return dt, di, L, dinv

    def run(self, th: list[float], par: dict[str, list[float]], *, max_iter: int, gtol: float):
        lam = 1e-3
        f, _ = self.logpost(th, par)
        it = 0
        gnorm = float("inf")
        for it in range(1, max_iter + 1):
            gt, At, items = self.derivs(th, par)
            gnorm = max([abs(x) for x in gt] + [abs(x) for g, _, _ in items.values() for x in g] + [0.0])
            if gnorm < gtol:
                return th, par, True, it - 1, gnorm
            accepted = False
            for _ in range(60):
                sol = self.solve(gt, At, items, lam)
                if sol is not None:
                    dt, di, _, _ = sol
                    th2 = [t + d for t, d in zip(th, dt)]
                    par2 = {i: [p + d for p, d in zip(par[i], di[i])] + par[i][len(di[i]):] for i in par}
                    f2, _ = self.logpost(th2, par2)
                    if f2 >= f - 1e-12 * max(1.0, abs(f)) and math.isfinite(f2):
                        th, par, f = th2, par2, f2
                        lam = max(lam / 3.0, 1e-12)
                        accepted = True
                        break
                lam = max(lam * 4.0, 1e-6)
            if not accepted:
                break
        gt, At, items = self.derivs(th, par)
        gnorm = max([abs(x) for x in gt] + [abs(x) for g, _, _ in items.values() for x in g] + [0.0])
        return th, par, gnorm < gtol, it, gnorm

    def covariance(self, th, par):
        """Marginal covariances at the point (λ = 0): θ variances, each free item's k×k block, and the solver pieces for
        cross-item covariances. None when the negative Hessian is not positive definite there."""
        gt, At, items = self.derivs(th, par)
        sol = self.solve(gt, At, items, 0.0)
        if sol is None:
            return None
        _, _, L, dinv = sol
        Sinv = _chol_inv(L)
        W = {}
        for i, (g, _D, cross) in items.items():
            k = len(g)
            Di = dinv[i]
            W[i] = [(r, [sum(Di[x][y] * c[y] for y in range(k)) for x in range(k)]) for r, c in cross]  # column r of D⁻¹Cᵀ

        def cov(i: str, j: str) -> list[list[float]]:
            ki, kj = len(items[i][0]), len(items[j][0])
            out = [[(dinv[i][x][y] if i == j else 0.0) for y in range(kj)] for x in range(ki)]
            for r, wi in W[i]:
                Sr = Sinv[r]
                for r2, wj in W[j]:
                    s = Sr[r2]
                    for x in range(ki):
                        for y in range(kj):
                            out[x][y] += wi[x] * s * wj[y]
            return out

        return [Sinv[r][r] for r in range(len(self.resp))], cov


def _map(counts: dict[tuple[str, str], tuple[float, float]], resp: list[str], fixed: Mapping[str, Item], free: Mapping[str, _Spec], *,
         mt: float, st: float, max_iter: int, gtol: float, hyper: tuple[float, float] | None = None):
    pb = _Problem(counts, resp, fixed, free, mt, st, hyper)
    th = [mt] * len(resp) + ([hyper[0]] if hyper else [])
    par = {i: [sp.b, sp.la] for i, sp in free.items()}
    th, par, ok, iters, gnorm = pb.run(th, par, max_iter=max_iter, gtol=gtol)
    return pb, th, par, ok, iters, gnorm


def fit(counts: Counts, *, anchors: Mapping[str, Item] | None = None, model: str = "auto", min_2pl: int = 15, sweeps: int = 500,
        tol: float = 1e-7) -> Fit:
    """The joint MAP. ``sweeps`` caps the solver's iterations; ``tol`` is the gradient-norm convergence test."""
    if model not in ("auto", "1pl", "2pl"):
        raise ValueError(f"model {model!r} is not auto | 1pl | 2pl")
    anchors = dict(anchors or {})
    for i, it in anchors.items():
        _check_item(i, it)
    c = _clean_counts(counts)
    resp = sorted({r for r, _ in counts})
    items = sorted({i for _, i in counts} | set(anchors))
    anchor_2pl = any(abs(a.a - 1.0) > 1e-9 for a in anchors.values())
    if model == "auto":
        two = anchor_2pl if anchors else len(resp) >= min_2pl
    else:
        two = model == "2pl"
    if anchors:
        mb = sum(a.b for a in anchors.values()) / len(anchors)
        mla = sum(math.log(a.a) for a in anchors.values()) / len(anchors)
        mt, st = 0.0, SD_THETA_ANCHORED
    else:
        mb, mla, mt, st = 0.0, 0.0, 0.0, SD_THETA
    # V14: a 1PL fit against 2PL anchors fixes fresh slopes at the anchors' geometric mean, never at a raw 1.0
    la_fixed = mla if (anchors and not two) else 0.0
    free = {i: _Spec("ab" if two else "b", mb, la_fixed if not two else mla, mb, SD_B, mla, SD_LOGA) for i in items if i not in anchors}
    hyper = (mb, SD_B) if anchors and free else None  # G2-2: the fresh items' mean difficulty is estimated, not assumed
    pb, th, par, ok, iters, gnorm = _map(c, resp, anchors, free, mt=mt, st=st, max_iter=sweeps, gtol=tol, hyper=hyper)
    cv = pb.covariance(th, par)
    theta = {r: th[k] for k, r in enumerate(resp)}
    if cv is not None:
        tvar, cov = cv
        theta_se = {r: math.sqrt(max(tvar[k], 0.0)) for k, r in enumerate(resp)}
    else:  # not positive definite at the point (unconverged): conditional SEs, flagged by converged=False
        cov = None
        _, At, _ = pb.derivs(th, par)
        theta_se = {r: 1.0 / math.sqrt(At[k]) for k, r in enumerate(resp)}
    out: dict[str, Item] = {}
    for i in items:
        if i in anchors:
            a = anchors[i]
            out[i] = Item(a.b, a.a, a.se_b, a.se_a)
            continue
        p = par[i]
        se_b = se_a = None
        if cov is not None:
            m = cov(i, i)
            se_b = math.sqrt(max(m[0][0], 0.0))
            if len(m) == 2:
                se_a = math.exp(p[1]) * math.sqrt(max(m[1][1], 0.0))
        out[i] = Item(p[0], math.exp(p[1]), se_b, se_a)
    f, ll = pb.logpost(th, par)
    priors = {"theta_mean": mt, "theta_sd": st, "b_mean": mb, "b_sd": SD_B, "loga_mean": mla, "loga_sd": SD_LOGA}
    if hyper:
        priors.update({"b_mean_hyper_mean": hyper[0], "b_mean_hyper_sd": hyper[1], "b_mean_fitted": th[len(resp)]})
    return Fit("2pl" if two else "1pl", theta, theta_se, out, sorted(anchors), ll, iters, ok and cv is not None, iters, gnorm, f, priors)


def ability(items: Mapping[str, Item], responses: Mapping[str, tuple[float, int]], *, lo: float = -12.0, hi: float = 12.0,
            prior_sd: float = SD_THETA_ANCHORED) -> tuple[float, float]:
    """θ and SE of one respondent against fixed (calibrated) items, by bisection on the monotone MAP gradient. The prior
    is N(0, ``prior_sd``²) on the items' scale (weak by default: G2-2). The SE is the posterior SE given the items plus,
    by the delta method, the items' calibration uncertainty (``se_b``), which a known-items SE leaves out (G2-2)."""
    for i, it in items.items():
        _check_item(i, it)
    rows = []
    for i, v in responses.items():
        if i not in items:
            continue
        s, n = float(v[0]), float(v[1])
        if not (math.isfinite(s) and math.isfinite(n)) or n < 0 or s < 0 or s > n:
            raise ValueError(f"responses[{i!r}] = {v!r}: need finite 0 <= successes <= trials")
        rows.append((items[i], s, n))

    def grad(t: float) -> float:
        return -t / prior_sd**2 + sum(it.a * (s - n * _sig(it.a * (t - it.b))) for it, s, n in rows)

    for _ in range(200):
        mid = (lo + hi) / 2
        if grad(mid) > 0:
            lo = mid
        else:
            hi = mid
    t = (lo + hi) / 2
    info = 1.0 / prior_sd**2 + sum(n * it.a**2 * _sig(it.a * (t - it.b)) * (1 - _sig(it.a * (t - it.b))) for it, s, n in rows)
    var = 1.0 / info
    # dθ/db_i = (n a² p(1-p)) / info (implicit function theorem on the gradient)
    var += sum((n * it.a**2 * _sig(it.a * (t - it.b)) * (1 - _sig(it.a * (t - it.b))) / info) ** 2 * it.se_b**2 for it, s, n in rows if it.se_b)
    return round(t, 6), round(math.sqrt(var), 6)


def information(it: Item, theta: float) -> float:
    p = _sig(it.a * (theta - it.b))
    return it.a * it.a * p * (1 - p)


def next_item(items: Mapping[str, Item], theta: float, *, exclude: Iterable[str] = ()) -> str | None:
    if not math.isfinite(theta):
        raise ValueError(f"theta {theta!r} is not finite")
    ex = set(exclude)
    pool = [(information(it, theta), i) for i, it in sorted(items.items()) if i not in ex]
    return max(pool)[1] if pool else None


# ---- drift (V6, G2-3) ------------------------------------------------------------------------------------------------------
def _p_two_sided(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def _adjust(p: Mapping[str, float], alpha: float, method: str = "holm") -> tuple[set[str], dict[str, float]]:
    """Multiplicity-adjusted p-values: Holm (family-wise error, the default: a quarantine is permanent), Benjamini–Hochberg
    (false discovery rate) or none."""
    order = sorted(p, key=lambda k: (p[k], k))
    m = len(order)
    adj: dict[str, float] = {}
    if method == "holm":
        run = 0.0
        for rank, k in enumerate(order):
            run = max(run, min(1.0, (m - rank) * p[k]))
            adj[k] = run
    elif method == "bh":
        run = 1.0
        for rank in reversed(range(m)):
            k = order[rank]
            run = min(run, m * p[k] / (rank + 1))
            adj[k] = min(1.0, run)
    elif method == "none":
        adj = dict(p)
    else:
        raise ValueError(f"method {method!r} is not holm | bh | none")
    return {k for k in order if adj[k] <= alpha}, adj


def drift_report(anchors: Mapping[str, Item], counts: Counts, *, alpha: float = 0.05, method: str = "holm", max_rounds: int = 20,
                 sweeps: int = 500) -> dict[str, Any]:
    """Every anchor's linked refit, contrast SE, z, p and Holm-adjusted p; the flagged set; the link constant; the
    refit's convergence. See the module docstring for the method."""
    for i, it in anchors.items():
        _check_item(i, it)
    c = _clean_counts(counts)
    resp = sorted({r for r, _ in counts})
    present = sorted(i for i in anchors if any(k[1] == i for k in c))
    if not present:
        return {"flagged": {}, "anchors": {}, "link": 0.0, "converged": True}
    others = sorted({i for _, i in c} - set(anchors))
    two = any(abs(a.a - 1.0) > 1e-9 for a in anchors.values())
    mb = sum(a.b for a in anchors.values()) / len(anchors)
    mla = sum(math.log(a.a) for a in anchors.values()) / len(anchors)
    free = {i: _Spec("b", anchors[i].b, math.log(anchors[i].a), mb, SD_B * 2.5, math.log(anchors[i].a), SD_LOGA) for i in present}
    free.update({i: _Spec("ab" if (two and len(resp) >= 15) else "b", mb, mla, mb, SD_B, mla, SD_LOGA) for i in others})
    pb, th, par, ok, iters, gnorm = _map(c, resp, {}, free, mt=0.0, st=SD_THETA_ANCHORED, max_iter=sweeps, gtol=1e-7)
    cv = pb.covariance(th, par)
    if cv is None:
        raise ArithmeticError("drift refit: the observed information is not positive definite at the MAP")
    _, cov = cv
    bnew = {i: par[i][0] for i in present}
    C = {(i, j): cov(i, j)[0][0] for i in present for j in present}
    se_anc = {i: anchors[i].se_b or 0.0 for i in present}
    d = {i: anchors[i].b - bnew[i] for i in present}  # each anchor's own estimate of the link constant

    def table(link_set: set[str], robust_start: bool) -> dict[str, dict[str, float]]:
        out = {}
        med = sorted(d[k] for k in present)
        med_v = (med[(len(med) - 1) // 2] + med[len(med) // 2]) / 2
        for j in present:
            ref = sorted(link_set - {j}) or sorted(set(present) - {j})
            if not ref:
                continue
            # leave-one-out precision weights (diagonal approximation); the variance below is the exact linear form
            w = {k: 1.0 / max(1e-12, C[(k, k)] + se_anc[k] ** 2) for k in ref}
            tw = sum(w.values())
            cw = {k: w[k] / tw for k in ref}
            link = med_v if robust_start else sum(cw[k] * d[k] for k in ref)
            # contrast: b_j + Σ c_k (b_anc_k − b_k) − b_anc_j; variance from the joint covariance + calibration SEs
            var = C[(j, j)] - 2 * sum(cw[k] * C[(j, k)] for k in ref) + sum(cw[k] * cw[l] * C[(k, l)] for k in ref for l in ref)
            var += se_anc[j] ** 2 + sum(cw[k] ** 2 * se_anc[k] ** 2 for k in ref)
            se = math.sqrt(max(var, 1e-12))
            linked = bnew[j] + link
            z = (linked - anchors[j].b) / se
            out[j] = {"anchor_b": anchors[j].b, "refit_b": linked, "se": se, "z": z, "p": _p_two_sided(z), "link": link}
        return out

    tab = table(set(present), True)
    flagged, adj = _adjust({k: v["p"] for k, v in tab.items()}, alpha, method)
    for _ in range(max_rounds):
        tab = table(set(present) - flagged, False)
        new, adj = _adjust({k: v["p"] for k, v in tab.items()}, alpha, method)
        if new == flagged:
            break
        flagged = new
    for k in tab:
        tab[k]["p_adj"] = adj[k]
    rnd = {k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in sorted(tab.items())}
    link_set = sorted(set(present) - flagged)
    return {"flagged": {k: rnd[k] for k in sorted(flagged)}, "anchors": rnd, "linking_set": link_set,
            "link": round(sum(d[k] for k in link_set) / len(link_set), 6) if link_set else None,
            "alpha": alpha, "method": method, "converged": ok, "iterations": iters, "grad_norm": gnorm}


def drift(anchors: Mapping[str, Item], counts: Counts, *, alpha: float = 0.05, method: str = "holm") -> dict[str, dict[str, float]]:
    """The anchors whose linked difficulty moved, at family-wise error ``alpha`` over the anchors (Holm; ``method="bh"``
    controls the false discovery rate instead, with more power). Each entry has anchor_b, refit_b (on the anchor scale),
    se (of the linked contrast), z, p and p_adj."""
    return drift_report(anchors, counts, alpha=alpha, method=method)["flagged"]


def item_qa(f: Fit, *, min_a: float = 0.2) -> list[str]:
    return sorted(i for i, it in f.items.items() if f.model == "2pl" and it.a < min_a)


def save_anchors(path: Path, items: Mapping[str, Item], *, meta: Mapping[str, Any] | None = None) -> Path:
    """Persist a bench version's anchor items (V12: so the next version's fresh items and abilities link to them)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"scheme": 1, "meta": dict(meta or {}),
           "items": {k: {"a": v.a, "b": v.b, "se_b": v.se_b, "se_a": v.se_a} for k, v in sorted(items.items())}}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)
    return path


def load_anchors(path: Path) -> dict[str, Item]:
    doc = json.loads(Path(path).read_text())
    out = {k: Item(b=float(v["b"]), a=float(v.get("a", 1.0)), se_b=v.get("se_b"), se_a=v.get("se_a")) for k, v in doc["items"].items()}
    for k, v in out.items():
        _check_item(k, v)
    return out
