"""Prediction-powered inference for adjudicated rates (part2-paper-core §5 "adjudicated rates are reported with PPI
intervals"; evidence doc §9.3; Angelopoulos et al. 2023).

A rate the panel decides (the share of ALT failures that were valid solutions, i.e. D6; the share of MUT survivors that
were wrong, i.e. D3) is an average of model judgements over N unlabelled probes. The panel is biased, so that average is
biased. A small labelled set with **executed truth** (the calibration anchors: oracle traces are valid, no-op and
minus-last-write traces are not) measures the bias, and the PPI mean estimator removes it:

    θ̂_PP = mean_N f(x̃) + mean_n (y − f(x))                      (the rectifier is the second term)
    CI    = θ̂_PP ± z · sqrt( Var_N f(x̃) / N + Var_n (y − f(x)) / n )

``f`` is the panel's soft score of "valid" (``vote_score``: each member's confidence as a probability of its verdict,
averaged; a member that failed counts 0.5). ``ppi_py`` is used when it is installed (its ``ppi_mean_ci``); otherwise the
normal-approximation interval above, which is the one ``ppi_py`` computes for the mean. ``naive`` (the uncorrected
mean of f) and ``classical`` (a Wilson interval from the labels alone) are reported beside it.

**Caveat, stated in every report that uses it.** PPI needs the labelled and unlabelled items drawn from one
distribution. The executed-truth anchors are oracle and degenerate traces, not suspects, so the interval is valid only
to the extent that the panel's error rate on anchors matches its error rate on suspects. The anchors are the only labels
with executed truth we have; a human-labelled sample of the queue is the fix.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.meta.recall import Z95, wilson


class PPIEstimate(CrucibleModel):
    estimate: float
    lo: float
    hi: float
    naive: float
    classical: tuple[float, float] | None = None  # Wilson interval of the labels alone
    n_unlabeled: int
    n_labeled: int
    method: str = "ppi-normal"
    note: str = "labels are executed-truth anchors (oracle / no-op / minus-last-write traces), not suspects"
    extra: dict[str, Any] = Field(default_factory=dict)


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _var(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def ppi_mean(f_unlabeled: Sequence[float], f_labeled: Sequence[float], y_labeled: Sequence[float], *, z: float = Z95) -> PPIEstimate:
    """The PPI mean estimate of E[y] over the unlabelled items, with its interval (module docstring)."""
    if len(f_labeled) != len(y_labeled):
        raise ValueError("labelled predictions and labels differ in length")
    if not f_unlabeled:
        raise ValueError("PPI needs at least one unlabelled item")
    n_u, n_l = len(f_unlabeled), len(f_labeled)
    naive = _mean(f_unlabeled)
    if n_l == 0:  # no labels: the uncorrected mean, with only its own sampling spread (flagged)
        hw = z * math.sqrt(_var(f_unlabeled) / n_u)
        return PPIEstimate(estimate=naive, lo=max(0.0, naive - hw), hi=min(1.0, naive + hw), naive=naive, n_unlabeled=n_u, n_labeled=0,
                           method="uncorrected (no labels)")
    rect = [y - f for f, y in zip(f_labeled, y_labeled, strict=True)]
    theta = naive + _mean(rect)
    k = sum(1 for y in y_labeled if y >= 0.5)
    classical = wilson(k, n_l, z)
    try:  # the reference implementation when it is installed (same estimator)
        import numpy as np
        from ppi_py import ppi_mean_ci  # type: ignore[import-not-found]

        alpha = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
        lo, hi = ppi_mean_ci(np.array(y_labeled, float), np.array(f_labeled, float), np.array(f_unlabeled, float), alpha=alpha)
        return PPIEstimate(estimate=min(1.0, max(0.0, theta)), lo=max(0.0, float(lo)), hi=min(1.0, float(hi)), naive=naive, classical=classical, n_unlabeled=n_u,
                           n_labeled=n_l, method="ppi_py.ppi_mean_ci")
    except ImportError:
        pass
    hw = z * math.sqrt(_var(f_unlabeled) / n_u + _var(rect) / n_l)
    # a rate lies in [0, 1]: the rectified point estimate is clipped there (the raw value is kept: it shows the judge's bias)
    return PPIEstimate(estimate=min(1.0, max(0.0, theta)), lo=max(0.0, theta - hw), hi=min(1.0, theta + hw), naive=naive, classical=classical,
                       n_unlabeled=n_u, n_labeled=n_l, extra={"raw": theta})


def vote_score(votes: Iterable[Mapping[str, Any]]) -> float:
    """The panel's soft probability that an attempt is valid: each member's confidence as the probability of its verdict
    (``1 - confidence`` for an "invalid" vote), averaged; a member with no verdict counts 0.5."""
    xs = []
    for v in votes:
        val, c = v.get("valid"), float(v.get("confidence") or 0.0)
        xs.append(0.5 if val is None else (c if val else 1.0 - c))
    return _mean(xs) if xs else 0.5


def adjudicated_rates(panel_rows: Sequence[Mapping[str, Any]], calibration_votes: Sequence[Mapping[str, Any]], *, z: float = Z95) -> dict[str, dict[str, Any]]:
    """Per falsifier class (and ``all``): the PPI estimate of the share of adjudicated probes that were VALID solutions,
    from the panel log (``Panel.log`` rows: ``cls``, ``votes``, ``decision``) and the calibration anchors' votes
    (``CalibrationReport.votes``: ``truth``, ``votes``). Also the raw decision counts."""
    f_lab = [vote_score(a.get("votes") or []) for a in calibration_votes]
    y_lab = [1.0 if a.get("truth") else 0.0 for a in calibration_votes]
    by: dict[str, list[Mapping[str, Any]]] = {}
    for r in panel_rows:
        by.setdefault(str(r.get("cls", "?")), []).append(r)
    by["all"] = list(panel_rows)
    out: dict[str, dict[str, Any]] = {}
    for k, rows in sorted(by.items()):
        if not rows:
            continue
        est = ppi_mean([vote_score(r.get("votes") or []) for r in rows], f_lab, y_lab, z=z)
        out[k] = {**est.model_dump(mode="json"), "decided_valid": sum(1 for r in rows if r.get("decision") == "valid"),
                  "decided_invalid": sum(1 for r in rows if r.get("decision") == "invalid"), "abstained": sum(1 for r in rows if r.get("decision") == "abstain")}
    return out
