"""The predictive-validity gate (DESIGN §6.5): a rubric item or milestone earns reward only if it predicts recorded
outcomes on held-out, *later* traces.

- ``auc``: Mann-Whitney AUC with average ranks for ties (0.5 = chance; a constant score gives 0.5).
- ``time_split``: the holdout is the latest fraction of traces by time, never a random sample (an item tuned on
  the future would look predictive for the wrong reason).
- ``validate``: keep an item iff its holdout AUC >= ``min_auc`` *and* its direction agrees on every subset with
  enough data of both classes (e.g. each repo); weight = max(0, 2·(AUC − 0.5)).
- ``bootstrap_ci`` / ``paired_bootstrap``: uncertainty on an AUC and on the difference of two AUCs on the same
  traces (used to claim Φ beats an LLM-judge baseline, M6 gate S2), resampled by *group* (e.g. task instance) so
  several attempts at one task do not count as independent evidence.

Pure Python and deterministic (seeded), so gate reports reproduce exactly.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any


def auc(scores: Sequence[float], labels: Sequence[int | bool]) -> float | None:
    """P(score of a positive > score of a negative), ties counted half. None without both classes."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels differ in length")
    pos = sum(1 for y in labels if y)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        r = (i + j) / 2 + 1  # average rank of the tie block (1-based)
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    rank_sum_pos = sum(ranks[i] for i, y in enumerate(labels) if y)
    return (rank_sum_pos - pos * (pos + 1) / 2) / (pos * neg)


def _groups(n: int, groups: Sequence[Hashable] | None) -> list[list[int]]:
    if groups is None:
        return [[i] for i in range(n)]
    by: dict[Hashable, list[int]] = {}
    for i, g in enumerate(groups):
        by.setdefault(g, []).append(i)
    return list(by.values())


def bootstrap_ci(scores: Sequence[float], labels: Sequence[int], *, groups: Sequence[Hashable] | None = None, n: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float | None, float | None]:
    """Percentile CI of the AUC under a (group) bootstrap."""
    rng = random.Random(seed)
    gs = _groups(len(scores), groups)
    vals = []
    for _ in range(n):
        idx = [i for g in (rng.choice(gs) for _ in gs) for i in g]
        a = auc([scores[i] for i in idx], [labels[i] for i in idx])
        if a is not None:
            vals.append(a)
    if not vals:
        return None, None
    vals.sort()
    return vals[int(alpha / 2 * len(vals))], vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]


def paired_bootstrap(a: Sequence[float], b: Sequence[float], labels: Sequence[int], *, groups: Sequence[Hashable] | None = None, n: int = 2000,
                     seed: int = 0) -> dict[str, float | None]:
    """AUC(a) − AUC(b) on the same traces: point estimate, 95% CI and the share of resamples where a ≤ b (a one-sided
    p-value for "a beats b")."""
    rng = random.Random(seed)
    gs = _groups(len(labels), groups)
    base_a, base_b = auc(a, labels), auc(b, labels)
    diffs = []
    for _ in range(n):
        idx = [i for g in (rng.choice(gs) for _ in gs) for i in g]
        ya = [labels[i] for i in idx]
        x, y = auc([a[i] for i in idx], ya), auc([b[i] for i in idx], ya)
        if x is not None and y is not None:
            diffs.append(x - y)
    if not diffs or base_a is None or base_b is None:
        return {"diff": None, "lo": None, "hi": None, "p_not_better": None}
    diffs.sort()
    return {"diff": base_a - base_b, "lo": diffs[int(0.025 * len(diffs))], "hi": diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))],
            "p_not_better": sum(1 for d in diffs if d <= 0) / len(diffs)}


def time_split(items: Sequence[Any], time: Callable[[Any], str | float], holdout: float = 0.3) -> tuple[list[Any], list[Any]]:
    """(train, holdout): the latest ``holdout`` fraction by time is held out; ties at the cut stay together."""
    if not 0 < holdout < 1:
        raise ValueError("holdout is a fraction in (0, 1)")
    ordered = sorted(items, key=time)
    if not ordered:
        return [], []
    cut = round(len(ordered) * (1 - holdout))
    cut = min(max(cut, 1), len(ordered) - 1) if len(ordered) > 1 else len(ordered)
    t_cut = time(ordered[cut]) if cut < len(ordered) else None
    while cut > 0 and t_cut is not None and time(ordered[cut - 1]) == t_cut:
        cut -= 1
    return ordered[:cut], ordered[cut:]


@dataclass
class Validity:
    item: str
    auc: float | None
    n: int
    positives: int
    ci: tuple[float | None, float | None]
    subsets: dict[str, float | None] = field(default_factory=dict)
    kept: bool = False
    weight: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"item": self.item, "auc": None if self.auc is None else round(self.auc, 4), "n": self.n, "positives": self.positives,
                "ci": [None if c is None else round(c, 4) for c in self.ci], "subsets": {k: None if v is None else round(v, 4) for k, v in self.subsets.items()},
                "kept": self.kept, "weight": round(self.weight, 4), "reason": self.reason}


def validate(item: str, scores: Sequence[float], labels: Sequence[int], *, subsets: Sequence[Hashable] | None = None,
             groups: Sequence[Hashable] | None = None, min_auc: float = 0.6, min_subset: int = 4) -> Validity:
    """The §6.5 keep-or-drop rule on held-out data. ``subsets`` (e.g. repo of each trace): the item's direction must
    agree on every subset with at least ``min_subset`` traces of each class; ``groups`` drives the bootstrap."""
    a = auc(scores, labels)
    v = Validity(item=item, auc=a, n=len(labels), positives=sum(1 for y in labels if y), ci=bootstrap_ci(scores, labels, groups=groups, n=500))
    if a is None:
        v.reason = "holdout lacks one class"
        return v
    if subsets is not None:
        for s in sorted(set(subsets), key=str):
            idx = [i for i, x in enumerate(subsets) if x == s]
            ys = [labels[i] for i in idx]
            if sum(ys) >= min_subset and len(ys) - sum(ys) >= min_subset:
                v.subsets[str(s)] = auc([scores[i] for i in idx], ys)
    flipped = [k for k, x in v.subsets.items() if x is not None and (x - 0.5) * (a - 0.5) < 0]
    if a < min_auc:
        v.reason = f"holdout AUC {a:.3f} < {min_auc}"
    elif flipped:
        v.reason = f"direction flips on subsets {flipped}"
    else:
        v.kept, v.weight, v.reason = True, max(0.0, 2 * (a - 0.5)), "predictive on later traces"
    return v
