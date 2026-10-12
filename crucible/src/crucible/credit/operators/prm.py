"""C5, a process reward model: a learned step score trained on labels from C1-C4, used **only** as a shaping baseline,
never as the main reward (DESIGN §6.1; P13: a learned scorer changes the objective and can be gamed).

This is the minimal reference implementation: an L2-regularised logistic regression (pure Python, deterministic)
over per-step features — ΔΦ, Φ, step kind, position in the episode, probe flags — predicting the episode outcome. It
gives (a) a baseline the gate can compare Φ and judges against, and (b) the interface a larger PRM (a fine-tuned
model) must satisfy: ``fit(rows)`` and ``score(features) -> [0, 1]``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

KINDS = ("investigate", "edit", "build", "test", "run", "act", "communicate", "submit", "message", "commit", "review", "ci", "state", "other")


def features(step: Mapping[str, object]) -> list[float]:
    kind = str(step.get("kind", "other"))
    f = [1.0, float(step.get("delta_phi", 0.0)), float(step.get("phi", 0.0)), float(step.get("position", 0.0)), float(bool(step.get("probe")))]
    f += [1.0 if kind == k else 0.0 for k in KINDS]
    return f


@dataclass
class LogisticPRM:
    l2: float = 1e-2
    lr: float = 0.5
    epochs: int = 300
    w: list[float] = field(default_factory=list)

    def fit(self, rows: Sequence[tuple[Mapping[str, object], int]]) -> LogisticPRM:
        xs = [features(s) for s, _ in rows]
        ys = [y for _, y in rows]
        d = len(xs[0]) if xs else 0
        self.w = [0.0] * d
        n = max(len(xs), 1)
        for _ in range(self.epochs):
            grad = [self.l2 * wi for wi in self.w]
            for x, y in zip(xs, ys, strict=True):
                p = self._p(x)
                for j in range(d):
                    grad[j] += (p - y) * x[j] / n
            self.w = [wi - self.lr * g for wi, g in zip(self.w, grad, strict=True)]
        return self

    def _p(self, x: Sequence[float]) -> float:
        z = sum(wi * xi for wi, xi in zip(self.w, x, strict=True))
        return 1 / (1 + math.exp(-max(-40.0, min(40.0, z))))

    def score(self, step: Mapping[str, object]) -> float | None:
        """P(success) for a step, or None: an unfit model or a non-finite feature is no score (M6 audit: an unfit model
        imputed 0.5, and a NaN feature scored 1.0 because ``min(40, nan)`` is 40)."""
        x = features(step)
        if not self.w or any(not math.isfinite(v) for v in x):
            return None
        return self._p(x)
