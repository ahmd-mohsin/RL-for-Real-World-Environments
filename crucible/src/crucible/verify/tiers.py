"""Fast and full verifier tiers (DESIGN §10.2, A12).

The fast tier (``Plan.run(tier="fast")``: checks marked ``tier="fast"``) grades every training rollout; the full tier
grades acceptance, benchmarks, and a sampled audit of training rollouts. A12 requires the two to agree: Spearman
correlation >= 0.8 between fast and full rewards on calibration rollouts, and the audit sample re-grades a fixed share
of rollouts with the full tier to catch divergence during training.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence


def _ranks(x: Sequence[float]) -> list[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Tie-aware Spearman rank correlation; None when either side is constant."""
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("need two equal-length samples of size >= 2")
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((y - mb) ** 2 for y in rb)
    return None if va == 0 or vb == 0 else cov / (va * vb) ** 0.5


def agreement(fast: Sequence[float], full: Sequence[float], *, required: float = 0.8) -> dict:
    rho = spearman(fast, full)
    false_pass = sum(1 for f, g in zip(fast, full, strict=True) if f > 0 and g == 0)  # the fast tier pays what the full tier rejects
    return {"spearman": None if rho is None else round(rho, 4), "required": required, "n": len(fast), "fast_pass_full_fail": false_pass,
            "ok": rho is not None and rho >= required}


def audit_sample(rollout_id: str, *, rate: float = 0.05, salt: str = "crucible-audit") -> bool:
    """Deterministic: whether this rollout is re-graded with the full tier (the policy cannot know which)."""
    h = int.from_bytes(hashlib.sha256(f"{salt}:{rollout_id}".encode()).digest()[:8], "big")
    return h / 2**64 < rate
