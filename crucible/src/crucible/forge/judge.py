"""The Judge (m13 G-13; audit 06 §4.2; GitSwarm's central *selector* as PATTERN: selecting gained 2.34 points where a
central *synthesizer* lost 3.87).

The Judge ranks candidate variants of the same seed for quality and diversity. It never accepts: only tasks M9 accepted
are ranked, and its output is an order, so no M9 verdict can change (a property the tests check over random pools).

- **Quality:** the L2 rate's closeness to the middle of the train band (most learning signal), plus small terms for
  graded gates and composition depth.
- **Diversity:** maximal marginal relevance against what is already selected (same operator or same seed costs
  ``redundancy``), and the YieldBook's diversity quota is respected: a pick that would break it is skipped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from crucible.forge.learnability import BANDS


@dataclass
class Variant:
    task_id: str
    seed: str
    operator: str
    accepted: bool  # the M9 verdict (read-only here)
    l2: float | None = None
    gates: int = 0
    depth: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


def quality(v: Variant) -> float:
    lo, hi = BANDS["train"].lo or 0.0, BANDS["train"].hi or 1.0
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    band = 0.0 if v.l2 is None else max(0.0, 1.0 - abs(v.l2 - mid) / half)
    return round(band + 0.02 * min(v.gates, 10) + 0.05 * min(v.depth, 4), 6)


def select(variants: Sequence[Variant], k: int, *, redundancy: float = 0.5, book: Any = None) -> list[str]:
    """Up to ``k`` accepted variants, best first, diverse (MMR) and within the book's quota."""
    pool = [v for v in variants if v.accepted]
    chosen: list[Variant] = []
    while pool and len(chosen) < k:
        def score(v: Variant) -> float:
            same = sum(1 for c in chosen if c.operator == v.operator) + sum(1 for c in chosen if c.seed == v.seed)
            return quality(v) - redundancy * same

        pool.sort(key=lambda v: (-score(v), v.task_id))
        pick = None
        for v in pool:
            if book is not None and v.task_id in getattr(book, "rows", {}):
                row = book.rows[v.task_id]
                if row.accepted and book._would_break(row):
                    continue
            pick = v
            break
        if pick is None:
            break
        chosen.append(pick)
        pool.remove(pick)
    return [c.task_id for c in chosen]
