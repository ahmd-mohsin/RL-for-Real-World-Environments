"""Invariants: properties that must hold over the whole end state (DESIGN §8.1). Standard library only.

An invariant is a query returning **violating** rows (``SELECT ... WHERE <bad>``): the check passes iff every
invariant returns none. This is EnterpriseOps-Gym's ``database_state`` verifier turned around: instead of asserting
the one expected row, assert that no row anywhere breaks the policy, which also catches collateral damage (P9: a
refund on the wrong ticket breaks the invariant even if the right ticket looks fine).

``query`` is any callable ``sql -> rows`` (sync or async): ``sqlite3`` inside a verifier image, or a World
``VerifierView`` on the host (``crucible.verify.state.view_query``).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from crucible.verify.runtime.plan import Verdict


@dataclass
class Invariant:
    id: str
    sql: str  # selects violating rows
    description: str = ""
    provenance: str | None = None  # policy document / signal / incident


async def check_invariants(query: Callable[[str], Any], invariants: list[Invariant], *, max_rows: int = 10) -> Verdict:
    broken = {}
    for inv in invariants:
        rows = query(inv.sql)
        if inspect.isawaitable(rows):
            rows = await rows
        rows = list(rows or [])
        if rows:
            broken[inv.id] = {"rows": [list(r) if isinstance(r, (list, tuple)) else r for r in rows[:max_rows]], "count": len(rows),
                              "description": inv.description}
    return Verdict(not broken, {"broken": broken, "checked": [i.id for i in invariants]})


def sqlite_query(path: str) -> Callable[[str], list]:
    """Read-only sqlite3 access for verifier images (``mode=ro``: a check can never modify the state it judges)."""
    import sqlite3

    def q(sql: str) -> list:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return con.execute(sql).fetchall()
        finally:
            con.close()

    return q
