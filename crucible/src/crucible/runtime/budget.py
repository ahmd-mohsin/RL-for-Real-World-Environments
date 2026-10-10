"""The budget ledger for runs (research m9-m10/02 §3.5; PORT of the Repo2RLEnv ``campaigns/budget.py`` design).

Every metered thing is an operation with a scope (``run``, ``run/policy/<id>``, ``run/task/<id>``, ``sample/<id>``):
reserve before dispatch, settle with the observed cost, or mark **uncertain** when the cost cannot be known exactly
(a provider answered without usage: the estimate is held as uncertain spend instead of aborting the run, B.5). A
reservation that would cross a scope's cap is denied with ``accounted_exhaustion``; the runner turns that into the
run stop rule (dispatch halts, in-flight attempts finish masked, ``resume`` continues after the cap is raised).

SQLite, so a crashed run's reservations survive: on reopen, ``reserved`` operations without a settle become
``uncertain`` (nothing is re-dispatched blindly).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class Exhausted(RuntimeError):
    def __init__(self, scope: str, msg: str):
        super().__init__(msg)
        self.scope = scope


class BudgetLedger:
    def __init__(self, path: Path, caps: dict[str, float], *, uncertain_tolerance: float = 0.1):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.caps = dict(caps)
        self.uncertain_tolerance = uncertain_tolerance
        self.db = sqlite3.connect(self.path)
        self.db.execute("CREATE TABLE IF NOT EXISTS ops (op TEXT PRIMARY KEY, scopes TEXT, reserved REAL, actual REAL, status TEXT, ts REAL)")
        self.db.execute("UPDATE ops SET status = 'uncertain', actual = reserved WHERE status = 'reserved'")  # crashed reservations
        self.db.commit()

    def _spent(self, scope: str) -> float:
        rows = self.db.execute("SELECT scopes, reserved, actual, status FROM ops").fetchall()
        tot = 0.0
        for scopes, reserved, actual, status in rows:
            if scope in scopes.split("|"):
                tot += reserved if status == "reserved" else (actual or 0.0)
        return tot

    def uncertain(self, scope: str = "run") -> float:
        return sum(a or 0.0 for s, a in self.db.execute("SELECT scopes, actual FROM ops WHERE status = 'uncertain'") if scope in s.split("|"))

    def reserve(self, op: str, scopes: list[str], usd: float) -> None:
        for sc in scopes:
            cap = self.caps.get(sc if sc in self.caps else sc.split("/", 1)[0])
            if cap is not None and self._spent(sc) + usd > cap + 1e-12:
                raise Exhausted(sc, f"accounted_exhaustion: {sc} would reach {self._spent(sc) + usd:.4f} > cap {cap:.4f}")
        self.db.execute("INSERT OR REPLACE INTO ops VALUES (?, ?, ?, NULL, 'reserved', ?)", (op, "|".join(scopes), usd, time.time()))
        self.db.commit()

    def settle(self, op: str, actual: float, *, uncertain: float = 0.0) -> None:
        status = "uncertain" if uncertain > 0 else "settled"
        self.db.execute("UPDATE ops SET actual = ?, status = ? WHERE op = ?", (actual + uncertain, status, op))
        self.db.commit()

    def release(self, op: str) -> None:
        """A reservation whose request was never sent (``delivery = not_sent``) costs nothing."""
        self.db.execute("UPDATE ops SET actual = 0, status = 'released' WHERE op = ?", (op,))
        self.db.commit()

    def increase_limit(self, scope: str, usd: float) -> None:
        self.caps[scope] = self.caps.get(scope, 0.0) + usd

    def summary(self) -> dict:
        return {"spent_run": round(self._spent("run"), 6), "uncertain_run": round(self.uncertain("run"), 6), "caps": self.caps}

    def over_uncertain_tolerance(self) -> bool:
        spent = self._spent("run")
        return spent > 0 and self.uncertain("run") / spent > self.uncertain_tolerance
