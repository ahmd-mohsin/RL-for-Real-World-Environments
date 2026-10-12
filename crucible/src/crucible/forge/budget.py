"""Reserve/settle budget ledger for a forge run (pattern: Repo2RLEnv ``campaigns/budget.py``).

A node reserves its estimate before it runs; afterwards it settles the observed cost. The ledger refuses a reservation
that would push committed + reserved spend over the cap, so a forge run never starts work it cannot pay for. Observed
spend comes from the LLM clients the roles use (``LLMClient.spend.usd``), so a role that under-estimates is still
accounted exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class Ledger:
    cap_usd: float
    observe: Callable[[], float] = lambda: 0.0  # total observed spend (sum over the roles' LLM clients)
    committed: float = 0.0
    reserved: dict[str, float] = field(default_factory=dict)
    entries: list[dict] = field(default_factory=list)
    _from_observed: float = 0.0  # committed spend attributed by observed-total deltas (clamp_to_observed)

    def observed(self) -> float:
        return float(self.observe())

    def reserve(self, who: str, usd: float) -> None:
        if self.exhausted:
            raise BudgetExhausted(f"{who}: the forge cap {self.cap_usd:.4f} USD is spent ({self.committed:.4f} committed)")
        if self.committed + sum(self.reserved.values()) + usd > self.cap_usd + 1e-12:
            raise BudgetExhausted(f"{who}: reserving {usd:.4f} USD exceeds the forge cap {self.cap_usd:.4f} "
                                  f"(committed {self.committed:.4f}, reserved {sum(self.reserved.values()):.4f})")
        self.reserved[who] = self.reserved.get(who, 0.0) + usd

    def settle(self, who: str, actual: float, *, clamp_to_observed: bool = False) -> float:
        """Book ``actual`` and release the reservation. Never raises (M8 audit V4: an over-cap settle raised from inside
        the node and aborted every concurrent plan, losing paid output): once committed passes the cap, ``exhausted``
        is True and every later ``reserve`` is refused, so the run winds down instead of crashing.
        ``clamp_to_observed``: the cost came from a delta of the global observed total (a role with no spend scope), which
        can include concurrent nodes' spend; such costs together never exceed what the clients observed (V3). Returns the booked cost."""
        self.reserved.pop(who, None)
        actual = max(0.0, actual)
        if clamp_to_observed:  # delta-attributed costs together never exceed the clients' observed total
            actual = min(actual, max(0.0, self.observed() - self._from_observed))
            self._from_observed += actual
        self.committed += actual
        self.entries.append({"who": who, "usd": round(actual, 6)})
        return actual

    @property
    def exhausted(self) -> bool:
        return self.committed > self.cap_usd + 1e-9
