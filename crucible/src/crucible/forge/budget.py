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

    def observed(self) -> float:
        return float(self.observe())

    def reserve(self, who: str, usd: float) -> None:
        if self.committed + sum(self.reserved.values()) + usd > self.cap_usd + 1e-12:
            raise BudgetExhausted(f"{who}: reserving {usd:.4f} USD exceeds the forge cap {self.cap_usd:.4f} "
                                  f"(committed {self.committed:.4f}, reserved {sum(self.reserved.values()):.4f})")
        self.reserved[who] = self.reserved.get(who, 0.0) + usd

    def settle(self, who: str, actual: float) -> None:
        self.reserved.pop(who, None)
        self.committed += max(0.0, actual)
        self.entries.append({"who": who, "usd": round(actual, 6)})
        if self.committed > self.cap_usd + 1e-9:
            raise BudgetExhausted(f"forge cap {self.cap_usd:.4f} USD exceeded ({self.committed:.4f} committed)")
