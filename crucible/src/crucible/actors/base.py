"""Actor base: what every non-policy actor shares (Responder protocol of ``crucible.world.world``).

State that must survive snapshot/restore (conversation memory, counters, failures) goes through
``state_dict`` / ``load_state_dict`` so a WorldRef captures what simulated people "remember".
Failures of a simulator (it broke character, cited nothing, its LLM was unavailable) are recorded
as ``failures``; episode runners mask such samples (NeMo-Gym's user_failure rule, DESIGN P16).
"""

from __future__ import annotations

from typing import Any

from crucible.core.schema.actions import Action
from crucible.world.world import ActorTurn


class ActorBase:
    actor_id: str
    reply_latency: str | float | None = None  # virtual time a reply takes ("15m", 600); step-mode clocks only

    def __init__(self, actor_id: str, *, reply_latency: str | float | None = None):
        self.actor_id = actor_id
        self.reply_latency = reply_latency
        self.history: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []
        self.finished = False
        # why this actor ended its part (its Done's reason), set by the World; saved so a restored World knows (M2 R-3)
        self.finished_reason: str | None = None

    async def respond(self, turn: ActorTurn) -> list[Action]:  # pragma: no cover - abstract
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        # the reason only once set: the state of an actor still talking, and so every WorldRef fingerprint, is unchanged
        return {"history": list(self.history), "failures": list(self.failures), "finished": self.finished,
                **({"finished_reason": self.finished_reason} if self.finished_reason else {})}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.history = list(state.get("history", []))
        self.failures = list(state.get("failures", []))
        self.finished = bool(state.get("finished", False))
        self.finished_reason = state.get("finished_reason")

    @property
    def failed(self) -> bool:
        return bool(self.failures)
