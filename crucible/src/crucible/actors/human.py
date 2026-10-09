"""Human and peer actors: messages to them leave the World through a transport and the reply comes back.

``HumanActor`` is a real person (expert in the loop: an approver, a reviewer answering live, an
operator). ``PeerActor`` is another agent (a deployed enterprise agent over A2A, a different model) that
the policy collaborates with. Both are recorded in the trace like any actor; their transport state
(e.g. the A2A context id) is part of the WorldRef.
"""

from __future__ import annotations

from typing import Any

from crucible.actors.base import ActorBase
from crucible.core.schema.actions import Action, Message
from crucible.world.world import ActorTurn


class TransportActor(ActorBase):
    kind = "transport"

    def __init__(self, actor_id: str, transport: Any, *, reply_latency: str | float | None = None, prefix_sender: bool = True):
        super().__init__(actor_id, reply_latency=reply_latency)
        self.transport = transport
        self.prefix_sender = prefix_sender

    async def respond(self, turn: ActorTurn) -> list[Action]:
        out: list[Action] = []
        for msg in turn.inbox:
            text = f"[from {msg.frm}] {msg.text}" if self.prefix_sender else msg.text
            try:
                reply = await self.transport.send(text)
            except Exception as e:  # noqa: BLE001 - an unreachable human/peer is infrastructure, not policy
                self.failures.append({"kind": "transport_unreachable", "reason": str(e)[:300], "at": turn.time})
                continue
            self.history.append({"from": msg.frm, "q": msg.text, "a": reply, "at": turn.time})
            out.append(Message(to=msg.frm, text=reply))
        return out

    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), "transport": self.transport.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.transport.load_state_dict(state.get("transport", {}))


class HumanActor(TransportActor):
    kind = "human"


class PeerActor(TransportActor):
    kind = "peer"
