"""Human and peer actors: messages to them leave the World through a transport and the reply comes back.

``HumanActor`` is a real person (expert in the loop: an approver, a reviewer answering live, an
operator). ``PeerActor`` is another agent (a deployed enterprise agent over A2A, a different model) that
the policy collaborates with. Both are recorded in the trace like any actor; their transport state
(e.g. the A2A context id) is part of the WorldRef.
"""

from __future__ import annotations

import asyncio
from typing import Any

from crucible.actors.base import ActorBase
from crucible.core.schema.actions import Action, Message
from crucible.world.world import ActorTurn

RESTORED = "Our conversation so far (continue from here; anything you remember beyond this did not happen):"


class TransportActor(ActorBase):
    """M2 audit (2026-10-12): a human who never answers held the World's step lock for the transport's whole timeout and,
    cut off by the episode timeout, left no failure, so the policy was scored on it. ``answer_timeout_s`` bounds the
    wait; a timeout or a cancelled episode records a masking failure (and the transport withdraws the question). A
    reply longer than ``max_reply_chars`` is cut (tool output is capped; external text was not). A transport with
    remote memory (A2A) gets a fresh context after a restore, opened with the restored conversation."""

    kind = "transport"

    def __init__(self, actor_id: str, transport: Any, *, reply_latency: str | float | None = None, prefix_sender: bool = True,
                 answer_timeout_s: float | None = None, max_reply_chars: int = 8000):
        super().__init__(actor_id, reply_latency=reply_latency)
        self.transport = transport
        self.prefix_sender = prefix_sender
        self.answer_timeout_s, self.max_reply_chars = answer_timeout_s, max_reply_chars
        self._replay = False

    async def respond(self, turn: ActorTurn) -> list[Action]:
        out: list[Action] = []
        for msg in turn.inbox:
            text = f"[from {msg.frm}] {msg.text}" if self.prefix_sender else msg.text
            if self._replay and self.history:
                text = RESTORED + "\n" + "\n".join(f"[from {h['from']}] {h['q']}\n[you] {h['a']}" for h in self.history) + "\n\n" + text
            self._replay = False
            try:
                reply = await asyncio.wait_for(self.transport.send(text), self.answer_timeout_s)
            except asyncio.CancelledError:  # the episode's own timeout: the policy did not cause the silence
                self.failures.append({"kind": "crucible:actor_timeout", "reason": f"{self.actor_id} had not answered when the episode ended",
                                      "at": turn.time})
                raise
            except TimeoutError as e:
                self.failures.append({"kind": "crucible:actor_timeout", "reason": f"{self.actor_id} did not answer: {e}"[:300], "at": turn.time})
                continue
            except Exception as e:  # noqa: BLE001 - an unreachable or failed human/peer is infrastructure, not policy
                self.failures.append({"kind": "transport_unreachable", "reason": f"{type(e).__name__}: {e}"[:300], "at": turn.time})
                continue
            if len(reply) > self.max_reply_chars:
                reply = reply[: self.max_reply_chars] + f"\n[... reply cut at {self.max_reply_chars} characters ...]"
            self.history.append({"from": msg.frm, "q": msg.text, "a": reply, "at": turn.time})
            out.append(Message(to=msg.frm, text=reply))
        return out

    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), "transport": self.transport.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        st = dict(state.get("transport", {}))
        if getattr(self.transport, "remote_memory", False):  # the remote side still remembers the abandoned branch
            st["context_id"] = None
            self._replay = True
        self.transport.load_state_dict(st)


class HumanActor(TransportActor):
    kind = "human"


class PeerActor(TransportActor):
    kind = "peer"
