"""Seat views for turn-based multi-agent Worlds (M1 item 7; moved here from ``train/seats.py`` so every episode can use them).

``MoveLog`` (a ``run_episode`` observer) records each executed move; ``SeatView`` (a policy wrapper) shows a seat the
moves the other seats made since it last acted, and its legal actions. ``run_episode`` wraps the seats of a multi-seat
turn-based World automatically: found live with two OpenRouter models playing Nim, a plain ``LLMToolPolicy`` seat was
never told the opponent's move or that it was its turn again, replied in text, and resigned the game after one move.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from crucible.core.schema.actions import Inbound, Observation, ToolCall


class MoveLog:
    """A ``run_episode`` observer: every executed move of a game seat, in order (``{seat, step, function, arguments}``).
    Refused calls (not your turn, illegal) are not moves; a simultaneous round's buffered moves are released together
    when the round resolves; messages are delivered by the World itself and are not repeated here."""

    def __init__(self, seats: Sequence[str]):
        self.seats = set(seats)
        self.moves: list[dict[str, Any]] = []
        self._held: list[dict[str, Any]] = []

    async def on_start(self, world: Any) -> None:
        self.moves, self._held = [], []

    async def on_step(self, world: Any, step: int, actor: str, action: Any, res: Any) -> None:
        if actor not in self.seats or res.info.get("refused"):
            return
        if isinstance(action, ToolCall):
            mv = {"seat": actor, "step": step, "function": action.function_name, "arguments": action.arguments}
            if res.info.get("waiting"):  # simultaneous: secret until the round resolves
                self._held.append(mv)
                return
            self.moves += [*self._held, mv]
            self._held = []
        elif self._held and not res.info.get("waiting"):  # a Done that closed the round
            self.moves += self._held
            self._held = []

    async def on_end(self, world: Any) -> dict[str, Any]:
        return {"moves": len(self.moves)}


def describe_move(m: dict[str, Any]) -> str:
    return f"played {m['function']} {json.dumps(m['arguments'], sort_keys=True)}"


class SeatView:
    """Wraps a seat's policy: before it acts, the moves the other seats made since its last action are appended to its
    observation's inbox (and, with ``show_legal``, its legal actions). Everything else is the wrapped policy's
    (``model_calls``, ``segments``, ``state_dict``, ...)."""

    def __init__(self, inner: Any, seat: str, log: MoveLog, *, reveal: bool = True, show_legal: bool = False,
                 describe: Callable[[dict[str, Any]], str] = describe_move):
        self.inner, self.actor_id, self.log = inner, seat, log
        self.reveal, self.show_legal, self.describe = reveal, show_legal, describe
        self._seen = 0

    def __getattr__(self, name: str) -> Any:
        if name == "inner":  # not yet set (copy / unpickling): never recurse
            raise AttributeError(name)
        return getattr(self.inner, name)

    async def act(self, obs: Observation) -> Any:
        new = [m for m in self.log.moves[self._seen:] if m["seat"] != self.actor_id]
        self._seen = len(self.log.moves)
        inbox = [Inbound(frm=m["seat"], text=self.describe(m)) for m in new] if self.reveal else []
        if self.show_legal and obs.legal is not None:
            inbox.append(Inbound(frm="game", text=f"your legal actions: {json.dumps(obs.legal, sort_keys=True)}"))
        if inbox:
            obs = obs.model_copy(update={"inbox": [*obs.inbox, *inbox]})
        return await self.inner.act(obs)
