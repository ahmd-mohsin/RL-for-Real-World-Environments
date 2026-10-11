"""Native turn order for Worlds with several policy actors (M1 item 7; P7 of the M1 taxonomy; OpenSpiel / OpenEnv game
semantics: current player, per-player legal actions, simultaneous moves, chance inside the game, per-player rewards).

``WorldSpec.turns`` (``TurnSpec``) picks one of three modes:

- ``sequential``: seats move in order, ``per_turn`` actions each (a tool call or a message is an action), then the
  next seat that is not done;
- ``simultaneous``: every seat still playing submits one action per round; nothing runs until the round is complete,
  so no seat sees another's move first. The round then resolves through the ``app``'s ``resolve_round(actions)``
  when it has one (a payoff matrix, an auction), else by executing the calls in seat order. A seat's results reach it
  on its next observation;
- ``app``: an App owns the rules. Its ``turn_state()`` returns ``{"to_move": [...], "legal": {seat: [...]},
  "terminal": bool, "rewards": {seat: float}}`` after every executed action (chance nodes, such as a dealt card or a
  die, are the App's, drawn from its seed).

In every mode a call from a seat that is not to move is refused (``not_your_turn``), and a call outside the seat's
``legal`` list is refused (``illegal_action``). Both are the policy's errors, and neither executes. A legal entry is a
qualified tool name, or ``{"function_name", "arguments"}`` matched exactly. ``Done`` is always accepted: the seat
leaves the rotation. Each policy observation carries ``to_move`` and the seat's ``legal`` list. Turn state (whose
move, the round's buffer, rewards) is part of every ``WorldRef``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from crucible.core.schema.actions import Message, ToolCall, ToolResult

if TYPE_CHECKING:
    from crucible.core.schema.world import TurnSpec
    from crucible.world.world import World


def _matches(entry: Any, call: ToolCall) -> bool:
    if isinstance(entry, str):
        return entry == call.function_name
    if isinstance(entry, dict):
        return entry.get("function_name") == call.function_name and (entry.get("arguments") is None or entry.get("arguments") == call.arguments)
    return False


class TurnManager:
    def __init__(self, world: World, spec: TurnSpec):
        self.world, self.spec = world, spec
        self.mode = spec.mode
        self.seats = list(spec.seats or [a.id for a in world.spec.policy_actors])
        self.to_move: list[str] = []
        self.legal: dict[str, list[Any]] = {}
        self.rewards: dict[str, float] = {}
        self.terminal = False
        self.used = 0  # actions taken in the current sequential turn
        self.round = 0
        self.buffer: dict[str, dict[str, Any]] = {}  # simultaneous: seat -> submitted action (json)
        self.deferred: dict[str, list[dict[str, Any]]] = {}  # seat -> results it has not seen yet (json)

    # ---- state ---------------------------------------------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        return {"to_move": list(self.to_move), "legal": {k: list(v) for k, v in self.legal.items()}, "rewards": dict(self.rewards),
                "terminal": self.terminal, "used": self.used, "round": self.round, "buffer": dict(self.buffer),
                "deferred": {k: list(v) for k, v in self.deferred.items()}}

    def restore(self, st: dict[str, Any] | None) -> None:
        st = st or {}
        self.to_move, self.legal, self.rewards = list(st.get("to_move", [])), dict(st.get("legal", {})), dict(st.get("rewards", {}))
        self.terminal, self.used, self.round = bool(st.get("terminal")), int(st.get("used", 0)), int(st.get("round", 0))
        self.buffer, self.deferred = dict(st.get("buffer", {})), {k: list(v) for k, v in st.get("deferred", {}).items()}

    def _playing(self) -> list[str]:
        return [s for s in self.seats if s not in self.world.done]

    async def begin(self) -> None:
        """The start of an episode (after start or a fresh reset)."""
        self.used, self.round, self.buffer, self.deferred, self.terminal = 0, 0, {}, {}, False
        if self.mode == "app":
            await self.pull()
        else:
            self.rewards = {}
            self.legal = {}
            playing = self._playing()
            self.to_move = playing[:1] if self.mode == "sequential" else playing

    async def pull(self) -> None:
        app = self.world.apps[self.spec.app or ""]
        st = await app.turn_state()  # type: ignore[attr-defined]
        unknown = sorted(set(st.get("to_move") or []) - set(self.seats))
        if unknown:
            raise RuntimeError(f"app {app.id!r} named unknown seats to move: {unknown}")
        self.to_move = [s for s in st.get("to_move") or [] if s not in self.world.done]
        self.legal = {k: list(v) for k, v in (st.get("legal") or {}).items()}
        self.rewards = {k: float(v) for k, v in (st.get("rewards") or {}).items()}
        self.terminal = bool(st.get("terminal"))

    # ---- the gate ------------------------------------------------------------------------------------------------
    def may_act(self, seat: str) -> bool:
        if seat not in self.seats:
            return True  # policy actors outside the game (a coach, an auditor) act freely
        if self.terminal or seat not in self.to_move:
            return False
        return not (self.mode == "simultaneous" and seat in self.buffer)

    def refusal(self, seat: str, action: ToolCall | Message) -> ToolResult | None:
        if seat not in self.seats:
            return None
        name = action.function_name if isinstance(action, ToolCall) else "message"
        cid = action.tool_call_id if isinstance(action, ToolCall) else f"msg-{self.world.step_index}"
        if self.terminal:
            return ToolResult(tool_call_id=cid, function_name=name, ok=False, content="The game is over.", error="not_your_turn", fault="policy")
        if not self.may_act(seat):
            who = ", ".join(self.to_move) or "nobody"
            why = "you already moved this round" if seat in self.buffer else f"{who} to move"
            return ToolResult(tool_call_id=cid, function_name=name, ok=False, content=f"It is not your turn ({why}).", error="not_your_turn", fault="policy")
        legal = self.legal.get(seat)
        if legal is not None and isinstance(action, ToolCall) and not any(_matches(e, action) for e in legal):
            return ToolResult(tool_call_id=cid, function_name=name, ok=False, content=f"Illegal action. Legal actions: {legal[:20]}", error="illegal_action",
                              fault="policy")
        return None

    # ---- bookkeeping after an executed action --------------------------------------------------------------------
    async def after_action(self, seat: str) -> None:
        if seat not in self.seats:
            return
        if self.mode == "app":
            await self.pull()
        elif self.mode == "sequential":
            self.used += 1
            if self.used >= self.spec.per_turn:
                self.advance(seat)

    def advance(self, seat: str) -> None:
        playing = self._playing()
        self.used = 0
        if not playing:
            self.to_move = []
            return
        i = self.seats.index(seat)
        nxt = next((s for s in self.seats[i + 1:] + self.seats[: i + 1] if s in playing), None)
        self.to_move = [nxt] if nxt else []

    async def on_done(self, seat: str) -> None:
        if seat not in self.seats:
            return
        if self.mode == "sequential" and seat in self.to_move:
            self.advance(seat)
        elif self.mode == "simultaneous":
            self.to_move = [s for s in self.to_move if s != seat]
        elif self.mode == "app":
            await self.pull()

    # ---- simultaneous rounds -------------------------------------------------------------------------------------
    def submit(self, seat: str, action: ToolCall | Message) -> bool:
        self.buffer[seat] = action.model_dump(mode="json")
        return all(s in self.buffer for s in self.to_move)

    def take_round(self) -> dict[str, dict[str, Any]]:
        out = {s: self.buffer[s] for s in self.seats if s in self.buffer}
        self.buffer = {}
        self.round += 1
        return out

    def end_round(self) -> None:
        self.to_move = self._playing()

    def defer(self, seat: str, results: list[ToolResult]) -> None:
        self.deferred.setdefault(seat, []).extend(r.model_dump(mode="json") for r in results)

    def take_deferred(self, seat: str) -> list[ToolResult]:
        return [ToolResult.model_validate(r) for r in self.deferred.pop(seat, [])]
