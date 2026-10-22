"""M2 R-3 (audit 2026-10-15): ``World.finished`` and ``World.failures`` follow a restore. Both were kept from before it,
so M6 ``branch_value`` continuations saw the previous continuation's user stop and responder fault, and a World restored
from a checkpoint taken after the user's DONE talked on to ``max_steps`` (probes p03 and v3_ng7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible.actors.base import ActorBase
from crucible.actors.llm import LLMResponse
from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode
from crucible.actors.simulated import DONE, Scenario, SimulatedActor
from crucible.core.schema.actions import Message
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.credit.operators.branch_value import branch_value
from crucible.world.world import World

SC = Scenario(persona="Lee, a customer", reason_for_contact="a double charge of 39.50 USD")


class SimLLM:
    """The user's first reply is a complaint, every later one ends the conversation."""

    model = "fake"

    async def complete(self, messages, *, tools=None, **kw):
        mine = sum(1 for m in messages if m.get("role") == "assistant")
        return LLMResponse(text="I was charged twice." if mine == 0 else f"Thanks. {DONE}")


class FlakyResponder(ActorBase):
    """Raises on its first message only (a counter outside its saved state), then answers."""

    def __init__(self, actor_id: str):
        super().__init__(actor_id)
        self.calls = 0

    async def respond(self, turn):
        if not turn.inbox:
            return []
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient backend error")
        return [Message(to=turn.inbox[-1].frm, text="ok")]


def world(tmp: Path, name: str, responder) -> World:
    spec = {"world": "t/sim", "apps": [], "actors": [{"id": "agent", "kind": "policy"}, {"id": "lee", "kind": "simulated"}]}
    w = World(WorldSpec.model_validate(spec), instructions={"agent": "Help."}, store=Store(tmp / f"s-{name}"), workdir=tmp / f"w-{name}")
    w.attach(responder)
    return w


def talk(_i: int = 0, n: int = 4) -> ScriptedPolicy:
    return ScriptedPolicy("agent", [Message(to="lee", text=f"message {j}") for j in range(n)])


@pytest.mark.parametrize(("name", "stops", "steps", "masked"), [
    ("finished", ["user_stop"] * 3, [2, 2, 2], 0),  # was: continuations 2 and 3 cut to 1 step by continuation 1's DONE
    ("failures", ["crucible:sim_failure", "max_steps", "max_steps"], [1, 4, 4], 1),  # was: all 3 masked by continuation 1's fault
])
async def test_each_branch_value_continuation_starts_from_the_branch_point(tmp_path, name, stops, steps, masked):
    w = world(tmp_path, name, SimulatedActor("lee", SC, SimLLM()) if name == "finished" else FlakyResponder("lee"))
    made: list[ScriptedPolicy] = []

    def factory(_i: int) -> ScriptedPolicy:
        made.append(talk())
        return made[-1]

    async def verify(_w) -> float:
        return 1.0

    await w.start()
    try:
        await w.reset()
        ref = await w.snapshot()
        out = await branch_value(w, ref, step=0, policy_factory=factory, verify=verify, k=3, limits=EpisodeLimits(max_steps=4))
        assert (out.stops, [p._i for p in made], out.masked) == (stops, steps, masked)
        assert w.finished == {} and w.failures == []  # left at the branch point
    finally:
        await w.close()


async def test_a_world_restored_after_the_users_done_still_knows_it(tmp_path):
    src = world(tmp_path, "src", SimulatedActor("lee", SC, SimLLM()))
    await src.start()
    try:
        await src.reset()
        early = await src.snapshot()
        await src.step("agent", Message(to="lee", text="hello"))
        await src.step("agent", Message(to="lee", text="refund issued"))
        assert src.finished == {"lee": "user_stop"}
        late = await src.snapshot()
        await src.restore(early)
        assert src.finished == {}  # a restore to before the DONE: the user is talking again
        res = await run_episode(src, {"agent": talk(n=5)}, reset=False, limits=EpisodeLimits(max_steps=5))
        assert (res.stop_condition, res.steps) == ("user_stop", 2)  # was: user_stop after 1 step, from the abandoned branch
    finally:
        await src.close()
    dst = world(tmp_path, "dst", SimulatedActor("lee", SC, SimLLM()))
    await dst.start()
    try:
        await dst.restore(late)  # a checkpoint resume in a fresh World
        assert dst.finished == {"lee": "user_stop"}
        res = await run_episode(dst, {"agent": talk(n=5)}, reset=False, limits=EpisodeLimits(max_steps=5))
        assert (res.stop_condition, res.steps, res.truncated) == ("user_stop", 1, False)  # was: max_steps after 5, truncated
    finally:
        await dst.close()


async def test_an_actor_still_talking_saves_the_state_it_saved_before():
    a = SimulatedActor("lee", SC, SimLLM())
    assert "finished_reason" not in a.state_dict()  # WorldRef fingerprints of unfinished conversations are unchanged
    a.finished_reason = "user_stop"
    b = SimulatedActor("lee", SC, SimLLM())
    b.load_state_dict(a.state_dict())
    assert b.finished_reason == "user_stop"
    b.load_state_dict({"history": [], "finished": True})  # a ref saved before R-3
    assert b.finished_reason is None
