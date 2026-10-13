"""M11 audit cases for the OpenEnv import and export, run inside ``crucible/.envs/openenv``:
``python _m11_openenv_cases.py <case> <tmpdir>`` prints ``OK`` on success."""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State

from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode
from crucible.core.schema.actions import ToolCall
from crucible.interop.openenv.import_env import OpenEnvTask


class PickAction(Action):
    n: int


class HiddenEnv(Environment):
    """Step 1 draws a hidden card into ``metadata`` (the text never shows it); step 2 ends the episode."""

    def __init__(self):
        super().__init__()
        self._state, self.card, self.n = State(), None, 0

    def reset(self, seed=None, episode_id=None, **kw) -> Observation:
        self._state, self.card, self.n = State(episode_id=episode_id, step_count=0), None, 0
        return Observation(done=False, reward=None, metadata={})

    def step(self, action, timeout_s=None, **kw) -> Observation:
        self.n += 1
        if action.n > 5:
            raise ValueError(f"action n={action.n} out of range")  # as Atari does for an out-of-range action_id
        if self.n == 1:
            self.card = uuid.uuid4().hex
            return Observation(done=False, reward=0.0, metadata={"card": self.card})
        return Observation(done=True, reward=float("nan") if action.n == 5 else 1.0, metadata={"card": self.card})

    @property
    def state(self) -> State:
        return self._state


def tc(i: int, n: int) -> ToolCall:
    return ToolCall(tool_call_id=f"t{i}", function_name="env.step", arguments={"n": n})


def task() -> OpenEnvTask:
    return OpenEnvTask("openenv/hidden/t@1", f"{__name__}:HiddenEnv", action_cls=f"{__name__}:PickAction")


async def import_cases(d: Path) -> None:
    from crucible.world.state.base import SnapshotMismatch

    # H3 V1: a restore that draws a different hidden card is refused
    w = await task().open(d / "a")
    try:
        await w.step("agent", tc(1, 1))
        ref = await w.snapshot()
        try:
            await w.restore(ref)
        except SnapshotMismatch:
            pass
        else:
            raise AssertionError("a restore with different hidden metadata was accepted")
    finally:
        await w.close()
    # H3 V2: an action the environment rejects by raising is the policy's error, scored, not masked
    w = await task().open(d / "b")
    try:
        ep = await run_episode(w, {"agent": ScriptedPolicy("agent", [tc(1, 99), tc(2, 1), tc(3, 1)])}, limits=EpisodeLimits(max_steps=10), reset=False)
        rep = await task().grade(w, d / "gb")
        assert not ep.masked and rep.infra["status"] == "ok" and rep.reward == 1.0, (ep.masked, ep.stop_condition, rep.infra)
    finally:
        await w.close()
    # H3 V3: a NaN terminal reward is masked, never scored
    w = await task().open(d / "c")
    try:
        await run_episode(w, {"agent": ScriptedPolicy("agent", [tc(1, 1), tc(2, 5)])}, limits=EpisodeLimits(max_steps=10), reset=False)
        rep = await task().grade(w, d / "gc")
        assert rep.infra["status"] == "masked", rep.infra
    finally:
        await w.close()


def export_cases(d: Path) -> None:
    """G1 D1, H3 V4/V5: a masked episode raises (its Ψ steps were already paid); finish after done pays nothing."""
    from crucible_pack_toy.runtime import HelpdeskTask
    from crucible_pack_toy.verify import GOLD
    from openenv.core.env_server.mcp_types import CallToolAction

    from crucible.interop.openenv.export import FINISH, CrucibleMaskedEpisode, make_environment

    class Flaky(HelpdeskTask):
        async def open(self, workdir):
            w = await super().open(workdir)
            app, orig = w.apps["tickets"], w.apps["tickets"].call

            async def call(name, args, *, actor, call_id):
                if name == "set_status":
                    raise ConnectionError("synthetic infra fault")
                return await orig(name, args, actor=actor, call_id=call_id)

            app.call = call
            return w

    env = make_environment(Flaky)()
    env.reset(seed=0, episode_id="m")
    raised = False
    for a in GOLD:
        try:
            env.step(CallToolAction(tool_name=a.function_name.replace(".", "__", 1), arguments=dict(a.arguments)))
        except CrucibleMaskedEpisode:
            raised = True
            break
    env.close()
    assert raised, "an infra fault in the episode did not mask it"
    env = make_environment(HelpdeskTask)()
    env.reset(seed=0, episode_id="ok")
    rewards = [env.step(CallToolAction(tool_name=a.function_name.replace(".", "__", 1), arguments=dict(a.arguments))).reward for a in GOLD]
    rewards += [env.step(CallToolAction(tool_name=FINISH, arguments={})).reward for _ in range(3)]
    after = env.step(CallToolAction(tool_name=GOLD[0].function_name.replace(".", "__", 1), arguments=dict(GOLD[0].arguments)))
    env.close()
    assert abs(sum(r for r in rewards if r is not None) - 1.0) < 1e-9, rewards
    assert after.done and after.reward == 0.0


if __name__ == "__main__":
    case, d = sys.argv[1], Path(sys.argv[2])
    d.mkdir(parents=True, exist_ok=True)
    asyncio.run(import_cases(d)) if case == "import" else export_cases(d)
    print("OK")
