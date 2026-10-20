"""M1 item 10a: typed spaces and the control App (``core/schema/spaces.py``, ``world/apps/control.py``).

Real Gymnasium environments (Pendulum: continuous torque; CartPole: discrete push) run as World Apps: the simulator owns
time (frame skip, a World in sim-clock mode follows simulation time), observations and per-tick rewards are numeric,
terminated and truncated end the episode, restore replays seed plus actions and checks every digest, and a simulator
that cannot be replayed refuses mid-episode restores. The language-model surface (a schema from the action space, wait,
macros) is checked here too.
"""

from __future__ import annotations

import json
import random

import pytest

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.spaces import Space, box, discrete, from_gymnasium, from_spec_dict
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.state.base import SnapshotMismatch
from crucible.world.world import World

gymnasium = pytest.importorskip("gymnasium")


# ----------------------------------------------------------------------------------------------- spaces
def test_spaces():
    b = box([2], low=-1.0, high=[1.0, 2.0])
    assert b.contains([0.5, 1.5]) and "outside" in b.problem([0.5, 3.0]) and "expected 2" in b.problem([1.0])
    assert "finite" in b.problem([float("nan"), 0.0]) and b.clip([5.0, -5.0]) == [1.0, -1.0]
    assert b.json_schema()["minItems"] == 2 and "bounds" in b.json_schema()["description"]  # non-uniform bounds are spelled out
    d = discrete(3, start=1)
    assert d.contains(3) and not d.contains(0) and not d.contains(True) and d.flatten(2) == [0.0, 1.0, 0.0]
    md = Space(kind="multi_discrete", nvec=[2, 3])
    assert md.contains([1, 2]) and not md.contains([2, 0]) and md.flat_dim() == 5
    dd = Space(kind="dict", spaces={"pos": box(2), "mode": discrete(2)})
    assert dd.flatten({"pos": [1.0, 2.0], "mode": 1}) == [0.0, 1.0, 1.0, 2.0]  # keys in order: mode, pos
    rng = random.Random(0)
    assert dd.contains(dd.sample(rng)) and b.contains(b.sample(rng))
    assert Space.model_validate_json(dd.model_dump_json()) == dd  # data, not a library object
    with pytest.raises(ValueError):
        box([2], low=1.0, high=0.0)
    pend = from_gymnasium(gymnasium.make("Pendulum-v1").action_space)
    assert pend.kind == "box" and pend.shape == [1] and pend.low == -2.0 and pend.high == 2.0
    assert from_gymnasium(gymnasium.make("CartPole-v1").action_space) == discrete(2)
    dm = from_spec_dict({"shape": [6], "dtype": "float64", "minimum": [-1.0] * 6, "maximum": [1.0] * 6})  # OpenEnv dm_control's export
    assert dm.shape == [6] and dm.low == -1.0 and dm.dtype == "float64"
    assert from_spec_dict({"position": {"shape": [3], "dtype": "float64"}, "velocity": {"shape": [2], "dtype": "float64"}}).flat_dim() == 5


# ----------------------------------------------------------------------------------------------- the control App
def _spec(env: str, *, clock: bool = True, **params) -> WorldSpec:
    d = {"world": "t/control", "seed": 3,
         "apps": [{"id": "sim", "template": "common/control", "state": {"adapter": "json_doc", "params": {"path": "sim.json"}}, "params": {"env": env, **params}}],
         "actors": [{"id": "agent", "kind": "policy", "tools": ["sim.*"]}]}
    if clock:
        d["clock"] = {"mode": "sim", "t0": "2026-01-05T09:00:00Z"}
    return WorldSpec.model_validate(d)


async def _world(tmp, spec):
    w = World(spec, store=Store(tmp / "s"), workdir=tmp / "w")
    await w.start()
    await w.reset()
    return w


async def _call(w, tool, i=1, **args):
    return (await w.step("agent", ToolCall(tool_call_id=f"c{i}", function_name=tool, arguments=args))).observation.tool_results[0]


async def test_pendulum_ticks_clock_and_restore(tmp_path):
    w = await _world(tmp_path, _spec("gymnasium:Pendulum-v1", frame_skip=2))
    try:
        (step,) = [t for t in w.tools_for("agent") if t.name == "step"]
        assert step.input_schema["properties"]["action"] == {"type": "array", "items": {"type": "number", "minimum": -2.0, "maximum": 2.0},
                                                             "minItems": 1, "maxItems": 1, "description": "1 numbers"}
        r = await _call(w, "sim.step", 1, action=[0.5])
        assert r.ok and r.data["ticks"] == 2 and r.data["tick"] == 2 and len(r.data["observation"]) == 3 and r.data["reward"] < 0
        assert json.loads(r.content)["time_s"] == 0.1 and w.clock.now().isoformat().endswith("09:00:00.100000+00:00")  # sim time moved
        bad = await _call(w, "sim.step", 2, action=[5.0])
        assert bad.error == "invalid_arguments" and bad.fault == "policy"  # the bounds are in the tool schema: the World's gate refuses it
        mid = await w.snapshot()
        r1 = await _call(w, "sim.step", 3, action=[-1.0])
        await _call(w, "sim.step", 4, action=[2.0])
        await w.restore(mid)  # the simulator is rebuilt from the seed and the journal is replayed, digests checked
        r2 = await _call(w, "sim.step", 5, action=[-1.0])
        assert r2.data["observation"] == r1.data["observation"] and r2.data["reward"] == r1.data["reward"]
    finally:
        await w.close()


async def test_episode_end_truncation_and_reseed(tmp_path):
    w = await _world(tmp_path, _spec("gymnasium:CartPole-v1", clock=False))
    try:
        start = (await _call(w, "sim.observe")).data["observation"]
        r = None
        for i in range(500):
            r = await _call(w, "sim.step", i, action=1)  # always push right: the pole falls
            if r.ends_episode:
                break
        assert r.ends_episode and r.data["terminated"] and w.done == {"agent"}
        await w.reset()  # a fresh episode: the initial journal is restored, the same seed gives the same start
        assert (await _call(w, "sim.observe")).data["observation"] == start
        await w.apps["sim"].reseed(99)
        assert (await _call(w, "sim.observe", 2)).data["observation"] != start
    finally:
        await w.close()
    w = await _world(tmp_path / "t", _spec("gymnasium:CartPole-v1", clock=False, max_episode_ticks=3, frame_skip=2))
    try:
        assert not (await _call(w, "sim.step", 1, action=0)).ends_episode
        r = await _call(w, "sim.step", 2, action=1)
        assert r.ends_episode and r.data["truncated"] and r.data["ticks"] == 1  # the cap cut the second skip short
    finally:
        await w.close()


async def test_language_model_surface_wait_and_macros(tmp_path):
    spec = _spec("gymnasium:Pendulum-v1", idle_action=[0.0], macros={"full_left": {"action": [-2.0], "ticks": 5, "description": "Full torque left for 0.25 s."}})
    w = await _world(tmp_path, spec)
    try:
        names = sorted(t.name for t in w.tools_for("agent"))
        assert names == ["full_left", "observe", "step", "wait"]
        r = await _call(w, "sim.wait", 1, ticks=20)
        assert r.data["ticks"] == 20 and json.loads(r.content)["time_s"] == 1.0  # one decision, a second of simulation
        r = await _call(w, "sim.full_left", 2)
        assert r.data["tick"] == 25
        j = w.apps["sim"].journal
        assert [s["tool"] for s in j["steps"]] == ["wait", "full_left"] and j["tick"] == 25
    finally:
        await w.close()


class Noisy:
    """A simulator whose dynamics ignore the seed: not replayable."""

    def __init__(self):
        import gymnasium as g

        self.action_space = g.spaces.Box(-1.0, 1.0, (1,))
        self.observation_space = g.spaces.Box(-10.0, 10.0, (1,))

    def reset(self, seed=None):
        return [0.0], {}

    def step(self, action):
        return [random.random()], 0.0, False, False, {}


async def test_non_deterministic_simulators(tmp_path):
    w = await _world(tmp_path, _spec("", clock=False, factory="tests.unit.test_control:Noisy"))
    try:
        mid = await w.snapshot()
        await _call(w, "sim.step", 1, action=[0.1])
        later = await w.snapshot()
        await w.restore(mid)  # the episode start is a fresh reset: fine
        with pytest.raises(SnapshotMismatch, match="not deterministic"):
            await w.restore(later)
    finally:
        await w.close()
    w = await _world(tmp_path / "n", _spec("", clock=False, factory="tests.unit.test_control:Noisy", restore="none"))
    try:
        await _call(w, "sim.step", 1, action=[0.1])
        with pytest.raises(SnapshotMismatch, match="episode start only"):
            await w.restore(await w.snapshot())
        await w.reset()  # a new episode still works
    finally:
        await w.close()


async def test_llm_tool_policy_drives_a_simulator_through_the_harness(tmp_path):
    """The language-model path end to end: LLMToolPolicy (replayed tool calls) -> step / wait tools -> ControlTask grade."""
    from crucible.actors.policy import EpisodeLimits, LLMToolPolicy
    from crucible.runtime.control_task import ControlTask
    from crucible.runtime.harness.native import NativeHarness, PolicyRef
    from crucible.runtime.inject import ReplayLLM
    from crucible.runtime.run import Runner, RunSpec

    task = ControlTask("t/pendulum/hold@1", env="gymnasium:Pendulum-v1", return_range=(-2000.0, 0.0), idle_action=[0.0], frame_skip=5, max_episode_ticks=60)
    acts = [{"function_name": "sim.step", "arguments": {"action": [1.0]}}, {"function_name": "sim.wait", "arguments": {"ticks": 30}},
            {"function_name": "sim.step", "arguments": {"action": [-1.0]}}]
    llm = ReplayLLM(acts)
    pol = PolicyRef(id="llm:replay", make=lambda a: {"agent": LLMToolPolicy("agent", llm)}, config={"script": "replayed tool calls"})  # M10 lab audit R-4
    await Runner(tmp_path / "runs").run(RunSpec(run_id="c", tasks=[task], policies=[pol], harnesses=[NativeHarness(limits=EpisodeLimits(max_steps=10))], k=1))
    rec = json.loads(next((tmp_path / "runs" / "c" / "samples").glob("*/a1/record.json")).read_text())
    assert rec["failure"] is None and 0.0 < rec["reward"] < 1.0, rec
    trace = json.loads(next((tmp_path / "runs" / "c" / "samples").glob("*/a1/trace.json")).read_text())
    calls = [e["data"]["function"] for e in trace["events"] if e.get("type") == "tool_call"]
    assert calls == ["sim.step", "sim.wait", "sim.step"]  # three decisions over 40 simulator ticks
