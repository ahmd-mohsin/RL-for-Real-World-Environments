"""Cases for ``test_openenv_import.py``, run inside ``crucible/.envs/openenv`` (OpenEnv + PortSimEnv + torch installed):
``python _openenv_import_cases.py <case> <tmpdir>`` prints ``OK`` on success."""

from __future__ import annotations

import asyncio
import json
import random
import sys
import zlib
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State

from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode
from crucible.core.schema.actions import ToolCall
from crucible.interop.openenv.catalog import portsim
from crucible.interop.openenv.import_env import OpenEnvTask


def tc(i: int, tool: str, args: dict) -> ToolCall:
    return ToolCall(tool_call_id=f"t{i}", function_name=f"env.{tool}", arguments=args)


def plans(index: int = 0):
    from berth_core import baselines, load_pack, plan_to_list

    t = load_pack().at("train", index)
    return t.reference["optimal_plan"], plan_to_list(baselines.serial(t))


async def episode(task, actions, d: Path, max_steps: int = 40):
    w = await task.open(d / "work")
    try:
        ep = await run_episode(w, {"agent": ScriptedPolicy("agent", actions)}, limits=EpisodeLimits(max_steps=max_steps), reset=False)
        rep = await task.grade(w, d / "grade")
        return ep, rep, w
    finally:
        await w.close()


def native_reward(plan, index: int = 0) -> float:
    from berth_openenv.environment import BerthPlanningEnvironment
    from openenv.core.env_server.mcp_types import CallToolAction

    e = BerthPlanningEnvironment()
    e.reset(split="train", index=index, episode_id="native")
    return e.step(CallToolAction(tool_name="submit_plan", arguments={"plan": plan})).reward


# ---- a generic (non-MCP) OpenEnv environment and a non-deterministic one ----------------------------------------
class AddAction(Action):
    """Add n to the running total; the episode ends when the total reaches 3 (reward 1 for exactly 3)."""

    n: int


class CounterEnv(Environment):
    def __init__(self):
        super().__init__()
        self.total, self._state = 0, State()

    def reset(self, seed=None, episode_id=None, **kw) -> Observation:
        self.total, self._state = 0, State(episode_id=episode_id, step_count=0)
        return Observation(done=False, reward=None, metadata={"goal": "reach exactly 3"})

    def step(self, action, timeout_s=None, **kw) -> Observation:
        self.total += action.n
        self._state.step_count += 1
        done = self.total >= 3
        return Observation(done=done, reward=(1.0 if self.total == 3 else 0.0) if done else 0.25, metadata={"total": self.total})

    @property
    def state(self) -> State:
        return self._state


class NoisyEnv(CounterEnv):
    def step(self, action, timeout_s=None, **kw) -> Observation:
        o = super().step(action)
        o.metadata["noise"] = random.random()
        return o


GENERIC = {"action_cls": f"{__name__}:AddAction", "show_metadata": True}

PNG = bytes.fromhex("89504e470d0a1a0a0000000d49484452")


def make_sim_env():
    """An MCP environment with an image tool, simulation time in its observations and a reset seed."""
    class SimMcpEnv(MCPEnvironment):
        def __init__(self):
            mcp = FastMCP("sim")
            self.t, self.seed = 0.0, None

            @mcp.tool
            def render() -> Image:
                """A frame of the simulation."""
                return Image(data=PNG, format="png")

            @mcp.tool
            def wait(hours: float) -> str:
                """Let the simulation run."""
                self.t += hours * 3600
                return f"t={self.t / 3600:.0f}h"

            @mcp.tool
            def seed() -> str:
                """The reset seed."""
                return str(self.seed)

            super().__init__(mcp)
            self._state = State()

        def reset(self, seed=None, episode_id=None, **kw):
            self.t, self.seed = 0.0, seed
            self._state = State(episode_id=episode_id, step_count=0)
            return Observation(done=False, reward=None, metadata={"goal": "watch"})

        async def step_async(self, action, timeout_s=None, **kw):
            o = await super().step_async(action, timeout_s=timeout_s, **kw)
            o.metadata = {**(o.metadata or {}), "sim": {"t_s": self.t}}
            return o

        def _step_impl(self, action, timeout_s=None, **kw):
            return Observation(done=False, reward=0.0)

        @property
        def state(self):
            return self._state

    return SimMcpEnv()


# ---- cases ------------------------------------------------------------------------------------------------------
async def portsim_rewards(d: Path) -> None:
    opt, ser = plans()
    _, rep, _ = await episode(portsim(), [tc(0, "get_situation", {}), tc(1, "submit_plan", {"plan": opt})], d / "o")
    assert rep.reward == 1.0 == native_reward(opt), rep.reward
    _, rep, _ = await episode(portsim(), [tc(0, "submit_plan", {"plan": ser})], d / "s")
    assert abs(rep.reward - native_reward(ser)) < 1e-12 and rep.reward <= 0.2 + 1e-9, rep.reward
    ep, rep, _ = await episode(portsim(), [], d / "n")  # stopped without a final answer: scored 0, not masked
    assert rep.reward == 0.0 and rep.infra["status"] == "ok" and not ep.masked
    _, rep, _ = await episode(portsim(), [tc(0, "submit_plan", {"plan": [{"ship": 0, "berth_hour": 0, "section": 2}]})], d / "b")
    assert rep.reward <= 0.2 and rep.infra["status"] == "ok"
    ep, rep, _ = await episode(portsim(), [tc(0, "submit_plan", {"plan": opt}), tc(1, "get_situation", {})], d / "x")
    assert ep.steps == 1 and rep.reward == 1.0  # submit_plan ended the episode: the World marked the agent done


async def restore(d: Path) -> None:
    opt, ser = plans()
    t = portsim()
    w = await t.open(d / "work")
    try:
        app = w.apps["env"]
        await w.step("agent", tc(0, "check_plan", {"plan": ser}))
        ref = await w.snapshot()
        await w.step("agent", tc(1, "check_plan", {"plan": opt}))
        assert app.env.state.checks_used == 2
        await w.restore(ref)  # rebuild + reset + re-issue the journal, digests verified
        assert app.env.state.checks_used == 1 and len(app.journal["steps"]) == 1
        assert (await w.verifier_view().fingerprints())["env"] == ref.apps["env"].fingerprint
        r = await w.step("agent", tc(2, "submit_plan", {"plan": opt}))
        assert r.done and (await t.grade(w, d / "g")).reward == 1.0
        await w.reset()
        assert app.env.state.checks_used == 0 and app.journal["steps"] == [] and not w.done
    finally:
        await w.close()


async def generic_env(d: Path) -> None:
    t = OpenEnvTask("openenv/counter/c@1", f"{__name__}:CounterEnv", instruction=lambda o: o.metadata["goal"], **GENERIC)
    w = await t.open(d / "w0")
    assert [x.qualified for x in w.tools_for("agent")] == ["env.step"] and w.instructions["agent"] == "reach exactly 3"
    schema = w.tools_for("agent")[0].input_schema
    assert schema["required"] == ["n"] and "metadata" not in schema["properties"]
    await w.close()
    _, rep, _ = await episode(t, [tc(0, "step", {"n": 1}), tc(1, "step", {"n": 2}), tc(2, "step", {"n": 5})], d / "a")
    assert rep.reward == 1.0
    _, rep, _ = await episode(t, [tc(0, "step", {"n": 4})], d / "b")
    assert rep.reward == 0.0
    s = OpenEnvTask("openenv/counter/c@1", f"{__name__}:CounterEnv", reward_mode="sum", reward_range=(0.0, 2.0), **GENERIC)
    _, rep, _ = await episode(s, [tc(0, "step", {"n": 1}), tc(1, "step", {"n": 2})], d / "c")
    assert abs(rep.reward - 0.625) < 1e-12, rep.reward  # (0.25 + 1.0) / 2
    _, rep, _ = await episode(t, [tc(0, "step", {"n": "x"})], d / "e")  # schema-invalid arguments: the policy's fault
    assert rep.reward == 0.0 and rep.infra["status"] == "ok"


async def nondeterministic(d: Path) -> None:
    from crucible.world.state.base import SnapshotMismatch

    t = OpenEnvTask("openenv/noisy/n@1", f"{__name__}:NoisyEnv", **GENERIC)
    w = await t.open(d / "w")
    try:
        await w.step("agent", tc(0, "step", {"n": 1}))
        ref = await w.snapshot()
        try:
            await w.restore(ref)
        except SnapshotMismatch as e:
            assert "not deterministic" in str(e)
        else:
            raise AssertionError("a non-deterministic environment restored silently")
    finally:
        await w.close()


async def cross_runtime(d: Path) -> None:
    """Native PortSim → Crucible import → (a) replay through the canonical channel, (b) re-export as an OpenEnv
    Environment: the same reward everywhere, and the re-export's step rewards telescope to it."""
    from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

    from crucible.interop.canonical import reward_doc, steps_rows
    from crucible.interop.openenv.export import make_environment
    from crucible.interop.replay import Action as RA
    from crucible.interop.replay import replay

    opt, _ = plans()
    acts = [RA("env.check_plan", {"plan": opt}), RA("env.submit_plan", {"plan": opt})]
    out = await replay(portsim(), acts, d / "replay", episode_key="openenv/portsim/train-0@1|e0")
    rj = reward_doc(d / "replay" / "reward.json")
    assert out["reward"] == 1.0 == rj["reward"] == native_reward(opt)
    rows = steps_rows(d / "replay" / "steps.jsonl")
    assert [r["milestones_new"] for r in rows][-1] and rows[-1]["phi"] == 1.0, rows

    Env = make_environment(lambda: portsim())
    e = Env()

    def run() -> float:
        e.reset(seed=0, episode_id="openenv/portsim/train-0@1|e0")
        tools = [x.name for x in e.step(ListToolsAction()).tools]
        assert "env__submit_plan" in tools and "crucible__finish" in tools
        total = 0.0
        for a in [CallToolAction(tool_name="env__check_plan", arguments={"plan": opt}), CallToolAction(tool_name="env__submit_plan", arguments={"plan": opt})]:
            o = e.step(a)
            total += o.reward or 0.0
        assert o.done and o.reward is not None and o.tool_name == "env__submit_plan"  # submit_plan itself ends the exported episode
        return total

    loop = asyncio.get_running_loop()
    total = await loop.run_in_executor(None, run)  # the exported Environment drives its own loop on a thread
    e.close()
    assert abs(total - 1.0) < 1e-12, total


async def m12_runtime(d: Path) -> None:
    """PortSim through the M12 runtime: a scripted engine emits submit_plan as tokens (oracle for half the rollouts,
    the serial baseline for the rest); LLMToolPolicy → EngineLLM → World → the environment's rubric → rollout/v1."""
    from crucible.core.schema.actions import openai_name
    from crucible.train.chat import ByteChatCodec
    from crucible.train.engine import EnginePool, ScriptedEngine, TinyPolicy
    from crucible.train.runtime import Runtime, RuntimeConfig
    from crucible.train.scheduler import SchedulerConfig
    from crucible.train.trainer import TorchTrainer, TrainerConfig
    from crucible.train.weights import InprocPublisher
    from crucible.train.world_env import world_rollout

    codec = ByteChatCodec()
    opt, ser = plans()

    def script(rid, k, prompt):
        plan = opt if zlib.crc32(rid.encode()) % 2 == 0 else ser
        body = json.dumps({"name": openai_name("env.submit_plan"), "arguments": {"plan": plan}})
        return list(f"<tool_call>{body}</tool_call>".encode()) + [codec.EOS]

    eng = EnginePool([ScriptedEngine(script, stop_ids=[codec.EOS], delay_s=0.001)])
    tr = TorchTrainer(TinyPolicy(codec.vocab, 16, seed=0), TrainerConfig(lr=1e-3, token_budget=1 << 16))
    rt = Runtime(RuntimeConfig(topology="colocated_partial", steps=2, max_turn_tokens=8192, warmup_batches=0), SchedulerConfig(n=2, B=2, max_staleness=2),
                 eng, tr, InprocPublisher(tr, eng), None, {"portsim": [f"p{i}" for i in range(50)]},
                 rollout_fn=world_rollout(lambda p: portsim(), codec, max_steps=4, max_tokens=8192, workroot=d))
    rep = await rt.run()
    assert len(rep.steps) == 2 and rep.masked_rollouts == 0
    rewards = sorted({round(r.reward, 6) for r in rep.trained_records})
    assert set(rewards) <= {round(native_reward(ser), 6), 1.0} and all(r.valid_for_training() for r in rep.trained_records), rewards
    assert all(len(r.turns) == 1 for r in rep.trained_records)  # one call: submit_plan ended the episode


async def mcp_image_clock_seed(d: Path) -> None:
    from crucible.core.ids import derive_seed

    ev = [{"id": "shift", "when": {"after": "3h"}, "actor": "agent", "action": {"type": "nl", "text": "shift change", "to": "agent"}}]
    t = OpenEnvTask("openenv/sim/s@1", f"{__name__}:make_sim_env", reset={"seed": "world"}, seed=11, clock_from="sim.t_s",
                    clock={"mode": "sim", "t0": "2026-01-05T00:00:00Z"}, events=ev)
    w = await t.open(d / "w")
    try:
        r = (await w.step("agent", tc(0, "render", {}))).observation.tool_results[0]
        assert r.ok and r.images and w.store.get_bytes(r.images[0].ref) == PNG
        assert (await w.step("agent", tc(1, "seed", {}))).observation.tool_results[0].content == str(derive_seed(11, "env"))
        o = (await w.step("agent", tc(2, "wait", {"hours": 2}))).observation
        assert o.time == "2026-01-05T02:00:00Z" and not o.inbox
        ref = await w.snapshot()
        o = (await w.step("agent", tc(3, "wait", {"hours": 2}))).observation
        assert o.time == "2026-01-05T04:00:00Z" and [m.text for m in o.inbox] == ["shift change"]
        await w.restore(ref)  # replay re-renders the image and re-checks its digest; the clock returns to 2h
        assert w.clock.now_iso() == "2026-01-05T02:00:00Z" and w.apps["env"].env.t == 7200.0
        j = w.apps["env"].journal
        assert j["steps"][0]["images"] == [r.images[0].ref]
    finally:
        await w.close()


CASES = {f.__name__: f for f in (portsim_rewards, restore, generic_env, nondeterministic, cross_runtime, m12_runtime, mcp_image_clock_seed)}

if __name__ == "__main__":
    asyncio.run(CASES[sys.argv[1]](Path(sys.argv[2])))
    print("OK")
