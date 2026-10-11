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


OPENENV_ENVS = Path(__file__).resolve().parents[3] / "RL-Environments" / "06-frameworks-and-env-code" / "meta-pytorch-openenv" / "envs"
CATALOG = {
    "grid_world": ("grid_world_env.server.grid_world_environment:GridWorldEnvironment", "grid_world_env.models:GridWorldAction",
                   [{"action": a} for a in ["RIGHT"] * 4 + ["DOWN"] * 4]),
    "connect4": ("connect4_env.server.connect4_environment:Connect4Environment", "connect4_env.models:Connect4Action",
                 [{"column": c} for c in [0, 1, 0, 1, 0, 1, 0]]),
    "maze": ("maze_env.server.maze_env_environment:MazeEnvironment", "maze_env.models:MazeAction", [{"action": a} for a in [1, 1, 3, 3, 1, 1, 3, 3]]),
    "wildfire": ("wildfire_env.server.wildfire_environment:WildfireEnvironment", "wildfire_env.models:WildfireAction",
                 [{"action": "water", "x": 5, "y": 5}, {"action": "break", "x": 4, "y": 4}, {"action": "wait"}] * 3),
}


async def openenv_catalog(d: Path) -> None:
    """OpenEnv's own environments (games, a maze, a stochastic fire simulation) through the generic importer: the same
    actions give the native rewards and termination; environments with a bare ``reset()`` work (signature-aware calls);
    a mid-episode restore of the simulation replays deterministically."""
    from crucible.interop.openenv.import_env import load

    sys.path.insert(0, str(OPENENV_ENVS))
    for name, (fac, act, acts) in CATALOG.items():
        env = load(fac)()
        env.reset()
        native, done = [], False
        for a in acts:
            o = env.step(load(act).model_validate(a))
            native.append(float(o.reward or 0.0))
            if o.done:
                done = True
                break
        t = OpenEnvTask(f"openenv/{name}/t@1", fac, action_cls=act, reward_mode="sum", reward_range=(-20.0, 20.0))
        w = await t.open(d / name)
        try:
            for i, a in enumerate(acts[:3]):
                await w.step("agent", tc(i, "step", a))
            if name == "wildfire":
                ref = await w.snapshot()
                await w.step("agent", tc(99, "step", acts[3]))
                await w.restore(ref)  # replay + digest check of a stochastic simulation
                assert len(w.apps["env"].journal["steps"]) == 3
            for i, a in enumerate(acts[3:], start=3):
                if w.apps["env"].journal["done"]:
                    break
                await w.step("agent", tc(i, "step", a))
            j = w.apps["env"].journal
            assert j["done"] == done and [round(r or 0.0, 9) for r in j["rewards"]] == [round(r, 9) for r in native], (name, j["rewards"], native)
        finally:
            await w.close()


async def two_player(d: Path) -> None:
    """m13 E pick 8: connect4 as a true two-player World. 100 seeded games of random legal moves: the imported reward and
    termination equal the native environment's; the outcome is attributed per seat (zero-sum, the winning mover +1); an
    out-of-turn move is refused and never executed; the game's end stops both seats; a mid-game restore replays."""
    import random

    from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode
    from crucible.interop.openenv.import_env import OpenEnvTask, load

    sys.path.insert(0, str(OPENENV_ENVS))
    fac, act, _ = CATALOG["connect4"]
    wins = 0
    for seed in range(100):
        rng = random.Random(seed)
        env = load(fac)()
        o = env.reset()
        moves = []
        while True:
            c = rng.choice(o.legal_actions)
            moves.append(c)
            o = env.step(load(act).model_validate({"column": c}))
            if o.done:
                break
        native_reward, mover = float(o.reward), ("p1" if len(moves) % 2 == 1 else "p2")
        t = OpenEnvTask("openenv/connect4/duel@1", fac, action_cls=act, players=["p1", "p2"], turn_map={1: "p1", -1: "p2"}, reward_range=(-1.0, 1.0))
        w = await t.open(d / f"g{seed}")
        try:
            if seed == 0:  # p2 cannot move first: refused, nothing executed or journaled
                r = await w.step("p2", tc(0, "step", {"column": 3}))
                assert r.observation.tool_results[0].error == "not_your_turn" and not w.apps["env"].journal["steps"]
            pol = {"p1": ScriptedPolicy("p1", [tc(i, "step", {"column": c}) for i, c in enumerate(moves[0::2])]),
                   "p2": ScriptedPolicy("p2", [tc(i, "step", {"column": c}) for i, c in enumerate(moves[1::2])])}
            ep = await run_episode(w, pol, limits=EpisodeLimits(max_steps=60), reset=False)
            j = w.apps["env"].journal
            assert j["done"] and j["final_reward"] == native_reward and len(j["steps"]) == len(moves), (seed, j["final_reward"], native_reward)
            assert [s_["actor"] for s_ in j["steps"]] == [("p1" if i % 2 == 0 else "p2") for i in range(len(moves))]
            assert w.done == {"p1", "p2"} and ep.done  # the game's end ends both seats
            ar = t.actor_rewards(j)
            assert ar[mover] == native_reward and ar["p2" if mover == "p1" else "p1"] == -native_reward
            wins += native_reward == 1.0
            rep = await t.grade(w, d / f"grade{seed}")
            assert next(c for c in rep.checks if c.id == "env_reward").detail["actor_rewards"] == ar  # the grade carries the per-seat outcome
        finally:
            await w.close()
    assert wins > 50  # random play mostly ends in a win, not a draw
    t = OpenEnvTask("openenv/connect4/duel@1", fac, action_cls=act, players=["p1", "p2"], turn_map={1: "p1", -1: "p2"})
    w = await t.open(d / "restore")
    try:
        for i, (who, c) in enumerate([("p1", 3), ("p2", 3), ("p1", 4)]):
            await w.step(who, tc(i, "step", {"column": c}))
        ref = await w.snapshot()
        await w.step("p2", tc(9, "step", {"column": 0}))
        await w.restore(ref)  # replay of the journal, digests verified; the turn is p2's again
        assert len(w.apps["env"].journal["steps"]) == 3 and w.apps["env"].turn == "p2"
        r = await w.step("p1", tc(10, "step", {"column": 0}))
        assert r.observation.tool_results[0].error == "not_your_turn"
    finally:
        await w.close()


def _start_connect4_server():
    """OpenEnv's own connect4 server (FastAPI + WebSocket) in a separate process; returns (process, port)."""
    import os as _os
    import socket
    import subprocess
    import time

    import httpx

    with socket.socket() as s_:
        s_.bind(("127.0.0.1", 0))
        port = s_.getsockname()[1]
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "connect4_env.server.app:app", "--port", str(port), "--log-level", "warning"],
                           env={**_os.environ, "PYTHONPATH": str(OPENENV_ENVS)}, cwd=str(OPENENV_ENVS), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code < 500:
                break
        except httpx.HTTPError:
            time.sleep(0.2)
    return srv, port


async def client_mode(d: Path) -> None:
    """m13 E (G7): an OpenEnv environment reached through its own server (OpenEnv's GenericEnvClient), not in-process:
    connect4's real FastAPI/WebSocket server in a separate process; the same moves give the native rewards and
    termination; restore replays over the wire (reset + re-issue, digests verified)."""
    from crucible.interop.openenv.import_env import OpenEnvTask, load

    sys.path.insert(0, str(OPENENV_ENVS))
    fac, act, moves = CATALOG["connect4"]
    env = load(fac)()
    env.reset()
    native = []
    for a in moves:
        o = env.step(load(act).model_validate(a))
        native.append(float(o.reward or 0.0))
        if o.done:
            break
    srv, port = _start_connect4_server()
    try:
        schema = {"type": "object", "properties": {"column": {"type": "integer"}}, "required": ["column"]}
        t = OpenEnvTask("openenv/connect4/remote@1", "", client={"base_url": f"http://127.0.0.1:{port}"}, action_schema=schema, reward_mode="sum",
                        reward_range=(-1.0, 1.0))
        w = await t.open(d / "remote")
        try:
            for i, a in enumerate(moves[:3]):
                await w.step("agent", tc(i, "step", a))
            ref = await w.snapshot()
            await w.step("agent", tc(50, "step", {"column": 6}))
            await w.restore(ref)  # a fresh server session, reset, the 3 journaled moves re-issued and their digests checked
            assert len(w.apps["env"].journal["steps"]) == 3
            for i, a in enumerate(moves[3:], start=3):
                if w.apps["env"].journal["done"]:
                    break
                await w.step("agent", tc(i, "step", a))
            j = w.apps["env"].journal
            assert j["done"] and [r or 0.0 for r in j["rewards"]] == native, (j["rewards"], native)
        finally:
            await w.close()
    finally:
        srv.kill()
        srv.wait()


async def image_fields(d: Path) -> None:
    """m13 E (G7): pixels in a non-MCP observation field become ToolResult.images (stored, in the replay digest), and the
    policy's text shows a placeholder, not base64."""
    import base64

    from crucible.interop.openenv.import_env import OpenEnvTask

    t = OpenEnvTask("openenv/pixels/t@1", "_openenv_import_cases:PixelEnv", action_cls="_openenv_import_cases:PixelAction", image_fields=["screenshot"],
                    reward_mode="sum", reward_range=(0.0, 10.0))
    w = await t.open(d / "px")
    try:
        r = (await w.step("agent", tc(0, "step", {"shade": 3}))).observation.tool_results[0]
        assert r.ok and len(r.images) == 1 and r.images[0].mime == "image/png" and "<image:screenshot>" in r.content and "iVBOR" not in r.content
        ref = await w.snapshot()
        await w.step("agent", tc(1, "step", {"shade": 9}))
        await w.restore(ref)  # replay re-renders the frame and re-checks its digest
        assert w.apps["env"].journal["steps"][0]["images"] == [r.images[0].ref]
        _ = base64
    finally:
        await w.close()


try:  # a pixel-observation environment for the image_fields case (OpenEnv types exist only inside crucible/.envs/openenv)
    from openenv.core.env_server.interfaces import Environment as _Env
    from openenv.core.env_server.types import Action as _Action
    from openenv.core.env_server.types import Observation as _Obs

    class PixelAction(_Action):
        shade: int = 0

    class PixelObservation(_Obs):
        screenshot: str = ""
        caption: str = ""

    class PixelEnv(_Env):
        def __init__(self):
            super().__init__()
            self.n = 0

        @staticmethod
        def _png(shade: int) -> str:
            import base64
            import struct
            import zlib

            raw = b"".join(b"\x00" + bytes([shade * 25 % 256] * 3) * 4 for _ in range(4))
            chunk = lambda t, data: struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data))
            png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
            return base64.b64encode(png).decode()

        def reset(self):
            self.n = 0
            return PixelObservation(screenshot=self._png(0), caption="start", reward=0.0, done=False)

        def step(self, action: PixelAction):
            self.n += 1
            return PixelObservation(screenshot=self._png(action.shade), caption=f"shade {action.shade}", reward=float(action.shade), done=self.n >= 5)

        @property
        def state(self):
            return {"n": self.n}
except ImportError:  # pragma: no cover - outside the OpenEnv environment
    pass


CASES = {f.__name__: f for f in (portsim_rewards, restore, generic_env, nondeterministic, cross_runtime, m12_runtime, mcp_image_clock_seed,
                                 openenv_catalog, two_player, client_mode, image_fields)}

if __name__ == "__main__":
    asyncio.run(CASES[sys.argv[1]](Path(sys.argv[2])))
    print("OK")
