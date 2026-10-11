"""M1 item 10d: simulators served elsewhere (``RemoteControlEnv`` in ``world/apps/control.py``, over ``world/remote.py``).

A point-mass simulator speaks OpenEnv's dm_control protocol over WebSocket: actions ``{"values": [...]}``, observations
``{"observations": {name: [floats]}}``, the state exporting ``action_spec`` / ``observation_spec`` /
``control_timestep``, and the episode ending at a time limit. Checked: spaces come from the server; the World's clock
follows the server's control timestep; the time limit is a truncation; restore rebuilds a new server session by replay;
a neural policy drives it through ``WorldEnv`` with dict observations flattened.
"""

from __future__ import annotations

import json
import random

import pytest

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.runtime.control_task import ControlTask
from crucible.world.gym import WorldEnv
from crucible.world.world import World

websockets = pytest.importorskip("websockets")
DT, LIMIT = 0.02, 50


class PointMassServer:
    async def handler(self, ws):
        x = v = 0.0
        t = 0
        async for raw in ws:
            m = json.loads(raw)
            typ, d = m.get("type"), m.get("data") or {}
            if typ == "close":
                break
            if typ == "reset":
                x, v, t = random.Random(d.get("seed", 0)).uniform(-1, 1), 0.0, 0
                await ws.send(json.dumps({"type": "observation", "data": {"observation": {"observations": {"position": [x], "velocity": [v]}}, "reward": None, "done": False}}))
            elif typ == "step":
                f = max(-1.0, min(1.0, float(d["values"][0])))
                v += f * DT
                x += v * DT
                t += 1
                obs = {"observations": {"position": [round(x, 12)], "velocity": [round(v, 12)]}}
                await ws.send(json.dumps({"type": "observation", "data": {"observation": obs, "reward": 1.0 - min(1.0, abs(x)), "done": t >= LIMIT}}))
            elif typ == "state":
                await ws.send(json.dumps({"type": "state", "data": {"step_count": t, "control_timestep": DT, "physics_timestep": DT / 4,
                                                                     "action_spec": {"shape": [1], "dtype": "float64", "minimum": [-1.0], "maximum": [1.0]},
                                                                     "observation_spec": {"position": {"shape": [1], "dtype": "float64"}, "velocity": {"shape": [1], "dtype": "float64"}}}}))

    async def __aenter__(self):
        from websockets.asyncio.server import serve

        self.server = await serve(self.handler, "127.0.0.1", 0)
        self.url = f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"
        return self

    async def __aexit__(self, *a):
        self.server.close()
        await self.server.wait_closed()


def _spec(url: str) -> WorldSpec:
    return WorldSpec.model_validate({"world": "t/pointmass", "seed": 5, "clock": {"mode": "sim", "t0": "2026-01-05T09:00:00Z"},
                                     "apps": [{"id": "sim", "template": "common/control", "state": {"adapter": "json_doc", "params": {"path": "sim.json"}},
                                               "params": {"client": {"base_url": url}, "frame_skip": 5, "idle_action": [0.0]}}],
                                     "actors": [{"id": "agent", "kind": "policy", "tools": ["sim.*"]}]})


async def _call(w, tool, i=1, **args):
    return (await w.step("agent", ToolCall(tool_call_id=f"c{i}", function_name=tool, arguments=args))).observation.tool_results[0]


async def test_remote_simulator_in_a_world(tmp_path):
    async with PointMassServer() as srv:
        w = World(_spec(srv.url), store=Store(tmp_path / "s"), workdir=tmp_path / "w")
        await w.start()
        try:
            await w.reset()
            app = w.apps["sim"]
            assert app.action_space.low == -1.0 and app.observation_space.kind == "dict" and app.sim_dt == DT
            r = await _call(w, "sim.step", 1, action=[1.0])
            assert r.data["ticks"] == 5 and json.loads(r.content)["time_s"] == 0.1 and w.clock.now().isoformat().endswith("09:00:00.100000+00:00")
            mid = await w.snapshot()
            r1 = await _call(w, "sim.step", 2, action=[-0.5])
            await w.restore(mid)  # a new server session, rebuilt by replay, digests checked
            r2 = await _call(w, "sim.step", 3, action=[-0.5])
            assert r2.data["observation"] == r1.data["observation"]
            r = await _call(w, "sim.wait", 4, ticks=100)
            assert r.ends_episode and r.data["truncated"] and not r.data["terminated"] and app.journal["tick"] == LIMIT  # the time limit
        finally:
            await w.close()


async def test_neural_policy_on_a_remote_simulator(tmp_path):
    async with PointMassServer() as srv:
        task = ControlTask("t/pointmass@1", client={"base_url": srv.url}, return_range=(0.0, float(LIMIT)), journal="lazy", clock=False)
        env = WorldEnv(task, workdir=tmp_path)
        try:
            obs, _ = await env.areset(seed=3)
            assert len(obs) == 2 and env.obs_dim == 2  # dict observation (position, velocity) flattened in key order
            done, n = False, 0
            while not done:
                obs, _r, term, trunc, info = await env.astep([-obs[0] - obs[1]])  # a PD controller toward the origin
                done, n = term or trunc, n + 1
            assert n == LIMIT and info["episode"]["return"] > 25
        finally:
            await env.aclose()

