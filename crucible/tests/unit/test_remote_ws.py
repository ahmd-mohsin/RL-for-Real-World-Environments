"""M1 item 9: a WebSocket client for remote environments (``world/remote.py``; OpenEnv's ``/ws`` protocol), and OpenEnv
client mode on it (``interop/openenv/import_env.ClientEnv``).

A protocol-level fake server checks request pairing, error classes, the desync guard after a timeout, and MCP-style tool
environments. A World over a remote counter environment checks journaling, restore by replay into a new session, and
recovery from a dropped connection mid-episode. The last test runs a real OpenEnv server (``create_app``) from the
separate OpenEnv venv and drives it from this one, which has no OpenEnv installed.
"""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.remote import RemoteActionError, RemoteEnvClient, RemoteEnvError, ws_url
from crucible.world.world import World

websockets = pytest.importorskip("websockets")


class FakeEnvServer:
    """A counter environment speaking OpenEnv's WebSocket protocol; ``mcp=True`` serves list_tools / call_tool actions."""

    def __init__(self, *, mcp: bool = False):
        self.mcp, self.sessions, self.drop_next, self.delay_on_step = mcp, 0, False, None
        self.server, self.port = None, None

    async def handler(self, ws):
        self.sessions += 1
        count, steps = 0, 0
        async for raw in ws:
            m = json.loads(raw)
            t, d = m.get("type"), m.get("data") or {}
            if t == "close":
                break
            if t == "reset":
                count, steps = int(d.get("start", 0)), 0
                await ws.send(json.dumps({"type": "observation", "data": {"observation": {"count": count, "metadata": {}}, "reward": None, "done": False}}))
            elif t == "step":
                steps += 1
                if self.drop_next:
                    self.drop_next = False
                    await ws.close()
                    return
                if self.delay_on_step == steps:
                    self.delay_on_step = None
                    await asyncio.sleep(0.5)
                if self.mcp and d.get("type") == "list_tools":
                    obs = {"tools": [{"name": "add", "description": "add n", "input_schema": {"type": "object", "properties": {"n": {"type": "integer"}}}}]}
                    await ws.send(json.dumps({"type": "observation", "data": {"observation": obs, "reward": None, "done": False}}))
                    continue
                n = (d.get("arguments") or {}).get("n") if self.mcp else d.get("n")
                if not isinstance(n, int):
                    await ws.send(json.dumps({"type": "error", "data": {"message": "n must be an integer", "code": "VALIDATION_ERROR"}}))
                    continue
                if n == 13:
                    await ws.send(json.dumps({"type": "error", "data": {"message": "unlucky", "code": "EXECUTION_ERROR"}}))
                    continue
                count += n
                obs = ({"tool_name": "add", "result": {"content": [{"type": "text", "text": f"count={count}"}]}, "error": None} if self.mcp
                       else {"count": count, "metadata": {}})
                await ws.send(json.dumps({"type": "observation", "data": {"observation": obs, "reward": float(count >= 10), "done": count >= 10}}))
            elif t == "state":
                await ws.send(json.dumps({"type": "state", "data": {"count": count, "step_count": steps}}))
            else:
                await ws.send(json.dumps({"type": "error", "data": {"message": f"Unknown message type: {t}", "code": "UNKNOWN_TYPE"}}))

    async def __aenter__(self):
        from websockets.asyncio.server import serve

        self.server = await serve(self.handler, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *a):
        self.server.close()
        await self.server.wait_closed()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def test_ws_url():
    assert ws_url("http://h:8000") == "ws://h:8000/ws" and ws_url("https://h/") == "wss://h/ws" and ws_url("ws://h/ws") == "ws://h/ws"
    with pytest.raises(ValueError):
        ws_url("h:8000")


async def test_protocol_errors_and_desync_guard():
    async with FakeEnvServer() as srv:
        c = RemoteEnvClient(srv.url, message_timeout_s=0.2)
        assert (await c.reset(start=2))["observation"]["count"] == 2
        assert (await c.step({"n": 3}))["observation"]["count"] == 5
        assert (await c.state())["count"] == 5
        with pytest.raises(RemoteActionError, match="integer"):
            await c.step({"n": "x"})  # the action itself was malformed: the policy's error
        with pytest.raises(RemoteEnvError) as e:
            await c.step({"n": 13})
        assert not isinstance(e.value, RemoteActionError) and e.value.code == "EXECUTION_ERROR"  # the environment failed
        srv.delay_on_step = 4  # the 4th step of this session answers late
        with pytest.raises(TimeoutError):
            await c.step({"n": 1})
        # the late reply must not be read as the next answer: the socket was dropped, the next request is a new session
        r = await c.reset(start=7)
        assert r["observation"]["count"] == 7 and c.session_lost and srv.sessions == 2
        await c.close()
    with pytest.raises(RemoteEnvError, match="cannot connect"):
        await RemoteEnvClient(srv.url, connect_timeout_s=1).reset()


def _spec(url: str, *, mcp: bool = False) -> WorldSpec:
    client = {"base_url": url, **({"mcp": True} if mcp else {})}
    params = {"client": client, "reset": {"start": 1}, **({} if mcp else {"action_schema": {"type": "object", "properties": {"n": {"type": "integer"}}}})}
    return WorldSpec.model_validate({"world": "t/remote", "apps": [{"id": "env", "template": "openenv/env", "state": {"adapter": "json_doc", "params": {"path": "env.json"}},
                                                                     "params": params}],
                                     "actors": [{"id": "agent", "kind": "policy", "tools": ["env.*"]}]})


async def _call(w, tool, i, **args):
    return (await w.step("agent", ToolCall(tool_call_id=f"c{i}", function_name=tool, arguments=args))).observation.tool_results[0]


async def test_world_over_a_remote_env_restore_and_reconnect(tmp_path):
    async with FakeEnvServer() as srv:
        w = World(_spec(srv.url), store=Store(tmp_path / "s"), workdir=tmp_path / "w")
        await w.start()
        try:
            await w.reset()
            assert json.loads((await _call(w, "env.step", 1, n=2)).content)["count"] == 3
            mid = await w.snapshot()
            r = await _call(w, "env.step", 2, n="x")
            assert r.error == "invalid_arguments"  # the World's own schema check stops it first
            r = await _call(w, "env.step", 2, m=1)  # passes the declared schema, refused by the server
            assert r.error == "invalid_action" and r.fault == "policy"  # the policy's error, not journaled
            await _call(w, "env.step", 3, n=4)  # count 7
            await w.restore(mid)  # a new server session, replayed from the journal, digests verified
            assert json.loads((await _call(w, "env.step", 4, n=1)).content)["count"] == 4
            sessions = srv.sessions
            srv.drop_next = True  # the server kills the socket on the next step
            r = await _call(w, "env.step", 5, n=6)
            assert r.ok and r.ends_episode and json.loads(r.content)["count"] == 10  # recovered: replayed, then retried
            assert srv.sessions == sessions + 1 and not w.failures
        finally:
            await w.close()


async def test_remote_mcp_environment(tmp_path):
    async with FakeEnvServer(mcp=True) as srv:
        w = World(_spec(srv.url, mcp=True), store=Store(tmp_path / "s"), workdir=tmp_path / "w")
        await w.start()
        try:
            await w.reset()
            assert [t.qualified for t in w.tools_for("agent")] == ["env.add"]  # tools discovered over the socket
            r = await _call(w, "env.add", 1, n=4)
            assert r.ok and r.content == "count=5"
        finally:
            await w.close()


OPENENV_PY = Path(__file__).resolve().parents[2] / ".envs" / "openenv" / "bin" / "python"
SERVER = '''
import sys, uvicorn
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State

class AddAction(Action):
    n: int

class CountObs(Observation):
    count: int = 0

class Counter(Environment):
    def __init__(self):
        super().__init__()
        self.count, self._state = 0, State(episode_id="e", step_count=0)
    def reset(self, seed=None, episode_id=None, start: int = 0, **kw):
        self.count = start
        return CountObs(count=self.count, done=False, reward=None)
    def step(self, action, **kw):
        self.count += action.n
        self._state.step_count += 1
        return CountObs(count=self.count, done=self.count >= 10, reward=float(self.count >= 10))
    @property
    def state(self):
        return self._state

app = create_app(Counter, AddAction, CountObs, env_name="counter")
uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
'''


@pytest.mark.skipif(not OPENENV_PY.exists(), reason="OpenEnv venv not built (crucible/.envs/openenv)")
async def test_real_openenv_server(tmp_path):
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        port = so.getsockname()[1]
    (tmp_path / "srv.py").write_text(SERVER)
    proc = subprocess.Popen([str(OPENENV_PY), str(tmp_path / "srv.py"), str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)  # noqa: ASYNC220
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                if proc.poll() is not None:
                    pytest.fail(proc.stderr.read().decode()[-1000:])
                await asyncio.sleep(0.1)
        w = World(_spec(f"http://127.0.0.1:{port}"), store=Store(tmp_path / "s"), workdir=tmp_path / "w")
        await w.start()
        try:
            await w.reset()
            r = await _call(w, "env.step", 1, n=4)
            assert r.ok and json.loads(r.content)["count"] == 5
            r = await _call(w, "env.step", 2, m=1)
            assert r.error == "invalid_action" and r.fault == "policy"  # OpenEnv's own validation error, classified
            r = await _call(w, "env.step", 3, n=5)
            assert r.ends_episode and r.data["reward"] == 1.0
        finally:
            await w.close()
    finally:
        proc.terminate()
        proc.wait(10)
    assert sys.modules.get("openenv") is None  # this side never imported OpenEnv
