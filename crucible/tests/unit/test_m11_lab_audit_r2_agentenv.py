"""M11 lab audit R-2: agentenv's gateway port is fenced from policy processes, a signed actor cannot be replayed, and the
agent is cut off before every grade.

- The gateway's control routes have no authentication (agentenv ``gateway.py:189-341``); ``gateway_bind`` bound the gateway
  itself, so an agent container on docker0 could re-enable a revoked tool or move the clock. Now the gateway listens on
  loopback only and each actor's ``AgentProxy`` (``/mcp`` only, role pinned) binds ``gateway_bind``.
- ``sign_actor`` was a static HMAC with no nonce, logged in the gateway's trajectory: a reader could replay it on the
  unauthenticated App-server port and call ``App.call`` directly. Now tokens are tool-bound and the World's own are
  single-use; a failing token is refused.
- ``A2AHarness`` and ``crucible_verify`` graded while the agent could still call tools. Now ``World.cut_policy`` makes every
  policy actor done (a later call is a refused, recorded ``late_call``), the World is quiesced, the A2A task cancelled in
  every terminal state, and ``quiesce.json`` records it.

The routing, token and World tests run without agentenv; the end-to-end ones through a real gateway need it."""

from __future__ import annotations

import asyncio
import json
import os
import socket
from types import SimpleNamespace

import httpx
import pytest

from crucible.core.schema.actions import Done, ToolCall
from crucible.interop.agentenv import world_backend as wb
from crucible.interop.agentenv.cut import late_call_flag, pre_grade_cut, receipt
from crucible.interop.agentenv.world_backend import AgentEnvBackend
from crucible.interop.replay import from_world_trace
from crucible.world.world import WorldError
from tests.unit.test_m10_lab_audit_r1_a2a import COMPLETED, Env, FakeTask, proto  # noqa: F401 -- proto is a fixture


def _backend(**params) -> AgentEnvBackend:
    """An unstarted backend: only the fields the code under test reads."""
    b = AgentEnvBackend.__new__(AgentEnvBackend)
    b.params, b._secret, b._spent, b.gateway_port, b.proxies = dict(params), os.urandom(32), set(), 45678, {}
    b.sandbox_spec, b.spec = None, SimpleNamespace(world="w", isolation=SimpleNamespace(policy_user=False))
    return b


def _call(n: int, fn: str = "tickets.list_tickets", **args) -> ToolCall:
    return ToolCall(tool_call_id=f"c{n}", function_name=fn, arguments=args)


async def _world(tmp_path):
    from crucible_pack_toy.runtime import HelpdeskTask

    return await HelpdeskTask().open(tmp_path / "w")


# ------------------------------------------------------------------------------------------- gateway bind (no agentenv)

async def test_the_gateway_listens_on_loopback_whatever_gateway_bind_says(tmp_path, monkeypatch):
    b = _backend(gateway_bind="172.17.0.1")
    b.servers, b.workdir = {}, tmp_path
    seen: dict = {}

    async def spawn(*argv, env, log):
        seen.update(env)
        raise RuntimeError("stop here")

    monkeypatch.setattr(wb, "spawn_logged", spawn)
    with pytest.raises(RuntimeError, match="stop here"):
        await b._start_gateway()
    assert seen["MCP_HOST"] == "127.0.0.1" and b.gateway_url == "http://127.0.0.1:45678"


async def test_only_the_agent_proxy_binds_gateway_bind(monkeypatch):
    made: list = []

    class Proxy:
        def __init__(self, upstream, actor, *, host):
            made.append((upstream, actor, host))
            self.mcp_url = f"http://{host}:1/mcp"

        async def start(self):
            pass

    monkeypatch.setattr(wb, "AgentProxy", Proxy)
    b = _backend(gateway_bind="172.17.0.1")
    b._world = SimpleNamespace(spec=SimpleNamespace(policy_actors=[SimpleNamespace(id="agent")]))
    assert await b.agent_mcp_url() == "http://172.17.0.1:1/mcp"
    assert made == [("http://127.0.0.1:45678", "agent", "172.17.0.1")]  # the proxy forwards to the loopback gateway


def test_gateway_isolation_names_what_fences_the_port(monkeypatch):
    b = _backend()
    monkeypatch.setattr(wb.sys, "platform", "linux")
    assert b.gateway_isolation == "bound_loopback"
    b.isolation = SimpleNamespace(egress=None)  # sandboxed, but no egress probe ran (a "public" world): not proven
    assert b.gateway_isolation == "bound_loopback"
    b.isolation = SimpleNamespace(egress={"isolated": True, "tested": [["172.17.0.1", 45678]]})
    assert b.gateway_isolation == "sandboxed"
    b.isolation = SimpleNamespace(egress={"isolated": True, "tested": [["172.17.0.1", 1]]})  # the probe never tried this port
    assert b.gateway_isolation == "bound_loopback"
    b.isolation = None
    monkeypatch.setattr(wb.sys, "platform", "darwin")  # Docker Desktop: host.docker.internal reaches the host's loopback
    assert b.gateway_isolation == "exposed"


# ------------------------------------------------------------------------------------------- actor tokens (no agentenv)

def test_a_world_call_token_is_single_use_and_bound_to_its_tool_and_arguments():
    b = _backend()
    args = {"order_id": "O-5001", "amount_usd": 129.99, "items": (1, 2)}
    t = b.sign_actor("agent", "tickets__issue_refund", args=args)
    assert b.verify_actor(t, "tickets__list_tickets", args) is None  # another tool: refused, and not spent by the attempt
    # a reader racing the gateway (it logs the token before it forwards the call) cannot change the arguments
    assert b.verify_actor(t, "tickets__issue_refund", {**args, "amount_usd": 9999.0}) is None
    wire = json.loads(json.dumps({**args, "__actor": t}))  # what the App server receives: JSON, the token among the arguments
    assert b.verify_actor(t, "tickets__issue_refund", wire) == "agent"
    assert b.verify_actor(t, "tickets__issue_refund", wire) is None  # a replay (read from the gateway's trajectory)


def test_a_trigger_token_is_reusable_for_its_one_tool_only():
    b = _backend()
    t = b.sign_actor("crucible-system", "tickets__get_ticket", once=False)
    assert [b.verify_actor(t, "tickets__get_ticket") for _ in range(3)] == ["crucible-system"] * 3  # the gateway checks again
    assert b.verify_actor(t, "tickets__issue_refund") is None


def test_forged_and_legacy_tokens_are_refused():
    b, other = _backend(), _backend()
    good = other.sign_actor("boss", "mail__send_email")  # another World's secret
    for v in (None, "boss", "boss:" + "0" * 32, "boss:1abc:" + "0" * 32, ":1abc:" + "0" * 32, good, 7):
        assert b.verify_actor(v, "mail__send_email") is None, v
    t = b.sign_actor("a:b", "x__y")  # an actor id with a colon still round-trips
    assert b.verify_actor(t, "x__y") == "a:b"
    t = b.sign_actor("agent", "x__y", once=False)
    assert b.verify_actor(t, "x__y", {"any": "templated arguments"}) == "agent"  # a trigger's: tool-bound, not argument-bound


# ------------------------------------------------------------------------------------------- the cut on a World (no agentenv)

async def test_a_call_after_the_cut_is_refused_and_recorded_as_a_late_call(tmp_path):
    w = await _world(tmp_path)
    try:
        assert (await w.step("agent", _call(1))).observation.tool_results[0].ok
        cut = await pre_grade_cut(w)
        assert cut["actor_done"] == ["agent"] and cut["quiesced"] is False  # the local backend has no sandbox to quiesce
        with pytest.raises(WorldError, match="late call"):
            await w.step("agent", _call(2, "tickets.issue_refund", order_id="O-5001", amount_usd=1.0))
        assert [c["action"] for c in w.late_calls] == ["tickets.issue_refund"]
        assert any(e.type == "late_call" for e in w.log) and any(e.type == "policy_cut" for e in w.log)
        assert [a.tool for a in from_world_trace(w.to_trace())] == ["tickets.list_tickets"]  # a late call is never replayed
        doc = receipt(w, {**cut, "a2a_state": "completed"}, tmp_path / "v" / "quiesce.json")
        assert json.loads((tmp_path / "v" / "quiesce.json").read_text()) == doc
        assert doc["late_calls"][0]["actor"] == "agent" and doc["a2a_state"] == "completed"
        assert late_call_flag(doc)["rule"] == "policy_late_call"
    finally:
        await w.close()


async def test_the_cut_waits_for_a_step_in_flight(tmp_path):
    w = await _world(tmp_path)
    try:
        await w._step_lock.acquire()  # a step is running
        task = asyncio.create_task(w.cut_policy())
        await asyncio.sleep(0.05)
        assert not task.done() and "agent" not in w.done
        w._step_lock.release()
        assert await task == ["agent"] and "agent" in w.done
    finally:
        await w.close()


async def test_lifting_the_cut_reopens_only_what_the_cut_closed(tmp_path):
    w = await _world(tmp_path)
    try:
        await w.cut_policy()
        w.lift_cut()
        assert (await w.step("agent", _call(1))).observation.tool_results[0].ok  # the next episode on a reused World acts
        await w.step("agent", Done(reason="finished"))
        await w.cut_policy()
        w.lift_cut()
        assert "agent" in w.done  # it declared done itself: still done
        with pytest.raises(WorldError, match="already declared done"):
            await w.step("agent", _call(2))
        assert w.late_calls == []
    finally:
        await w.close()


async def test_a_new_task_step_ends_the_cut(tmp_path):
    w = await _world(tmp_path)
    try:
        await w.cut_policy()
        await w.begin_task_step("second", 1, "go on")
        assert (await w.step("agent", _call(1))).observation.tool_results[0].ok
        await w.step("agent", Done(reason="finished"))
        with pytest.raises(WorldError, match="already declared done"):  # its own Done, not a late call
            await w.step("agent", _call(2))
        assert w.late_calls == []
    finally:
        await w.close()


async def test_a_restore_ends_the_cut(tmp_path):
    w = await _world(tmp_path)
    try:
        ref = await w.snapshot()
        await w.cut_policy()
        await w.restore(ref)
        assert (await w.step("agent", _call(1))).observation.tool_results[0].ok
    finally:
        await w.close()


# ------------------------------------------------------------------------------------------- A2AHarness (fake agentenv protocol)

async def test_the_a2a_harness_cuts_before_the_grade_and_cancels_a_completed_task(tmp_path, monkeypatch, proto):  # noqa: F811
    env = Env(tmp_path, monkeypatch)

    async def poll():
        return COMPLETED

    proto.script["poll"] = poll
    worlds: list = []
    task = FakeTask()
    opened = task.open

    async def open_(workdir):
        w = await opened(workdir)
        worlds.append(w)
        return w

    async def on_grade():
        w = worlds[0]
        w.events.append("grade")
        w.late_calls.append({"actor": "agent", "action": "tickets.issue_refund", "at": None})  # the World refused it

    cancel = proto.cancel_a2a_task

    async def cancel_logged(url, task_id):
        worlds[0].events.append("cancel")
        await cancel(url, task_id)

    proto.cancel_a2a_task = cancel_logged
    task.open, task.on_grade = open_, on_grade
    res = await env.run(env.harness(), task)
    assert res.failure is None and res.reward == 1.0
    assert worlds[0].events == ["cut", "quiesce", "cancel", "grade"]  # the cut, then the cancel, then the grade
    assert proto.calls["cancel"] == ["t1"]  # a completed task is cancelled too (only timeout/unreachable were)
    q = json.loads((tmp_path / "a1" / "quiesce.json").read_text())
    assert q["a2a_state"] == "completed" and q["a2a_cancel_sent"] is True and q["actor_done"] == ["agent"] and q["gateway"] == "bound_loopback"
    assert [c["action"] for c in q["late_calls"]] == ["tickets.issue_refund"]
    assert res.grade["quiesce"]["late_calls"] == 1 and [f["rule"] for f in res.flags] == ["policy_late_call"]
    assert res.artifacts["quiesce"] == "quiesce.json"


async def test_processes_that_survive_the_quiesce_end_the_attempt_scored_after_the_cancel_and_the_receipt(tmp_path, monkeypatch, proto):  # noqa: F811
    env = Env(tmp_path, monkeypatch)

    async def poll():
        return COMPLETED

    proto.script["poll"] = poll
    task = FakeTask()
    opened = task.open

    async def open_(workdir):
        w = await opened(workdir)

        async def quiesce():
            raise WorldError("policy processes survived quiesce: {'ok': False, 'left': [4242]}")

        w.quiesce = quiesce
        return w

    task.open = open_
    res = await env.run(env.harness(), task)
    assert res.failure is not None and res.failure.kind == "crucible:policy_misbehavior" and not res.failure.mask and task.graded == 0
    assert [f["rule"] for f in res.flags] == ["policy_process_survived"] and proto.calls["cancel"] == ["t1"]
    q = json.loads((tmp_path / "a1" / "quiesce.json").read_text())
    assert "survived" in q["quiesce_error"] and q["quiesced"] is False


async def test_a_timed_out_task_is_not_cancelled_twice(tmp_path, monkeypatch, proto):  # noqa: F811
    env = Env(tmp_path, monkeypatch)

    async def poll():
        raise TimeoutError

    proto.script["poll"] = poll
    res = await env.run(env.harness(), FakeTask())
    assert proto.calls["cancel"] == ["t1"]  # _ask's own cancel only
    assert json.loads((tmp_path / "a1" / "quiesce.json").read_text())["a2a_state"] == "timeout"
    assert res.grade["quiesce"]["late_calls"] == 0 and res.flags == []


# ------------------------------------------------------------------------------------------- through a real gateway (agentenv)

def _lan_ip() -> str | None:
    """A non-loopback address of this host (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            ip = s.getsockname()[0]
    except OSError:
        return None
    return None if ip.startswith("127.") else ip


async def test_the_gateway_port_is_unreachable_off_loopback_and_the_proxy_still_serves_mcp(tmp_path):
    pytest.importorskip("agent_env")
    from crucible_pack_toy.runtime import HelpdeskTask

    ip = _lan_ip()
    if ip is None:
        pytest.skip("no non-loopback address on this host")
    from crucible_pack_toy.runtime import PACK_ROOT, helpdesk_spec, responders

    from crucible.core.store import Store
    from crucible.world.world import World

    def factory(workdir):
        w = World(helpdesk_spec(), instructions={"agent": "x"}, store=Store(workdir / "s"), workdir=workdir / "w", pack_root=PACK_ROOT,
                  backend="agentenv", backend_params={"gateway_bind": ip})
        for r in responders():
            w.attach(r)
        return w

    w = await HelpdeskTask(world_factory=factory, backend="agentenv").open(tmp_path / "w")
    try:
        port = w.backend.gateway_port
        async with httpx.AsyncClient(timeout=10) as c:
            for method, path, body in (("POST", "/tools/enable", {"role": "agent", "tools": "*"}),
                                       ("PUT", "/clock/set-time", {"virtual_time": "2030-01-01T00:00:00Z"}), ("GET", "/trajectory", None)):
                with pytest.raises(httpx.ConnectError):
                    await c.request(method, f"http://{ip}:{port}{path}", json=body)
            url = await w.backend.agent_url("agent")
            assert url.startswith(f"http://{ip}:")
            r = await c.post(f"{url}/step", json={"action": "list_tools"})
            assert r.status_code == 200 and r.json()["tools"]
            assert (await c.get(f"{url}/trajectory")).status_code == 404
    finally:
        await w.close()


async def test_a_replayed_actor_token_on_an_app_server_port_is_refused(tmp_path):
    pytest.importorskip("agent_env")
    from mcp.client.session import ClientSession

    from crucible.world.packaging.mcp_compat import http_streams
    from crucible_pack_toy.runtime import HelpdeskTask

    w = await HelpdeskTask(backend="agentenv").open(tmp_path / "w")
    try:
        assert (await w.step("agent", _call(1))).observation.tool_results[0].ok
        traj = (await w.backend.http.get("/trajectory")).text
        tokens = {v for v in traj.split('"') if v.startswith("agent:1")}  # the signed actors a trajectory reader sees
        assert tokens, "the World's own call carries a signed actor through the gateway"
        srv = w.backend.servers["tickets"]
        async with http_streams(srv.url, headers={}, timeout_s=30) as (r, wr), ClientSession(r, wr) as s:
            await s.initialize()
            res = await s.call_tool("tickets__list_tickets", {"__actor": next(iter(tokens))})
        assert res.isError and "refused" in res.content[0].text
        assert any(e.type == "actor_token_refused" for e in w.log)
    finally:
        await w.close()


async def test_a_late_call_through_the_agent_proxy_is_refused_and_the_grade_is_the_state_at_the_cut(tmp_path):
    pytest.importorskip("agent_env")
    from crucible_pack_toy.runtime import HelpdeskTask
    from crucible_pack_toy.verify import GOLD

    task = HelpdeskTask(backend="agentenv")
    w = await task.open(tmp_path / "w")
    try:
        url = await w.backend.agent_url("agent")
        async with httpx.AsyncClient(timeout=60) as c:
            async def call(a):
                return await c.post(f"{url}/step", json={"action": "call_tool", "tool_name": a.function_name.replace(".", "__", 1), "arguments": a.arguments})

            for a in GOLD[:2]:
                assert (await call(a)).status_code == 200
            before = await task.grade(w, tmp_path / "g0")
            cut = await pre_grade_cut(w)
            for a in GOLD[2:]:  # the agent keeps going after it reported completion
                r = await call(a)
                assert r.status_code == 200 and r.json().get("isError")
            after = await task.grade(w, tmp_path / "g1")  # a re-grade sees no new policy call either
        assert after.reward == before.reward < 1.0
        doc = receipt(w, cut, tmp_path / "quiesce.json")
        assert [c["action"] for c in doc["late_calls"]] == [a.function_name for a in GOLD[2:]]
    finally:
        await w.close()
