"""A remote World or grader host that stops answering (``world.hostreach``, 2026-10-26): when this Mac lost its network, every
in-flight kernel episode's tool calls hung until the 9,000 s attempt clock and all 8 episodes of a measurement were masked
after $29.5 of sampling. Here: a hung call is cut fast with a typed error, the episode pauses with its clocks held and
resumes the same episode when the host is back (a replayable call runs again, a policy's cut write comes back as
interrupted), a host that stays down masks the attempt as ``host_unreachable``, a launcher holds new episodes, and the
reason reaches the record, the run journal, the swarm journal and the framework-health clusters.

Every host here is a fake transport: a probe and a Docker CLI that read one ``Net`` switch. No network, no Docker, no model."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world import hostreach
from crucible.world.exec import HOST_UNREACHABLE, DockerExec, ExecOut, HostUnreachable, LocalExec
from crucible.world.hostreach import HostReach, ReachRule

FAST = ReachRule(probe_timeout_s=0.1, heartbeat_s=0.05, misses=2, fresh_s=0.0, pause_s=5.0, backoff_s=0.02, backoff_cap_s=0.05)


class Net:
    """The fake network: down until ``up_at`` (monotonic), or for good with ``up_at = inf``."""

    def __init__(self, down_for: float = 0.0):
        self.up_at = time.monotonic() + down_for

    @property
    def up(self) -> bool:
        return time.monotonic() >= self.up_at

    def drop(self, for_s: float) -> None:
        self.up_at = time.monotonic() + for_s

    async def probe(self) -> bool:
        if not self.up:
            await asyncio.sleep(10)  # a lost network: the probe hangs, its own timeout ends it
        return True


def host(net: Net, rule: ReachRule = FAST, key: str = "docker:ssh://lab", name: str = "lab") -> HostReach:
    return hostreach.register(key, HostReach(name, net.probe, rule))


class FakeDocker(DockerExec):
    """A ``DockerExec`` on a remote daemon whose CLI reads ``net``: while it is down a call either fails to connect (``mode``
    "refuse") or hangs (``mode`` "hang", the tunnel stalled mid-command)."""

    def __init__(self, net: Net, mode: str = "refuse"):
        super().__init__("c1", docker_host="ssh://lab")
        self.net, self.mode, self.calls = net, mode, 0

    async def _cli(self, argv, *, timeout_s, stdin):
        self.calls += 1
        if not self.net.up:
            if self.mode == "hang":
                await asyncio.sleep(3600)
            return ExecOut(1, "", "Cannot connect to the Docker daemon at unix:///tmp/crucible-docker.sock. Is the docker daemon running?")
        return ExecOut(0, "ok\n")


# ---- fail fast ---------------------------------------------------------------------------------------------------------------
def test_the_default_rule_cuts_a_hung_call_within_about_a_minute_and_pauses_at_most_30_minutes():
    r = ReachRule()
    assert r.detect_s <= 60 and r.pause_s == 1800
    from crucible.factory.lanes import kernel_episode as ke

    argv = " ".join(ke.host_of("rtx5090").ssh_argv("true"))
    assert "ConnectTimeout=15" in argv and "ServerAliveInterval=15" in argv and "ServerAliveCountMax=4" in argv  # a dead session: 60 s


async def test_a_hung_call_is_cut_with_a_typed_error_once_the_host_stops_answering():
    net = Net(down_for=float("inf"))
    reach = host(net)
    t0 = time.monotonic()
    with pytest.raises(HostUnreachable) as ei:
        await reach.guard(lambda: asyncio.sleep(3600), what="docker exec")
    assert time.monotonic() - t0 < 2 and ei.value.dispatched and ei.value.host == "lab" and reach.down
    assert str(ei.value).startswith(HOST_UNREACHABLE) and ei.value.failure_kind == "transport_unreachable" and ei.value.detail == HOST_UNREACHABLE


async def test_a_slow_call_on_a_host_that_answers_is_never_cut():
    reach = host(Net())
    assert await reach.guard(lambda: asyncio.sleep(0.3, result="done")) == "done" and not reach.down


async def test_a_failed_command_on_a_reachable_host_is_the_command_s_own():
    """The words alone never decide: a CLI error on a host that answers its probe stays the command's result."""
    reach = host(Net())
    assert not await reach.confirm("Cannot connect to the Docker daemon") and not reach.down


# ---- pause, then retry -----------------------------------------------------------------------------------------------------------
async def test_an_outage_holds_every_registered_clock_and_gives_back_what_it_had_left():
    net = Net(down_for=0.6)
    reach = host(net)
    async with asyncio.timeout(0.3) as attempt_clock:  # shorter than the outage: it would fire without the hold
        with hostreach.clock(attempt_clock), hostreach.episode() as led:
            assert await reach.outage("lost", what="test")
            left = attempt_clock.when() - asyncio.get_running_loop().time()
    assert 0.2 < left <= 0.3 and not reach.down
    assert led.rows and led.rows[0]["resumed"] and led.paused_s >= 0.5 and led.to_json()["reason"] == HOST_UNREACHABLE


async def test_a_connect_failure_pauses_and_the_same_call_runs_once_the_host_is_back():
    net = Net(down_for=0.4)
    host(net)
    ex = FakeDocker(net)
    with hostreach.episode() as led:
        out = await ex.run("echo hi")
    assert out.ok and out.stdout == "ok\n" and ex.calls >= 2
    assert led.to_json()["resumed"] == 1 and led.to_json()["gave_up"] == 0 and led.to_json()["hosts"] == ["lab"]


async def test_a_call_cut_mid_way_is_replayed_only_when_replayable():
    net = Net()
    host(net)
    ex = FakeDocker(net, mode="hang")
    for replay, ok in ((True, True), (False, False)):
        net.drop(0.5)
        with hostreach.replayable(replay):
            if ok:  # a read, a snapshot, a grade: the same when repeated
                assert (await ex.run("cat f")).ok
            else:  # a policy's write or shell call may have partly run: it comes back to the caller as resumed
                with pytest.raises(HostUnreachable) as ei:
                    await ex.run("echo x >> f")
                assert ei.value.resumed and ei.value.dispatched and ei.value.paused_s > 0


async def test_a_host_that_stays_down_ends_in_host_unreachable_within_the_pause_budget():
    net = Net(down_for=float("inf"))
    host(net, ReachRule(**{**FAST.__dict__, "pause_s": 0.5}))
    t0 = time.monotonic()
    with hostreach.episode() as led, pytest.raises(HostUnreachable) as ei:
        await FakeDocker(net).run("true")
    assert not ei.value.resumed and 0.4 < time.monotonic() - t0 < 3 and "pause budget" in str(ei.value)
    assert led.to_json()["gave_up"] >= 1
    # a later attempt does not wait again for an outage that already outlasted the budget
    t1 = time.monotonic()
    with hostreach.episode(), pytest.raises(HostUnreachable):
        await FakeDocker(net).run("true")
    assert time.monotonic() - t1 < 0.5


# ---- the World and the episode ---------------------------------------------------------------------------------------------------
class RemoteFs(LocalExec):
    """A workspace App's exec on a 'remote' host: every snippet goes through the host's reach and reads ``net``."""

    def __init__(self, reach: HostReach, net: Net, *, cut: bool = False):
        super().__init__()
        self.reach, self.net, self.cut = reach, net, cut

    async def run_python(self, code, args, *, timeout_s=600):
        async def once():
            if not self.net.up and await self.reach.confirm("transport"):
                raise HostUnreachable(self.reach.name, "docker exec: Cannot connect", dispatched=self.cut)
            return await LocalExec.run_python(self, code, args, timeout_s=timeout_s)

        return await self.reach.call(once, what="run_python")


async def world_with(tmp_path: Path, exec_):
    from crucible.world.world import World

    spec = WorldSpec.model_validate({"world": "test/outage", "actors": [{"id": "agent", "kind": "policy", "tools": ["ws.*"]}],
                                     "apps": [{"id": "ws", "template": "common/workspace", "state": {"adapter": "fs", "params": {"root": "ws"}}}]})
    w = World(spec, store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    await w.start()
    await w.reset()
    (w.workdir / "ws").mkdir(parents=True, exist_ok=True)
    w.apps["ws"].ctx.extra["policy_exec"] = exec_
    return w


async def test_the_episode_resumes_after_an_outage_and_the_agent_clock_does_not_run_out(tmp_path: Path):
    from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode

    net = Net()
    reach = host(net)
    w = await world_with(tmp_path, RemoteFs(reach, net))
    try:
        await w.step("agent", ToolCall(tool_call_id="w0", function_name="ws.write_file", arguments={"path": "a.txt", "content": "hello"}))
        net.drop(0.6)  # longer than the agent clock below
        pol = ScriptedPolicy("agent", [ToolCall(tool_call_id="r1", function_name="ws.read_file", arguments={"path": "a.txt"})])
        with hostreach.episode() as led:
            ep = await run_episode(w, {"agent": pol}, limits=EpisodeLimits(max_steps=5, timeout_s=0.4), reset=False)
        assert ep.stop_condition != "timeout" and not ep.masked and not w.failures, (ep.stop_condition, w.failures)
        assert led.to_json()["resumed"] == 1 and led.paused_s >= 0.5  # paused longer than the whole agent clock, and finished
        r = await w._call_tool("agent", ToolCall(tool_call_id="r2", function_name="ws.read_file", arguments={"path": "a.txt"}))
        assert r.ok and "hello" in r.content  # the same World, the agent's file still there
    finally:
        await w.close()


async def test_a_cut_write_comes_back_to_the_agent_as_interrupted_and_the_episode_goes_on(tmp_path: Path):
    net = Net()
    reach = host(net)
    w = await world_with(tmp_path, RemoteFs(reach, net, cut=True))
    try:
        net.drop(0.3)
        r = await w._call_tool("agent", ToolCall(tool_call_id="w1", function_name="ws.write_file", arguments={"path": "b.txt", "content": "x"}))
        assert not r.ok and r.error == "interrupted" and r.fault == "tool" and hostreach.INTERRUPTED in r.content
        assert not w.failures  # not masked: the agent decides what to do
    finally:
        await w.close()


async def test_a_host_that_does_not_come_back_masks_the_episode_as_host_unreachable(tmp_path: Path):
    net = Net()
    reach = host(net, ReachRule(**{**FAST.__dict__, "pause_s": 0.3}))
    w = await world_with(tmp_path, RemoteFs(reach, net))
    try:
        net.drop(float("inf"))
        r = await w._call_tool("agent", ToolCall(tool_call_id="r2", function_name="ws.read_file", arguments={"path": "a.txt"}))
        assert not r.ok and r.fault == "infra"
        f = w.failures[0]
        assert f.kind == "transport_unreachable" and f.detail == HOST_UNREACHABLE and f.masks and HOST_UNREACHABLE in f.reason
    finally:
        await w.close()


async def test_a_dead_looking_world_on_an_unreachable_host_is_an_outage_not_a_lost_container(monkeypatch):
    from types import SimpleNamespace

    from crucible.factory import envs

    net = Net(down_for=0.3)
    reach = host(net)
    seen = []

    async def running(name, env=None, *, timeout_s=60.0):
        seen.append(net.up)
        return net.up

    monkeypatch.setattr(envs, "container_running", running)
    w = SimpleNamespace(_crucible_container="c1", _crucible_env={}, _crucible_reach=reach, _infra=lambda *a, **k: None)
    assert await envs.world_alive(w) is True and seen == [False, True]  # paused for the host, then asked again
    net.drop(float("inf"))
    reach.rule = ReachRule(**{**FAST.__dict__, "pause_s": 0.2})
    faults = []
    w._infra = lambda kind, why, failure_detail=None: faults.append((kind, failure_detail))
    watch = envs.InfraWatch(None)
    await watch.on_step(w, 1, "agent", None, None)
    assert faults == [("transport_unreachable", HOST_UNREACHABLE)]  # never env_container_lost


# ---- the Runner: clock, record, journal ------------------------------------------------------------------------------------------
async def test_the_runner_pauses_its_attempt_clock_and_records_the_outage(tmp_path: Path):
    from crucible.runtime.harness.base import HarnessResult
    from crucible.runtime.infra import classify_exception
    from crucible.runtime.run import OUTAGE_META, Runner, RunSpec

    net = Net()
    reach = host(net)

    class T:
        id = "t@1"

    class P:
        id = "p"
        estimate_usd = 0.0

    class H:
        """A harness whose one remote call meets an outage longer than the attempt clock (resumed), or one that never ends."""

        id = "remote@1"

        async def run(self, attempt):
            try:
                await FakeDocker(net).run("true")
            except HostUnreachable as e:
                return HarnessResult(failure=classify_exception(e, "agent"))
            return HarnessResult(reward=1.0, steps=1)

    net.drop(0.6)
    s = await asyncio.wait_for(Runner(tmp_path).run(RunSpec(run_id="ok", tasks=[T()], policies=[P()], harnesses=[H()], k=1, attempt_timeout_s=0.4,
                                                            breaker=None, retry_backoff_s=0.0)), 30)
    rec = json.loads(next((tmp_path / "ok").glob("samples/*/a1/record.json")).read_text())
    assert s.scored == 1 and not rec.get("mask_sample") and rec["meta"][OUTAGE_META]["resumed"] == 1  # the clock (0.4 s) did not fire
    ev = [json.loads(x) for x in (tmp_path / "ok" / "journal.jsonl").read_text().splitlines()]
    assert any(e["event"] == "host_outage" and e["hosts"] == ["lab"] and not e["masked"] for e in ev)

    reach.rule = ReachRule(**{**FAST.__dict__, "pause_s": 0.3})
    net.drop(float("inf"))
    await asyncio.wait_for(Runner(tmp_path).run(RunSpec(run_id="down", tasks=[T()], policies=[P()], harnesses=[H()], k=1, attempt_timeout_s=30,
                                                        breaker=None, retry_backoff_s=0.0)), 30)
    rec = json.loads(next((tmp_path / "down").glob("samples/*/a1/record.json")).read_text())
    f = rec["failure"]
    assert rec["mask_sample"] and f["kind"] == "transport_unreachable" and f["detail"] == HOST_UNREACHABLE and f["blame"] in (None, "infra")
    assert rec["meta"][OUTAGE_META]["gave_up"] >= 1


# ---- fleet stop, the kernel lane, the journals ---------------------------------------------------------------------------------
async def test_a_launcher_holds_new_episodes_while_the_host_is_down(monkeypatch):
    from crucible.factory.lanes import kernel_episode as ke

    h = ke.host_of("rtx5090")
    net = Net(down_for=0.3)
    reach = host(net, key=f"docker:{h.docker_host}", name="rtx5090")
    assert ke.host_reach(h) is reach and await ke.launch_gate(h)  # not known down: at once
    reach._mark_down("lost")
    assert await ke.launch_gate(h, max_s=5)  # back within the wait
    net.drop(float("inf"))
    reach._mark_down("lost")
    assert not await ke.launch_gate(h, max_s=0.2)  # still down: launch nothing


async def test_a_kernel_grade_on_an_unreachable_host_is_masked_host_unreachable_and_the_name_is_the_host_s(monkeypatch):
    from crucible.factory.lanes import kernel_episode as ke
    from crucible.runtime.infra import classify_verifier

    h = ke.host_of("rtx5090")
    net = Net(down_for=float("inf"))
    host(net, ReachRule(**{**FAST.__dict__, "pause_s": 0.2}), key=f"docker:{h.docker_host}", name="rtx5090")

    async def fake_bundle(host, *, migration=False):
        return f"{host.work}/bundle"

    async def dead_ssh(host, script, stdin, timeout):
        return ke.Remote(255, "", "ssh: connect to host 100.100.101.36 port 22: Operation timed out", 15.0)

    monkeypatch.setattr(ke, "ensure_episode_bundle", fake_bundle)
    monkeypatch.setattr(ke, "_ssh_once", dead_ssh)
    doc = await ke.HostGrader(h).grade({"id": "kd-bug-x-1", "milestones": []}, "p")
    assert doc["infra"]["status"] == "masked" and doc["infra"]["failure_kind"] == HOST_UNREACHABLE and "rtx5090" in doc["infra"]["reason"]
    f = classify_verifier(doc["infra"])
    assert f.kind == "transport_unreachable" and f.detail == HOST_UNREACHABLE and f.who == "infra"  # regradable later, never the policy's 0


def test_the_swarm_journal_counts_outages_per_host(tmp_path: Path):
    from crucible.factory.swarm_journal import SwarmJournal, render, tables

    j = SwarmJournal(tmp_path / "journal")
    stop = hostreach.listen(j.host)
    try:
        hostreach._emit({"event": "host_down", "host": "rtx5090", "why": "lost"})
        hostreach._emit({"event": "outage", "host": "rtx5090", "what": "docker exec", "paused_s": 12.5, "resumed": True})
        hostreach._emit({"event": "outage", "host": "rtx5090", "what": "ssh", "paused_s": 1800.0, "resumed": False})
    finally:
        stop()
    j.host({"event": "launch_held", "host": "rtx5090", "task": "kd-1", "arm": "student"})
    t = tables(tmp_path / "journal")["hosts"]["rtx5090"]
    assert t == {"reason": HOST_UNREACHABLE, "down": 1, "outages": 2, "resumed": 1, "gave_up": 1, "paused_s": 1812.5, "launches_held": 1}
    d = {"tables": tables(tmp_path / "journal"), "bottlenecks": [], "models_with_failures": [], "builder_credit": {"attempts": 0, "with_progress": 0,
                                                                                                                      "failed_below_phi_1": []}}
    assert "rtx5090 (host_unreachable)" in render(d)


def test_framework_health_groups_host_outages_per_host():
    from crucible.factory import framework_health as fh
    from crucible.factory import train_monitor as tm
    from crucible.factory.traceaudit import EpisodeBundle, ToolStep

    def view(i: int, rec: dict, *, masked: bool, steps=()):
        b = EpisodeBundle(id=f"r/s{i}/a1", dir="/x", run="r", task="kd-bug-x-1", window="kd-bug-x-1", policy="p", model="m", family="qwen", patch="",
                          steps=list(steps), milestones={}, masked=masked, reward=None if masked else 0.0, stop="agent_done")
        lane, lang, _ = tm.task_meta("kd-bug-x-1")
        e = tm.Episode(id=b.id, dir=f"/nonexistent/{i}", task="kd-bug-x-1", lane=lane, lang=lang, split="train", cohort="c", step=None, policy="p",
                       family="qwen", masked=masked, passed=False, reward=b.reward, phi0=0.0, phi_final=0.0, stop="agent_done", tool_calls=len(steps),
                       compactions=0, milestones={}, model="m")
        return fh.View(e, b, rec, {}, [], [], None, "g")

    gone = {"failure": {"kind": "transport_unreachable", "stage": "agent", "detail": HOST_UNREACHABLE,
                        "reason": f"repo.shell: {HOST_UNREACHABLE}: docker exec in c on rtx5090: the host did not answer again within the pause budget"},
            "meta": {"host_outages": {"hosts": ["rtx5090"], "resumed": 0, "gave_up": 1, "paused_s": 1800.0}}}
    resumed = {"meta": {"host_outages": {"hosts": ["rtx5090"], "resumed": 1, "gave_up": 0, "paused_s": 40.0}}}
    cut = ToolStep(1, "c1", "repo.shell", {"command": "make"}, result=f"repo.shell was interrupted: {hostreach.INTERRUPTED} and came back after 40s.",
                   ok=False, error="interrupted", fault="tool")
    vs = [view(i, gone, masked=True) for i in range(8)] + [view(9, resumed, masked=False, steps=[cut])]
    ds = fh.cluster([h for v in vs for h in fh.per_attempt(v)], vs)
    masked = [d for d in ds if d.kind == "host_unreachable" and "(masked)" in d.signature]
    assert len(masked) == 1 and masked[0].signature == f"{HOST_UNREACHABLE}: rtx5090 (masked)" and len(masked[0].episodes) == 8
    assert masked[0].component == "sandbox" and masked[0].severity == "HIGH" and masked[0].impact.get("samples_lost") == 8
    assert any(d.kind == "host_unreachable" and "resumed" in d.signature and d.severity == "LOW" for d in ds)
    assert not [d for d in ds if d.kind in ("masked_failure", "tool_error")]  # one fact, one cluster: not a generic mask or a tool defect
    assert not fh.health_report(vs)["pause"]  # infrastructure: HIGH, never a CRITICAL pause of training
