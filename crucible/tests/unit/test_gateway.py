"""m13 D-1: the model gateway. Sticky routing that queues before spilling, failures returned as aborts (re-issued, never
masked), eviction that drops only the dead engine's sessions, health-driven revival with weight catch-up, and the
multi-process acceptance: three engine-server processes behind the gateway, one SIGKILLed mid-run."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time

import httpx
import pytest

from crucible.train.engine import GenRequest, ScriptedEngine, TinyPolicy
from crucible.train.gateway import EngineDown, GatewayPool, HTTPEngine


class Flaky(ScriptedEngine):
    def __init__(self, *a, down=False, **kw):
        super().__init__(*a, **kw)
        self.down, self.healthy = down, True

    async def generate(self, req):
        if self.down:
            raise EngineDown("gone")
        return await super().generate(req)

    async def health(self):
        return self.healthy


def _req(rid, k=1):
    return GenRequest(rid=rid, call_id=f"{rid}:c{k}", prompt_ids=[1, 2], max_tokens=2)


async def test_sticky_queues_then_spills_and_eviction_keeps_other_sessions():
    script = lambda rid, k, p: [3, 4]
    a, b, c = (Flaky(script, capacity=2) for _ in range(3))
    gw = GatewayPool([a, b, c], spill_factor=2.0)
    first = {}
    for r in ("r1", "r2", "r3"):  # new sessions go to the least-loaded live engine (load counted by generate)
        first[r] = gw.route(r)
        gw.inflight[first[r]] += 1
    assert sorted(first.values()) == [0, 1, 2]
    gw.inflight = [0, 0, 0]
    gw.inflight[first["r1"]] = 3  # under spill_factor × capacity = 4: still sticky (it queues)
    assert gw.route("r1") == first["r1"]
    gw.inflight[first["r1"]] = 4  # at the threshold: spill and re-pin
    assert gw.route("r1") != first["r1"] and gw.stats["spilled"] == 1
    gw.inflight = [0, 0, 0]
    dead = gw.route("r2")
    gw.evict(dead)
    others = [r for r, i in gw.sessions.items() if i != dead]
    assert all(gw.sessions[r] != dead for r in gw.sessions) and others  # only the dead engine's sessions were dropped


async def test_failed_call_is_an_abort_and_the_engine_is_evicted():
    script = lambda rid, k, p: [3, 4]
    bad, good = Flaky(script, down=True), Flaky(script)
    gw = GatewayPool([bad, good])
    gw.sessions["r"] = 0
    res = await gw.generate(_req("r"))
    assert res.finish == "abort" and not gw.alive[0] and gw.stats["failed_calls"] == 1
    res = await gw.generate(_req("r", 2))
    assert res.finish != "abort" and gw.sessions["r"] == 1 and gw.version == good.version


async def test_health_revives_only_after_catching_up_on_weights():
    script = lambda rid, k, p: [3]
    a, b = Flaky(script), Flaky(script)
    gw = GatewayPool([a, b], failure_threshold=2)
    a.healthy = False
    await gw.check_health()
    await gw.check_health()
    assert not gw.alive[0]
    await gw.load_weights({"w": 1}, 5)  # published while engine 0 was out
    assert b.version == 5 and a.version == 0 and gw.version == 5  # a dead engine does not pin the version
    a.healthy = True
    await gw.check_health()
    assert gw.alive[0] and a.version == 5 and gw.stats["revived"] == 1


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def engine_servers():
    procs, urls = [], []
    for _ in range(3):
        port = _free_port()
        procs.append(subprocess.Popen([sys.executable, "-m", "crucible.train.gateway", "engine", "--port", str(port), "--seed", "1", "--capacity", "64",
                                       "--delay", "0.002"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True))
        urls.append(f"http://127.0.0.1:{port}")
    deadline = time.time() + 60
    for u in urls:
        while time.time() < deadline:
            try:
                if httpx.get(u + "/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        else:
            pytest.fail(f"engine server {u} did not start")
    yield procs, urls
    for p in procs:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
        p.wait()


async def test_three_engine_processes_one_killed_mid_run(engine_servers, tmp_path):
    from crucible.train.envs import CopyTask
    from crucible.train.mismatch import Correction
    from crucible.train.runtime import Runtime, RuntimeConfig
    from crucible.train.scheduler import SchedulerConfig
    from crucible.train.trainer import TorchTrainer, TrainerConfig
    from crucible.train.weights import CheckpointPublisher

    procs, urls = engine_servers
    gw = GatewayPool([HTTPEngine(u, capacity=64) for u in urls], health_interval_s=0.5)
    tr = TorchTrainer(TinyPolicy(32, 48, seed=1), TrainerConfig(lr=1e-2, correction=Correction("dis")))
    rt = Runtime(RuntimeConfig(topology="colocated_partial", steps=10, max_turn_tokens=3, max_wall_s=300), SchedulerConfig(n=4, B=4, max_staleness=2),
                 gw, tr, CheckpointPublisher(tr, gw, tmp_path / "ckpt"), lambda: CopyTask(turns=3, symbols=8, latency_s=0.003),
                 {"copy": [f"p{i}" for i in range(4000)]})
    gw.start_health()

    async def killer():
        while len(rt.report.steps) < 3:
            await asyncio.sleep(0.05)
        os.killpg(procs[1].pid, signal.SIGKILL)  # mid-run, with calls in flight

    k = asyncio.create_task(killer())
    try:
        rep = await rt.run()
    finally:
        k.cancel()
        await gw.stop_health()
    assert len(rep.steps) == 10 and rep.stop_reason == "steps" and rep.publishes == 10
    assert rep.masked_rollouts == 0 and not rep.failures  # the dead engine's calls came back as aborts and were re-issued
    assert not gw.alive[1] and gw.alive[0] and gw.alive[2] and gw.stats["evictions"] == 1
    assert gw.affinity({1}) >= 0.99  # sessions on surviving engines kept their engine
    assert all(r.reward is not None for r in rep.trained_records)
