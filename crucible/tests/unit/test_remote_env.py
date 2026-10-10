"""m13 D-3: remote Worlds over the NeMo-Gym resources protocol. Rewards identical to the in-process World; capacity
overload masked (never scored 0); abandoned sessions reaped; the M12 runtime trains on remote Worlds."""

from __future__ import annotations

import asyncio
import os
import random
import signal
import socket
import subprocess
import sys
import time

import httpx
import pytest
from crucible_pack_toy.runtime import HelpdeskTask
from crucible_pack_toy.verify import GOLD

from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode
from crucible.interop.nemo_gym.client import remote_episode


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(*extra):
    port = _port()
    p = subprocess.Popen([sys.executable, "-m", "crucible.interop.nemo_gym.server", "--tasks", "crucible_pack_toy.runtime:nemo_tasks", "--port", str(port),
                          *extra], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            if httpx.get(url + "/health", timeout=1).status_code == 200:
                return p, url
        except httpx.HTTPError:
            time.sleep(0.2)
    os.killpg(p.pid, signal.SIGKILL)
    pytest.fail("env server did not start")


@pytest.fixture
def servers():
    procs = [_serve(), _serve()]
    yield [u for _p, u in procs]
    for p, _u in procs:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()


def _actions(seed):
    r = random.Random(seed)
    acts = [a for a in GOLD if r.random() < 0.7]
    r.shuffle(acts)
    return acts


async def test_remote_rewards_equal_in_process(servers, tmp_path):
    used = set()
    for seed in range(30):
        acts = _actions(seed)
        task = HelpdeskTask()
        w = await task.open(tmp_path / f"local{seed}")
        try:
            await run_episode(w, {"agent": ScriptedPolicy("agent", list(acts))}, limits=EpisodeLimits(max_steps=40), reset=False)
            local = (await task.grade(w, tmp_path / f"g{seed}")).reward
        finally:
            await w.close()
        ep = await remote_episode(servers, "toy/helpdesk", ScriptedPolicy("agent", list(acts)), rid=f"seed{seed}", max_steps=40)
        assert not ep.masked and ep.reward == local, (seed, ep.reward, local)
        used.add(ep.server)
    assert len(used) == 2  # sessions spread across both servers (affinity by rollout id)


async def test_overload_is_masked_and_abandoned_sessions_are_reaped():
    p, url = _serve("--max-concurrent", "1", "--queue-timeout", "0.3", "--idle-timeout", "0.6", "--reap-every", "0.2")
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(url + "/seed_session", json={"resources_session_id": "held", "task_id": "toy/helpdesk"})
            assert r.status_code == 200  # the only slot, then abandoned (never closed)
        ep = await remote_episode([url], "toy/helpdesk", ScriptedPolicy("agent", list(GOLD)), rid="late")
        assert ep.masked and ep.failure_kind == "env_admission_timeout" and ep.reward is None
        await asyncio.sleep(1.5)
        async with httpx.AsyncClient() as c:
            h = (await c.get(url + "/health")).json()
        assert h["sessions"] == 0 and h["stats"]["reaped"] == 1
        ep = await remote_episode([url], "toy/helpdesk", ScriptedPolicy("agent", list(GOLD)), rid="after")
        assert not ep.masked and ep.reward == 1.0  # the reaped slot is usable again
    finally:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()


async def test_runtime_trains_on_remote_worlds(servers):
    import json

    from crucible.interop.nemo_gym.client import remote_world_rollout
    from crucible.train.chat import ByteChatCodec
    from crucible.train.engine import EnginePool, ScriptedEngine, TinyPolicy
    from crucible.train.mismatch import Correction
    from crucible.train.runtime import Runtime, RuntimeConfig
    from crucible.train.scheduler import SchedulerConfig
    from crucible.train.trainer import TorchTrainer, TrainerConfig
    from crucible.train.weights import InprocPublisher

    call = '<tool_call>' + json.dumps({"name": "tickets__list_tickets", "arguments": {"status": "open"}}) + '</tool_call>'

    def script(rid, k, prompt):
        return list((call if k == 0 else "all done").encode()) + [256]

    codec = ByteChatCodec()
    eng = EnginePool([ScriptedEngine(script, capacity=256, stop_ids=(256,))])
    tr = TorchTrainer(TinyPolicy(258, 16, seed=0), TrainerConfig(lr=1e-3, correction=Correction("dis")))
    rt = Runtime(RuntimeConfig(topology="colocated_partial", steps=2, max_turn_tokens=400, stop_ids=(256,), warmup_batches=0, max_wall_s=120),
                 SchedulerConfig(n=2, B=2, max_staleness=2), eng, tr, InprocPublisher(tr, eng), None, {"helpdesk": [f"p{i}" for i in range(100)]},
                 rollout_fn=remote_world_rollout(servers, lambda prompt: "toy/helpdesk", codec, max_steps=4, max_tokens=400))
    rep = await rt.run()
    assert len(rep.steps) == 2 and rep.masked_rollouts == 0 and rep.trained_records
    for r in rep.trained_records:
        ok, why = r.valid_for_training()
        assert ok, why
        assert r.reward is not None and len(r.turns) == 2
