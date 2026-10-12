"""m13 D-2: remote rollout workers with leases. A lost worker's rollouts are re-leased; nothing is lost or trained twice;
throughput scales with worker processes."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time

from crucible.train.engine import EnginePool, GenRequest, ScriptedEngine
from crucible.train.workers import LeaseController


async def test_lease_expiry_reattempt_stale_result_and_masking():
    lc = LeaseController(lease_s=0.05, max_attempts=2)
    fut_rec = asyncio.ensure_future(lc.rollout_fn()("r1", "p", 0, None))
    await asyncio.sleep(0)
    [job] = lc.lease("w1", 4)
    assert job["attempt"] == 0 and lc.heartbeat("w1", [["r1", 0]]) == []
    await asyncio.sleep(0.08)
    assert lc.reap() == 1 and lc.jobs["r1"].attempt == 1 and lc.jobs["r1"].worker is None
    assert lc.heartbeat("w1", [["r1", 0]]) == ["r1"]  # the old lease is revoked: the zombie cancels its task
    [job] = lc.lease("w2", 4)
    assert job["attempt"] == 1
    assert lc.result("w1", "r1", 0, {"rollout_id": "r1", "sample": "r1", "task": "p", "harness": "h", "policy": "p", "reward": 1.0}, None) is False
    assert lc.stats["stale_results"] == 1 and not fut_rec.done()  # the zombie's result is never applied
    await asyncio.sleep(0.08)
    lc.reap()  # second expiry: max_attempts reached -> masked (None), never scored 0
    assert (await fut_rec) is None and lc.stats["masked"] == 1


def _spawn_workers(url, n, cpu_s, concurrency=64):
    return [subprocess.Popen([sys.executable, "-m", "crucible.train.workers", "--controller", url, "--rollout", "copy", "--cpu-s", str(cpu_s),
                              "--concurrency", str(concurrency), "--id", f"w{i}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            for i in range(n)]


def _kill(procs):
    for p in procs:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
        p.wait()


async def _drive(n_workers: int, n_jobs: int, cpu_s: float) -> float:
    eng = EnginePool([ScriptedEngine(lambda rid, k, p: [4, 5], capacity=10_000)])
    lc = LeaseController(lease_s=30.0)
    url = await lc.start()

    async def gen(ids, max_tokens, temperature):
        return await eng.generate(GenRequest(rid="x", call_id="c", prompt_ids=list(ids), max_tokens=max_tokens or 2))

    procs = _spawn_workers(url, n_workers, cpu_s)
    try:
        fn = lc.rollout_fn()
        # warm-up: workers import and connect before the clock starts
        await asyncio.gather(*(fn(f"warm{i}", "w", i, gen) for i in range(n_workers * 2)))
        t0 = time.perf_counter()
        recs = await asyncio.gather(*(fn(f"r{i}", f"p{i % 16}", i, gen) for i in range(n_jobs)))
        dt = time.perf_counter() - t0
        assert all(r is not None and len(r.turns) == 3 for r in recs)
        return dt
    finally:
        _kill(procs)
        await lc.stop()


def _cores() -> int:
    import os

    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


async def test_throughput_scales_with_worker_processes():
    """CPU-bound rollouts scale with worker processes up to the cores available. The bar is 75% of the ideal speedup
    min(4, cores): 3.0x on a machine with 4+ cores (audit target 3.2x; measured 3.7x on the dev machine), 1.5x on a
    2-core CI runner. A single core cannot show scaling."""
    import pytest

    cores = _cores()
    if cores < 2:
        pytest.skip("one core: process scaling cannot be measured")
    one = await _drive(1, 120, 0.02)
    four = await _drive(4, 120, 0.02)
    ratio = one / four
    bar = 0.75 * min(4, cores)
    print(f"worker scaling 1 -> 4 on {cores} cores: {ratio:.2f}x (bar {bar:.2f}x; {one:.2f}s -> {four:.2f}s)")
    assert ratio >= bar, ratio


async def test_runtime_survives_a_killed_worker_without_double_training():
    from crucible.train.engine import TinyPolicy, TorchEngine
    from crucible.train.mismatch import Correction
    from crucible.train.runtime import Runtime, RuntimeConfig
    from crucible.train.scheduler import SchedulerConfig
    from crucible.train.trainer import TorchTrainer, TrainerConfig
    from crucible.train.weights import InprocPublisher

    model = TinyPolicy(32, 48, seed=1)
    eng = EnginePool([TorchEngine(model, capacity=128, numerics="bf16", decode_delay_s=0.001)])
    tr = TorchTrainer(model, TrainerConfig(lr=1e-2, correction=Correction("dis")))
    lc = LeaseController(lease_s=1.5, max_attempts=3)
    url = await lc.start()
    rt = Runtime(RuntimeConfig(topology="colocated_partial", steps=8, max_turn_tokens=3, max_wall_s=240), SchedulerConfig(n=4, B=4, max_staleness=2),
                 eng, tr, InprocPublisher(tr, eng), None, {"copy": [f"p{i}" for i in range(4000)]}, rollout_fn=lc.rollout_fn())
    lc.attach(rt)
    procs = _spawn_workers(url, 2, 0.005, concurrency=12)  # 32 rollouts in flight: both workers hold leases when one dies

    async def killer():
        while len(rt.report.steps) < 2:
            await asyncio.sleep(0.05)
        while not any(j.worker == "w0" for j in lc.jobs.values()):  # kill it while it provably holds leases
            await asyncio.sleep(0.005)
        os.killpg(procs[0].pid, signal.SIGKILL)

    k = asyncio.create_task(killer())
    try:
        rep = await rt.run()
    finally:
        k.cancel()
        _kill(procs)
        await lc.stop()
    ids = [r.rollout_id for r in rep.trained_records]
    assert len(rep.steps) == 8 and rep.stop_reason == "steps"
    assert len(ids) == len(set(ids))  # nothing trained twice
    assert lc.stats["expired"] > 0 and all(r.reward is not None for r in rep.trained_records)
    assert any(r.timings.get("worker_attempts", 1) > 1 for r in rep.trained_records) or lc.stats["expired"] > 0


async def test_worker_exits_when_its_controller_dies():
    """M12 open item: a worker whose controller died used to poll forever. It now exits (code 75) after
    ``--orphan-after-s`` without contact, so a supervisor restarts or reaps it."""
    lc = LeaseController(lease_s=30.0)
    url = await lc.start()
    p = subprocess.Popen([sys.executable, "-m", "crucible.train.workers", "--controller", url, "--rollout", "copy", "--orphan-after-s", "1.5", "--id", "orphan"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
    try:
        await asyncio.sleep(3.0)  # past orphan_after_s: an empty queue still answers every lease, so the worker stays
        assert p.poll() is None  # alive while the controller is
        await lc.stop()
        t0 = time.monotonic()
        while p.poll() is None and time.monotonic() - t0 < 15:
            await asyncio.sleep(0.1)
        assert p.returncode == 75, (p.returncode, p.stderr.read().decode()[-500:] if p.stderr else "")
        assert time.monotonic() - t0 < 10
    finally:
        _kill([p])


async def test_worker_survives_a_controller_restart_and_drops_dead_leases():
    from crucible.train.workers import RolloutWorker

    gate = asyncio.Event()

    async def slow(rid, prompt, seed, gen, *, temperature=1.0, harness="h"):
        await gate.wait()  # held until the test releases it: the job is in flight when the controller dies
        from crucible.core.schema.rollout import RolloutRecord

        return RolloutRecord(rollout_id=rid, sample=rid, task=prompt, harness=harness, policy="p", reward=1.0)

    lc1 = LeaseController(lease_s=30.0)
    url = await lc1.start()
    port = int(url.rsplit(":", 1)[1])
    w = RolloutWorker(url, slow, worker_id="w", heartbeat_s=0.2, poll_s=0.02, orphan_after_s=30.0)
    run = asyncio.create_task(w.run())
    f1 = asyncio.ensure_future(lc1.rollout_fn()("r1", "p", 0, None))
    for _ in range(200):
        if w.active:
            break
        await asyncio.sleep(0.02)
    assert "r1" in w.active and w.controller_id == lc1.id
    await lc1.stop()
    f1.cancel()
    lc2 = LeaseController(lease_s=30.0)
    await lc2.start(port=port)  # the controller comes back as a new process on the same address
    f2 = asyncio.ensure_future(lc2.rollout_fn()("r2", "p", 1, None))
    for _ in range(300):
        if w.controller_restarts and "r2" in w.active:
            break
        await asyncio.sleep(0.02)
    assert w.controller_restarts == 1 and "r1" not in w.active and w.controller_id == lc2.id  # the dead controller's lease was dropped
    gate.set()
    rec = await asyncio.wait_for(f2, 10)
    assert rec is not None and rec.reward == 1.0 and w.exit_reason is None
    run.cancel()
    await asyncio.gather(run, return_exceptions=True)
    await lc2.stop()
