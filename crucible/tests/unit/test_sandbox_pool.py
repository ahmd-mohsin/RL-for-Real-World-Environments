"""m13 S10: the sandbox warm pool. Acquire from a warm pool takes under 10% of a cold ``docker run``; fresh mode never
hands a used container out again; reuse mode resets by hygiene and verifies (a planted process, file and user are all
gone, and a container that cannot be cleaned is destroyed); a pooled agentenv World provisions through the pool."""

from __future__ import annotations

import statistics
import time

import pytest

from crucible.core.schema.world import SandboxSpec, WorldSpec
from crucible.world.pool import PooledSandbox, SandboxPool, close_pools, pool_for, signature
from crucible.world.sandbox import DockerSandbox

pytestmark = [pytest.mark.docker]
IMAGE = "python:3.13-slim"


@pytest.fixture(scope="module", autouse=True)
def _image():
    import subprocess

    if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, check=False).returncode != 0:
        subprocess.run(["docker", "pull", "-q", IMAGE], capture_output=True, check=False, timeout=600)


def _spec() -> WorldSpec:
    return WorldSpec.model_validate({"world": "test/pool", "network": "no-network", "apps": [], "actors": [{"id": "agent", "kind": "policy"}]})


def test_signature_ignores_names_but_not_hardening():
    sb = SandboxSpec(image=IMAGE)
    assert signature(sb, _spec(), pids_limit=64, memory=None) == signature(sb, _spec(), pids_limit=64, memory=None)
    assert signature(sb, _spec(), pids_limit=64, memory=None) != signature(sb, _spec(), pids_limit=128, memory=None)


async def test_warm_acquire_is_under_a_tenth_of_cold():
    sb, spec = SandboxSpec(image=IMAGE), _spec()
    cold = []
    for _ in range(4):
        s = DockerSandbox(sb, spec)
        t0 = time.perf_counter()
        await s.start()
        cold.append(time.perf_counter() - t0)
        await s.close()
    pool = SandboxPool(sb, spec, size=4)
    await pool.warm()
    warm, used = [], set()
    try:
        for _ in range(8):
            ps = PooledSandbox(pool)
            await ps.start()
            warm.append(ps.acquire_s)
            assert ps._sbx.name not in used  # fresh mode: never the same container twice
            used.add(ps._sbx.name)
            assert (await ps.exec.run("echo hi", timeout_s=30)).stdout.strip() == "hi"
            await ps.close()
            await pool.warm()  # the refill keeps up between rollouts
    finally:
        await pool.close()
    assert statistics.median(warm) < 0.1 * statistics.median(cold), (warm, cold)
    assert pool.stats["hits"] == 8 and pool.stats["destroyed"] == 8 and pool.stats["reused"] == 0


async def test_reuse_mode_hygiene_is_verified():
    pool = SandboxPool(SandboxSpec(image=IMAGE), _spec(), size=1, reuse=True)
    try:
        s = await pool.acquire()  # a miss: the pool was never warmed
        await s.exec.run("useradd -m leftover; mkdir -p /crucible && echo x > /crucible/f && echo y > /tmp/t; (nohup sleep 999 >/dev/null 2>&1 &)",
                         timeout_s=60, user="0")
        for t in list(pool._tasks):  # let the refill land, then drop it so the released container can come back
            await t
        while not pool._ready.empty():
            await pool._ready.get_nowait().close()
        await pool.release(s)
        assert pool.stats["reused"] == 1 and pool.stats["dirty"] == 0
        again = await pool.acquire()
        assert again.name == s.name
        out = await again.exec.run("ls -A /crucible /tmp; id leftover 2>&1; ps -eo args | grep -v grep | grep -c 'sleep 999'", timeout_s=30, user="0")
        lines = out.stdout.split()
        assert "f" not in lines and "t" not in lines and "no such user" in out.stdout and lines[-1] == "0", out.stdout

        async def never_clean(sbx):  # a container that cannot be verified clean is destroyed, never pooled
            pool.stats["dirty"] += 1
            return False

        pool._clean = never_clean
        before = pool.stats["destroyed"]
        await pool.release(again)
        assert pool.stats["destroyed"] == before + 1 and pool.stats["dirty"] == 1
    finally:
        await pool.close()


async def test_open_sandbox_uses_the_pool():
    from crucible.interop.agentenv.sandbox import open_sandbox

    sb = SandboxSpec(image=IMAGE, pool=2)
    a = open_sandbox(sb, _spec())
    b = open_sandbox(sb, _spec())
    try:
        assert isinstance(a, PooledSandbox) and a.pool is b.pool is pool_for(sb, _spec())
        await a.start()
        assert (await a.exec.run("echo pooled", timeout_s=30)).stdout.strip() == "pooled"
        await a.close()
    finally:
        await close_pools()
