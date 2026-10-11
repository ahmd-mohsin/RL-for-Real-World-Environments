"""M1 item 6: health checks, reset hooks and the run-serially flag (``world/health.py``; Harbor healthcheck semantics,
TheAgentCompany resets and ``-j 1``).
"""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from crucible.actors.policy import EpisodeLimits, ScriptedPolicy
from crucible.core.schema.world import HealthcheckSpec, WorldSpec
from crucible.core.store import Store
from crucible.runtime.harness.native import NativeHarness, PolicyRef
from crucible.runtime.run import Runner, RunSpec
from crucible.world.health import HealthcheckError, ResetHookError, SerialLock, run_probe
from crucible.world.world import World


def _spec(**app_extra) -> WorldSpec:
    app = {"id": "svc", "template": "common/json_api", "state": {"adapter": "json_doc", "params": {"path": "svc/doc.json"}},
           "params": {"collections": ["items"]}, **app_extra.pop("app", {})}
    return WorldSpec.model_validate({"world": app_extra.pop("world", "t/health"), "apps": [app],
                                     "actors": [{"id": "agent", "kind": "policy", "tools": ["svc.*"]}], **app_extra})


async def test_healthcheck_waits_through_start_period(tmp_path):
    flag = tmp_path / "ready"
    subprocess.Popen([sys.executable, "-c", f"import time, pathlib; time.sleep(0.6); pathlib.Path({str(flag)!r}).write_text('1')"])  # noqa: ASYNC220 -- a background writer
    s = _spec(app={"healthcheck": {"command": f"test -f {flag}", "start_period_s": 5, "start_interval_s": 0.1, "retries": 1}})
    w = World(s, store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    await w.start()
    try:
        assert w.health["svc"]["healthy"] and w.health["svc"]["attempts"] > 1  # failures in the start period did not count
    finally:
        await w.close()


async def test_unhealthy_world_does_not_start_and_is_masked(tmp_path):
    s = _spec(app={"healthcheck": {"command": "false", "retries": 2, "interval_s": 0}})
    w = World(s, store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    with pytest.raises(HealthcheckError, match="2 consecutive failures"):
        await w.start()
    assert w.closed  # a failed start cleans up

    class Task:
        id, world_id = "t/health/unhealthy@1", "t/health"

        async def open(self, workdir):
            w = World(s, store=Store(Path(workdir) / "s"), workdir=Path(workdir) / "w")
            await w.start()
            return w

    await Runner(tmp_path / "runs").run(RunSpec(run_id="r", tasks=[Task()], policies=[PolicyRef("scripted:noop", lambda a: {"agent": ScriptedPolicy("agent", [])})],
                                                harnesses=[NativeHarness(limits=EpisodeLimits(max_steps=2))], k=1))
    rec = json.loads(next((tmp_path / "runs" / "r" / "samples").glob("*/a*/record.json")).read_text())
    assert rec["failure"]["kind"] == "crucible:backend_error" and rec["mask_sample"]  # not ready is never the policy's 0


async def test_http_probe(tmp_path):
    from crucible.world.exec import LocalExec

    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        port = so.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(tmp_path)],  # noqa: ASYNC220
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        (tmp_path / "ok.txt").write_text("ok")
        ex = LocalExec()
        for _ in range(50):
            ok, _ = await run_probe(ex, HealthcheckSpec(http=f"http://127.0.0.1:{port}/ok.txt"))
            if ok:
                break
            await asyncio.sleep(0.1)
        assert ok
        assert not (await run_probe(ex, HealthcheckSpec(http=f"http://127.0.0.1:{port}/missing")))[0]
        assert (await run_probe(ex, HealthcheckSpec(http=f"http://127.0.0.1:{port}/missing", expect_status=[404])))[0]
        bad = await run_probe(ex, HealthcheckSpec(http="http://127.0.0.1:1/", timeout_s=1))
        assert not bad[0] and "status None" in bad[1]
    finally:
        proc.terminate()
        proc.wait(5)
    with pytest.raises(ValueError, match="exactly one"):
        HealthcheckSpec(command="true", http="http://x")


async def test_reset_hook_runs_on_fresh_resets_only(tmp_path):
    log = tmp_path / "resets.log"
    s = _spec(app={"reset_hook": {"command": f"echo reset >> {log}"}})
    w = World(s, store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    await w.start()
    try:
        await w.reset()
        await w.reset()
        mid = await w.snapshot()
        await w.reset(start=mid)  # a mid-episode restore (branching, resume) is not a new episode
        assert log.read_text().count("reset") == 2
    finally:
        await w.close()
    s = _spec(app={"reset_hook": {"command": "exit 3"}})
    w = World(s, store=Store(tmp_path / "s2"), workdir=tmp_path / "w2")
    await w.start()
    try:
        with pytest.raises(ResetHookError, match="reset hook failed"):
            await w.reset()
    finally:
        await w.close()


async def test_serial_worlds_never_overlap(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_LOCK_DIR", str(tmp_path / "locks"))
    spans: dict[str, list[float]] = {}

    async def episode(name: str, serial: bool, world: str):
        w = World(_spec(world=world, serial=serial), store=Store(tmp_path / name / "s"), workdir=tmp_path / name / "w")
        await w.start()
        t0 = time.monotonic()
        try:
            await asyncio.sleep(0.4)
        finally:
            await w.close()
        spans[name] = [t0, time.monotonic()]

    await asyncio.gather(episode("a", True, "t/shared"), episode("b", True, "t/shared"))
    (a0, a1), (b0, b1) = spans["a"], spans["b"]
    assert a1 <= b0 + 0.05 or b1 <= a0 + 0.05  # one after the other
    spans.clear()
    await asyncio.gather(episode("c", False, "t/free"), episode("d", False, "t/free"))
    (c0, c1), (d0, d1) = spans["c"], spans["d"]
    assert c0 < d1 and d0 < c1  # non-serial worlds overlap


async def test_serial_lock_across_processes(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_LOCK_DIR", str(tmp_path / "locks"))
    held = SerialLock("t/shared")
    await held.acquire()
    code = (f"import asyncio, os; os.environ['CRUCIBLE_LOCK_DIR'] = {str(tmp_path / 'locks')!r}\n"
            "from crucible.world.health import SerialLock\n"
            "l = SerialLock('t/shared', wait_s=0.5)\n"
            "try:\n    asyncio.run(l.acquire()); print('acquired')\nexcept TimeoutError:\n    print('busy')\n")
    run = lambda: subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False)
    out = await asyncio.to_thread(run)
    assert out.stdout.strip() == "busy", out.stderr
    held.release()
    out = await asyncio.to_thread(run)
    assert out.stdout.strip() == "acquired"
