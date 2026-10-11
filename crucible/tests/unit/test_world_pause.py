"""m13 S6: pause/resume. A wall clock frozen for the pause continues without a jump; a paused sandbox uses no CPU and
keeps its files, and its processes continue afterwards; a sandboxed agentenv World paused and resumed keeps its
snapshot fingerprints and its virtual time."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from datetime import timedelta

import pytest

from crucible.core.schema.world import ClockSpec, SandboxSpec, WorldSpec
from crucible.world.clock import VirtualClock
from crucible.world.sandbox import DockerSandbox


def test_wall_clock_freezes_and_continues_without_a_jump():
    c = VirtualClock(ClockSpec(mode="wall", t0="2026-01-01T00:00:00Z", rate=3600))
    time.sleep(0.05)
    c.pause()
    at = c.now()
    time.sleep(0.2)  # 720 virtual seconds at this rate, if the pause leaked
    assert c.now() == at
    c.resume()
    assert timedelta() <= c.now() - at < timedelta(seconds=60)
    step = VirtualClock(ClockSpec(mode="step", t0="2026-01-01T00:00:00Z"))
    step.pause()
    step.on_step()  # step clocks are not frozen by pause (they never move by themselves)
    assert step.now() > step.t0


def _cpu(name: str) -> float:
    out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{json .}}", name], capture_output=True, text=True, check=False).stdout
    return float(json.loads(out)["CPUPerc"].rstrip("%"))


@pytest.mark.docker
async def test_paused_sandbox_uses_no_cpu_and_keeps_state():
    spec = WorldSpec.model_validate({"world": "test/pause", "network": "no-network", "apps": [], "actors": [{"id": "agent", "kind": "policy"}]})
    s = DockerSandbox(SandboxSpec(image="python:3.13-slim"), spec)
    await s.start()
    try:
        await s.exec.run("echo state > /tmp/f; (nohup python3 -c 'import time\nn=0\nwhile True:\n n+=1\n if n%10**6==0: open(\"/tmp/n\",\"w\").write(str(n))' >/dev/null 2>&1 &)",
                         timeout_s=30)
        await asyncio.sleep(2)
        busy = _cpu(s.name)
        await s.pause()
        await asyncio.sleep(2)
        paused = _cpu(s.name)
        await s.resume()
        before = (await s.exec.run("cat /tmp/n", timeout_s=30)).stdout
        await asyncio.sleep(1.5)
        after = (await s.exec.run("cat /tmp/n; echo; cat /tmp/f", timeout_s=30)).stdout.split()
        assert busy > 30 and paused == 0.0, (busy, paused)
        assert int(after[0]) > int(before) and after[1] == "state"  # the process continued; files kept
    finally:
        await s.close()
