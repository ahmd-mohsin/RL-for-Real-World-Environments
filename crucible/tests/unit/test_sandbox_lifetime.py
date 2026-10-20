"""A sandbox container's end by itself is infrastructure, its end by the policy's hand is not (M1 audit 2026-10-14 R-1, R-6).
The bounded life, the owner and expiry labels, the perf guard's orphans and the sweep are M10 lab audit R-6's
(``world.lifecycle``, tests ``test_m10_lab_audit_r6_*``), which replaced M1 R-6's ``sleep <timeout_s + 900>`` and pid owners
on the 2026-10-20 merge into dev; here: the sandbox tells its exec when its keeper ends, renewals included, and a policy that
kills its own container is scored."""

from __future__ import annotations

import asyncio
import subprocess
import time
import uuid

import pytest

from crucible.core.schema.world import SandboxSpec, WorldSpec
from crucible.world import lifecycle
from crucible.world import sandbox as sandbox_mod
from crucible.world.exec import DockerExec, ExecOut
from crucible.world.sandbox import DockerSandbox


def world() -> WorldSpec:
    return WorldSpec.model_validate({"world": "test/life", "apps": [], "actors": [{"id": "agent", "kind": "policy", "tools": []}]})


async def test_the_exec_learns_when_the_keeper_ends_and_each_renewal(monkeypatch):
    async def docker(argv, *, cwd=None, env=None, timeout_s=None, stdin=None, sessions=None):
        return ExecOut(0, "", "")

    monkeypatch.setattr(sandbox_mod, "_run_subprocess", docker)
    monkeypatch.setattr(lifecycle, "_run_subprocess", docker)
    monkeypatch.setattr(lifecycle, "_install_reaper", lambda: None)
    monkeypatch.setattr(lifecycle, "_HOSTS", set())
    sbx = DockerSandbox(SandboxSpec(image="x"), world(), lifetime_s=600, renewable=True)
    await sbx.start()
    try:
        assert isinstance(sbx.exec, DockerExec) and sbx.exec.expires_at == sbx.deadline and sbx.deadline > time.time() + 500
        assert await sbx.renew(3600)  # a loss before the renewed deadline is still the policy's or the daemon's, never the TTL's
        assert sbx.exec.expires_at == sbx.deadline and sbx.deadline > time.time() + 3500
    finally:
        await sbx.close()


def _docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=120, check=False)


@pytest.mark.docker
async def test_a_policy_killing_its_own_container_is_scored_not_masked():
    name = f"crucible-r6-kill-{uuid.uuid4().hex[:8]}"
    ex = DockerExec(name)  # a sandbox makes its exec before the container: events are read from then
    assert _docker("run", "-d", "--rm", "--init", "--name", name, "python:3.12-slim", "sleep", "300").returncode == 0
    try:
        await ex.run("for p in /proc/[0-9]*; do [ \"$(cat $p/comm 2>/dev/null)\" = sleep ] && kill ${p#/proc/}; done; true")
        for _ in range(50):  # until --rm has deleted it: the case R-1 alone took for an outage
            if not _docker("ps", "-aq", "--filter", f"name=^{name}$").stdout.strip():
                break
            await asyncio.sleep(0.2)
        out = await ex.run("true")  # deleted, but it died from inside (die, destroy; no kill): the episode's doing
        assert out.rc != 0 and "No such container" in out.stderr  # scored: no EnvContainerLost
    finally:
        _docker("rm", "-f", name)
