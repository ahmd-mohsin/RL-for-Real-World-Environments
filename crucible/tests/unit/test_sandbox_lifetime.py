"""Sandbox containers end on their own (M1 audit 2026-10-14 R-6): the main process sleeps for the sandbox's lifetime, not
forever, every container names its owner, and the perf guard no longer waits on containers a dead process left behind."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import uuid

import pytest

from crucible.core.schema.world import SandboxSpec, WorldSpec
from crucible.runtime.perf_guard import PerfGuard, local_probe, sample
from crucible.world.exec import DockerExec
from crucible.world.sandbox import DockerSandbox, owner_dead, owner_id

HOST = socket.gethostname()


def world() -> WorldSpec:
    return WorldSpec.model_validate({"world": "test/life", "apps": [], "actors": [{"id": "agent", "kind": "policy", "tools": []}]})


def dead_pid() -> int:
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


def test_a_sandbox_sleeps_for_its_lifetime_and_names_its_owner():
    sbx = DockerSandbox(SandboxSpec(image="x", timeout_s=600), world())
    a = sbx.argv()
    assert a[-3:] == ["x", "sleep", "1500"] and sbx.lifetime_s == 600 + 900 and "--rm" in a
    assert f"crucible.owner={HOST}:{os.getpid()}" in a and owner_id() == f"{HOST}:{os.getpid()}"


def test_owner_dead_only_for_a_gone_process_on_this_host():
    assert owner_dead(f"{HOST}:{dead_pid()}")
    assert not owner_dead(owner_id()) and not owner_dead(f"{HOST}:1")  # alive (pid 1 always is)
    assert not owner_dead(f"elsewhere:{dead_pid()}") and not owner_dead("") and not owner_dead(f"{HOST}:x")  # unknown


def test_the_perf_guard_ignores_containers_of_dead_owners():
    dead = f"{HOST}:{dead_pid()}"

    def probe(cmd: str):
        if cmd.startswith("docker ps"):
            return 0, (f"crucible-sbx-a|{dead}|\ncrucible-rlm-b||{dead}\ncrucible-sbx-c|{owner_id()}|\ncrucible-grade-d||\n"
                       "crucible-sbx-e|elsewhere:7|\nharbor-main-1||\n")
        return 0, ""

    s = sample(probe)
    assert s["crucible_containers"] == 3  # c (alive), d (no label), e (another host); a and b were left by a dead process
    assert "3 other crucible" in " ".join(PerfGuard(probe=probe).busy(s))


def _docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=120, check=False)


@pytest.mark.docker
def test_the_perf_guard_reads_owner_labels_from_docker():
    tag = uuid.uuid4().hex[:8]
    base = sample(local_probe)["crucible_containers"]
    names = []
    try:
        for kind, owner in (("dead", f"{HOST}:{dead_pid()}"), ("live", owner_id())):
            names.append(f"crucible-r6-{kind}-{tag}")
            assert _docker("run", "-d", "--rm", "--name", names[-1], "--label", f"crucible.owner={owner}", "python:3.12-slim", "sleep", "120").returncode == 0
            assert sample(local_probe)["crucible_containers"] == base + (kind == "live")  # the dead owner's never counts
    finally:
        _docker("rm", "-f", *names)


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
