"""Swap capped at the memory limit (M1 audit 2026-10-14 R-26): Docker's default lets a container with ``--memory m`` swap
as much again, so a policy's runaway allocation thrashes the host instead of being stopped at its limit."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from crucible.core.schema.world import SandboxSpec, WorldSpec

WORLD = WorldSpec.model_validate({"world": "test/r26", "apps": [], "actors": [{"id": "agent", "kind": "policy", "tools": []}]})


def test_a_sandbox_caps_swap_at_its_memory():
    from crucible.world.sandbox import DockerSandbox

    a = DockerSandbox(SandboxSpec(image="x", memory_mb=512), WORLD).argv()
    assert a[a.index("--memory") + 1] == a[a.index("--memory-swap") + 1] == "512m"
    assert "--memory-swap" not in DockerSandbox(SandboxSpec(image="x"), WORLD).argv()  # no limit, nothing to cap


def test_harbors_compose_overlay_caps_swap_too():
    pytest.importorskip("harbor")
    from crucible.interop.harbor.gpu_docker import GpuDockerEnvironment

    env = object.__new__(GpuDockerEnvironment)  # only the overlay: no Harbor environment is built
    env.__dict__.update(_overlay_path=None, _gpu_mode="none", _gpu_devices="all", _network_none=True, _init=True, _shm_size=None,
                        _pids_limit=None, _cap_add=[], _mem_limit="2g", _security_opt=[])
    svc = json.loads(Path(env._ensure_overlay()).read_text())["services"]["main"]
    assert svc["mem_limit"] == svc["memswap_limit"] == "2g"


@pytest.mark.docker
async def test_a_real_sandbox_has_no_swap_past_its_limit():
    from crucible.world.sandbox import DockerSandbox

    sbx = DockerSandbox(SandboxSpec(image="python:3.12-slim", memory_mb=64), WORLD)
    await sbx.start()
    try:
        out = await asyncio.to_thread(subprocess.run, ["docker", "inspect", "-f", "{{.HostConfig.Memory}} {{.HostConfig.MemorySwap}}", sbx.name],
                                      capture_output=True, text=True, check=True)
        assert out.stdout.split() == [str(64 << 20)] * 2
    finally:
        await sbx.close()
