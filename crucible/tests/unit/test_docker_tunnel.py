"""``world/docker_tunnel.py``: Docker CLI calls with an ``ssh://`` DOCKER_HOST share one SSH tunnel per host. Unit: the
rewrite applies only to docker calls with an ssh host and can be turned off. Live (``CRUCIBLE_VM_DOCKER_HOST``): 40
concurrent CLI calls over the tunnel all succeed (per-call SSH hung 5 of 24 under sshd's MaxStartups throttling)."""

from __future__ import annotations

import asyncio
import os

import pytest

from crucible.world.docker_tunnel import rewrite_env

HOST = os.environ.get("CRUCIBLE_VM_DOCKER_HOST")


def test_rewrite_only_for_docker_over_ssh(monkeypatch):
    env = {"DOCKER_HOST": "tcp://h:2375", "X": "1"}
    assert rewrite_env(["docker", "ps"], env) is env  # not ssh
    assert rewrite_env(["bash", "-c", "docker ps"], {"DOCKER_HOST": "ssh://u@h"}) == {"DOCKER_HOST": "ssh://u@h"}  # not the docker CLI
    monkeypatch.setenv("CRUCIBLE_DOCKER_SSH_TUNNEL", "0")
    assert rewrite_env(["docker", "ps"], {"DOCKER_HOST": "ssh://u@h"}) == {"DOCKER_HOST": "ssh://u@h"}  # opted out


@pytest.mark.skipif(not HOST, reason="needs an ssh:// docker host (CRUCIBLE_VM_DOCKER_HOST)")
async def test_concurrent_cli_calls_share_one_tunnel():
    from crucible.world.exec import _run_subprocess

    outs = await asyncio.gather(*(_run_subprocess(["docker", "version", "--format", "{{.Server.Version}}"], cwd=None, env={"DOCKER_HOST": HOST}, timeout_s=60,
                                                  stdin=None) for _ in range(40)))
    assert all(o.ok for o in outs) and len({o.stdout.strip() for o in outs}) == 1
