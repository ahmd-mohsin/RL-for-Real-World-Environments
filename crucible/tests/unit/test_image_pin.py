"""Pinned images (M1 audit 2026-10-14 R-14): a World refuses to start in an image other than its pinned ``image_digest``
(``crucible world pin``), so a tag re-pushed or rebuilt under the same name can no longer change a task unseen."""

from __future__ import annotations

import asyncio
import subprocess
import uuid

import pytest
from click.testing import CliRunner

from crucible.cli.main import cli
from crucible.core.schema.run import DEFAULT_BLAME, NON_RETRYABLE_KINDS
from crucible.core.schema.world import SandboxSpec, WorldSpec
from crucible.runtime.infra import StageFailure, boundary
from crucible.world.sandbox import DockerSandbox, StaleImage, check_pinned


def test_the_pin_check():
    check_pinned(None, "layers:a", "x")  # nothing pinned
    check_pinned("layers:a", "layers:a", "x")
    with pytest.raises(StaleImage, match="spec pins layers:a"):
        check_pinned("layers:a", "layers:b", "x")
    with pytest.raises(StaleImage, match="identity unreadable"):
        check_pinned("layers:a", None, "x")  # a pin that cannot be checked is not a pass


def test_a_stale_image_is_a_typed_infra_failure_never_retried():
    with pytest.raises(StageFailure) as e, boundary("admission"):
        raise StaleImage("re-pushed")
    f = e.value.failure
    assert f.kind == "crucible:stale_image" and DEFAULT_BLAME[f.kind] == "infra" and f.kind in NON_RETRYABLE_KINDS


def _docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=120, check=False)


@pytest.mark.docker
async def test_a_re_tagged_image_fails_start():
    tag = f"crucible-r14-{uuid.uuid4().hex[:8]}:v1"
    world = WorldSpec.model_validate({"world": "test/r14", "apps": [], "actors": [{"id": "agent", "kind": "policy", "tools": []}]})
    assert _docker("tag", "python:3.12-slim", tag).returncode == 0
    try:
        out = await asyncio.to_thread(CliRunner().invoke, cli, ["world", "pin", tag])  # the CLI runs its own event loop
        assert out.exit_code == 0 and out.output.startswith("layers:"), out.output
        pin = out.output.strip()
        ok = DockerSandbox(SandboxSpec(image=tag, image_digest=pin), world)
        await ok.start()
        await ok.close()
        assert _docker("tag", "python:3.13-slim", tag).returncode == 0  # the same name, another image
        stale = DockerSandbox(SandboxSpec(image=tag, image_digest=pin), world)
        with pytest.raises(StaleImage, match=f"spec pins {pin}"):
            await stale.start()
        assert not _docker("ps", "-aq", "--filter", f"name=^{stale.name}$").stdout.strip()  # its container is gone
    finally:
        _docker("rmi", tag)


def test_pinning_an_absent_image_is_a_user_error():
    out = CliRunner().invoke(cli, ["world", "pin", f"crucible-no-such-image-{uuid.uuid4().hex[:8]}:v0"])
    assert out.exit_code != 0 and "is not on" in out.output
