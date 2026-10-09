"""GPU / sandbox options without credentials: the Docker sandbox's flags for the lab RTX 5090, Modal's GPU spec,
agentenv's sandbox exec adapter (driven through a fake provider sandbox that runs commands locally), and the
Harbor backend handing GPU types and the network policy to a cloud provider (Modal)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from crucible.core.schema.world import SandboxSpec, WorldSpec


def world(gpus: int = 1, gpu_type: str | None = None, network: str = "no-network", policy_user: str | None = "agent") -> WorldSpec:
    return WorldSpec.model_validate({
        "world": "test/sbx", "network": network, "isolation": {"policy_user": policy_user, "pids_limit": 512},
        "apps": [{"id": "repo", "template": "common/workspace", "state": {"adapter": "fs", "params": {"root": "/w"}}},
                 {"id": "gpu", "template": "common/toolchain", "state": {"adapter": "process", "params": {"cwd": "/w"}},
                  "resources": {"gpus": gpus, "gpu_type": gpu_type}}],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["repo.*"]}],
    })


def test_docker_sandbox_flags_for_the_lab_gpu_host():
    from crucible.world.sandbox import DockerSandbox

    sb = DockerSandbox(SandboxSpec(image="task04-pcl-gpu-cuda12:v1", gpu_host="lab5090", memory_mb=16384), world(), pids_limit=512)
    a = sb.argv()
    assert sb.docker_host == "ssh://ahmed@100.100.101.36"
    assert "--runtime=nvidia" in a and "NVIDIA_VISIBLE_DEVICES=all" in a and "--gpus" not in a  # --gpus fails on that host
    assert a[a.index("--network") + 1] == "none" and a[a.index("--pids-limit") + 1] == "512" and "SYS_RESOURCE" in a
    assert a[-3:] == ["task04-pcl-gpu-cuda12:v1", "sleep", "infinity"] and "--init" in a and a[a.index("--memory") + 1] == "16384m"
    b = DockerSandbox(SandboxSpec(image="x", gpu_mode="devices"), world(gpus=2, network="public", policy_user=None)).argv()
    assert b[b.index("--gpus") + 1] == "2" and "--network" not in b and "SYS_RESOURCE" not in b


def test_modal_gpu_spec():
    from crucible.interop.agentenv.sandbox import modal_gpu_spec

    assert modal_gpu_spec(SandboxSpec(provider="modal", image="x"), world(gpus=1, gpu_type="H100")) == "H100"
    assert modal_gpu_spec(SandboxSpec(provider="modal", image="x"), world(gpus=2, gpu_type="A100")) == "A100:2"
    assert modal_gpu_spec(SandboxSpec(provider="modal", image="x"), world(gpus=1)) == "any"
    assert modal_gpu_spec(SandboxSpec(provider="modal", image="x", gpu="L4"), world(gpus=1, gpu_type="H100")) == "L4"
    assert modal_gpu_spec(SandboxSpec(provider="modal", image="x"), world(gpus=0)) is None


class _Stream:
    def __init__(self, data: bytes):
        self.data = data

    async def read(self) -> bytes:
        return self.data


class _Proc:
    def __init__(self, rc: int, out: bytes, err: bytes):
        self.rc, self.stdout, self.stderr = rc, _Stream(out), _Stream(err)

    async def wait(self) -> int:
        return self.rc


class FakeProviderSandbox:
    """The agentenv ``Sandbox`` surface Crucible uses (exec + write_file_from_text), backed by local bash in a root dir.
    Paths are rewritten under ``root`` so the adapter's absolute paths stay inside the test's tmp dir."""

    def __init__(self, root: Path):
        self.root = root
        self.commands: list[tuple[str, ...]] = []

    def _map(self, s: str) -> str:
        import re

        return re.sub(r"/sbx(?=/|\b)", str(self.root), s)

    async def exec(self, *command: str):
        self.commands.append(command)
        p = await asyncio.create_subprocess_exec(*[self._map(c) for c in command], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await p.communicate()
        return _Proc(p.returncode or 0, out, err)

    async def write_file_from_text(self, content: str, destination_path: str) -> None:
        Path(self._map(destination_path)).write_text(content)

    async def terminate(self) -> None:
        pass


async def test_agentenv_sandbox_exec_adapter(tmp_path: Path):
    from crucible.interop.agentenv.sandbox import AgentEnvSandboxExec

    fake = FakeProviderSandbox(tmp_path)
    ex = AgentEnvSandboxExec(fake)
    ex.staging_dir = "/sbx/stage"
    r = await ex.run("echo $GREETING from $(pwd)", cwd="/sbx", env={"GREETING": "hi there"})
    assert r.ok and r.stdout.strip() == f"hi there from {tmp_path}"
    r = await ex.run("echo should-not-run", cwd="/sbx/missing")
    assert not r.ok and "should-not-run" not in r.stdout
    r = await ex.run("cat", stdin=b"\x00binary\xffstdin")  # staged through a base64 file, removed afterwards
    assert r.stdout.encode("latin-1", "replace")[:7] == b"\x00binary" and not list((tmp_path / "stage").glob(".crucible_stdin_*"))
    src = tmp_path / "blob.bin"
    src.write_bytes(bytes(range(256)) * 50)
    await ex.push_file(src, "/sbx/deep/dir/blob.bin")
    assert (tmp_path / "deep/dir/blob.bin").read_bytes() == src.read_bytes()
    await ex.pull_file("/sbx/deep/dir/blob.bin", tmp_path / "back.bin")
    assert (tmp_path / "back.bin").read_bytes() == src.read_bytes()
    import shutil

    if shutil.which("timeout"):  # every Linux image has it; a bare macOS host does not (then the client-side deadline applies)
        r = await ex.run("sleep 30", timeout_s=1)
        assert r.timed_out and r.duration_s < 10
    assert all(c[:2] == ("bash", "-c") for c in fake.commands)  # container sandboxes: no docker-exec prefix


async def test_agentenv_sandbox_uses_agentenv_providers(monkeypatch):
    """Modal gets agentenv's own ModalSandboxProvider with the World's GPU spec and an enforced empty allowlist."""
    pytest.importorskip("agent_env")
    from agent_env.providers.sandbox_providers import modal_sandbox
    from agent_env.providers.sandbox_providers.sandbox import NetworkMode

    from crucible.interop.agentenv.sandbox import AgentEnvSandbox

    seen: dict = {}

    class FakeModal:
        def __init__(self, gpu=None, **kw):
            seen["gpu"] = gpu

        async def create_container(self, *, image_name, port, env, cpu=1.0, memory=8192, disk_size_gb=10, timeout=7200, attribution=None,
                                   network_policy=None, command=None, expose_externally=True, **kw):
            seen.update(image=image_name, policy=network_policy, command=command, expose=expose_externally, memory=memory)
            return FakeProviderSandbox(Path("/tmp"))

    monkeypatch.setattr(modal_sandbox, "ModalSandboxProvider", FakeModal)
    sbx = AgentEnvSandbox(SandboxSpec(provider="modal", image="ghcr.io/acme/world:1", memory_mb=32768), world(gpus=1, gpu_type="H100"))
    await sbx.start()
    assert seen["gpu"] == "H100" and seen["image"] == "ghcr.io/acme/world:1" and seen["command"] == ["sleep", "infinity"] and seen["memory"] == 32768
    assert seen["policy"].mode == NetworkMode.ALLOWLIST and not seen["policy"].allow_hosts and seen["expose"] is False
    assert "gpu=H100" in sbx.description
    with pytest.raises(RuntimeError, match="no GPU path"):
        AgentEnvSandbox(SandboxSpec(provider="e2b", image="x"), world(gpus=1)).provider()


async def test_harbor_backend_passes_gpu_types_and_network_to_cloud_providers(monkeypatch, tmp_path: Path):
    pytest.importorskip("harbor")
    from harbor.environments.factory import EnvironmentFactory
    from harbor.models.task.config import NetworkMode

    from crucible.interop.harbor.world_backend import HarborBackend

    seen: dict = {}

    def fake_create(cls, type, **kw):
        seen.update(type=type, **kw)
        raise RuntimeError("stop after create")

    monkeypatch.setattr(EnvironmentFactory, "create_environment", classmethod(fake_create))
    spec = world(gpus=2, gpu_type="H100").model_copy(update={"backend": "harbor"})
    b = HarborBackend(spec, {"environment_type": "modal", "docker_image": "ghcr.io/acme/world:1"}, workdir=tmp_path)
    with pytest.raises(RuntimeError, match="stop after create"):
        await b.provision()
    cfg = seen["task_env_config"]
    assert seen["type"].value == "modal" and cfg.gpus == 2 and cfg.gpu_types == ["H100"] and cfg.docker_image == "ghcr.io/acme/world:1"
    assert seen["network_policy"].network_mode == NetworkMode.NO_NETWORK


def test_worldrefs_restore_only_into_their_image():
    from crucible.world.sandbox import check_image
    from crucible.world.state.base import SnapshotMismatch

    check_image({"image": "layers:aa"}, "layers:aa")
    check_image({}, "layers:aa")  # refs from before images were recorded, or from non-container worlds
    with pytest.raises(SnapshotMismatch, match="taken in image"):
        check_image({"image": "layers:aa"}, "layers:bb")
