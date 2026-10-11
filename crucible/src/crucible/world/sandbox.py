"""Container sandboxes a backend can put a World's state and commands into (no framework imports here).

``DockerSandbox`` runs one long-lived container from an image on this machine's Docker daemon or a remote one
(``gpu_host`` from ``crucible.toml``, e.g. the lab RTX 5090 over SSH), with the same hardening as the Harbor
overlay: ``--init``, no network for no-network worlds, a pids limit, ``SYS_RESOURCE`` (so the guard can pin the
policy's OOM score) and GPUs through the nvidia runtime or ``--gpus``. Cloud sandboxes (Modal and the other
agentenv providers) live in ``crucible.interop.agentenv.sandbox`` behind the same small interface.
"""

from __future__ import annotations

import uuid
from typing import Protocol

from crucible.core.schema.world import SandboxSpec, WorldSpec
from crucible.world.exec import DockerExec, Exec, _run_subprocess


class Sandbox(Protocol):
    exec: Exec
    gpus: int
    description: str

    async def start(self) -> None: ...

    async def close(self) -> None: ...


def gpu_request(spec: WorldSpec) -> tuple[int, str | None]:
    """(count, type) the World's Apps ask for: the largest count, and the type if all GPU apps agree on one."""
    gpu_apps = [a.resources for a in spec.apps if a.resources.gpus]
    types = {r.gpu_type for r in gpu_apps if r.gpu_type}
    return max((r.gpus for r in gpu_apps), default=0), (types.pop() if len(types) == 1 else None)


def resolve_docker_host(sb: SandboxSpec) -> str | None:
    if sb.docker_host:
        return sb.docker_host
    if sb.gpu_host:
        from crucible.core.config import get_config

        host = get_config().gpu_hosts.get(sb.gpu_host)
        if host is None:
            raise RuntimeError(f"gpu_host {sb.gpu_host!r} is not configured in crucible.toml [gpu_hosts]")
        return host.docker_host or f"ssh://{host.ssh}"
    return None


async def image_id(ref: str, docker_host: str | None = None, *, container: bool = False) -> str | None:
    """Digest of the filesystem layers of an image (or of a running container's image): what a WorldRef was taken in.
    Layers, not the image id: compose stamps project labels into images it builds, so two identical builds differ in id."""
    import hashlib

    env = {"DOCKER_HOST": docker_host} if docker_host else None
    if container:
        out = await _run_subprocess(["docker", "inspect", "-f", "{{.Image}}", ref], cwd=None, env=env, timeout_s=60, stdin=None)
        if not out.ok or not out.stdout.strip():
            return None
        ref = out.stdout.strip()
    out = await _run_subprocess(["docker", "image", "inspect", "-f", "{{json .RootFS.Layers}}", ref], cwd=None, env=env, timeout_s=60, stdin=None)
    if not out.ok or not out.stdout.strip():
        return None
    return "layers:" + hashlib.sha256(out.stdout.strip().encode()).hexdigest()


def check_image(state: dict, current: str | None) -> None:
    """A WorldRef restores only into the image it was taken in (DESIGN B.5: stale images silently change behavior)."""
    taken = state.get("image")
    if taken and current and taken != current:
        from crucible.world.state.base import SnapshotMismatch

        raise SnapshotMismatch(f"WorldRef was taken in image {taken[:19]}, this world runs {current[:19]}")


class DockerSandbox:
    def __init__(self, sb: SandboxSpec, spec: WorldSpec, *, pids_limit: int | None = None, memory: str | None = None):
        self.sb, self.spec = sb, spec
        self.gpus, self.gpu_type = gpu_request(spec)
        self.docker_host = resolve_docker_host(sb)
        self.name = f"crucible-sbx-{spec.world.replace('/', '-')[:40]}-{uuid.uuid4().hex[:8]}"
        self.pids_limit, self.memory = pids_limit, memory
        self.exec: Exec = DockerExec(self.name, docker_host=self.docker_host)
        self.description = f"docker {sb.image} on {self.docker_host or 'local daemon'}" + (f", {self.gpus} GPU(s)" if self.gpus else "")

    def argv(self) -> list[str]:
        a = ["docker", "run", "-d", "--rm", "--init", "--name", self.name, "--label", f"crucible.world={self.spec.world}", "--label", "crucible.sandbox=1"]
        if self.spec.network == "no-network":
            a += ["--network", "none"]
        if self.pids_limit:
            a += ["--pids-limit", str(self.pids_limit)]
        if self.spec.isolation.policy_user:
            a += ["--cap-add", "SYS_RESOURCE"]
        if self.gpus:
            if self.sb.gpu_mode == "runtime":  # hosts where --gpus fails (the lab 5090: missing MPS mount)
                a += ["--runtime=nvidia", "-e", "NVIDIA_VISIBLE_DEVICES=all", "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility"]
            else:
                a += ["--gpus", str(self.gpus) if self.gpus else "all"]
        if self.sb.cpus:
            a += ["--cpus", str(self.sb.cpus)]
        mem = self.memory or (f"{self.sb.memory_mb}m" if self.sb.memory_mb else None)
        if mem:
            a += ["--memory", mem]
        return [*a, "--entrypoint", "", self.sb.image, "sleep", "infinity"]

    def _env(self) -> dict[str, str] | None:
        return {"DOCKER_HOST": self.docker_host} if self.docker_host else None

    image_id: str | None = None

    async def start(self) -> None:
        out = await _run_subprocess(self.argv(), cwd=None, env=self._env(), timeout_s=1800, stdin=None)
        if not out.ok:
            raise RuntimeError(f"sandbox container failed to start ({self.description}): {(out.stderr or out.stdout)[-800:]}")
        self.image_id = await image_id(self.name, self.docker_host, container=True)

    async def pause(self) -> None:
        """Freeze every process in the container (cgroup freezer): no CPU, memory and files kept as they are (m13 S6)."""
        out = await _run_subprocess(["docker", "pause", self.name], cwd=None, env=self._env(), timeout_s=120, stdin=None)
        if not out.ok:
            raise RuntimeError(f"docker pause {self.name}: {(out.stderr or out.stdout)[-300:]}")

    async def resume(self) -> None:
        out = await _run_subprocess(["docker", "unpause", self.name], cwd=None, env=self._env(), timeout_s=120, stdin=None)
        if not out.ok:
            raise RuntimeError(f"docker unpause {self.name}: {(out.stderr or out.stdout)[-300:]}")

    async def close(self) -> None:
        await _run_subprocess(["docker", "rm", "-f", self.name], cwd=None, env=self._env(), timeout_s=120, stdin=None)
