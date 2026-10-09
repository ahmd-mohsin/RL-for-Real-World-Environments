"""agentenv sandbox providers as Crucible sandboxes: Modal (cloud GPUs: H100, A100, L4, ...), E2B, Sail and any
provider registered under ``[sandbox.providers.<name>]`` in agentenv's config.

``AgentEnvSandbox`` provisions one container from the World's image with agentenv's own provider API
(``SandboxProvider.create_container``); Modal gets its GPU through ``ModalSandboxProvider(gpu=...)`` (agentenv
switches to Modal's V1 sandbox factory, the one that can attach a GPU). ``AgentEnvSandboxExec`` adapts the
provider's ``Sandbox.exec`` / ``write_file_from_text`` to Crucible's ``Exec``: deadlines are enforced inside the
sandbox, stdin and binary files travel base64-encoded, and a container on a VM-backed provider is reached the way
agentenv's own ``run_code`` step reaches it (``sudo docker exec -u 0 <container>``).

Network: no-network worlds get an empty allowlist (``NetworkPolicy(mode=allowlist)``), allowlist worlds their
``allowed_hosts``; providers that cannot enforce a policy are refused by agentenv before anything is created.
Needs credentials for the provider (Modal: ``modal token new`` or ``MODAL_TOKEN_ID``/``MODAL_TOKEN_SECRET``).
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import shlex
import time
import uuid
from pathlib import Path
from typing import Any

from crucible.core.schema.world import SandboxSpec, WorldSpec
from crucible.world.exec import Exec, ExecError, ExecOut, deadline_hit, with_deadline
from crucible.world.sandbox import gpu_request


def modal_gpu_spec(sb: SandboxSpec, spec: WorldSpec) -> str | None:
    """Modal's GPU string: ``sb.gpu`` verbatim, else ``<type>[:count]`` from the Apps' resources ("any" if untyped)."""
    if sb.gpu:
        return sb.gpu
    count, gtype = gpu_request(spec)
    if not count:
        return None
    return f"{gtype or 'any'}:{count}" if count > 1 else (gtype or "any")


def network_policy(spec: WorldSpec):
    from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy

    if spec.network == "no-network":
        return NetworkPolicy(mode=NetworkMode.ALLOWLIST)
    if spec.network == "allowlist":
        return NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=tuple(spec.allowed_hosts))
    return None


class AgentEnvSandboxExec(Exec):
    name = "agentenv-sandbox"

    def __init__(self, sandbox: Any, prefix: tuple[str, ...] = ()):
        self.sandbox = sandbox
        self.prefix = prefix

    async def _raw(self, script: str) -> tuple[int, bytes, bytes]:
        proc = await self.sandbox.exec(*self.prefix, "bash", "-c", script)
        out, err = await asyncio.gather(proc.stdout.read(), proc.stderr.read())
        return await proc.wait(), out or b"", err or b""

    async def run(self, cmd, *, cwd=None, env=None, timeout_s=None, user=None, stdin=None) -> ExecOut:
        lines = []  # one statement per line: a failed cd must stop the command, not just the next export
        if cwd:
            lines.append(f"cd {shlex.quote(cwd)} || exit 97")
        lines += [f"export {k}={shlex.quote(str(v))}" for k, v in (env or {}).items()]
        script = "\n".join([*lines, cmd])
        tmp = None
        if stdin is not None:
            tmp = f"{self.staging_dir}/.crucible_stdin_{uuid.uuid4().hex}"
            await self._write_bytes(stdin, tmp)
            script = f"{{ {script} ; }} < {tmp}; __rc=$?; rm -f {tmp}; exit $__rc"
        if user:
            script = f"runuser -u {shlex.quote(user)} -- bash -c {shlex.quote(script)}"
        t0 = time.monotonic()
        try:
            rc, out, err = await asyncio.wait_for(self._raw(with_deadline(script, timeout_s)), timeout=(timeout_s + 30) if timeout_s else None)
        except TimeoutError:
            return ExecOut(rc=-9, stdout="", stderr="sandbox exec did not return", duration_s=time.monotonic() - t0, timed_out=True)
        res = ExecOut(rc=rc, stdout=out.decode(errors="replace"), stderr=err.decode(errors="replace"), duration_s=time.monotonic() - t0)
        if deadline_hit(res.rc, res.duration_s, timeout_s):
            res.timed_out, res.rc = True, -9
        return res

    async def _write_bytes(self, data: bytes, dest: str) -> None:
        staged = f"{dest}.b64"
        parent = str(Path(dest).parent)
        rc, _, err = await self._raw(f"mkdir -p {shlex.quote(parent)}")
        if rc != 0:
            raise ExecError(f"mkdir {parent} failed: {err.decode(errors='replace')[-300:]}")
        await self.sandbox.write_file_from_text(base64.b64encode(data).decode(), staged)
        rc, _, err = await self._raw(f"base64 -d < {shlex.quote(staged)} > {shlex.quote(dest)} && rm -f {shlex.quote(staged)}")
        if rc != 0:
            raise ExecError(f"writing {dest} failed: {err.decode(errors='replace')[-300:]}")

    async def push_file(self, src: Path, dest: str) -> None:
        await self._write_bytes(Path(src).read_bytes(), dest)

    async def pull_file(self, src: str, dest: Path) -> None:
        rc, out, err = await self._raw(f"base64 < {shlex.quote(src)} | tr -d '\\n'")  # GNU, BSD and busybox alike
        if rc != 0:
            raise ExecError(f"reading {src} failed: {err.decode(errors='replace')[-300:]}")
        Path(dest).write_bytes(base64.b64decode(out))


class AgentEnvSandbox:
    """One container from an agentenv sandbox provider, holding the World's state and commands."""

    def __init__(self, sb: SandboxSpec, spec: WorldSpec):
        self.sb, self.spec = sb, spec
        self.gpus, self.gpu_type = gpu_request(spec)
        self.gpu = modal_gpu_spec(sb, spec) if sb.provider == "modal" else None
        self._sandbox: Any = None
        self.exec: Exec | None = None
        self.description = f"agentenv:{sb.provider} {sb.image}" + (f", gpu={self.gpu}" if self.gpu else (f", {self.gpus} GPU(s)" if self.gpus else ""))

    def provider(self) -> Any:
        if self.sb.provider == "modal":
            from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandboxProvider

            return ModalSandboxProvider(gpu=self.gpu)
        if self.gpus:
            raise RuntimeError(f"agentenv provider {self.sb.provider!r} has no GPU path; use provider 'modal' or a docker GPU host")
        from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider

        return build_sandbox_provider(self.sb.provider)

    async def start(self) -> None:
        from agent_env.providers.sandbox_providers.sandbox import VmSandbox

        provider = self.provider()
        kwargs: dict[str, Any] = {
            "image_name": self.sb.image, "port": 8080, "env": {}, "cpu": self.sb.cpus or 2.0, "memory": self.sb.memory_mb or 8192,
            "timeout": self.sb.timeout_s, "attribution": {"crucible_world": self.spec.world}, "network_policy": network_policy(self.spec),
        }
        params = inspect.signature(provider.create_container).parameters
        if "command" in params:
            kwargs["command"] = ["sleep", "infinity"]
        if "expose_externally" in params:
            kwargs["expose_externally"] = False
        self._sandbox = await provider.create_container(**kwargs)
        prefix = ("sudo", "docker", "exec", "-u", "0", self._sandbox.container_name) if isinstance(self._sandbox, VmSandbox) else ()
        self.exec = AgentEnvSandboxExec(self._sandbox, prefix)

    async def close(self) -> None:
        if self._sandbox is not None:
            await self._sandbox.terminate()
            self._sandbox = None


def open_sandbox(sb: SandboxSpec, spec: WorldSpec, *, pids_limit: int | None = None, memory: str | None = None):
    """The sandbox for a SandboxSpec: our Docker driver (local or a GPU host), or an agentenv provider."""
    if sb.provider == "docker":
        from crucible.world.sandbox import DockerSandbox

        return DockerSandbox(sb, spec, pids_limit=pids_limit, memory=memory)
    return AgentEnvSandbox(sb, spec)
