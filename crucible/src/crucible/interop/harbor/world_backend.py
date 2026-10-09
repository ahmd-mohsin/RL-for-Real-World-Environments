"""Harbor backend: a World's Apps live in a Harbor sandbox environment (DEPEND: harbor.environments).

The container is created by Harbor's own ``EnvironmentFactory`` (Docker / GPU Docker here; any of
Harbor's ~35 providers through ``environment_type`` / ``import_path``), and every App executes through
Harbor's ``exec`` / ``upload_file`` / ``download_file``. A remote Docker daemon (our lab RTX 5090) is
used by setting ``docker_host`` (``ssh://user@host``), which Harbor's compose subprocess inherits.

Params:
  task_dir          Harbor task directory (resources, environment dir, network mode) — optional
  environment_dir   used when no task_dir (needs a Dockerfile or ``docker_image``)
  docker_image      prebuilt image (skips the build; Harbor's prebuilt compose path)
  docker_host       e.g. "ssh://ahmed@100.100.101.36"
  gpu_mode          runtime | devices | none (default: runtime when any App needs GPUs, else none)
  network_none      default: True when the world's network is "no-network"
  base_path         where relative App paths resolve in the container (default "/")
  user              default exec user; keep_containers; start_timeout_s
"""

from __future__ import annotations

import asyncio
import os
import shlex
import uuid
from pathlib import Path
from typing import Any

from harbor.environments.factory import EnvironmentFactory
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from crucible.core.schema.world import AppSpec
from crucible.world.backends.base import Backend, BackendCapabilities
from crucible.world.exec import Exec, ExecOut, deadline_hit, with_deadline

GPU_IMPORT_PATH = "crucible.interop.harbor.gpu_docker:GpuDockerEnvironment"


class HarborExec(Exec):
    """Exec over a Harbor ``BaseEnvironment`` (stderr is merged into stdout by Harbor's Docker exec)."""

    name = "harbor"

    def __init__(self, env: Any, *, python: str = "python3", default_user: str | None = None):
        self.env = env
        self.python = python
        self.default_user = default_user

    async def run(self, cmd, *, cwd=None, env=None, timeout_s=None, user=None, stdin=None) -> ExecOut:
        import time

        if stdin is not None:
            # Harbor's exec has no stdin: stage the bytes in a file and redirect.
            tmp = f"{self.staging_dir}/.crucible_stdin_{uuid.uuid4().hex}"  # args may be secret (forbidden SHAs, gold paths)
            await self._put_bytes(stdin, tmp)
            cmd = f"{{ {cmd} ; }} < {tmp}; __rc=$?; rm -f {tmp}; exit $__rc"
        t0 = time.monotonic()
        try:  # the in-container deadline is the real one; Harbor's client-side timeout is only a backstop
            r = await self.env.exec(with_deadline(cmd, timeout_s), cwd=cwd, env=env, timeout_sec=int(timeout_s) + 15 if timeout_s else None,
                                    user=user or self.default_user)
        except RuntimeError as e:
            if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                return ExecOut(rc=-9, stdout="", stderr=str(e), duration_s=time.monotonic() - t0, timed_out=True)
            raise
        out = ExecOut(rc=r.return_code, stdout=r.stdout or "", stderr=r.stderr or "", duration_s=time.monotonic() - t0)
        if deadline_hit(out.rc, out.duration_s, timeout_s):
            out.timed_out, out.rc = True, -9
        return out

    async def _put_bytes(self, data: bytes, dest: str) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
        try:
            await self.env.upload_file(f.name, dest)
        finally:
            Path(f.name).unlink(missing_ok=True)

    async def push_file(self, src: Path, dest: str) -> None:
        await self.env.exec(f"mkdir -p {shlex.quote(str(Path(dest).parent))}")
        await self.env.upload_file(src, dest)

    async def pull_file(self, src: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        await self.env.download_file(src, dest)


class HarborBackend(Backend):
    name = "harbor"

    def __init__(self, spec, params, *, workdir):
        super().__init__(spec, params, workdir=workdir)
        self.env: Any = None
        self._exec: HarborExec | None = None
        self.session_id = f"crucible-{spec.world.replace('/', '-')[:40]}-{uuid.uuid4().hex[:8]}".lower()
        self._needs_gpu = any(a.resources.gpus for a in spec.apps)
        self._prev_docker_host: str | None = None

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(isolation="container", gpus=self._needs_gpu, network_enforced=self._network_none(),
                                   notes=[f"harbor env {type(self.env).__name__ if self.env else '?'}", f"docker_host={self.params.get('docker_host') or 'local'}"])

    def _network_none(self) -> bool:
        return bool(self.params.get("network_none", self.spec.network == "no-network"))

    def validate(self) -> list[str]:
        problems = []
        for a in self.spec.apps:
            if a.container not in (None, "main"):
                problems.append(f"app {a.id!r} asks for container {a.container!r}; the harbor backend runs one 'main' container")
            if a.template == "common/sqlite_app" and a.params.get("tools_module"):
                problems.append(f"app {a.id!r}: python tool modules run host-side; serve them in-container via the agentenv backend")
        if not (self.params.get("task_dir") or self.params.get("environment_dir") or self.params.get("docker_image")):
            problems.append("harbor backend needs task_dir, environment_dir or docker_image")
        return problems

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        base = self.world.pack_root if self.world is not None and self.world.pack_root else Path.cwd()
        return (base / path).resolve()

    def _env_config(self) -> tuple[Path, EnvironmentConfig, str]:
        if self.params.get("task_dir"):
            from crucible.interop.harbor.task_io import load_harbor_task

            t = load_harbor_task(self._resolve(self.params["task_dir"]), require_tests=False)
            cfg = t.config.environment.model_copy(deep=True)
            env_dir = t.task.paths.environment_dir
            name = t.task.short_name
        else:
            env_dir = self._resolve(self.params["environment_dir"]) if self.params.get("environment_dir") else self.workdir / "empty-env"
            env_dir.mkdir(parents=True, exist_ok=True)
            cfg = EnvironmentConfig()
            name = self.spec.world.replace("/", "-")
        if self.params.get("docker_image"):
            cfg.docker_image = self.params["docker_image"]
        gpus = max((a.resources.gpus for a in self.spec.apps), default=0)
        if gpus:
            cfg.gpus = gpus
        for key in ("cpus", "memory_mb"):
            if self.params.get(key):
                setattr(cfg, key, self.params[key])
        return env_dir, cfg, name

    async def provision(self) -> None:
        docker_host = self.params.get("docker_host")
        if not docker_host and self.params.get("gpu_host"):
            from crucible.core.config import get_config

            host = get_config().gpu_hosts.get(self.params["gpu_host"])
            if host is None:
                raise RuntimeError(f"gpu_host {self.params['gpu_host']!r} is not configured in crucible.toml [gpu_hosts]")
            docker_host = host.docker_host or f"ssh://{host.ssh}"
            self.params["docker_host"] = docker_host
        if docker_host:
            current = os.environ.get("DOCKER_HOST")
            if current and current != docker_host:
                raise RuntimeError(f"DOCKER_HOST already set to {current!r} in this process; refusing to switch to {docker_host!r}")
            self._prev_docker_host = current
            os.environ["DOCKER_HOST"] = docker_host  # Harbor's compose + docker CLI subprocesses inherit it
        env_dir, cfg, name = self._env_config()
        trial = TrialPaths(trial_dir=self.workdir / "harbor_trial")
        trial.mkdir()
        gpu_mode = self.params.get("gpu_mode", "runtime" if self._needs_gpu else "none")
        common = {"environment_dir": env_dir, "environment_name": name, "session_id": self.session_id, "trial_paths": trial, "task_env_config": cfg}
        iso = self.spec.isolation
        # Always our overlay subclass: GPUs (if any), no-network, init, and the policy sandbox's resource caps.
        self.env = EnvironmentFactory.create_environment_from_import_path(
            GPU_IMPORT_PATH, **common, gpu_mode=gpu_mode, network_none=self._network_none(), pids_limit=iso.pids_limit,
            cap_add=["SYS_RESOURCE"] if iso.policy_user else [], mem_limit=iso.memory,
            security_opt=self.params.get("security_opt", []),  # opt-in: the guard sets no_new_privs on the policy tree itself
            keep_containers=bool(self.params.get("keep_containers")))
        await asyncio.wait_for(self.env.start(force_build=bool(self.params.get("force_build"))), float(self.params.get("start_timeout_s", 1800)))
        self._exec = HarborExec(self.env, default_user=self.params.get("user"))
        probe = await self._exec.run("command -v python3 >/dev/null && command -v git >/dev/null && echo ok", timeout_s=60)
        if "ok" not in probe.stdout:
            raise RuntimeError("sandbox image needs python3 and git for Crucible's state adapters")

    async def after_apps_started(self, world) -> None:
        """Isolation (Karotte-style), after Apps seeded their files as root (``crucible.world.guard.isolate``)."""
        if not self.spec.isolation.policy_user:
            return
        from crucible.world.guard.isolate import isolate_world

        assert self._exec is not None
        self.isolation = await isolate_world(world, self._exec, base_path=self.params.get("base_path", "/"),
                                             root_only=self.params.get("root_only", []))

    def exec_for(self, app: AppSpec) -> Exec:
        assert self._exec is not None, "provision() first"
        return self._exec

    def base_path(self, app: AppSpec) -> str:
        return self.params.get("base_path", "/")

    async def teardown(self) -> None:
        try:
            if self.env is not None:
                await asyncio.shield(self.env.stop(delete=not self.params.get("keep_containers")))
        finally:
            if self.params.get("docker_host"):
                if self._prev_docker_host is None:
                    os.environ.pop("DOCKER_HOST", None)
                else:
                    os.environ["DOCKER_HOST"] = self._prev_docker_host
