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
  require_state_tools  probe for python3 + git (default True; False for in-place Harbor tasks on arbitrary images)
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import uuid
from pathlib import Path
from typing import Any

from harbor.environments.factory import EnvironmentFactory
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from crucible.core.schema.world import AppSpec
from crucible.world.backends.base import Backend, BackendCapabilities
from crucible.world.exec import CONTAINER_GONE, EnvContainerLost, Exec, ExecOut, capped_output, deadline_hit, with_deadline

GPU_IMPORT_PATH = "crucible.interop.harbor.gpu_docker:GpuDockerEnvironment"

# image pulls and Dockerfile builds that fail (M13 audit B12): the deterministic causes are never retried
_PULL = re.compile(r"pull access denied|manifest unknown|manifest for \S+ not found|repository does not exist|failed to resolve (?:source metadata|reference)"
                   r"|not found: manifest|no matching manifest|unauthorized|denied: requested access|toomanyrequests|error pulling image"
                   r"|failed to pull|No such image|requested access to the resource is denied|TLS handshake timeout|i/o timeout"
                   r"|lookup [\w.-]+(?: on \S+)?: no such host|dial tcp [^:]+:\d+: connect: (?:network is unreachable|connection refused)", re.IGNORECASE)
_PULL_FIXED = re.compile(r"pull access denied|manifest unknown|manifest for \S+ not found|repository does not exist|not found: manifest"
                         r"|no matching manifest|requested access to the resource is denied|No such image", re.IGNORECASE)
_BUILD = re.compile(r"failed to solve|failed to compute cache key|did not complete successfully|COPY failed|failed to build|ERROR \[|"
                    r"dockerfile parse error|unknown instruction|failed to read dockerfile", re.IGNORECASE)
_BUILD_FIXED = re.compile(r"failed to compute cache key|COPY failed|: not found|dockerfile parse error|unknown instruction|failed to read dockerfile", re.IGNORECASE)


def classify_start_error(text: str) -> tuple[str, bool] | None:
    """A failed environment start -> (failure kind, retryable), or None when it is neither a pull nor a build failure."""
    cmd = re.search(r"Command: (.*?)\. Return code", text, re.DOTALL)  # Harbor's compose failure message names the subcommand
    sub = next((t for t in (cmd.group(1).split() if cmd else []) if t in ("build", "up", "pull", "create", "start", "run", "exec", "down")), "")
    if _PULL.search(text) and sub in ("", "up", "pull", "build", "create"):
        return "crucible:env_image_unavailable", not _PULL_FIXED.search(text)
    if sub == "build" or (not cmd and _BUILD.search(text)):
        return "crucible:env_build_failed", not _BUILD_FIXED.search(text)
    return None


# Worlds sharing one remote daemon in a process share ``DOCKER_HOST`` by reference count: the first sets it, the last
# restores what was there before (M11 audit H1 V8: the first World to tear down unset it under the others)
_DOCKER_HOST: dict[str, object] = {"value": None, "users": 0, "before": None}


def _acquire_docker_host(host: str) -> None:
    current = os.environ.get("DOCKER_HOST")
    if _DOCKER_HOST["users"]:
        if _DOCKER_HOST["value"] != host:
            raise RuntimeError(f"DOCKER_HOST is held for {_DOCKER_HOST['value']!r} by another World in this process; refusing {host!r}")
    else:
        if current and current != host:
            raise RuntimeError(f"DOCKER_HOST already set to {current!r} in this process; refusing to switch to {host!r}")
        _DOCKER_HOST.update(value=host, before=current)
        os.environ["DOCKER_HOST"] = host
    _DOCKER_HOST["users"] = int(_DOCKER_HOST["users"]) + 1  # type: ignore[call-overload]


def _release_docker_host(host: str) -> None:
    if not _DOCKER_HOST["users"] or _DOCKER_HOST["value"] != host:
        return
    _DOCKER_HOST["users"] = int(_DOCKER_HOST["users"]) - 1  # type: ignore[call-overload]
    if not _DOCKER_HOST["users"]:
        if _DOCKER_HOST["before"] is None:
            os.environ.pop("DOCKER_HOST", None)
        else:
            os.environ["DOCKER_HOST"] = str(_DOCKER_HOST["before"])
        _DOCKER_HOST.update(value=None, before=None)


class HarborExec(Exec):
    """Exec over a Harbor ``BaseEnvironment`` (stderr is merged into stdout by Harbor's Docker exec)."""

    name = "harbor"

    def __init__(self, env: Any, *, python: str = "python3", default_user: str | None = None, lost: Any = None):
        self.env = env
        self.python = python
        self.default_user = default_user
        self.lost = lost  # async () -> reason when the container is gone or was stopped from outside, else None

    async def run(self, cmd, *, cwd=None, env=None, timeout_s=None, user=None, stdin=None) -> ExecOut:
        import time

        if stdin is not None:
            # Harbor's exec has no stdin: stage the bytes in a file and redirect.
            tmp = f"{self.staging_dir}/.crucible_stdin_{uuid.uuid4().hex}"  # args may be secret (forbidden SHAs, gold paths)
            await self._put_bytes(stdin, tmp)
            cmd = f"{{ {cmd} ; }} < {tmp}; __rc=$?; rm -f {tmp}; exit $__rc"
        t0 = time.monotonic()
        try:  # the in-container deadline is the real one; Harbor's client-side timeout is only a backstop. Harbor keeps every byte
            # its exec receives, so the output is cut to a head and a tail inside the container (M1 audit R-2)
            r = await self.env.exec(capped_output(with_deadline(cmd, timeout_s)), cwd=cwd, env=env,
                                    timeout_sec=int(timeout_s) + 15 if timeout_s else None, user=user or self.default_user)
        except RuntimeError as e:
            if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                return ExecOut(rc=-9, stdout="", stderr=str(e), duration_s=time.monotonic() - t0, timed_out=True)
            raise
        out = ExecOut(rc=r.return_code, stdout=r.stdout or "", stderr=r.stderr or "", duration_s=time.monotonic() - t0)
        if out.rc != 0 and self.lost is not None and CONTAINER_GONE.search((out.stdout + out.stderr)[-2000:]):
            # compose answers a removed or stopped container with an ordinary non-zero exit ("service \"main\" is not
            # running"): the policy saw it as its own failed command and the grade blamed it (M13 audit B2). The host
            # decides: a container that is gone or was stopped from outside is the infrastructure's, never the policy's
            why = await self.lost()
            if why:
                raise EnvContainerLost(why)
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
        from crucible.world.sandbox import gpu_request

        gpus, gpu_type = gpu_request(self.spec)
        if gpus:
            cfg.gpus = gpus
            if gpu_type or self.params.get("gpu_types"):  # cloud providers (Modal, GKE, Daytona, ...) pick the GPU class from this
                cfg.gpu_types = list(self.params.get("gpu_types") or [gpu_type])
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
            _acquire_docker_host(docker_host)  # Harbor's compose + docker CLI subprocesses inherit it
        env_dir, cfg, name = self._env_config()
        import time as _time

        self._started_at = _time.time()
        trial = TrialPaths(trial_dir=self.workdir / "harbor_trial")
        trial.mkdir()
        gpu_mode = self.params.get("gpu_mode", "runtime" if self._needs_gpu else "none")
        common = {"environment_dir": env_dir, "environment_name": name, "session_id": self.session_id, "trial_paths": trial, "task_env_config": cfg}
        remote = bool(docker_host) or self.params.get("environment_type", "docker") != "docker"
        if not remote:
            # Harbor's Trial bind-mounts /logs/{verifier,agent,artifacts} from the trial dir (Trial._agent_env_mounts); without
            # them every verifier / oracle redirect into /logs fails and Harbor tasks yield no reward (found by A-1)
            from harbor.trial.trial import _harbor_log_bind_mount

            common["mounts"] = [_harbor_log_bind_mount(source=d.resolve().absolute().as_posix(), target=t, os=cfg.os)
                                for d, t in ((trial.verifier_dir, "/logs/verifier"), (trial.agent_dir, "/logs/agent"), (trial.artifacts_dir, "/logs/artifacts"))]
        iso = self.spec.isolation
        etype = self.params.get("environment_type", "docker")
        if etype != "docker":
            # Any Harbor provider (modal, daytona, gke, e2b, runloop, ...): GPUs come from cfg.gpus / cfg.gpu_types; the guard
            # hardens the sandbox the same way afterwards. Provider-specific options go in backend_params.environment_kwargs.
            from harbor.models.task.config import NetworkMode, NetworkPolicy

            mode = {"no-network": NetworkMode.NO_NETWORK, "allowlist": NetworkMode.ALLOWLIST}.get(self.spec.network, NetworkMode.PUBLIC)
            policy = NetworkPolicy(network_mode=mode, allowed_hosts=list(self.spec.allowed_hosts) if mode == NetworkMode.ALLOWLIST else [])
            # Harbor refuses providers that cannot enforce the policy (_validate_network_policy_support) before creating anything
            self.env = EnvironmentFactory.create_environment(EnvironmentType(etype), **common, network_policy=policy, **self.params.get("environment_kwargs", {}))
            await self._start_env()
            self._exec = HarborExec(self.env, default_user=self.params.get("user"))
            if self.params.get("require_state_tools", True):
                probe = await self._exec.run("command -v python3 >/dev/null && command -v git >/dev/null && echo ok", timeout_s=120)
                if "ok" not in probe.stdout:
                    raise RuntimeError(f"sandbox image needs python3 and git for Crucible's state adapters ({etype})")
            return
        # Local or remote Docker: our overlay subclass (GPUs via the nvidia runtime, no-network, init, the policy's caps).
        # Allowlist worlds and phase network changes need Harbor's egress-control sidecar, which only starts when the policy
        # is passed (m13 audit A-3: it never was, so allowlist worlds could not run on Docker). The overlay's blanket
        # network_mode=none then steps aside: the sidecar enforces the policy.
        from harbor.models.task.config import NetworkMode, NetworkPolicy

        phases = [NetworkPolicy.model_validate(x) for x in self.params.get("phase_network_policies", [])]
        egress = self.spec.network == "allowlist" or bool(phases)
        if egress:
            mode = {"no-network": NetworkMode.NO_NETWORK, "allowlist": NetworkMode.ALLOWLIST}.get(self.spec.network, NetworkMode.PUBLIC)
            common["network_policy"] = NetworkPolicy(network_mode=mode, allowed_hosts=list(self.spec.allowed_hosts) if mode == NetworkMode.ALLOWLIST else [])
            common["phase_network_policies"] = phases
        self.env = EnvironmentFactory.create_environment_from_import_path(
            GPU_IMPORT_PATH, **common, gpu_mode=gpu_mode, network_none=self._network_none() and not egress, pids_limit=iso.pids_limit,
            cap_add=["SYS_RESOURCE"] if iso.policy_user else [], mem_limit=iso.memory,
            security_opt=self.params.get("security_opt", []),  # opt-in: the guard sets no_new_privs on the policy tree itself
            keep_containers=bool(self.params.get("keep_containers")), logs_mounted=not remote)
        if egress and not getattr(self.env, "_enable_egress_control", False):  # fail closed: never run an unenforced allowlist
            raise RuntimeError(f"network {self.spec.network!r} needs Harbor's egress control, which this Docker daemon's kernel does not support "
                               "(nftables fib inet)")
        await self._start_env()
        self._exec = HarborExec(self.env, default_user=self.params.get("user"), lost=self.container_lost)
        if self.params.get("image_digest"):  # the task image's pin (M1 audit R-14): a re-pushed tag never runs
            from crucible.world.sandbox import check_pinned

            check_pinned(self.params["image_digest"], await self._image_id(), f"task container {self.session_id}-main-1")
        if remote:  # no host bind mounts on a remote daemon: the dirs live in the container and Harbor downloads them
            await self._exec.run("mkdir -p /logs/verifier /logs/agent /logs/artifacts", timeout_s=60, user="root")
        if self.params.get("require_state_tools", True):  # in-place Harbor tasks (process adapter) run on any image
            probe = await self._exec.run("command -v python3 >/dev/null && command -v git >/dev/null && echo ok", timeout_s=60)
            if "ok" not in probe.stdout:
                raise RuntimeError("sandbox image needs python3 and git for Crucible's state adapters")

    async def _start_env(self) -> None:
        """Harbor's ``start`` (build or pull, then ``up``). An image that cannot be pulled or a Dockerfile that does not build
        is the environment's failure, named as such with its whole cause kept: the full compose output goes to
        ``env_start_error.log`` in the World's workdir and the failure keeps its tail and that path (M13 audit B12: these
        were ``agent_run_error`` with harness or unknown blame, the cause cut at 500 characters)."""
        try:
            await asyncio.wait_for(self.env.start(force_build=bool(self.params.get("force_build"))), float(self.params.get("start_timeout_s", 1800)))
        except RuntimeError as e:
            text = str(e)
            got = classify_start_error(text)
            if got is None:
                raise
            from crucible.core.schema.run import Failure
            from crucible.runtime.infra import StageFailure

            kind, retryable = got
            log = self.workdir / "env_start_error.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            from crucible.core.secrets import scrub_secrets

            log.write_text(scrub_secrets(text))
            what = "the task image could not be pulled" if kind.endswith("image_unavailable") else "the task's Dockerfile did not build"
            m = (_PULL if kind.endswith("image_unavailable") else _BUILD).search(text)
            cause = text[max(0, m.start() - 200): m.end() + 300] if m else ""
            reason = f"{what} ({'transient' if retryable else 'deterministic'}): …{cause}… | tail: {text[-900:]} | full log: {log}"
            raise StageFailure(Failure(kind=kind, stage="seed", reason=reason[-1990:] if len(reason) > 1990 else reason, retryable=retryable,
                                       detail="pull" if kind.endswith("image_unavailable") else "build")) from e

    # ---- host-side container facts (Docker only; M13 audit B1, B2) -------------------------------------------------
    @property
    def docker_cli(self) -> bool:
        """The container is a Docker container the host CLI can inspect (local or remote daemon)."""
        return self.params.get("environment_type", "docker") == "docker" and self.env is not None

    async def docker(self, *args: str, timeout: float = 120.0) -> tuple[int, bytes, str]:
        proc = await asyncio.create_subprocess_exec("docker", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                                                    stdin=asyncio.subprocess.DEVNULL)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout)
        except TimeoutError:
            proc.kill()
            return -9, b"", f"docker {args[0]} timed out after {timeout:.0f}s"
        return proc.returncode or 0, out, err.decode("utf-8", "replace")

    async def main_container(self) -> str | None:
        """The main service's container id, found once (while it exists) and remembered: a removed container keeps its id."""
        if getattr(self, "_cid", None):
            return self._cid
        if not self.docker_cli:
            return None
        from harbor.environments.docker.docker import _sanitize_docker_compose_project_name

        project = _sanitize_docker_compose_project_name(self.env.session_id)
        rc, out, _ = await self.docker("ps", "-aq", "--no-trunc", "--filter", f"label=com.docker.compose.project={project}",
                                       "--filter", "label=com.docker.compose.service=main", timeout=60)
        cid = out.decode().split()[0] if rc == 0 and out.strip() else None
        self._cid = cid
        return cid

    async def container_state(self) -> dict[str, Any]:
        """Host-side liveness: ``{"supported", "exists", "running", "status", "exit_code", "oom", "outside", "events", "error"}``.
        ``outside``: the daemon API stopped, killed or removed it (``docker stop`` / ``kill`` / ``rm``), which no process
        in the container can do; a container that died without one exited from inside (its own main process)."""
        if not self.docker_cli:
            return {"supported": False}
        cid = await self.main_container()
        if cid is None:
            return {"supported": True, "exists": False, "running": False, "outside": True, "events": [], "error": "no main container"}
        rc, out, err = await self.docker("inspect", "--format", "{{json .State}}", cid, timeout=60)
        if rc != 0:
            daemon = "no such" not in err.lower()
            return {"supported": True, "exists": False if not daemon else None, "running": False, "outside": True, "events": await self._events(cid),
                    "error": err.strip()[-300:]}
        import json as _json

        st = _json.loads(out.decode() or "{}")
        state = {"supported": True, "exists": True, "running": bool(st.get("Running")) and not st.get("Paused"), "status": st.get("Status"),
                 "exit_code": st.get("ExitCode"), "oom": bool(st.get("OOMKilled")), "events": []}
        if not state["running"]:
            state["events"] = await self._events(cid)
            state["outside"] = bool(set(state["events"]) & {"kill", "stop", "destroy", "pause"}) or bool(st.get("Paused"))
        return state

    async def _events(self, cid: str) -> list[str]:
        import time as _time

        rc, out, _ = await self.docker("events", "--since", str(int(getattr(self, "_started_at", _time.time() - 86400)) - 5), "--until",
                                       str(int(_time.time()) + 1), "--filter", f"container={cid}", "--format", "{{.Action}}", timeout=30)
        return [x.split(":")[0].strip() for x in out.decode().splitlines() if x.strip()] if rc == 0 else []

    async def container_lost(self) -> str | None:
        """Why the container is lost to the episode (gone, stopped from outside, daemon unreachable), or None."""
        st = await self.container_state()
        if not st.get("supported"):
            return None
        if st.get("exists") is None:
            return f"the Docker daemon cannot be reached: {st.get('error', '')}"
        if not st.get("exists"):
            return f"the task container was removed ({', '.join(st.get('events') or []) or st.get('error', '')})"
        if not st.get("running") and st.get("outside"):
            return f"the task container was stopped from outside ({', '.join(st.get('events') or [])}; status {st.get('status')})"
        return None

    async def networks(self) -> dict[str, Any] | None:
        """The main container's attached networks (host-side), or None when unknown."""
        cid = await self.main_container()
        if cid is None:
            return None
        rc, out, _ = await self.docker("inspect", "--format", "{{json .NetworkSettings.Networks}}", cid, timeout=60)
        if rc != 0:
            return None
        import json as _json

        return _json.loads(out.decode() or "null") or {}

    async def after_apps_started(self, world) -> None:
        """Isolation (Karotte-style), after Apps seeded their files as root (``crucible.world.guard.isolate``)."""
        if not self.spec.isolation.policy_user:
            return
        from crucible.world.guard.isolate import isolate_world

        assert self._exec is not None
        self.isolation = await isolate_world(world, self._exec, base_path=self.params.get("base_path", "/"),
                                             root_only=self.params.get("root_only", []))

    async def _image_id(self) -> str | None:
        if getattr(self, "_image", None) is None and self.params.get("environment_type", "docker") == "docker":
            from crucible.world.sandbox import image_id

            self._image = await image_id(f"{self.session_id}-main-1", self.params.get("docker_host"), container=True) or ""
        return getattr(self, "_image", None) or None

    async def snapshot_state(self) -> dict[str, Any]:
        img = await self._image_id()
        return {"image": img} if img else {}

    async def restore_state(self, state: dict[str, Any]) -> None:
        from crucible.world.sandbox import check_image

        check_image(state or {}, await self._image_id())

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
                _release_docker_host(self.params["docker_host"])
