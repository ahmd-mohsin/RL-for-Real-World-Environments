"""Apptainer sandboxes and World backend: containers on HPC hosts without Docker (Marlowe; swarm-pipeline/05-marlowe.md).

The same contract as ``world.sandbox.DockerSandbox`` (``exec``, ``gpus``, ``description``, ``start``, ``close``) over an
**Apptainer instance**: one long-lived, unprivileged container per World, started with

    apptainer instance start --containall --cleanenv --no-home --net --network none --writable-tmpfs --bind <ws>:<ws> <image> <name>

and every command run as ``apptainer exec --cleanenv instance://<name> bash -c ...``. ``runtime.harness.sandbox.ExecSandbox``
over ``ApptainerSandbox.exec`` (with ``alive=sandbox.alive``) is the slime four-method ``Sandbox`` (exec, read_file,
write_file, async context, ``SandboxTerminatedError``): a command into an instance that is gone raises ``ExecError``, which
``ExecSandbox`` turns into ``SandboxTerminatedError`` once the liveness probe agrees (V8: infrastructure, masked).

**Isolation achieved** (measured on the Marlowe login node, apptainer 1.5.4, non-setuid, user namespaces):
- network: ``--net --network none`` is allowed for unprivileged users: only loopback, a connect to a public address fails
  with ENETUNREACH (``egress_probe`` checks it inside the sandbox);
- filesystem: ``--containall`` gives a private /tmp and /dev, no host home, /users or /scratch; the image is read-only
  and ``--writable-tmpfs`` takes writes outside the bound workspace; only the workspace is bound, at its host path;
- environment: ``--cleanenv`` on start *and* on every exec (an exec without it carries the host environment in);
- processes: its own PID namespace; ``instance stop`` ends every process the policy left behind;
- identity: the invoking user, no root, no setuid, no capabilities (no ``policy_user`` demotion: there is no root to
  demote from);
- limits: cgroup flags (``--memory``, ``--cpus``, ``--pids-limit``) need a delegated cgroup; they failed on the login
  node, so ``limits`` are applied when the host allows them (``cgroups="auto"`` retries without them and says so) and the
  Slurm job's cgroup bounds the rest;
- GPUs: ``--nv`` binds the host driver and the job's GPUs into the container.

**Instance state.** Apptainer keeps instance files under ``~/.apptainer``; ``configdir`` (``APPTAINER_CONFIGDIR``)
moves them (on Marlowe the home directory was over quota and every ``instance start`` failed there).

``ApptainerBackend`` (registry ``world_backend: apptainer``) runs a World's Apps in one instance whose work directory is
bound at its host path, so host paths and container paths agree (as in the window lane's Docker World).
``open_workspace_world`` is the window-lane pattern: the App state on the host, the policy's tools in the container.
"""

from __future__ import annotations

import base64
import json
import shlex
import shutil
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Self

from crucible.core.schema.world import AppSpec, WorldSpec
from crucible.world.backends.base import Backend, BackendCapabilities
from crucible.world.exec import Exec, ExecError, ExecOut, _run_subprocess, deadline_hit, with_deadline

Runner = Callable[..., Awaitable[ExecOut]]
PREFIX = "crucible-apt-"
# apptainer's own wording when the instance a command targets does not exist (stopped, or the node lost it)
GONE_WORDS = ("no instance found", "instance not found", "while setting image/instance")
DEFAULT_ENV = {"HOME": "/tmp", "LANG": "C.UTF-8"}


def _host_env(configdir: str | None) -> dict[str, str] | None:
    return {"APPTAINER_CONFIGDIR": configdir} if configdir else None


class ApptainerExec(Exec):
    """``apptainer exec --cleanenv instance://<name>``: commands in a running instance as the invoking user.

    ``env`` is exported inside the command (``--env`` splits values on commas). Files travel on stdin (``cat >``) and back
    as base64 (binary-safe), so paths inside the instance's private /tmp work like the bound workspace."""

    name = "apptainer"

    def __init__(self, instance: str, *, apptainer: str = "apptainer", env: Mapping[str, str] | None = None, configdir: str | None = None,
                 python: str = "python3", runner: Runner | None = None):
        self.instance, self.apptainer, self.configdir, self.python = instance, apptainer, configdir, python
        self.env = {**DEFAULT_ENV, **dict(env or {})}
        self._runner = runner or _run_subprocess

    def argv(self, cmd: str, *, cwd: str | None = None, env: Mapping[str, str] | None = None, timeout_s: float | None = None) -> list[str]:
        exports = "".join(f"export {k}={shlex.quote(str(v))}; " for k, v in {**self.env, **dict(env or {})}.items())
        return [self.apptainer, "exec", "--cleanenv", "--pwd", cwd or "/", f"instance://{self.instance}", "bash", "-c",
                exports + with_deadline(cmd, timeout_s)]

    async def run(self, cmd, *, cwd=None, env=None, timeout_s=None, user=None, stdin=None) -> ExecOut:
        if user not in (None, "root"):
            raise ExecError(f"apptainer runs every command as the invoking user (unprivileged): cannot switch to {user!r}")
        out = await self._runner(self.argv(cmd, cwd=cwd, env=env, timeout_s=timeout_s), cwd=None, env=_host_env(self.configdir),
                                 timeout_s=timeout_s + 15 if timeout_s else None, stdin=stdin)
        if out.rc != 0 and any(w in (out.stderr or "").lower() for w in GONE_WORDS):
            raise ExecError(f"apptainer instance {self.instance} is gone: {out.stderr.strip()[-300:]}", out)
        if deadline_hit(out.rc, out.duration_s, timeout_s):
            out.timed_out, out.rc = True, -9
        return out

    async def push_file(self, src: Path, dest: str) -> None:
        q = shlex.quote(dest)
        out = await self.run(f"mkdir -p \"$(dirname {q})\" && cat > {q}", stdin=Path(src).read_bytes(), timeout_s=600)
        if not out.ok:
            raise ExecError(f"copy to {dest} in {self.instance} failed: {(out.stderr or out.stdout)[-500:]}", out)

    async def pull_file(self, src: str, dest: Path) -> None:
        q = shlex.quote(src)
        out = await self.run(f"base64 -w0 {q} 2>/dev/null || {self.python} -c 'import base64,sys;sys.stdout.write(base64.b64encode(open(sys.argv[1],\"rb\").read()).decode())' {q}",
                             timeout_s=600)
        if not out.ok:
            raise ExecError(f"copy from {src} in {self.instance} failed: {(out.stderr or out.stdout)[-500:]}", out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(base64.b64decode(out.stdout.strip()))


class ApptainerSandbox:
    """One Apptainer instance from a SIF (or any image ``apptainer`` accepts: ``docker://``, a sandbox directory).

    ``binds``: ``(host, container, "rw"|"ro")``; ``network``: ``"none"`` (loopback only) or ``"host"``; ``gpus``: > 0 adds
    ``--nv``; ``overlay``: a writable overlay image instead of ``--writable-tmpfs``; ``limits``: ``memory`` (e.g. ``"4G"``),
    ``cpus``, ``pids``; ``cgroups``: ``"auto"`` (try the limits, start without them when the host refuses, recorded in
    ``self.cgroups_applied`` and the description), ``"on"`` (refuse to start without them) or ``"off"``."""

    def __init__(self, image: str, *, name: str | None = None, binds: Sequence[tuple[str, str, str]] = (), network: str = "none", gpus: int = 0,
                 overlay: str | None = None, limits: Mapping[str, Any] | None = None, cgroups: str = "auto", env: Mapping[str, str] | None = None,
                 apptainer: str = "apptainer", configdir: str | None = None, runner: Runner | None = None, label: str = ""):
        if network not in ("none", "host"):
            raise ValueError(f"network: none | host (got {network!r})")
        if cgroups not in ("auto", "on", "off"):
            raise ValueError(f"cgroups: auto | on | off (got {cgroups!r})")
        for b in binds:
            if len(b) != 3 or b[2] not in ("rw", "ro"):
                raise ValueError(f"bind {b!r}: (host, container, rw|ro)")
        self.image, self.binds, self.network, self.gpus, self.overlay = image, list(binds), network, int(gpus), overlay
        self.limits, self.cgroups, self.apptainer, self.configdir = dict(limits or {}), cgroups, apptainer, configdir
        self.name = name or f"{PREFIX}{(label or 'w').replace('/', '-')[:24]}-{uuid.uuid4().hex[:10]}"
        self._runner = runner or _run_subprocess
        self.exec: ApptainerExec = ApptainerExec(self.name, apptainer=apptainer, env=env, configdir=configdir, runner=self._runner)
        self.cgroups_applied: bool | None = None
        self.started = False

    @property
    def description(self) -> str:
        lim = {None: "", True: ", cgroup limits", False: ", no cgroup limits (host refused; the job's cgroup bounds it)"}[self.cgroups_applied]
        return (f"apptainer instance {self.name} of {self.image} (containall, cleanenv, no-home, network {self.network}"
                + (f", {self.gpus} GPU(s) via --nv" if self.gpus else "") + lim + ")")

    def _limit_flags(self) -> list[str]:
        f: list[str] = []
        if self.limits.get("memory"):
            f += ["--memory", str(self.limits["memory"])]
        if self.limits.get("cpus"):
            f += ["--cpus", str(self.limits["cpus"])]
        if self.limits.get("pids"):
            f += ["--pids-limit", str(int(self.limits["pids"]))]
        return f

    def argv(self, *, with_limits: bool = True) -> list[str]:
        a = [self.apptainer, "instance", "start", "--containall", "--cleanenv", "--no-home"]
        if self.network == "none":
            a += ["--net", "--network", "none"]
        if self.gpus:
            a += ["--nv"]
        a += ["--overlay", self.overlay] if self.overlay else ["--writable-tmpfs"]
        for host, cont, mode in self.binds:
            a += ["--bind", f"{host}:{cont}:{mode}"]
        if with_limits and self.cgroups != "off":
            a += self._limit_flags()
        return [*a, self.image, self.name]

    async def _call(self, argv: list[str], timeout_s: float = 300) -> ExecOut:
        return await self._runner(argv, cwd=None, env=_host_env(self.configdir), timeout_s=timeout_s, stdin=None)

    async def start(self) -> None:
        if self.configdir:
            Path(self.configdir).mkdir(parents=True, exist_ok=True)
        limits = bool(self._limit_flags()) and self.cgroups != "off"
        out = await self._call(self.argv(with_limits=limits), timeout_s=1800)
        if not out.ok and limits and self.cgroups == "auto":
            out = await self._call(self.argv(with_limits=False), timeout_s=1800)
            limits = False
        if not out.ok:
            raise RuntimeError(f"apptainer instance failed to start ({self.description}): {(out.stderr or out.stdout)[-800:]}")
        self.cgroups_applied = limits if self._limit_flags() else None
        self.started = True

    async def alive(self) -> bool:
        out = await self._call([self.apptainer, "instance", "list", "--json", self.name], timeout_s=60)
        if not out.ok:
            return False
        try:
            return any(i.get("instance") == self.name for i in json.loads(out.stdout or "{}").get("instances") or [])
        except json.JSONDecodeError:
            return self.name in (out.stdout or "")

    async def close(self) -> None:
        if self.started:
            await self._call([self.apptainer, "instance", "stop", self.name], timeout_s=120)
            self.started = False

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def isolation_report(self, must_hide: Sequence[str] = ()) -> dict[str, Any]:
        """Measured inside the instance (evidence, not configuration): the uid, a public connect, the network interfaces,
        the processes visible, whether the root is writable (the ephemeral overlay), and which host paths that must stay
        outside are visible: the host home, /cm/shared, /etc/slurm and ``must_hide`` (e.g. the code and the venv). The
        parents of a bound workspace exist inside as empty mount points, so a path counts only when it exists."""
        hide = sorted({str(Path.home()), "/cm/shared", "/etc/slurm", *must_hide} - {b[1] for b in self.binds})
        probe = ("import json,os,socket,sys\n"
                 "s=socket.socket();s.settimeout(3)\n"
                 "try:\n s.connect(('1.1.1.1',53));net='open'\nexcept Exception as e:\n net='blocked: '+type(e).__name__\n"
                 "ifs=[l.split(':')[0].strip() for l in open('/proc/net/dev').read().splitlines()[2:]]\n"
                 "def w(p):\n try:\n  open(p,'w').close();os.unlink(p);return True\n except OSError:\n  return False\n"
                 "print(json.dumps({'uid':os.getuid(),'public_connect':net,'interfaces':ifs,"
                 "'host_paths_visible':[p for p in json.loads(sys.argv[1]) if os.path.exists(p)],"
                 "'pids_visible':len([p for p in os.listdir('/proc') if p.isdigit()]),'usr_writable':w('/usr/.crucible_probe'),"
                 "'env_keys':sorted(os.environ)}))\n")
        out = await self.exec.run(f"{self.exec.python} -c {shlex.quote(probe)} {shlex.quote(json.dumps(hide))}", timeout_s=60)
        if not out.ok:
            return {"ok": False, "error": (out.stderr or out.stdout)[-400:]}
        r = json.loads(out.stdout.strip().splitlines()[-1])
        r["network_none"] = self.network == "none"
        net_ok = self.network != "none" or (r["public_connect"].startswith("blocked") and set(r["interfaces"]) <= {"lo"})
        # host variables that must never arrive (--cleanenv): the scheduler's, Crucible's, Hugging Face's (the image's own, e.g. GPG_KEY, stay)
        r["host_env_leaked"] = [k for k in r["env_keys"] if k.startswith(("SLURM_", "CRU_", "HF_", "CRUCIBLE_", "OPENROUTER", "AWS_", "SSH_"))]
        r["ok"] = net_ok and not r["host_paths_visible"] and not r["host_env_leaked"]  # usr_writable: the ephemeral overlay, never the SIF
        r["cgroups_applied"] = self.cgroups_applied
        return r


async def run_once(image: str, cmd: str, *, binds: Sequence[tuple[str, str, str]] = (), cwd: str = "/", network: str = "none", gpus: int = 0,
                   env: Mapping[str, str] | None = None, timeout_s: float | None = 1800, apptainer: str = "apptainer", runner: Runner | None = None) -> ExecOut:
    """One command in a fresh container (no instance): a grader's separate environment (V4), or a GPU probe."""
    a = [apptainer, "exec", "--containall", "--cleanenv", "--no-home", "--writable-tmpfs", "--pwd", cwd]
    if network == "none":
        a += ["--net", "--network", "none"]
    if gpus:
        a += ["--nv"]
    for host, cont, mode in binds:
        a += ["--bind", f"{host}:{cont}:{mode}"]
    exports = "".join(f"export {k}={shlex.quote(str(v))}; " for k, v in {**DEFAULT_ENV, **dict(env or {})}.items())
    out = await (runner or _run_subprocess)([*a, image, "bash", "-c", exports + with_deadline(cmd, timeout_s)], cwd=None, env=None,
                                            timeout_s=timeout_s + 30 if timeout_s else None, stdin=None)
    if deadline_hit(out.rc, out.duration_s, timeout_s):
        out.timed_out, out.rc = True, -9
    return out


class ApptainerBackend(Backend):
    """A World's Apps in one Apptainer instance. ``backend_params``: ``image`` (required: a SIF path or ``docker://...``),
    ``binds`` (extra ``[host, container, mode]``), ``limits``, ``cgroups``, ``configdir``, ``apptainer``, ``keep_workdir``.
    The work directory's ``apps`` folder is bound read-write at its host path; GPUs follow the Apps' resources."""

    name = "apptainer"

    def __init__(self, spec: WorldSpec, params: dict[str, Any], *, workdir: Path, runner: Runner | None = None):
        super().__init__(spec, params, workdir=workdir)
        self.apps_dir = (workdir / "apps").resolve()
        gpus = max((a.resources.gpus for a in spec.apps if a.resources.gpus), default=0)
        self.sandbox = ApptainerSandbox(
            str(params.get("image") or ""), label=spec.world, network="none" if spec.network == "no-network" else "host", gpus=gpus,
            binds=[(str(self.apps_dir), str(self.apps_dir), "rw"), *[tuple(b) for b in params.get("binds") or []]],
            limits=params.get("limits"), cgroups=params.get("cgroups", "auto"), configdir=params.get("configdir"),
            apptainer=params.get("apptainer", "apptainer"), runner=runner)

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(isolation="container", gpus=True, network_enforced=self.spec.network == "no-network",
                                   notes=["apptainer: unprivileged user namespace; commands run as the invoking user (no policy_user demotion)"])

    def validate(self) -> list[str]:
        problems = []
        if not self.params.get("image"):
            problems.append("apptainer backend needs backend_params.image (a SIF path or docker://...)")
        if self.spec.isolation.policy_user:
            problems.append("apptainer runs unprivileged: isolation.policy_user (a demoted uid inside the container) is not available")
        return problems

    async def provision(self) -> None:
        self.apps_dir.mkdir(parents=True, exist_ok=True)
        await self.sandbox.start()

    def exec_for(self, app: AppSpec) -> Exec:
        return self.sandbox.exec

    def base_path(self, app: AppSpec) -> str:
        return str(self.apps_dir)

    async def teardown(self) -> None:
        await self.sandbox.close()
        if not self.params.get("keep_workdir"):
            shutil.rmtree(self.apps_dir, ignore_errors=True)


async def open_workspace_world(spec: WorldSpec, *, image: str, workspace: Path, workdir: Path, instruction: str, app: str = "repo",
                               actor: str = "agent", workspace_mode: str = "rw", env: Mapping[str, str] | None = None, gpus: int = 0,
                               limits: Mapping[str, Any] | None = None, configdir: str | None = None, runner: Runner | None = None) -> Any:
    """The window lane's World on Apptainer (``factory.lanes.windows_world.open_world`` with Docker): the App's state on
    the host, its policy tools (``policy_exec``) in an instance that binds ``workspace`` at the same path. Closing the
    World stops the instance. ``world._crucible_sandbox`` is the ``ApptainerSandbox``; ``world._crucible_ws`` the
    workspace."""
    from crucible.core.store import Store
    from crucible.world.world import World

    if workspace_mode not in ("rw", "ro"):
        raise ValueError(f"workspace_mode is rw or ro, got {workspace_mode!r}")
    workdir, ws = Path(workdir).resolve(), Path(workspace).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    sb = ApptainerSandbox(image, label=spec.world, binds=[(str(ws), str(ws), workspace_mode)], network="none" if spec.network == "no-network" else "host",
                          gpus=gpus, limits=limits, env=env, configdir=configdir, runner=runner)
    await sb.start()
    w = World(spec, instructions={actor: instruction}, store=Store(workdir / "s"), workdir=workdir / "w")
    try:
        await w.start()
        w.apps[app].ctx.extra["policy_exec"] = sb.exec
        await w.reset()
    except BaseException:
        await sb.close()
        raise
    w._crucible_ws, w._crucible_sandbox = ws, sb
    inner = w.close

    async def close() -> None:
        try:
            await inner()
        finally:
            await sb.close()

    w.close = close
    return w


def instances(*, apptainer: str = "apptainer", configdir: str | None = None) -> list[str]:
    """Names of this user's running Crucible instances (``crucible-apt-*``): what a cleanup or a leak check looks for."""
    import subprocess

    env = None
    if configdir:
        import os

        env = {**os.environ, "APPTAINER_CONFIGDIR": configdir}
    r = subprocess.run([apptainer, "instance", "list", "--json"], capture_output=True, text=True, timeout=60, check=False, env=env)
    try:
        return [i["instance"] for i in json.loads(r.stdout or "{}").get("instances") or [] if str(i.get("instance", "")).startswith(PREFIX)]
    except json.JSONDecodeError:
        return []


__all__ = ["ApptainerBackend", "ApptainerExec", "ApptainerSandbox", "instances", "open_workspace_world", "run_once"]
