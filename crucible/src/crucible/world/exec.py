"""Execution surfaces: where an App's state lives and its commands run.

State adapters and Apps talk to an ``Exec``, never to a backend directly. ``LocalExec`` runs on this
machine (CI, toy worlds, trusted scripted policies; **no isolation**). Container execs (Harbor /
docker, remote GPU hosts) implement the same interface in ``crucible.interop``.

Every Exec can run Python snippets (``run_python``): canonicalization code (fingerprints, dumps,
tarballs) is written once in Python and runs identically locally and in any container with
``python3``, so a fingerprint means the same thing on every backend.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExecOut:
    rc: int
    stdout: str
    stderr: str = ""
    duration_s: float = 0.0
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.rc == 0 and not self.timed_out


class ExecError(RuntimeError):
    def __init__(self, msg: str, out: ExecOut | None = None):
        super().__init__(msg)
        self.out = out


# what ``docker exec`` / ``docker compose exec`` print for a container that is no longer there or no longer running
CONTAINER_GONE = re.compile(r"service \"?main\"? is not running|No such container|is not running|container [0-9a-f]+ is (?:not running|paused)"
                            r"|Cannot connect to the Docker daemon|error during connect", re.IGNORECASE)


class EnvContainerLost(RuntimeError):
    """The task container was removed or stopped from outside the episode (or its daemon is unreachable): infrastructure,
    masked as ``crucible:env_container_lost`` (M13 audit B2, M1 audit R-1). Raised only after the host confirmed it."""

    failure_kind = "crucible:env_container_lost"


def container_lost_cause(e: BaseException | None) -> EnvContainerLost | None:
    """The ``EnvContainerLost`` that is ``e`` or sits in its cause / context chain (an App wrapped it in a ToolError), or None."""
    seen: set[int] = set()
    while e is not None and id(e) not in seen:
        if isinstance(e, EnvContainerLost):
            return e
        seen.add(id(e))
        e = e.__cause__ or e.__context__
    return None


class Exec(ABC):
    name: str = "exec"
    is_local: bool = False
    python: str = "python3"
    staging_dir: str = "/tmp"  # where snippets and state payloads are staged; root-only once a guard isolates the world

    @abstractmethod
    async def run(
        self,
        cmd: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
        user: str | None = None,
        stdin: bytes | None = None,
    ) -> ExecOut: ...

    @abstractmethod
    async def push_file(self, src: Path, dest: str) -> None:
        """Copy a host file into the exec's filesystem (creating parent dirs)."""

    @abstractmethod
    async def pull_file(self, src: str, dest: Path) -> None:
        """Copy a file from the exec's filesystem to the host."""

    async def run_python(self, code: str, args: dict[str, Any], *, timeout_s: float | None = 600) -> Any:
        """Run ``code`` (which must define ``main(args) -> json-able``) and return its result."""
        prog = code + "\n\nimport json as _j, sys as _s\n_r = main(_j.loads(_s.stdin.read()))\n_s.stdout.write('\\n__CRUCIBLE_JSON__' + _j.dumps(_r))\n"
        script = f"{self.staging_dir}/.crucible_py_{os.getpid()}_{time.monotonic_ns()}.py"
        await self.write_text(script, prog)
        try:
            out = await self.run(f"{self.python} {shlex.quote(script)}", stdin=json.dumps(args).encode(), timeout_s=timeout_s)
        finally:
            await self.run(f"rm -f {shlex.quote(script)}")
        if not out.ok or "__CRUCIBLE_JSON__" not in out.stdout:
            raise ExecError(f"python snippet failed (rc={out.rc}): {(out.stderr or out.stdout)[-2000:]}", out)
        return json.loads(out.stdout.rsplit("__CRUCIBLE_JSON__", 1)[1])

    async def write_text(self, path: str, text: str) -> None:
        tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f".crucible_push_{os.getpid()}_{time.monotonic_ns()}"
        tmp.write_text(text)
        try:
            await self.push_file(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    async def read_text(self, path: str) -> str:
        tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f".crucible_pull_{os.getpid()}_{time.monotonic_ns()}"
        try:
            await self.pull_file(path, tmp)
            return tmp.read_text(errors="replace")
        finally:
            tmp.unlink(missing_ok=True)

    async def close(self) -> None:
        pass


_SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|COOKIE|SESSION)", re.IGNORECASE)


def safe_environ(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The host environment minus anything that looks like a credential: every subprocess Crucible starts (shells,
    tool servers, the agentenv gateway) gets this, so an API key in the operator's shell never reaches a policy.
    Explicit ``extra`` entries (an App's own ``env`` params, e.g. DATABASE_URL) are kept."""
    env = {k: v for k, v in os.environ.items() if not _SECRET_NAME.search(k)}
    return {**env, **(extra or {})}


def with_deadline(cmd: str, timeout_s: float | None) -> str:
    """Enforce a timeout *inside* the sandbox: killing the local ``docker exec``/compose client leaves the command
    running in the container (Harbor's exec, ``docker exec``). GNU/busybox ``timeout`` kills its process group."""
    if not timeout_s:
        return cmd
    q = shlex.quote(cmd)  # images without coreutils/busybox timeout fall back to the client-side deadline
    return f"if command -v timeout >/dev/null 2>&1; then timeout -s KILL {max(1, math.ceil(timeout_s))} bash -c {q}; else bash -c {q}; fi"


def deadline_hit(rc: int, duration_s: float, timeout_s: float | None) -> bool:
    return bool(timeout_s) and rc in (124, 137) and duration_s >= 0.9 * float(timeout_s)


async def spawn_logged(*argv: str, log: Path | None, **kw: Any) -> asyncio.subprocess.Process:
    """Start a long-running helper process (server, gateway) in its own session, stdout+stderr appended to ``log``.

    The log is opened with ``os.open`` and the parent's descriptor is closed once the child holds it, so
    servers started and stopped many times (sweeps, tests) do not leak file handles.
    """
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644) if log is not None else asyncio.subprocess.DEVNULL
    try:
        return await asyncio.create_subprocess_exec(*argv, stdout=fd, stderr=fd, start_new_session=True, **kw)
    finally:
        if log is not None:
            os.close(fd)


# Commands are waited on with a blocking ``Popen.communicate`` in this pool, never through asyncio's child watcher.
# Python 3.12 on Linux watches children with pidfds registered via ``add_reader``; if any fd is ever closed while
# registered with the loop, the selector keeps a stale key and a later pidfd that reuses the number is silently never
# added to epoll, so the child's exit is lost and the await hangs forever (lab M9 accept gate: a zombie ``mkdir``,
# the gate idle for an hour). ``waitpid`` in a thread cannot lose an exit. Threads only wait: the pool is wide.
_WAITERS = ThreadPoolExecutor(max_workers=256, thread_name_prefix="crucible-exec")


def _blocking_run(argv: list[str], cwd: str | None, env: dict[str, str], timeout_s: float | None, stdin: bytes | None,
                  started: list[subprocess.Popen], sessions: set[int] | None = None) -> tuple[int, bytes, bytes, bool]:
    p = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, start_new_session=True)
    started.append(p)
    if sessions is not None:
        sessions.add(p.pid)  # the session (and process group) id: what the command leaves behind is killed with it
    try:
        out, err = p.communicate(stdin, timeout=timeout_s)
        return p.returncode or 0, out, err, False
    except subprocess.TimeoutExpired:
        _killpg(p)
        out, err = p.communicate()
        return -9, out, err, True


def _killpg(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


async def _run_subprocess(argv: list[str], *, cwd: str | None, env: dict[str, str] | None, timeout_s: float | None, stdin: bytes | None,
                          sessions: set[int] | None = None) -> ExecOut:
    t0 = time.monotonic()
    started: list[subprocess.Popen] = []
    full_env = safe_environ(env)
    if argv and Path(argv[0]).name == "docker" and full_env.get("DOCKER_HOST", "").startswith("ssh://"):
        from crucible.world.docker_tunnel import rewrite_env  # one shared SSH tunnel per remote daemon (sshd throttles per-call SSH)

        full_env = await asyncio.get_running_loop().run_in_executor(_WAITERS, rewrite_env, argv, full_env)
    fut = asyncio.get_running_loop().run_in_executor(_WAITERS, _blocking_run, argv, cwd, full_env, timeout_s, stdin, started, sessions)
    try:
        rc, out, err, timed_out = await asyncio.shield(fut)
    except asyncio.CancelledError:  # the caller gave up: kill the command's session (no orphan keeps running)
        for p in started:
            _killpg(p)
        raise
    return ExecOut(rc=rc, stdout=out.decode(errors="replace"), stderr=err.decode(errors="replace"), duration_s=time.monotonic() - t0, timed_out=timed_out)


class LocalExec(Exec):
    """Runs on this machine. Paths are host paths. No isolation: never give an untrusted policy a shell here."""

    name = "local"
    is_local = True

    def __init__(self, python: str | None = None):
        import sys

        self.python = python or sys.executable
        self.sessions: set[int] = set()  # every command's session: ``kill_sessions`` ends what they left running

    async def run(self, cmd, *, cwd=None, env=None, timeout_s=None, user=None, stdin=None) -> ExecOut:
        if user is not None:
            raise ExecError("LocalExec cannot switch users")
        return await _run_subprocess(["bash", "-c", cmd], cwd=cwd, env=env, timeout_s=timeout_s, stdin=stdin, sessions=self.sessions)

    def kill_sessions(self) -> int:
        """Kill every process group a command of this executor started (background jobs a policy left: M10 audit V5,
        a ``nohup ... &`` outlived the World and masked the next perf grade). Returns how many groups were alive."""
        alive = 0
        for pid in list(self.sessions):
            try:
                os.killpg(pid, signal.SIGKILL)
                alive += 1
            except (ProcessLookupError, PermissionError):
                pass
            self.sessions.discard(pid)
        return alive

    async def push_file(self, src: Path, dest: str) -> None:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copyfile, src, dest)

    async def pull_file(self, src: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copyfile, src, dest)

    async def run_python(self, code: str, args: dict[str, Any], *, timeout_s: float | None = 600) -> Any:
        # Same code path as containers, but in-process: exec the snippet's main() directly.
        ns: dict[str, Any] = {"__name__": "crucible_snippet"}
        exec(compile(code, "<crucible-snippet>", "exec"), ns)  # noqa: S102 - our own constant snippets
        try:
            return await asyncio.to_thread(lambda: json.loads(json.dumps(ns["main"](json.loads(json.dumps(args))))))
        except Exception as e:  # same contract as containers: a failing snippet is an ExecError
            raise ExecError(f"python snippet failed: {type(e).__name__}: {e}") from e


class DockerExec(Exec):
    """``docker exec`` into a running container (optionally on a remote daemon via DOCKER_HOST)."""

    name = "docker"

    def __init__(self, container: str, *, docker_host: str | None = None, default_user: str | None = None, python: str = "python3"):
        self.container = container
        self.docker_host = docker_host
        self.default_user = default_user
        self.python = python

    def _env(self) -> dict[str, str] | None:
        return {"DOCKER_HOST": self.docker_host} if self.docker_host else None

    async def run(self, cmd, *, cwd=None, env=None, timeout_s=None, user=None, stdin=None) -> ExecOut:
        argv = ["docker", "exec"]
        if stdin is not None:
            argv.append("-i")
        if cwd:
            argv += ["-w", cwd]
        for k, v in (env or {}).items():
            argv += ["-e", f"{k}={v}"]
        u = user or self.default_user
        if u:
            argv += ["-u", u]
        argv += [self.container, "bash", "-c", with_deadline(cmd, timeout_s)]
        out = await _run_subprocess(argv, cwd=None, env=self._env(), timeout_s=timeout_s + 15 if timeout_s else None, stdin=stdin)
        if deadline_hit(out.rc, out.duration_s, timeout_s):
            out.timed_out, out.rc = True, -9
        return await self._unless_lost(out)

    async def lost(self) -> str | None:
        """Why the container is lost to the episode (removed, paused or stopped from outside, daemon unreachable), asked of
        the daemon on the same ``DOCKER_HOST``, or None. ``docker stop`` / ``kill`` / ``rm`` / ``pause`` go through the daemon
        API, which no process in the container can do; a container whose own main process exited (the policy killed it, or
        it ran out of memory) is the episode's doing, not lost (the rule Harbor's ``container_lost`` applies)."""
        st = await _run_subprocess(["docker", "inspect", "--format", "{{json .State}}", self.container], cwd=None, env=self._env(),
                                   timeout_s=60, stdin=None)
        if not st.ok:
            if "no such" in st.stderr.lower():
                return f"the task container {self.container} was removed"
            return f"the Docker daemon cannot be reached: {st.stderr.strip()[-300:]}"
        state = json.loads(st.stdout or "{}")
        if state.get("Paused"):
            return f"the task container {self.container} was paused from outside"
        if state.get("Running"):
            return None
        ev = await _run_subprocess(["docker", "events", "--since", state.get("StartedAt") or "0", "--until", str(int(time.time()) + 1),
                                    "--filter", f"container={self.container}", "--format", "{{.Action}}"], cwd=None, env=self._env(),
                                   timeout_s=30, stdin=None)
        outside = sorted({a.split(":")[0].strip() for a in ev.stdout.splitlines()} & {"kill", "stop", "destroy", "pause"}) if ev.ok else []
        return f"the task container {self.container} was stopped from outside ({', '.join(outside)}; status {state.get('Status')})" if outside else None

    async def _unless_lost(self, out: ExecOut) -> ExecOut:
        """``out``, unless it is the docker CLI reporting a lost container and the daemon confirms it: then EnvContainerLost.
        The words alone never decide (a command can print ``No such container`` and exit 1), so the host is asked."""
        if out.rc != 0 and CONTAINER_GONE.search((out.stdout + out.stderr)[-2000:]) and (why := await self.lost()):
            raise EnvContainerLost(why)
        return out

    async def push_file(self, src: Path, dest: str) -> None:
        parent = str(Path(dest).parent)
        await self.run(f"mkdir -p {shlex.quote(parent)}")
        out = await _run_subprocess(["docker", "cp", str(src), f"{self.container}:{dest}"], cwd=None, env=self._env(), timeout_s=600, stdin=None)
        if not (await self._unless_lost(out)).ok:
            raise ExecError(f"docker cp to {dest} failed: {out.stderr[-500:]}", out)

    async def pull_file(self, src: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        out = await _run_subprocess(["docker", "cp", f"{self.container}:{src}", str(dest)], cwd=None, env=self._env(), timeout_s=600, stdin=None)
        if not (await self._unless_lost(out)).ok:
            raise ExecError(f"docker cp from {src} failed: {out.stderr[-500:]}", out)


class SSHExec(Exec):
    """Host processes on a remote machine over SSH (m13 S4: Docker-less GPU hosts such as a Runpod pod, which is itself a
    container without user namespaces). One multiplexed connection (``ControlMaster``) carries every command; files go
    by ``scp`` over it. **No isolation**: commands run as the SSH user with resource limits only (``ulimit``: processes,
    address space, file size, CPU seconds), so never give an untrusted policy a shell here unless explicitly opted in."""

    name = "ssh"

    def __init__(self, target: str, *, port: int | None = None, key: str | None = None, rlimits: dict[str, int] | None = None, python: str = "python3",
                 control_dir: str | None = None, staging_dir: str = "/tmp", known_hosts: str | None = None):
        import hashlib
        import tempfile

        self.target, self.port, self.key, self.python, self.staging_dir = target, port, key, python, staging_dir
        self.rlimits, self.known_hosts = dict(rlimits or {}), known_hosts
        cd = Path(control_dir or tempfile.gettempdir())
        self.control = str(cd / f"crucible-ssh-{hashlib.sha256(f'{target}:{port}'.encode()).hexdigest()[:12]}")

    def _opts(self) -> list[str]:
        o = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30", "-o", "ControlMaster=auto",
             "-o", f"ControlPath={self.control}", "-o", "ControlPersist=120"]
        if self.key:
            o += ["-i", self.key, "-o", "IdentitiesOnly=yes"]
        if self.known_hosts:
            o += ["-o", f"UserKnownHostsFile={self.known_hosts}"]
        return o

    def _limits(self) -> str:
        flags = {"nproc": "-u", "as_mb": "-v", "fsize_mb": "-f", "cpu_s": "-t", "nofile": "-n"}
        parts = []
        for k, v in self.rlimits.items():
            if k not in flags:
                raise ExecError(f"unknown rlimit {k!r} (one of {sorted(flags)})")
            n = int(v) * 1024 if k == "as_mb" else int(v) * 2048 if k == "fsize_mb" else int(v)  # ulimit -v is KiB, -f is 512-byte blocks
            parts.append(f"ulimit {flags[k]} {n}")
        return " && ".join(parts)

    async def run(self, cmd, *, cwd=None, env=None, timeout_s=None, user=None, stdin=None) -> ExecOut:
        if user is not None:
            raise ExecError("SSHExec runs as the SSH user and cannot switch users (no isolation on ssh_exec hosts)")
        inner = with_deadline(cmd, timeout_s)
        pre = [f"cd {shlex.quote(cwd)}"] if cwd else []
        pre += [f"export {k}={shlex.quote(str(v))}" for k, v in (env or {}).items()]
        lim = self._limits()
        script = " && ".join([*pre, *([lim] if lim else []), inner]) if (pre or lim) else inner
        argv = ["ssh", *self._opts(), *(["-p", str(self.port)] if self.port else []), self.target, f"bash -c {shlex.quote(script)}"]
        out = await _run_subprocess(argv, cwd=None, env=None, timeout_s=timeout_s + 30 if timeout_s else None, stdin=stdin)
        if out.rc == 255 and ("Connection" in out.stderr or "ssh:" in out.stderr):
            raise ExecError(f"ssh to {self.target} failed: {out.stderr[-300:]}", out)
        if deadline_hit(out.rc, out.duration_s, timeout_s):
            out.timed_out, out.rc = True, -9
        return out

    async def push_file(self, src: Path, dest: str) -> None:
        await self.run(f"mkdir -p {shlex.quote(str(Path(dest).parent))}")
        out = await _run_subprocess(["scp", "-q", *self._opts(), *(["-P", str(self.port)] if self.port else []), str(src), f"{self.target}:{dest}"],
                                    cwd=None, env=None, timeout_s=600, stdin=None)
        if not out.ok:
            raise ExecError(f"scp to {dest} failed: {out.stderr[-500:]}", out)

    async def pull_file(self, src: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        out = await _run_subprocess(["scp", "-q", *self._opts(), *(["-P", str(self.port)] if self.port else []), f"{self.target}:{src}", str(dest)],
                                    cwd=None, env=None, timeout_s=600, stdin=None)
        if not out.ok:
            raise ExecError(f"scp from {src} failed: {out.stderr[-500:]}", out)

    async def close(self) -> None:
        await _run_subprocess(["ssh", *self._opts(), *(["-p", str(self.port)] if self.port else []), "-O", "exit", self.target], cwd=None, env=None,
                              timeout_s=30, stdin=None)
