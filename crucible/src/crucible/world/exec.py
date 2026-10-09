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
import shlex
import shutil
import time
from abc import ABC, abstractmethod
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


def with_deadline(cmd: str, timeout_s: float | None) -> str:
    """Enforce a timeout *inside* the sandbox: killing the local ``docker exec``/compose client leaves the command
    running in the container (Harbor's exec, ``docker exec``). GNU/busybox ``timeout`` kills its process group."""
    if not timeout_s:
        return cmd
    return f"timeout -s KILL {max(1, math.ceil(timeout_s))} bash -c {shlex.quote(cmd)}"


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


async def _run_subprocess(argv: list[str], *, cwd: str | None, env: dict[str, str] | None, timeout_s: float | None, stdin: bytes | None) -> ExecOut:
    t0 = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        env={**os.environ, **(env or {})},
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin), timeout=timeout_s)
    except TimeoutError:
        try:
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass
        out, err = await proc.communicate()
        return ExecOut(rc=-9, stdout=out.decode(errors="replace"), stderr=err.decode(errors="replace"), duration_s=time.monotonic() - t0, timed_out=True)
    return ExecOut(rc=proc.returncode or 0, stdout=out.decode(errors="replace"), stderr=err.decode(errors="replace"), duration_s=time.monotonic() - t0)


class LocalExec(Exec):
    """Runs on this machine. Paths are host paths. No isolation: never give an untrusted policy a shell here."""

    name = "local"
    is_local = True

    def __init__(self, python: str | None = None):
        import sys

        self.python = python or sys.executable

    async def run(self, cmd, *, cwd=None, env=None, timeout_s=None, user=None, stdin=None) -> ExecOut:
        if user is not None:
            raise ExecError("LocalExec cannot switch users")
        return await _run_subprocess(["bash", "-c", cmd], cwd=cwd, env=env, timeout_s=timeout_s, stdin=stdin)

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
        return out

    async def push_file(self, src: Path, dest: str) -> None:
        parent = str(Path(dest).parent)
        await self.run(f"mkdir -p {shlex.quote(parent)}")
        out = await _run_subprocess(["docker", "cp", str(src), f"{self.container}:{dest}"], cwd=None, env=self._env(), timeout_s=600, stdin=None)
        if not out.ok:
            raise ExecError(f"docker cp to {dest} failed: {out.stderr[-500:]}", out)

    async def pull_file(self, src: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        out = await _run_subprocess(["docker", "cp", f"{self.container}:{src}", str(dest)], cwd=None, env=self._env(), timeout_s=600, stdin=None)
        if not out.ok:
            raise ExecError(f"docker cp from {src} failed: {out.stderr[-500:]}", out)
