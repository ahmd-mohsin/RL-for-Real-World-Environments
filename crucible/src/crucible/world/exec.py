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
import collections
import json
import math
import os
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import threading
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


# daemon API actions on a container that no process inside it can cause: from outside the episode
OUTSIDE_ACTIONS = frozenset({"kill", "stop", "pause"})


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


HOST_UNREACHABLE = "host_unreachable"  # the Failure detail (and reason prefix) of a remote host that stopped answering
# what the Docker CLI prints when it never reached the daemon (nothing was dispatched), and any transport failure of a call
DOCKER_NOT_DISPATCHED = re.compile(r"Cannot connect to the Docker daemon|dial unix [^\n]*connect: (?:connection refused|no such file)", re.IGNORECASE)
DOCKER_TRANSPORT = re.compile(DOCKER_NOT_DISPATCHED.pattern + r"|error during connect|unexpected EOF|connection reset by peer|broken pipe"
                              r"|use of closed network connection|i/o timeout|context deadline exceeded", re.IGNORECASE)
SSH_NOT_DISPATCHED = re.compile(r"Connection (?:timed out|refused)|Could not resolve hostname|No route to host|Network is unreachable"
                                r"|Operation timed out|connect to host", re.IGNORECASE)


class HostUnreachable(ExecError):
    """A remote host (a GPU World's Docker daemon, an SSH target) stopped answering, as its probe confirmed
    (``world.hostreach``, 2026-10-26). Infrastructure: the World and the Runner record it as ``transport_unreachable`` with
    detail ``host_unreachable`` and mask the attempt (V8). ``dispatched``: the operation may have reached the host before it
    was cut, so running it again could repeat it; ``resumed``: the host came back after the pause, but the operation was cut
    mid-way and is not replayable (a policy's shell or write call): the World tells the agent and the episode goes on;
    ``paused_s``: how long the episode was paused for it."""

    failure_kind = "transport_unreachable"
    detail = HOST_UNREACHABLE

    def __init__(self, host: str, why: str, *, dispatched: bool = False, resumed: bool = False, paused_s: float = 0.0):
        super().__init__(why if why.startswith(HOST_UNREACHABLE) else f"{HOST_UNREACHABLE}: {why}")
        self.host, self.why, self.dispatched, self.resumed, self.paused_s = host, why, dispatched, resumed, paused_s


def host_unreachable_cause(e: BaseException | None) -> HostUnreachable | None:
    """The ``HostUnreachable`` that is ``e`` or sits in its cause / context chain (an App wrapped it in a ToolError), or None."""
    seen: set[int] = set()
    while e is not None and id(e) not in seen:
        if isinstance(e, HostUnreachable):
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


# Output kept per stream of one command (M1 audit R-2): the first OUT_HEAD and the last OUT_TAIL bytes. A policy's `yes`
# no longer grows the orchestrator until the OOM killer takes every World in it. The tail is generous because run_python
# returns its result at the end of stdout; a shell tool shows the policy 30k characters of it anyway.
OUT_HEAD = 1 << 20
OUT_TAIL = 16 << 20
EXIT_GRACE_S = 1.0  # after the command exits, background jobs still holding its pipes get this long, not the timeout


def capped_output(cmd: str) -> str:
    """``cmd`` with its output cut inside the sandbox to the first ``OUT_HEAD`` and last ``OUT_TAIL`` bytes, stderr merged into
    stdout, for a transport that keeps all it receives (Harbor's exec, which merges them anyway). The exit status is the
    command's; nothing marks the cut, so an output that fits passes byte for byte. An image without head/tail: unchanged."""
    return (f"if command -v head >/dev/null 2>&1 && command -v tail >/dev/null 2>&1; then "
            f"{{ {cmd} ; }} 2>&1 | {{ head -c {OUT_HEAD}; tail -c {OUT_TAIL}; }}; exit ${{PIPESTATUS[0]}}; else {cmd}; fi")


class _Kept:
    """The first ``OUT_HEAD`` and last ``OUT_TAIL`` bytes of one stream, and how many fell between."""

    def __init__(self) -> None:
        self.head = bytearray()
        self.tail: collections.deque[bytes] = collections.deque()
        self.size = 0  # bytes held in ``tail``
        self.dropped = 0

    def add(self, b: bytes) -> None:
        if len(self.head) < OUT_HEAD:
            n = OUT_HEAD - len(self.head)
            self.head += b[:n]
            b = b[n:]
        if b:
            self.tail.append(b)
            self.size += len(b)
            while self.size - len(self.tail[0]) >= OUT_TAIL:  # whole chunks out while the rest still fills the tail
                x = self.tail.popleft()
                self.size -= len(x)
                self.dropped += len(x)

    def value(self) -> bytes:
        t = b"".join(self.tail)
        cut = max(0, len(t) - OUT_TAIL)
        if not self.dropped + cut:
            return bytes(self.head) + t
        return bytes(self.head) + b"\n...[%d bytes dropped]...\n" % (self.dropped + cut) + t[cut:]


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
    """Run ``argv`` to its exit or ``timeout_s``, reading both pipes as they fill into a head and a tail (``_Kept``): memory
    per call stays bounded whatever the command prints, and background jobs still holding the pipes after the command
    exited hold the call for ``EXIT_GRACE_S``, not until the timeout (agentenv ``local_sandbox.py:349-438``; M1 audit R-2)."""
    p = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, start_new_session=True)
    started.append(p)
    if sessions is not None:
        sessions.add(p.pid)  # the session (and process group) id: what the command leaves behind is killed with it
    if stdin is not None:  # fed from its own thread, as communicate does: a large input never blocks the reads
        threading.Thread(target=_feed, args=(p, stdin), daemon=True).start()
    assert p.stdout is not None and p.stderr is not None
    kept = {p.stdout.fileno(): _Kept(), p.stderr.fileno(): _Kept()}
    deadline = time.monotonic() + timeout_s if timeout_s else math.inf
    timed_out = _drain(p, kept, deadline)
    if not timed_out:
        try:  # the pipes closed (or the grace ran out) while the command may still run
            p.wait(timeout=None if deadline == math.inf else max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
    if timed_out:
        _killpg(p)
        _drain(p, kept, time.monotonic() + EXIT_GRACE_S)  # what the killed group printed last
    rc = p.wait()
    p.stdout.close()
    p.stderr.close()
    out, err = (k.value() for k in kept.values())
    return (-9 if timed_out else rc or 0), out, err, timed_out


def _feed(p: subprocess.Popen, data: bytes) -> None:
    assert p.stdin is not None
    try:
        p.stdin.write(data)
    except OSError:  # the command exited or closed its stdin without reading it all
        pass
    finally:
        try:
            p.stdin.close()
        except OSError:
            pass


def _drain(p: subprocess.Popen, kept: dict[int, _Kept], deadline: float) -> bool:
    """Read ``p``'s pipes into ``kept`` until both close or ``p`` exited ``EXIT_GRACE_S`` ago (False), or ``deadline`` (True)."""
    with selectors.DefaultSelector() as sel:
        for fd in kept:
            sel.register(fd, selectors.EVENT_READ)
        exited = None
        while sel.get_map():
            now = time.monotonic()
            if exited is None and p.poll() is not None:
                exited = now
            if now >= deadline:
                return True
            if exited is not None and now - exited >= EXIT_GRACE_S:
                return False
            for key, _ in sel.select(min(0.2, deadline - now)):
                chunk = os.read(key.fd, 65536)
                if chunk:
                    kept[key.fd].add(chunk)
                else:
                    sel.unregister(key.fd)
    return False


def _killpg(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


async def _run_subprocess_raw(argv: list[str], *, cwd: str | None, env: dict[str, str] | None, timeout_s: float | None, stdin: bytes | None,
                              sessions: set[int] | None = None) -> tuple[int, bytes, bytes, bool, float]:
    """``_run_subprocess`` without the decoding: ``(rc, stdout, stderr, timed_out, duration_s)``, the output as the bytes the
    command wrote. M10 lab audit R-5 (critic amendment 1): a caller that reads a binary stream (the grader guard's
    ``docker cp <cid>:<path> -`` tar, a ``docker diff`` read with ``surrogateescape``) must not get it back through
    ``errors="replace"``, which turns every non-UTF-8 byte into U+FFFD and corrupts the tar."""
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
    return rc, out, err, timed_out, time.monotonic() - t0


async def _run_subprocess(argv: list[str], *, cwd: str | None, env: dict[str, str] | None, timeout_s: float | None, stdin: bytes | None,
                          sessions: set[int] | None = None) -> ExecOut:
    """M10 lab audit R-5 item 4: the one path for every host CLI call an attempt makes (git, docker, rsync, ssh, bash): the
    command is a ``Popen`` waited on in ``_WAITERS`` (a blocking ``communicate`` in the pool, never asyncio's child watcher:
    see the note above ``_WAITERS``), so the event loop keeps stepping every other attempt while it runs, in its own
    session. Past ``timeout_s`` its whole session is killed (``_killpg``) and the result is ``timed_out`` (rc -9); a caller
    that is cancelled kills the session too (no orphan keeps running). ``env`` is merged over ``safe_environ()``
    (credential-named variables of the operator's shell are dropped), and an ``ssh://`` ``DOCKER_HOST`` goes through
    ``docker_tunnel.rewrite_env``. ``cwd``, ``env``, ``timeout_s`` and ``stdin`` have no defaults: a caller says what each is
    (``timeout_s=None`` is an explicit "unbounded"). A binary stdout is ``_run_subprocess_raw``'s."""
    rc, out, err, timed_out, duration_s = await _run_subprocess_raw(argv, cwd=cwd, env=env, timeout_s=timeout_s, stdin=stdin, sessions=sessions)
    return ExecOut(rc=rc, stdout=out.decode(errors="replace"), stderr=err.decode(errors="replace"), duration_s=duration_s, timed_out=timed_out)


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

    def __init__(self, container: str, *, docker_host: str | None = None, default_user: str | None = None, python: str = "python3",
                 reach: Any = None):
        self.container = container
        self.docker_host = docker_host
        self.default_user = default_user
        self.python = python
        self.since = time.time()  # the container's events are read from here (a removed one has no StartedAt left)
        self.expires_at: float | None = None  # set by the sandbox that runs it for a fixed lifetime (sandbox.DockerSandbox)
        self._reach = reach  # a remote daemon's ``hostreach.HostReach`` (None: the process's one for ``docker_host``, on first use)

    def _env(self) -> dict[str, str] | None:
        return {"DOCKER_HOST": self.docker_host} if self.docker_host else None

    @property
    def reach(self) -> Any:
        """The remote daemon's reachability (``world.hostreach``), shared by every exec on that ``DOCKER_HOST``; None for the
        local daemon, whose calls run as before."""
        if self._reach is None and self.docker_host:
            from crucible.world import hostreach

            self._reach = hostreach.for_docker_host(self.docker_host)
        return self._reach

    async def _cli(self, argv: list[str], *, timeout_s: float | None, stdin: bytes | None) -> ExecOut:
        """One Docker CLI call (the transport seam a test replaces with a fake)."""
        return await _run_subprocess(argv, cwd=None, env=self._env(), timeout_s=timeout_s, stdin=stdin)

    async def _remote(self, argv: list[str], *, timeout_s: float | None, stdin: bytes | None, what: str, replay: bool | None = None) -> ExecOut:
        """``argv`` on a remote daemon under its ``HostReach`` (2026-10-26): a hung call is cut once the host stops answering its
        probes, a transport failure the probe confirms is ``HostUnreachable``, and either pauses the episode (its clocks held)
        until the host is back. Then the call runs again when nothing reached the daemon or the call is replayable, else the
        typed error goes up (``hostreach.HostReach.call``). The local daemon (no reach): the call as it was."""
        reach = self.reach
        if reach is None:
            return await self._cli(argv, timeout_s=timeout_s, stdin=stdin)

        async def once() -> ExecOut:
            try:
                out = await self._cli(argv, timeout_s=timeout_s, stdin=stdin)
            except RuntimeError as e:  # the shared SSH tunnel did not come up (``docker_tunnel.local_docker_host``): nothing was sent
                if await reach.confirm(str(e)):
                    raise HostUnreachable(reach.name, f"{what}: {e}", dispatched=False) from e
                raise
            if not out.ok:
                tail = (out.stdout + out.stderr)[-2000:].strip()
                if (out.timed_out or DOCKER_TRANSPORT.search(tail)) and await reach.confirm(tail[-300:] or "the call timed out"):
                    raise HostUnreachable(reach.name, f"{what}: {tail[-300:] or 'the call timed out'}", dispatched=not DOCKER_NOT_DISPATCHED.search(tail))
            return out

        return await reach.call(once, what=what, replay=replay)

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
        out = await self._remote(argv, timeout_s=timeout_s + 15 if timeout_s else None, stdin=stdin, what=f"docker exec in {self.container}")
        if deadline_hit(out.rc, out.duration_s, timeout_s):
            out.timed_out, out.rc = True, -9
        return await self._unless_lost(out)

    async def lost(self) -> str | None:
        """Why the container is lost to the episode (removed, paused or stopped from outside, its lifetime over, daemon
        unreachable), asked of the daemon on the same ``DOCKER_HOST``, or None. ``docker stop`` / ``kill`` / ``rm`` /
        ``pause`` go through the daemon API, which no process in the container can do; a container whose own main process
        exited (the policy killed it, or it ran out of memory) is the episode's doing, not lost (the rule Harbor's
        ``container_lost`` applies). ``--rm`` deletes such a container at once, so a removed one is the episode's only when
        its events show it died with no kill or stop before its lifetime ran out (M1 audit R-6); no evidence: lost."""
        st = await _run_subprocess(["docker", "inspect", "--format", "{{json .State}}", self.container], cwd=None, env=self._env(),
                                   timeout_s=60, stdin=None)
        expired = self.expires_at is not None and time.time() >= self.expires_at
        if not st.ok:
            if "no such" not in st.stderr.lower():
                return f"the Docker daemon cannot be reached: {st.stderr.strip()[-300:]}"
            acts = await self._actions(str(int(self.since) - 1))
            if acts is not None and "die" in acts and not acts & OUTSIDE_ACTIONS and not expired:
                return None
            return f"the task container {self.container} " + ("reached the end of its lifetime" if expired else "was removed")
        state = json.loads(st.stdout or "{}")
        if state.get("Paused"):
            return f"the task container {self.container} was paused from outside"
        if state.get("Running"):
            return None
        if expired:
            return f"the task container {self.container} reached the end of its lifetime"
        outside = sorted((await self._actions(state.get("StartedAt") or "0") or set()) & OUTSIDE_ACTIONS)
        return f"the task container {self.container} was stopped from outside ({', '.join(outside)}; status {state.get('Status')})" if outside else None

    async def _actions(self, since: str) -> set[str] | None:
        """The container's event actions since ``since`` (``die``, ``kill``, ...), or None when the daemon gives none."""
        ev = await _run_subprocess(["docker", "events", "--since", since, "--until", str(int(time.time()) + 1), "--filter",
                                    f"container={self.container}", "--format", "{{.Action}}"], cwd=None, env=self._env(), timeout_s=30, stdin=None)
        return {a.split(":")[0].strip() for a in ev.stdout.splitlines() if a.strip()} if ev.ok else None

    async def _unless_lost(self, out: ExecOut) -> ExecOut:
        """``out``, unless it is the docker CLI reporting a lost container and the daemon confirms it: then EnvContainerLost.
        The words alone never decide (a command can print ``No such container`` and exit 1), so the host is asked."""
        if out.rc != 0 and CONTAINER_GONE.search((out.stdout + out.stderr)[-2000:]) and (why := await self.lost()):
            raise EnvContainerLost(why)
        return out

    async def push_file(self, src: Path, dest: str) -> None:
        parent = str(Path(dest).parent)
        await self.run(f"mkdir -p {shlex.quote(parent)}")
        out = await self._remote(["docker", "cp", str(src), f"{self.container}:{dest}"], timeout_s=600, stdin=None, what=f"docker cp into {self.container}",
                                 replay=True)  # the same copy again
        if not (await self._unless_lost(out)).ok:
            raise ExecError(f"docker cp to {dest} failed: {out.stderr[-500:]}", out)

    async def pull_file(self, src: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        out = await self._remote(["docker", "cp", f"{self.container}:{src}", str(dest)], timeout_s=600, stdin=None, what=f"docker cp from {self.container}",
                                 replay=True)
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
        self._reach: Any = None

    @property
    def reach(self) -> Any:
        """The host's reachability (``world.hostreach``), probed with ``ssh ... true`` over the same options."""
        if self._reach is None:
            from crucible.world import hostreach

            self._reach = hostreach.for_ssh(f"{self.target}:{self.port or 22}", ["ssh", *self._opts(), *(["-p", str(self.port)] if self.port else []),
                                                                                 self.target, "true"], name=self.target)
        return self._reach

    def _opts(self) -> list[str]:
        # 2026-10-26: a host that stops answering fails a connect within ConnectTimeout and a live session within
        # ServerAliveInterval x ServerAliveCountMax (60 s); without them a lost network held a call until the attempt clock
        o = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
             "-o", "ServerAliveCountMax=4", "-o", "ControlMaster=auto", "-o", f"ControlPath={self.control}", "-o", "ControlPersist=120"]
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
        reach = self.reach

        async def once() -> ExecOut:
            out = await _run_subprocess(argv, cwd=None, env=None, timeout_s=timeout_s + 30 if timeout_s else None, stdin=stdin)
            if out.rc == 255 and ("Connection" in out.stderr or "ssh:" in out.stderr):
                why = f"ssh to {self.target} failed: {out.stderr[-300:]}"
                if await reach.confirm(why):  # the host does not answer a probe either: an outage, paused for (hostreach)
                    raise HostUnreachable(reach.name, why, dispatched=not SSH_NOT_DISPATCHED.search(out.stderr))
                raise ExecError(why, out)
            return out

        out = await reach.call(once, what=f"ssh to {self.target}")
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
