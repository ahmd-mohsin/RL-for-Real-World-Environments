"""Host side of the RLM kernel (m14): process launch, the host-request bridge, the journal, and exact restore by replay.

The kernel (``repl.py``) is a separate process; model-written code never runs in this process.

**Where it runs (M14 audit H3 V1, G1 D8).** The default launcher is ``DockerLauncher``: a container with no network, a
read-only root, every capability dropped, an unprivileged user, removed after the episode. ``ExecLauncher`` streams the
kernel through any argv prefix, e.g. ``docker exec -i <sandbox>`` (the World's own sandbox). ``LocalLauncher`` runs the
kernel on the host and refuses to start unless built with ``allow_host=True``: then it gets an allow-listed environment
(no ``SSH_AUTH_SOCK``, proxies, database URLs or tokens), ``HOME`` and the working directory in a fresh temporary
directory, a copy of ``repl.py`` there (no repository path in its argv), its own process group killed on close. Host
mode does **not** isolate: any process of the same user can read the parent's environment (``/proc/<pid>/environ``,
``ps -E``). It is for trusted code only (unit tests, a person's own agent); RL rollouts and evaluations never use it.

**Protocol hygiene (H3 V2, G1 D12, D13).** Every frame the host reads is validated (known event, field types, size caps,
no lone surrogates); an invalid frame is dropped and counted on the cell (``CellResult.protocol_errors``), so a cell
that forges or garbles protocol traffic only hurts its own output, which is the policy's scored outcome. A blocked cell
is first interrupted (namespace kept), then killed; ``close`` and a kill also stop the kernel's process group inside a
sandbox (``kill``: H3 V3, G1 D7), where killing the local ``docker exec`` client leaves the kernel running.

**Infra faults carry their kind (H3 V6, G1 D6).** A launcher or kernel that cannot start raises ``KernelStartError`` with
a ``failure_kind`` (``crucible:env_container_lost``, ``crucible:env_image_unavailable``, ``crucible:backend_error``), so
the episode is masked, never scored as the policy's error. A host handler that must end the episode as an infra fault
raises ``HostAbort`` (it propagates out of ``execute``; any other handler exception becomes the cell's exception).

**Journal and restore.** Every executed cell is journaled with the host replies it received, in order. ``replay`` starts a
fresh kernel with the same seed and re-executes the cells, answering each host request from the record (a request of a
different type than recorded is a divergence: ``KernelDivergence``); the final namespace digest must equal the recorded
one or the restore raises ``SnapshotMismatch``. Restore is exact only for deterministic cells (a cell that keeps
``time.time()`` or ``uuid4()`` diverges; ``RLMPolicy`` then keeps a live kernel and tells the model, H3 V10). An
in-progress cell (a checkpoint taken while the cell waits on a host reply) replays from its recorded replies and then
continues live through the caller's handler.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPL_PATH = Path(__file__).with_name("repl.py")
PROTOCOL = 3  # 3: a per-session frame key (``hello``) and an HMAC on every frame (M14 campaign B10)
Handler = Callable[[dict[str, Any]], Awaitable[Any]]
MAX_FRAME = 64 * 1024 * 1024  # host-side line cap (the kernel's own frames are capped well below it)
MAX_FIELD = 4 * 1024 * 1024  # one text field of a done event (the kernel caps each at ~1 MiB plus a marker)

# The only variables a host-mode kernel inherits (G1 D8: a deny-list let SSH_AUTH_SOCK, DATABASE_URL, HTTP_PROXY, GH_PAT through)
HOST_ENV_ALLOW = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "SYSTEMROOT")


class KernelError(RuntimeError):
    pass


class KernelDivergence(KernelError):
    """A replayed cell asked the host for something other than what the journal recorded."""


class KernelStartError(KernelError):
    """The kernel's sandbox or process could not start (or was lost): an infrastructure fault, masked."""

    def __init__(self, msg: str, failure_kind: str = "crucible:backend_error"):
        super().__init__(msg)
        self.failure_kind = failure_kind


class HostAbort(KernelError):
    """Raised by a host handler to end the episode as an infra fault (e.g. a sub-LLM provider down): never a cell error."""

    def __init__(self, msg: str, failure_kind: str = "provider_unavailable"):
        super().__init__(msg)
        self.failure_kind = failure_kind


class IsolationRefused(KernelError):
    """Model code would run on the host without ``allow_host=True``."""

    failure_kind = "crucible:harness_unsupported"


def classify_sandbox_error(text: str) -> str:
    """The failure kind of a Docker / sandbox error message."""
    t = (text or "").lower()
    if any(s in t for s in ("unable to find image", "pull access denied", "manifest unknown", "repository does not exist",
                            "no such image", "manifest for")):
        return "crucible:env_image_unavailable"
    if any(s in t for s in ("cannot connect to the docker daemon", "no such container", "is not running", "is paused",
                            "container not found", "is restarting", "error during connect", "docker: not found",
                            "no such file or directory: 'docker'")):
        return "crucible:env_container_lost"
    return "crucible:backend_error"


@dataclass
class CellResult:
    stdout: str = ""
    stderr: str = ""
    result: str | None = None
    error: str | None = None
    timed_out: bool = False
    finished: Any = None
    digest: str | None = None
    killed: bool = False
    interrupted: bool = False
    host_calls: int = 0
    protocol_errors: int = 0  # invalid frames dropped while this cell ran (forged or garbled by the cell itself)

    @property
    def ok(self) -> bool:
        return self.error is None and not self.timed_out and not self.killed


@dataclass
class JournalEntry:
    code: str
    timeout_s: float | None
    replies: list[dict[str, Any]] = field(default_factory=list)  # [{"request": data, "reply": data}]
    digest: str | None = None
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "timeout_s": self.timeout_s, "replies": self.replies, "digest": self.digest, "done": self.done}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JournalEntry:
        return cls(d["code"], d.get("timeout_s"), list(d.get("replies") or []), d.get("digest"), bool(d.get("done")))


def scrub(x: Any, depth: int = 0) -> Any:
    """Lone surrogates replaced (they broke the next chat-codec call: H3 V2); nesting bounded."""
    if isinstance(x, str):
        try:
            x.encode("utf-8")
            return x
        except UnicodeEncodeError:
            return x.encode("utf-8", "replace").decode("utf-8")
    if depth > 64:
        return None
    if isinstance(x, dict):
        return {scrub(str(k), depth + 1): scrub(v, depth + 1) for k, v in x.items()}
    if isinstance(x, list):
        return [scrub(v, depth + 1) for v in x]
    return x


def _valid(ev: Any) -> str | None:
    """Why a kernel frame is invalid (None: valid)."""
    if not isinstance(ev, dict):
        return "not an object"
    kind = ev.get("event")
    if kind == "ready":
        return None if isinstance(ev.get("protocol"), int) else "ready without a protocol version"
    if kind == "protocol_error":
        return None if isinstance(ev.get("error"), str) else "protocol_error without error"
    if kind == "host_request":
        d = ev.get("data")
        if not isinstance(ev.get("id"), str) or len(ev["id"]) > 128:
            return "host_request id"
        return None if isinstance(d, dict) and isinstance(d.get("type"), str) and len(d["type"]) <= 128 else "host_request data"
    if kind == "done":
        if not isinstance(ev.get("id"), str) or len(ev["id"]) > 128:
            return "done id"
        for k in ("stdout", "stderr", "result", "error", "finished"):
            v = ev.get(k)
            if v is not None and (not isinstance(v, str) or len(v) > MAX_FIELD):
                return f"done.{k}"
        if "timed_out" in ev and not isinstance(ev["timed_out"], bool):
            return "done.timed_out"
        dg = ev.get("digest")
        if dg is not None and (not isinstance(dg, str) or len(dg) > 64):
            return "done.digest"
        nm = ev.get("names")
        if nm is not None and (not isinstance(nm, list) or not all(isinstance(n, str) for n in nm)):
            return "done.names"
        return None
    return f"unknown event {str(kind)[:40]!r}"


class LocalLauncher:
    """The kernel as a host subprocess. Refuses to start without ``allow_host=True`` (model code on the host can read
    every same-user process's environment and the user's files). With it: an allow-listed environment, ``HOME``, ``TMPDIR``
    and the working directory in a fresh temporary directory holding a copy of ``repl.py``, the base interpreter (not a
    project virtualenv's path), its own session, killed as a group on close."""

    def __init__(self, python: str | None = None, *, allow_host: bool = False):
        self.python = python or os.path.realpath(getattr(sys, "_base_executable", None) or sys.executable)
        self.allow_host = allow_host
        self._cwd: str | None = None

    def env(self, seed: int) -> dict[str, str]:
        env = {k: os.environ[k] for k in HOST_ENV_ALLOW if k in os.environ and "@" not in os.environ[k]}
        env.update({"HOME": str(self._cwd), "TMPDIR": str(self._cwd), "CRUCIBLE_RLM_SEED": str(seed), "PYTHONUNBUFFERED": "1",
                    "PYTHONHASHSEED": str(seed), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        return env

    async def start(self, seed: int) -> asyncio.subprocess.Process:
        if not self.allow_host:
            raise IsolationRefused("LocalLauncher runs model-written code on the host: pass allow_host=True (trusted code only), "
                                   "or use DockerLauncher / ExecLauncher")
        import tempfile

        if self._cwd is None:
            self._cwd = tempfile.mkdtemp(prefix="crucible-rlm-")
            shutil.copyfile(REPL_PATH, Path(self._cwd) / "crucible_rlm_repl.py")
        return await asyncio.create_subprocess_exec(self.python, "-u", "-s", "crucible_rlm_repl.py", stdin=asyncio.subprocess.PIPE,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=self.env(seed),
                                                    limit=MAX_FRAME, start_new_session=True, cwd=self._cwd)

    async def kill(self, proc: Any, pgid: int | None) -> bool:
        """The kernel's process group (its children included); True when it is gone."""
        g = pgid or (proc.pid if proc is not None else None)
        if g is None:
            return True
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(g, signal.SIGKILL)
        return True

    async def aclose(self) -> None:
        if self._cwd:
            shutil.rmtree(self._cwd, ignore_errors=True)
            self._cwd = None


class ExecLauncher:
    """The kernel inside a sandbox: ``prefix`` streams stdio (``["docker", "exec", "-i", name]``); ``push`` copies
    ``repl.py`` to ``remote_path`` first (``Exec.push_file``). ``kill`` stops the kernel's process group inside the
    sandbox (killing the local client does not)."""

    def __init__(self, prefix: list[str], *, push: Callable[[Path, str], Awaitable[None]] | None = None, remote_path: str = "/tmp/crucible_rlm_repl.py",
                 python: str = "python3"):
        self.prefix, self.push, self.remote_path, self.python = list(prefix), push, remote_path, python
        self._pushed = False

    async def start(self, seed: int) -> asyncio.subprocess.Process:
        if self.push is not None and not self._pushed:
            try:
                await self.push(REPL_PATH, self.remote_path)
            except KernelStartError:
                raise
            except Exception as e:
                raise KernelStartError(f"pushing the kernel failed: {e}", classify_sandbox_error(str(e))) from e
            self._pushed = True
        argv = [*self.prefix, "env", f"CRUCIBLE_RLM_SEED={seed}", f"PYTHONHASHSEED={seed}", self.python, "-u", self.remote_path]
        try:
            return await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.PIPE, limit=MAX_FRAME)
        except OSError as e:
            raise KernelStartError(f"cannot start {argv[0]}: {e}", "crucible:env_container_lost") from e

    async def run(self, *argv: str, timeout: float = 20) -> tuple[int, str]:
        p = await asyncio.create_subprocess_exec(*self.prefix, *argv, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(p.communicate(), timeout=timeout)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                p.kill()
            return -1, "timeout"
        return p.returncode or 0, out.decode(errors="replace")

    async def kill(self, proc: Any, pgid: int | None) -> bool:
        """SIGKILL the kernel's group inside the sandbox and wait until it is gone (True), or report it is not (False)."""
        if not pgid or pgid <= 1:
            return False
        code = (f"import os,signal,time\ng={int(pgid)}\ntry:\n    os.killpg(g,signal.SIGKILL)\nexcept OSError:\n    pass\n"
                "for _ in range(50):\n    try:\n        os.killpg(g,0)\n    except OSError:\n        print('gone'); break\n    time.sleep(0.1)\n")
        _rc, out = await self.run(self.python, "-c", code)
        return "gone" in out


# ---- RLM container lifecycle (M14 campaign B4) ----------------------------------------------------------------------------
# Kernel containers leaked when a process ended with rollouts in flight: a ``docker run`` still running when the rollout was
# cancelled created its container after ``aclose`` had removed nothing, and ``sleep infinity`` kept it forever. Now every
# container is labelled with its owner (host and pid), the start and the removal are shielded from cancellation, the
# process removes its own containers at exit (atexit and SIGTERM), containers whose owner is dead or whose lifetime passed
# are reaped when a process first starts one, and each container ends by itself after ``lifetime_s`` (``sleep <n>``).
OWNER_LABEL, EXPIRES_LABEL = "crucible.rlm.owner", "crucible.rlm.expires"
DEFAULT_LIFETIME_S = 6 * 3600
_LIVE: dict[str, str] = {}  # container name -> docker binary: started by this process, not yet removed
_REAPER: dict[str, Any] = {"installed": False, "stale_checked": set()}


def owner_id() -> str:
    import socket

    return f"{socket.gethostname()}:{os.getpid()}"


def _docker_sync(docker: str, *argv: str, timeout: float = 30) -> str:
    import subprocess

    try:
        r = subprocess.run([docker, *argv], capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout


def reap_own() -> list[str]:
    """Remove every RLM container this process started and still holds (atexit, SIGTERM): the tracked names, and any
    container labelled with this process as its owner (one whose ``docker run`` finished after its rollout was gone)."""
    removed: list[str] = []
    by_docker: dict[str, set[str]] = {}
    for name, docker in list(_LIVE.items()):
        by_docker.setdefault(docker, set()).add(name)
    by_docker.setdefault("docker", set())
    for docker, names in by_docker.items():
        if not shutil.which(docker):
            continue
        names |= {n for n in _docker_sync(docker, "ps", "-a", "--filter", f"label={OWNER_LABEL}={owner_id()}", "--format", "{{.Names}}").split() if n}
        if names:
            _docker_sync(docker, "rm", "-f", *sorted(names))
            removed += sorted(names)
        for n in names:
            _LIVE.pop(n, None)
    return removed


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def reap_stale(docker: str = "docker", *, now: float | None = None) -> list[str]:
    """Remove RLM containers left by processes that are gone: labelled with an owner on this host whose pid is dead, or
    past their ``expires`` time. Containers of live processes, other hosts' processes and unlabelled ones are left alone."""
    import socket
    import time

    if not shutil.which(docker):
        return []
    now = time.time() if now is None else now
    host = socket.gethostname()
    out = _docker_sync(docker, "ps", "-a", "--filter", f"label={OWNER_LABEL}", "--format",
                       f'{{{{.Names}}}}\t{{{{.Label "{OWNER_LABEL}"}}}}\t{{{{.Label "{EXPIRES_LABEL}"}}}}')
    stale = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        name, owner, expires = parts
        h, _, pid = owner.rpartition(":")
        try:
            expired = bool(expires) and float(expires) < now
        except ValueError:
            expired = False
        dead = h == host and pid.isdigit() and int(pid) != os.getpid() and not _pid_alive(int(pid))
        if expired or dead:
            stale.append(name)
    if stale:
        _docker_sync(docker, "rm", "-f", *stale)
    return stale


def _on_sigterm(signum: int, frame: Any) -> None:
    reap_own()
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGTERM)


def _install_reaper() -> None:
    """Once per process: remove this process's containers at exit and on SIGTERM (when nothing else handles it)."""
    if _REAPER["installed"]:
        return
    import atexit
    import threading

    _REAPER["installed"] = True
    atexit.register(reap_own)
    if threading.current_thread() is threading.main_thread():
        with contextlib.suppress(ValueError, OSError):
            if signal.getsignal(signal.SIGTERM) in (signal.SIG_DFL, None):
                signal.signal(signal.SIGTERM, _on_sigterm)


async def _finish(task: asyncio.Future, timeout: float) -> None:
    """Let a shielded task finish after its awaiter was cancelled (bounded)."""
    with contextlib.suppress(BaseException):
        await asyncio.wait_for(asyncio.shield(task), timeout)


class DockerLauncher:
    """The kernel in its own container: no network, read-only root (a ``/tmp`` tmpfs), every capability dropped,
    ``no-new-privileges``, an unprivileged user, and memory and pid limits. The container lives as long as the launcher,
    so a journal replay (a restore) reuses it; ``aclose`` removes it. Model-written code in RL rollouts runs here. One
    launcher serves one episode: never share an instance across concurrent rollouts (H3 V12, G1 D9).

    Lifecycle (M14 campaign B4): labelled ``crucible.rlm=1`` and with its owner (``host:pid``) and expiry; ``start`` and
    ``aclose`` finish their ``docker run`` / ``docker rm`` even when the rollout is cancelled; the process removes its
    containers at exit; and the container ends by itself after ``lifetime_s`` whatever happens to its owner."""

    def __init__(self, image: str = "python:3.12-slim", *, memory: str = "1g", pids: int = 256, cpus: str = "1", docker: str = "docker",
                 lifetime_s: int = DEFAULT_LIFETIME_S):
        self.image, self.memory, self.pids, self.cpus, self.docker = image, memory, pids, cpus, docker
        self.lifetime_s = int(lifetime_s)
        self.name: str | None = None
        self._exec: ExecLauncher | None = None

    async def _run(self, *argv: str) -> None:
        try:
            p = await asyncio.create_subprocess_exec(self.docker, *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        except OSError as e:
            raise KernelStartError(f"docker is not available: {e}", "crucible:env_container_lost") from e
        _, err = await p.communicate()
        if p.returncode != 0:
            text = err.decode(errors="replace")[-400:]
            raise KernelStartError(f"docker {argv[0]} failed: {text}", classify_sandbox_error(text))

    async def start(self, seed: int) -> asyncio.subprocess.Process:
        if self._exec is None:
            import time

            _install_reaper()
            if self.docker not in _REAPER["stale_checked"]:  # once per process and docker binary: what dead processes left
                _REAPER["stale_checked"].add(self.docker)
                await asyncio.to_thread(reap_stale, self.docker)
            name = self.name = f"crucible-rlm-{uuid.uuid4().hex[:12]}"
            _LIVE[name] = self.docker  # tracked before it exists: the exit reaper removes it whatever happens to this coroutine
            run = asyncio.ensure_future(self._run(
                "run", "-d", "--rm", "--name", name, "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=256m",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65534:65534", "--memory", self.memory,
                "--pids-limit", str(self.pids), "--cpus", self.cpus, "--workdir", "/tmp", "--label", "crucible.rlm=1",
                "--label", f"{OWNER_LABEL}={owner_id()}", "--label", f"{EXPIRES_LABEL}={int(time.time()) + self.lifetime_s}",
                "--env", "HOME=/tmp", self.image, "sleep", str(self.lifetime_s)))  # a bounded life, never ``sleep infinity``
            try:
                await asyncio.shield(run)
            except asyncio.CancelledError:  # the rollout was cancelled mid-``docker run``: let it finish, then remove what it made
                await _finish(run, 120)
                await self.aclose()
                raise
            except BaseException:
                await self.aclose()  # a failed run may still have left a container behind
                raise

            async def push(src: Path, dest: str) -> None:  # docker cp cannot write into a tmpfs: stream the file through the shell
                p = await asyncio.create_subprocess_exec(self.docker, "exec", "-i", self.name, "sh", "-c", f"cat > {dest}", stdin=asyncio.subprocess.PIPE,
                                                         stderr=asyncio.subprocess.PIPE)
                _, err = await p.communicate(src.read_bytes())
                if p.returncode != 0:
                    text = err.decode(errors="replace")[-300:]
                    raise KernelStartError(f"pushing the kernel failed: {text}", classify_sandbox_error(text))

            self._exec = ExecLauncher([self.docker, "exec", "-i", self.name], push=push)
        return await self._exec.start(seed)

    async def kill(self, proc: Any, pgid: int | None) -> bool:
        return await self._exec.kill(proc, pgid) if self._exec is not None else True

    async def _rm(self, name: str) -> None:
        try:
            await self._run("rm", "-f", name)
        except Exception:  # noqa: BLE001 -- the exit reaper and the container's own lifetime remain
            return
        _LIVE.pop(name, None)

    async def aclose(self) -> None:
        if self.name:
            name, self.name, self._exec = self.name, None, None
            rm = asyncio.ensure_future(self._rm(name))
            try:
                await asyncio.shield(rm)  # a cancelled rollout still removes its container (B4)
            except asyncio.CancelledError:
                await _finish(rm, 60)
                raise


def default_launcher() -> Any:
    """The isolated kernel (``DockerLauncher``) where Docker exists; otherwise refuse: model code never silently runs on
    the host (H3 V1)."""
    if shutil.which("docker"):
        return DockerLauncher()
    raise IsolationRefused("no isolated RLM kernel is available (docker not found): pass a launcher (DockerLauncher, ExecLauncher), "
                           "or LocalLauncher(allow_host=True) for trusted code only")


class Kernel:
    def __init__(self, launcher: Any = None, *, seed: int = 0, cell_timeout_s: float = 120.0, kill_grace_s: float = 10.0,
                 interrupt_grace_s: float = 2.0):
        self.launcher = launcher if launcher is not None else default_launcher()
        self.seed, self.cell_timeout_s, self.kill_grace_s, self.interrupt_grace_s = seed, cell_timeout_s, kill_grace_s, interrupt_grace_s
        self.proc: asyncio.subprocess.Process | None = None
        self.pgid: int | None = None
        self.journal: list[JournalEntry] = []
        self.protocol_errors = 0
        self.replay_divergences = 0
        self._lenient = False
        self.orphaned = False  # a kill could not confirm the in-sandbox kernel was gone
        self._mac: Any = None  # HMAC-SHA256 under this session's key (B10): frames without a valid MAC are dropped

    # ---- process ------------------------------------------------------------------------------------------------------
    async def start(self) -> None:
        import hashlib
        import hmac
        import secrets

        try:
            self.proc = await self.launcher.start(self.seed)
        except KernelError:
            raise
        except Exception as e:
            raise KernelStartError(f"the RLM kernel's launcher failed: {type(e).__name__}: {e}", classify_sandbox_error(str(e))) from e
        key = secrets.token_bytes(32)  # a fresh key per kernel process: a frame from another session never verifies
        self._mac = hmac.new(key, digestmod=hashlib.sha256)
        with contextlib.suppress(Exception):  # a kernel that died at once is reported by the read below
            await self._send({"type": "hello", "key": key.hex()})
        ev = await self._read(timeout=60)
        if ev is None or ev.get("event") != "ready" or ev.get("protocol") != PROTOCOL:
            err = ""
            p = self.proc
            if p is not None and p.stderr is not None:
                with contextlib.suppress(Exception):
                    err = (await asyncio.wait_for(p.stderr.read(4000), timeout=2)).decode(errors="replace")
            await self.close()
            raise KernelStartError(f"the RLM kernel did not start: {ev} {err[-400:]}", classify_sandbox_error(err))
        self.pgid = ev.get("pgid") if isinstance(ev.get("pgid"), int) else None

    async def close(self) -> None:
        p, self.proc = self.proc, None
        pgid, self.pgid = self.pgid, None
        if p is None:
            return
        with contextlib.suppress(Exception):
            if p.returncode is None:
                p.stdin.write(b'{"type":"shutdown"}\n')
                await p.stdin.drain()
                await asyncio.wait_for(p.wait(), timeout=3)
        if p.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                p.kill()
            with contextlib.suppress(Exception):
                await p.wait()
        kill = getattr(self.launcher, "kill", None)
        if kill is not None:  # the kernel inside a sandbox outlives its killed client (H3 V3): stop its process group there
            try:
                gone = await kill(p, pgid)
            except Exception:  # noqa: BLE001 -- a lost sandbox has nothing left to kill
                gone = True
            self.orphaned = self.orphaned or not gone

    async def _send(self, obj: dict[str, Any]) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(obj, default=str) + "\n").encode())
        await self.proc.stdin.drain()

    async def _line(self, timeout: float | None) -> bytes | None:
        """One frame (b"" for an oversize frame, which is discarded); None at EOF."""
        assert self.proc is not None and self.proc.stdout is not None
        r = self.proc.stdout
        try:
            return await asyncio.wait_for(r.readuntil(b"\n"), timeout=timeout)
        except asyncio.IncompleteReadError:
            return None
        except asyncio.LimitOverrunError as e:
            await r.readexactly(e.consumed)
            while True:
                try:
                    await asyncio.wait_for(r.readuntil(b"\n"), timeout=timeout)
                    return b""
                except asyncio.LimitOverrunError as e2:
                    await r.readexactly(e2.consumed)
                except asyncio.IncompleteReadError:
                    return None

    async def _read(self, timeout: float | None) -> dict[str, Any] | None:
        """The next valid frame; ``{"event": "__timeout__"}`` when none came in time; None when the kernel is gone."""
        while True:
            try:
                line = await self._line(timeout)
            except TimeoutError:
                return {"event": "__timeout__"}
            if line is None:
                return None
            try:
                ev = json.loads(line) if line else None
            except (ValueError, RecursionError):
                ev = None
            why = "oversize or malformed frame" if ev is None else (_valid(ev) or self._unauthentic(ev))
            if why is not None:
                self.protocol_errors += 1
                continue
            return scrub(ev)

    def _unauthentic(self, ev: dict[str, Any]) -> str | None:
        """Why a frame's MAC does not verify (None: it does). B10: a cell wrote a well-formed ``done`` on the protocol
        descriptor and the host accepted it."""
        import hmac

        mac = ev.pop("mac", None)
        if self._mac is None:
            return "no session key"
        if not isinstance(mac, str):
            return "unauthenticated frame"
        m = self._mac.copy()
        m.update(json.dumps(ev, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("ascii"))
        return None if hmac.compare_digest(m.hexdigest(), mac) else "bad frame MAC"

    # ---- execution ----------------------------------------------------------------------------------------------------
    async def execute(self, code: str, handler: Handler, *, timeout_s: float | None = None, entry: JournalEntry | None = None) -> CellResult:
        """Run one cell. ``handler(data)`` answers its host requests. ``entry``: continue a journaled, partly executed cell
        (its recorded replies are served first, then ``handler`` takes over)."""
        if self.proc is None:
            await self.start()
        t = self.cell_timeout_s if timeout_s is None else timeout_s
        if entry is None:
            entry = JournalEntry(code, t)
            self.journal.append(entry)
        recorded = list(entry.replies)
        served = 0
        cid = uuid.uuid4().hex
        errors0 = self.protocol_errors
        await self._send({"type": "execute", "id": cid, "code": code, "timeout_s": t})
        res = CellResult()
        interrupted = False
        while True:
            wait = None if not t else (self.kill_grace_s if interrupted else t + self.interrupt_grace_s)
            ev = await self._read(timeout=wait)
            if ev is not None and ev.get("event") == "__timeout__" and not interrupted:
                interrupted = True  # the cell blocks the kernel's loop past its deadline: interrupt first (state kept), then kill
                with contextlib.suppress(Exception):
                    await self._send({"type": "interrupt", "id": cid})
                continue
            if ev is None or ev.get("event") == "__timeout__":  # wedged (the interrupt did not stop it) or dead
                await self.close()
                res.killed, res.timed_out = True, ev is not None
                res.error = "the kernel stopped responding and was killed" if ev is not None else "the kernel process exited"
                res.protocol_errors = self.protocol_errors - errors0
                return res
            kind = ev.get("event")
            if kind == "host_request":
                data = ev.get("data") or {}
                if served < len(recorded):
                    rec = recorded[served]
                    if (rec.get("request") or {}).get("type") != data.get("type") and self._lenient:
                        self.replay_divergences += 1
                        reply = {"__error__": "replay: the cell asked for another request than before"}
                    elif (rec.get("request") or {}).get("type") != data.get("type"):
                        raise KernelDivergence(f"replay diverged: the cell asked for {data.get('type')!r}, the journal recorded "
                                               f"{(rec.get('request') or {}).get('type')!r}")
                    else:
                        reply = rec.get("reply")
                    served += 1
                else:
                    try:
                        reply = await handler(data)
                    except (KernelDivergence, HostAbort):
                        raise
                    except Exception as e:  # noqa: BLE001 -- a host-side failure is the cell's exception, not ours
                        reply = {"__error__": f"{type(e).__name__}: {e}"}
                    entry.replies.append({"request": data, "reply": reply})
                res.host_calls += 1
                await self._send({"type": "host_reply", "id": ev["id"], "data": reply})
            elif kind == "done" and ev.get("id") == cid:
                res.stdout, res.stderr, res.result = ev.get("stdout") or "", ev.get("stderr") or "", ev.get("result")
                res.error, res.timed_out, res.finished, res.digest = ev.get("error"), bool(ev.get("timed_out")), ev.get("finished"), ev.get("digest")
                res.interrupted = interrupted
                res.protocol_errors = self.protocol_errors - errors0
                entry.digest, entry.done = res.digest, True
                return res
            elif kind in ("done", "ready"):  # a frame for no request in flight: forged by the cell
                self.protocol_errors += 1

    async def query(self, kind: str) -> Any:
        if self.proc is None:
            raise KernelError(f"no kernel to ask for {kind}")
        cid = uuid.uuid4().hex
        await self._send({"type": kind, "id": cid})
        while True:
            ev = await self._read(timeout=30)
            if ev is None or ev.get("event") == "__timeout__":
                raise KernelError(f"no reply to {kind}")
            if ev.get("event") == "done" and ev.get("id") == cid:
                return ev.get(kind)

    async def digest(self) -> str:
        return await self.query("digest")

    async def names(self) -> list[str]:
        return await self.query("names")

    # ---- journal ------------------------------------------------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        return {"seed": self.seed, "cell_timeout_s": self.cell_timeout_s, "journal": [e.to_dict() for e in self.journal]}

    async def replay(self, journal: list[dict[str, Any]] | list[JournalEntry], *, live: Handler | None = None, strict: bool = True) -> JournalEntry | None:
        """Fresh kernel, then every completed cell from the record (digest-checked). Returns the unfinished last entry, if
        any, *without* executing it: the caller continues it with ``execute(code, handler, entry=...)``.

        ``strict=False`` (the fallback after a divergence): a digest mismatch is counted (``replay_divergences``), not
        raised, and a host request the journal does not hold gets an error reply (that cell fails), never a live action."""
        await self.close()
        entries = [e if isinstance(e, JournalEntry) else JournalEntry.from_dict(e) for e in journal]
        self.journal = []
        self.replay_divergences = 0
        await self.start()

        async def refuse(data: dict[str, Any]) -> Any:
            if not strict:
                self.replay_divergences += 1
                raise RuntimeError("replay: this request is not in the journal (the cell took another path than before)")
            raise KernelDivergence(f"replay asked for a host request beyond the journal: {data.get('type')!r}")

        pending = None
        for e in entries:
            if not e.done:
                pending = JournalEntry(e.code, e.timeout_s, list(e.replies), None, False)
                self.journal.append(pending)
                break
            copy = JournalEntry(e.code, e.timeout_s, list(e.replies), None, False)
            self.journal.append(copy)
            self._lenient = not strict
            try:
                r = await self.execute(e.code, live or refuse, timeout_s=e.timeout_s, entry=copy)
            finally:
                self._lenient = False
            if r.killed:
                raise KernelDivergence(f"replay: cell {len(self.journal)} did not complete ({r.error})")
            if e.digest is not None and r.digest != e.digest and not strict:
                self.replay_divergences += 1
            elif e.digest is not None and r.digest != e.digest:
                from crucible.world.state.base import SnapshotMismatch

                raise SnapshotMismatch(f"RLM kernel replay diverged after cell {len(self.journal)}: digest {r.digest} != recorded {e.digest}")
        return pending
