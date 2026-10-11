"""Host side of the RLM kernel (m14): process launch, the host-request bridge, the journal, and exact restore by replay.

The kernel (``repl.py``) is a separate process; model-written code never runs in this process. ``LocalLauncher`` starts it
as a subprocess with a credential-free environment (``exec.safe_environ``); ``ExecLauncher`` streams it through any argv
prefix, e.g. ``docker exec -i <sandbox>``, after pushing ``repl.py`` there (the World's Docker / gVisor / pooled sandbox).

**Journal and restore.** Every executed cell is journaled with the host replies it received, in order. ``replay`` starts a
fresh kernel with the same seed and re-executes the cells, answering each host request from the record (a request of a
different type than recorded is a divergence: ``KernelDivergence``); the final namespace digest must equal the recorded
one or the restore is refused. An in-progress cell (a checkpoint taken while the cell waits on a host reply) replays from
its recorded replies and then continues live through the caller's handler. Prime Agent pickles the namespace with dill
(best effort) and loses its factory state; replay is exact for everything the cells compute from their inputs.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPL_PATH = Path(__file__).with_name("repl.py")
Handler = Callable[[dict[str, Any]], Awaitable[Any]]


class KernelError(RuntimeError):
    pass


class KernelDivergence(KernelError):
    """A replayed cell asked the host for something other than what the journal recorded."""


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
    host_calls: int = 0

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


class LocalLauncher:
    """A host subprocess with a credential-free environment, started in an empty directory. For trusted code only (tests,
    a person's own agent): the host filesystem is still reachable. RL rollouts use ``DockerLauncher`` (the m14 live
    smoke's model probed the host's working directory for policy files)."""

    def __init__(self, python: str | None = None):
        self.python = python or sys.executable
        self._cwd: str | None = None

    async def start(self, seed: int) -> asyncio.subprocess.Process:
        import tempfile

        from crucible.world.exec import safe_environ

        self._cwd = self._cwd or tempfile.mkdtemp(prefix="crucible-rlm-")
        env = safe_environ({"CRUCIBLE_RLM_SEED": str(seed), "PYTHONUNBUFFERED": "1", "PYTHONHASHSEED": str(seed)})
        return await asyncio.create_subprocess_exec(self.python, "-u", str(REPL_PATH), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE, env=env, limit=64 * 1024 * 1024, start_new_session=True,
                                                    cwd=self._cwd)

    async def aclose(self) -> None:
        if self._cwd:
            import shutil

            shutil.rmtree(self._cwd, ignore_errors=True)
            self._cwd = None


class DockerLauncher:
    """The kernel in its own container: no network, read-only root (a ``/tmp`` tmpfs), every capability dropped,
    ``no-new-privileges``, an unprivileged user, and memory and pid limits. The container lives as long as the launcher,
    so a journal replay (a restore) reuses it; ``aclose`` removes it. Model-written code in RL rollouts runs here."""

    def __init__(self, image: str = "python:3.12-slim", *, memory: str = "1g", pids: int = 256, cpus: str = "1", docker: str = "docker"):
        self.image, self.memory, self.pids, self.cpus, self.docker = image, memory, pids, cpus, docker
        self.name: str | None = None
        self._exec: ExecLauncher | None = None

    async def _run(self, *argv: str) -> None:
        p = await asyncio.create_subprocess_exec(self.docker, *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, err = await p.communicate()
        if p.returncode != 0:
            raise RuntimeError(f"docker {argv[0]} failed: {err.decode(errors='replace')[-400:]}")

    async def start(self, seed: int) -> asyncio.subprocess.Process:
        if self._exec is None:
            self.name = f"crucible-rlm-{uuid.uuid4().hex[:12]}"
            await self._run("run", "-d", "--rm", "--name", self.name, "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=256m",
                            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65534:65534", "--memory", self.memory,
                            "--pids-limit", str(self.pids), "--cpus", self.cpus, "--workdir", "/tmp", "--label", "crucible.rlm=1", self.image,
                            "sleep", "infinity")

            async def push(src: Path, dest: str) -> None:  # docker cp cannot write into a tmpfs: stream the file through the shell
                p = await asyncio.create_subprocess_exec(self.docker, "exec", "-i", self.name, "sh", "-c", f"cat > {dest}", stdin=asyncio.subprocess.PIPE,
                                                         stderr=asyncio.subprocess.PIPE)
                _, err = await p.communicate(src.read_bytes())
                if p.returncode != 0:
                    raise RuntimeError(f"pushing the kernel failed: {err.decode(errors='replace')[-300:]}")

            self._exec = ExecLauncher([self.docker, "exec", "-i", self.name], push=push)
        return await self._exec.start(seed)

    async def aclose(self) -> None:
        if self.name:
            name, self.name, self._exec = self.name, None, None
            with contextlib.suppress(Exception):
                await self._run("rm", "-f", name)


class ExecLauncher:
    """The kernel inside a sandbox: ``prefix`` streams stdio (``["docker", "exec", "-i", name]``); ``push`` copies
    ``repl.py`` to ``remote_path`` first (``Exec.push_file``)."""

    def __init__(self, prefix: list[str], *, push: Callable[[Path, str], Awaitable[None]] | None = None, remote_path: str = "/tmp/crucible_rlm_repl.py",
                 python: str = "python3"):
        self.prefix, self.push, self.remote_path, self.python = list(prefix), push, remote_path, python
        self._pushed = False

    async def start(self, seed: int) -> asyncio.subprocess.Process:
        if self.push is not None and not self._pushed:
            await self.push(REPL_PATH, self.remote_path)
            self._pushed = True
        argv = [*self.prefix, "env", f"CRUCIBLE_RLM_SEED={seed}", f"PYTHONHASHSEED={seed}", self.python, "-u", self.remote_path]
        return await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE, limit=64 * 1024 * 1024)


class Kernel:
    def __init__(self, launcher: Any = None, *, seed: int = 0, cell_timeout_s: float = 120.0, kill_grace_s: float = 10.0):
        self.launcher = launcher or LocalLauncher()
        self.seed, self.cell_timeout_s, self.kill_grace_s = seed, cell_timeout_s, kill_grace_s
        self.proc: asyncio.subprocess.Process | None = None
        self.journal: list[JournalEntry] = []

    # ---- process ------------------------------------------------------------------------------------------------------
    async def start(self) -> None:
        self.proc = await self.launcher.start(self.seed)
        ev = await self._read(timeout=60)
        if ev is None or ev.get("event") != "ready":
            await self.close()
            raise KernelError(f"the RLM kernel did not start: {ev}")

    async def close(self) -> None:
        p, self.proc = self.proc, None
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

    async def _send(self, obj: dict[str, Any]) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(obj, default=str) + "\n").encode())
        await self.proc.stdin.drain()

    async def _read(self, timeout: float | None) -> dict[str, Any] | None:
        assert self.proc is not None and self.proc.stdout is not None
        while True:
            try:
                line = await asyncio.wait_for(self.proc.stdout.readline(), timeout=timeout)
            except TimeoutError:
                return {"event": "__timeout__"}
            if not line:
                return None
            try:
                return json.loads(line)
            except ValueError:
                continue

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
        await self._send({"type": "execute", "id": cid, "code": code, "timeout_s": t})
        res = CellResult()
        while True:
            ev = await self._read(timeout=(t + self.kill_grace_s) if t else None)
            if ev is None or ev.get("event") == "__timeout__":  # wedged (a blocking call ignored the deadline) or dead
                await self.close()
                res.killed, res.timed_out = True, ev is not None
                res.error = "the kernel stopped responding and was killed" if ev is not None else "the kernel process exited"
                return res
            kind = ev.get("event")
            if kind == "host_request":
                data = ev.get("data") or {}
                if served < len(recorded):
                    rec = recorded[served]
                    if (rec.get("request") or {}).get("type") != data.get("type"):
                        raise KernelDivergence(f"replay diverged: the cell asked for {data.get('type')!r}, the journal recorded "
                                               f"{(rec.get('request') or {}).get('type')!r}")
                    reply = rec.get("reply")
                    served += 1
                else:
                    try:
                        reply = await handler(data)
                    except KernelDivergence:
                        raise
                    except Exception as e:  # noqa: BLE001 -- a host-side failure is the cell's exception, not ours
                        reply = {"__error__": f"{type(e).__name__}: {e}"}
                    entry.replies.append({"request": data, "reply": reply})
                res.host_calls += 1
                await self._send({"type": "host_reply", "id": ev["id"], "data": reply})
            elif kind == "done" and ev.get("id") == cid:
                res.stdout, res.stderr, res.result = ev.get("stdout") or "", ev.get("stderr") or "", ev.get("result")
                res.error, res.timed_out, res.finished, res.digest = ev.get("error"), bool(ev.get("timed_out")), ev.get("finished"), ev.get("digest")
                entry.digest, entry.done = res.digest, True
                return res

    async def query(self, kind: str) -> Any:
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

    async def replay(self, journal: list[dict[str, Any]] | list[JournalEntry], *, live: Handler | None = None) -> JournalEntry | None:
        """Fresh kernel, then every completed cell from the record (digest-checked). Returns the unfinished last entry, if
        any, *without* executing it: the caller continues it with ``execute(code, handler, entry=...)``."""
        await self.close()
        entries = [e if isinstance(e, JournalEntry) else JournalEntry.from_dict(e) for e in journal]
        self.journal = []
        await self.start()

        async def refuse(data: dict[str, Any]) -> Any:
            raise KernelDivergence(f"replay asked for a host request beyond the journal: {data.get('type')!r}")

        pending = None
        for e in entries:
            if not e.done:
                pending = JournalEntry(e.code, e.timeout_s, list(e.replies), None, False)
                self.journal.append(pending)
                break
            copy = JournalEntry(e.code, e.timeout_s, list(e.replies), None, False)
            self.journal.append(copy)
            r = await self.execute(e.code, live or refuse, timeout_s=e.timeout_s, entry=copy)
            if e.digest is not None and r.digest != e.digest:
                from crucible.world.state.base import SnapshotMismatch

                raise SnapshotMismatch(f"RLM kernel replay diverged after cell {len(self.journal)}: digest {r.digest} != recorded {e.digest}")
        return pending
