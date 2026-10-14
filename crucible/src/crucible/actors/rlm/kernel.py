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

**Snapshots and bounded restores (M16 §3.4, R12, H11, B4, B5).** ``snapshot(dest)`` has the kernel dill its namespace
(``repl.py``: one pickle stream, 256 MiB total, 16 MiB per name, the rest reported ``lost``) and stream it back in signed
chunks; the host writes the bytes to ``dest``, checks their sha256 against the kernel's, and never unpickles them.
``replay(journal, snapshot=S)`` restores the blob into a fresh kernel, checks the namespace digest against the one taken
at the snapshot, and replays only ``journal[S.journal_len:]``; any failure falls back to a full replay, then to a
lenient one, and ``last_restore`` says which path ran. A journal that holds only the cells since the snapshot (a B5
checkpoint: pass ``offset=S.journal_len``) can only be restored through the snapshot.

*Bootstrap vs snapshot.* A cell executed with ``bootstrap=True`` (the harness's own setup: ``world``, ``rlm``, ``history``,
their classes) is journaled with the flag, and every snapshot carries the bootstrap cells. A restore re-runs them first
(their host replies from the record), then loads the blob: what the bootstrap bound is a persistent reference in the
blob (resolved to the fresh kernel's own objects, so ``host`` and ``world`` are the new session's), while the bootstrap's
data (``OBS``, ``INBOX``, ``HISTORY``...) is saved by value like any variable and replaces the fresh empty values. A name
the snapshot lost is removed after the restore (the bootstrap's own names keep their fresh value, so the harness still
works) and reported in ``Snapshot.lost``.

*Digests (B4).* The per-cell digest recorded in the journal is the kernel's bounded fingerprint (O(names), not
O(everything seen)); the full digest (complete, no 100k-character cut) is taken at snapshots and by ``digest()``, both
recorded on the journal entry (``full_digest``) and checked again when a replay passes that cell. The clock and OS
entropy a cell read travel in its entry (``entropy``) and are served again on replay, so ``time.time()`` and
``uuid4()`` replay exactly; ``datetime.now()`` does not (C clock) and takes the lenient path.

*Kernel lost between cells (B10).* The kernel acknowledges each cell (``started``) before its code runs; a kernel that
died or was wedged before that returns ``CellResult(lost_between_cells=True)``: the cell's code never ran.

*Lifetime (B9).* ``DockerLauncher(lifetime_s=...)`` bounds the container (the caller derives it from the episode's wall
budget, ``lifetime_for_episode``); with ``renewable=True`` its life is extended by ``renew()``.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import hashlib
import json
import os
import shutil
import signal
import sys
import time
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
SNAPSHOT_MAX_BYTES = 256 * 1024 * 1024  # Prime Agent repl.py:40-41 (MIT)
SNAPSHOT_MAX_VAR_BYTES = 16 * 1024 * 1024
SNAPSHOT_CHUNK = 4 * 1024 * 1024  # raw bytes per restore_chunk request (the kernel sends snapshot_chunk frames of the same size)
MAX_CHUNK_FIELD = 6 * 1024 * 1024  # base64 of one chunk
ENTROPY_MAX_READS, ENTROPY_MAX_BYTES = 4_096, 65_536

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
    stray_requests: int = 0  # (M16 hunt H11) host requests from a task of an earlier cell, refused while this cell ran
    # M16 B10: the kernel was gone (or wedged by earlier background work) before this cell's code started: nothing of it
    # ran, which the policy states as such (never "did not finish in time"). ``killed`` is True as well.
    lost_between_cells: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and not self.timed_out and not self.killed


@dataclass
class JournalEntry:
    code: str
    timeout_s: float | None
    replies: list[dict[str, Any]] = field(default_factory=list)  # [{"request": data, "reply": data}]
    digest: str | None = None  # the kernel's per-cell fingerprint after the cell (bounded, M16 B4)
    done: bool = False
    entropy: dict[str, Any] | None = None  # clock / OS-random reads, served again on replay (M16 B4)
    full_digest: str | None = None  # the full namespace digest after the cell, when one was taken (snapshot, digest())
    bootstrap: bool = False  # a host-written setup cell, re-run before a snapshot is loaded (M16 §3.4)

    def to_dict(self) -> dict[str, Any]:
        d = {"code": self.code, "timeout_s": self.timeout_s, "replies": self.replies, "digest": self.digest, "done": self.done}
        if self.entropy:
            d["entropy"] = self.entropy
        if self.full_digest:
            d["full_digest"] = self.full_digest
        if self.bootstrap:
            d["bootstrap"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JournalEntry:
        return cls(d["code"], d.get("timeout_s"), list(d.get("replies") or []), d.get("digest"), bool(d.get("done")),
                   d.get("entropy") or None, d.get("full_digest"), bool(d.get("bootstrap")))

    def fresh(self) -> JournalEntry:
        """A copy to re-execute: the record (code, replies, entropy, flag) without the outcome."""
        return JournalEntry(self.code, self.timeout_s, list(self.replies), None, False, self.entropy, None, self.bootstrap)


@dataclass
class JournalReply:
    """A handler's reply whose journal record is smaller than the reply (M16 B5: ``history.*`` replies of up to 256 KiB
    were copied into every journal and checkpoint). The cell receives ``reply``; the journal keeps ``{"request", "record"}``
    and a replay rebuilds the reply with ``resolve(request, record)`` (``Kernel.execute`` / ``replay``)."""

    reply: Any
    record: Any


Resolver = Callable[[dict[str, Any], Any], Any]  # (request, record) -> reply, sync or async


@dataclass
class Snapshot:
    """A kernel namespace snapshot (M16 §3.4). ``path`` is a host-side blob outside the kernel's reach that the host never
    unpickles; ``journal_len`` the cells it covers (absolute: ``kernel.journal_offset + len(kernel.journal)``);
    ``digest`` the full namespace digest at snapshot time and ``saved_digest`` the same over the saved names only;
    ``lost`` the names that could not be saved (``lost_reasons`` says why), reported to the model after a restore;
    ``bootstrap`` the bootstrap cells (journal dicts) a restore re-runs before loading the blob."""

    path: Path
    journal_len: int
    digest: str
    lost: list[str]
    bytes: int
    sha256: str = ""
    saved_digest: str = ""
    lost_reasons: dict[str, str] = field(default_factory=dict)
    bootstrap: list[dict[str, Any]] = field(default_factory=list)
    seed: int = 0
    python: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "journal_len": self.journal_len, "digest": self.digest, "lost": list(self.lost), "bytes": self.bytes,
                "sha256": self.sha256, "saved_digest": self.saved_digest, "lost_reasons": dict(self.lost_reasons), "bootstrap": list(self.bootstrap),
                "seed": self.seed, "python": self.python, "seconds": self.seconds}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Snapshot:
        return cls(Path(d["path"]), int(d["journal_len"]), str(d["digest"]), list(d.get("lost") or []), int(d.get("bytes") or 0),
                   str(d.get("sha256") or ""), str(d.get("saved_digest") or ""), dict(d.get("lost_reasons") or {}), list(d.get("bootstrap") or []),
                   int(d.get("seed") or 0), str(d.get("python") or ""), float(d.get("seconds") or 0.0))


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


def _strs(x: Any, n: int = 1_000_000, each: int = 4096) -> bool:
    return isinstance(x, list) and len(x) <= n and all(isinstance(s, str) and len(s) <= each for s in x)


def _valid_extras(ev: dict[str, Any]) -> str | None:
    """The M16 fields of a ``done`` frame: the cell's entropy record and the snapshot / restore / describe replies."""
    en = ev.get("entropy")
    if en is not None:
        if not isinstance(en, dict) or set(en) - {"t", "n", "u", "over"}:
            return "done.entropy"
        t, n, u = en.get("t", []), en.get("n", []), en.get("u", [])
        if not (isinstance(t, list) and len(t) <= ENTROPY_MAX_READS and all(type(x) is float for x in t)):
            return "done.entropy.t"
        if not (isinstance(n, list) and len(n) <= ENTROPY_MAX_READS and all(type(x) is int for x in n)):
            return "done.entropy.n"
        if not (_strs(u, ENTROPY_MAX_READS, 2 * ENTROPY_MAX_BYTES) and sum(map(len, u)) <= 2 * ENTROPY_MAX_BYTES):
            return "done.entropy.u"
        if not isinstance(en.get("over", False), bool):
            return "done.entropy.over"
    sn = ev.get("snapshot")
    if sn is not None:
        ok = (isinstance(sn, dict) and isinstance(sn.get("sha256"), str) and len(sn["sha256"]) == 64 and type(sn.get("bytes")) is int
              and type(sn.get("chunks")) is int and all(isinstance(sn.get(k), str) and len(sn[k]) <= 64 for k in ("digest", "saved_digest"))
              and _strs(sn.get("saved")) and isinstance(sn.get("lost"), dict)
              and all(isinstance(k, str) and isinstance(v, str) and len(v) <= 1024 for k, v in sn["lost"].items())
              and isinstance(sn.get("python"), str) and len(sn["python"]) <= 32)
        if not ok:
            return "done.snapshot"
    rs = ev.get("restore")
    if rs is not None and not (isinstance(rs, dict) and isinstance(rs.get("digest"), str) and len(rs["digest"]) <= 64 and _strs(rs.get("restored"))):
        return "done.restore"
    ds = ev.get("describe")
    if ds is not None:
        ok = isinstance(ds, dict) and type(ds.get("total")) is int and isinstance(ds.get("entries"), list) and all(
            isinstance(e, dict) and isinstance(e.get("name"), str) and isinstance(e.get("type"), str) and len(e["type"]) <= 512
            and type(e.get("size")) is int and isinstance(e.get("unit"), str) and isinstance(e.get("boot"), bool) for e in ds["entries"])
        if not ok:
            return "done.describe"
    return None


def _valid(ev: Any) -> str | None:
    """Why a kernel frame is invalid (None: valid)."""
    if not isinstance(ev, dict):
        return "not an object"
    kind = ev.get("event")
    if kind == "started":
        return None if isinstance(ev.get("id"), str) and len(ev["id"]) <= 128 else "started id"
    if kind == "snapshot_chunk":
        if not isinstance(ev.get("id"), str) or len(ev["id"]) > 128 or type(ev.get("seq")) is not int:
            return "snapshot_chunk id"
        return None if isinstance(ev.get("data"), str) and len(ev["data"]) <= MAX_CHUNK_FIELD else "snapshot_chunk data"
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
        return _valid_extras(ev)
    return f"unknown event {str(kind)[:40]!r}"


_PYLIB: dict[str, Any] = {}
PYLIB_NAME = "crucible_pylib.zip"


def pylib_zip() -> Path | None:
    """dill's pure-Python package (BSD-3-Clause, its licence included) as a zip pushed next to the kernel script, which
    reads it into memory at start and removes it (``repl._load_pylib``): the Docker kernel has no network and the host
    kernel runs the base interpreter without the project's site-packages. Built once per process; None when dill is not
    installed (snapshots are then unavailable and callers fall back to journal replay)."""
    if "path" in _PYLIB:
        return _PYLIB["path"]
    import importlib.util

    path = None
    try:
        spec = importlib.util.find_spec("dill")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.submodule_search_locations:
        import atexit
        import tempfile
        import zipfile
        from importlib import metadata

        src = Path(next(iter(spec.submodule_search_locations)))
        tmp = Path(tempfile.mkdtemp(prefix="crucible-rlm-pylib-"))
        path = tmp / PYLIB_NAME
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(src.rglob("*.py")):
                rel = f.relative_to(src)
                if rel.parts[0] != "tests":
                    z.write(f, (Path("dill") / rel).as_posix())
            with contextlib.suppress(Exception):
                lic = metadata.distribution("dill").read_text("LICENSE")
                if lic:
                    z.writestr("dill/LICENSE", lic)
        atexit.register(shutil.rmtree, tmp, True)
    _PYLIB["path"] = path
    return path


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
        lib = pylib_zip()  # the venv's dill, as the Docker kernel gets it; each kernel consumes (removes) its copy (M16)
        if lib is not None:
            shutil.copyfile(lib, Path(self._cwd) / PYLIB_NAME)
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
        self.lib_path = remote_path.rsplit("/", 1)[0] + "/" + PYLIB_NAME if "/" in remote_path else PYLIB_NAME
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
        lib = pylib_zip()
        if self.push is not None and lib is not None:  # next to the kernel, which loads it into memory and removes it (M16)
            with contextlib.suppress(Exception):  # no dill in the sandbox only costs snapshots (callers keep journal replay)
                await self.push(lib, self.lib_path)
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
RENEW_SLACK_S = 3600  # a renewable container's expiry label lies this far past its start deadline (a label cannot be renewed)
CTL_DIR = "/crucible-ctl"  # renewable containers: a root-owned tmpfs holding the deadline (the kernel's user cannot write it)
# PID 1 of a renewable container: exits at the later of its start deadline and the one ``renew`` wrote
LIFE_LOOP = ('d=$1; while :; do n=$(cat ' + CTL_DIR + '/deadline 2>/dev/null); [ "${n:-0}" -gt "$d" ] 2>/dev/null && d=$n; '
             '[ "$(date +%s)" -lt "$d" ] || exit 0; sleep 2; done')


def lifetime_for_episode(wall_s: float | None, *, margin_s: float = 900.0, floor_s: float = 600.0) -> int:
    """The kernel container's lifetime for an episode with wall budget ``wall_s`` (M16 B9: a fixed 6 h ended long
    episodes): the budget plus a margin for restores and grading; the 6 h default when the episode has no wall budget."""
    if not wall_s or wall_s <= 0:
        return DEFAULT_LIFETIME_S
    return int(max(floor_s, float(wall_s) + margin_s))
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
    past their ``expires`` time unless their owner is a live process on this host (a renewed container outlives its
    label, M16 B9). Containers of live processes, other hosts' unexpired ones and unlabelled ones are left alone."""
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
        here = h == host and pid.isdigit()
        dead = here and int(pid) != os.getpid() and not _pid_alive(int(pid))
        live_here = here and not dead
        if dead or (expired and not live_here):
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
    containers at exit; and the container ends by itself after ``lifetime_s`` whatever happens to its owner.

    Lifetime per episode (M16 B9): the caller passes ``lifetime_s`` (``lifetime_for_episode(wall_s)``). With
    ``renewable=True`` PID 1 is a small shell loop that ends at the later of the start deadline and the deadline
    ``renew(lifetime_s)`` writes into a root-owned tmpfs (``/crucible-ctl``, written through ``docker exec --user 0``; the
    kernel's user cannot extend its own life); otherwise PID 1 is ``sleep <lifetime_s>`` and the life is fixed."""

    def __init__(self, image: str = "python:3.12-slim", *, memory: str = "1g", pids: int = 256, cpus: str = "1", docker: str = "docker",
                 lifetime_s: int = DEFAULT_LIFETIME_S, renewable: bool = False):
        self.image, self.memory, self.pids, self.cpus, self.docker = image, memory, pids, cpus, docker
        self.lifetime_s = int(lifetime_s)
        self.renewable = bool(renewable)
        self.deadline: float | None = None  # epoch seconds at which the container ends by itself
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
            _install_reaper()
            if self.docker not in _REAPER["stale_checked"]:  # once per process and docker binary: what dead processes left
                _REAPER["stale_checked"].add(self.docker)
                await asyncio.to_thread(reap_stale, self.docker)
            name = self.name = f"crucible-rlm-{uuid.uuid4().hex[:12]}"
            _LIVE[name] = self.docker  # tracked before it exists: the exit reaper removes it whatever happens to this coroutine
            deadline = int(time.time()) + self.lifetime_s
            self.deadline = float(deadline)
            ctl = ["--tmpfs", f"{CTL_DIR}:rw,size=64k,mode=0755"] if self.renewable else []
            life = ["sh", "-c", LIFE_LOOP, "crucible-rlm-life", str(deadline)] if self.renewable else ["sleep", str(self.lifetime_s)]
            run = asyncio.ensure_future(self._run(
                "run", "-d", "--rm", "--name", name, "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=256m", *ctl,
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65534:65534", "--memory", self.memory,
                "--pids-limit", str(self.pids), "--cpus", self.cpus, "--workdir", "/tmp", "--label", "crucible.rlm=1",
                "--label", f"{OWNER_LABEL}={owner_id()}", "--label", f"{EXPIRES_LABEL}={deadline + (RENEW_SLACK_S if self.renewable else 0)}",
                "--env", "HOME=/tmp", self.image, *life))  # a bounded life, never ``sleep infinity``
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

    async def renew(self, lifetime_s: float | None = None) -> float:
        """Extend the container's life to ``now + lifetime_s`` (default: the launcher's ``lifetime_s``); returns the new
        deadline (epoch s). Only a renewable container (``renewable=True``) can be renewed; never shortens it."""
        if not self.renewable:
            raise KernelError("this kernel container's lifetime is fixed: build DockerLauncher(renewable=True) to renew it")
        if self.name is None:
            raise KernelError("no kernel container to renew")
        deadline = int(time.time() + (self.lifetime_s if lifetime_s is None else float(lifetime_s)))
        await self._run("exec", "--user", "0", self.name, "sh", "-c", f"echo {deadline} > {CTL_DIR}/deadline")
        self.deadline = max(self.deadline or 0.0, float(deadline))
        return self.deadline

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
        self.python: str | None = None  # the kernel interpreter's version (a snapshot loads only into the same one)
        self.journal: list[JournalEntry] = []
        self.journal_offset = 0  # cells before journal[0]: > 0 after a restore from a snapshot (the prefix lives in the blob)
        self.last_snapshot: Snapshot | None = None
        self.last_restore: dict[str, Any] | None = None  # {"mode", "from", "cells", "seconds", "divergences", "lost", "fallback"}
        self.describe_total = 0
        self.protocol_errors = 0
        self.replay_divergences = 0
        self._lenient = False
        self._boot: list[JournalEntry] = []  # the bootstrap cells, kept for every snapshot (also when the journal is a suffix)
        self.resolve: Resolver | None = None  # the default resolver of journaled records (``JournalReply``)
        self.orphaned = False  # a kill could not confirm the in-sandbox kernel was gone
        self._mac: Any = None  # HMAC-SHA256 under this session's key (B10): frames without a valid MAC are dropped

    # ---- process ------------------------------------------------------------------------------------------------------
    async def start(self) -> None:
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
        self.python = ev.get("python") if isinstance(ev.get("python"), str) else None

    def alive(self) -> bool:
        """The kernel process is running and its channel is open (a kernel that died between cells is not)."""
        p = self.proc
        return p is not None and p.returncode is None and not (p.stdout is not None and p.stdout.at_eof())

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
    async def execute(self, code: str, handler: Handler, *, timeout_s: float | None = None, entry: JournalEntry | None = None,
                      bootstrap: bool = False, resolve: Resolver | None = None) -> CellResult:
        """Run one cell. ``handler(data)`` answers its host requests. ``entry``: continue a journaled, partly executed cell
        (its recorded replies are served first, then ``handler`` takes over). ``bootstrap``: a host-written setup cell
        (``world``, ``rlm``, ``history``...): journaled with the flag and re-run before a snapshot is loaded (M16 §3.4).

        A handler may return ``JournalReply(reply, record)``: the cell gets ``reply``, the journal keeps ``record``. A
        recorded entry that holds a ``record`` is served ``resolve(request, record)`` (default: ``self.resolve``); no
        resolver, or one that raises, is a divergence (strict: ``KernelDivergence``; lenient: an error reply, counted)."""
        if self.proc is None:
            await self.start()
        t = self.cell_timeout_s if timeout_s is None else timeout_s
        if entry is None:
            entry = JournalEntry(code, t, bootstrap=bool(bootstrap))
            self.journal.append(entry)
        recorded = list(entry.replies)
        served = 0
        cid = uuid.uuid4().hex
        errors0 = self.protocol_errors
        req: dict[str, Any] = {"type": "execute", "id": cid, "code": code, "timeout_s": t}
        if entry.bootstrap:
            req["bootstrap"] = True
        if entry.entropy:
            req["entropy"] = entry.entropy  # replay: the clock and random bytes the cell read the first time (M16 B4)
        res = CellResult()
        started = False
        try:
            await self._send(req)
        except (OSError, RuntimeError, AssertionError):  # the kernel's channel is closed: it died after the last cell
            pass
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
                res.killed = True
                if not started:  # M16 B10: nothing of this cell ran
                    res.lost_between_cells = True
                    res.error = ("the kernel was lost before this cell ran: its code did not run" if ev is None else
                                 "the kernel was unresponsive before this cell ran (earlier background work blocked it) and was "
                                 "killed: this cell's code did not run")
                else:
                    res.timed_out = ev is not None
                    res.error = "the kernel stopped responding and was killed" if ev is not None else "the kernel process exited"
                res.protocol_errors = self.protocol_errors - errors0
                return res
            kind = ev.get("event")
            if kind == "started" and ev.get("id") == cid and not started:
                started = True
            elif kind == "host_request":
                data = ev.get("data") or {}
                if ev.get("cell") != cid:
                    # (M16 hunt H11) a task an earlier cell started (asyncio.create_task) and that outlived it: its World, sub-LLM,
                    # agent or history request was served here and credited to this cell's model call, and a replay (where the
                    # task fires at another moment) went lenient. Refused with an error the stray task sees; never journaled
                    res.stray_requests += 1
                    with contextlib.suppress(Exception):
                        await self._send({"type": "host_reply", "id": ev["id"], "data": {"__error__": (
                            f"host request {data.get('type')!r} refused: it comes from a task of an earlier python call, not the cell "
                            "running now (await your tasks inside the cell that started them)")}})
                    continue
                if served < len(recorded):
                    rec = recorded[served]
                    if (rec.get("request") or {}).get("type") != data.get("type") and self._lenient:
                        self.replay_divergences += 1
                        reply = {"__error__": "replay: the cell asked for another request than before"}
                    elif (rec.get("request") or {}).get("type") != data.get("type"):
                        raise KernelDivergence(f"replay diverged: the cell asked for {data.get('type')!r}, the journal recorded "
                                               f"{(rec.get('request') or {}).get('type')!r}")
                    elif "record" in rec and "reply" not in rec:
                        reply = await self._resolve(resolve or self.resolve, data, rec["record"])
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
                    if isinstance(reply, JournalReply):  # M16 B5: journal the small record, send the reply
                        entry.replies.append({"request": data, "record": reply.record})
                        reply = reply.reply
                    else:
                        entry.replies.append({"request": data, "reply": reply})
                res.host_calls += 1
                await self._send({"type": "host_reply", "id": ev["id"], "data": reply})
            elif kind == "done" and ev.get("id") == cid:
                res.stdout, res.stderr, res.result = ev.get("stdout") or "", ev.get("stderr") or "", ev.get("result")
                res.error, res.timed_out, res.finished, res.digest = ev.get("error"), bool(ev.get("timed_out")), ev.get("finished"), ev.get("digest")
                res.interrupted = interrupted
                res.protocol_errors = self.protocol_errors - errors0
                entry.digest, entry.done = res.digest, True
                entry.entropy = ev.get("entropy") or None
                if entry.bootstrap and res.ok and entry not in self._boot:
                    self._boot.append(entry)
                return res
            elif kind in ("done", "ready", "started", "snapshot_chunk"):  # a frame for no request in flight: forged by the cell
                self.protocol_errors += 1

    async def _resolve(self, resolve: Resolver | None, data: dict[str, Any], record: Any) -> Any:
        """The reply a journaled ``record`` stands for (``JournalReply``), rebuilt by the caller's resolver."""
        try:
            if resolve is None:
                raise LookupError("no resolver for a journaled record")
            reply = resolve(data, record)
            if asyncio.iscoroutine(reply):
                reply = await reply
            return reply
        except (KernelDivergence, HostAbort):
            raise
        except Exception as e:
            if not self._lenient:
                raise KernelDivergence(f"replay could not rebuild a {data.get('type')!r} reply from its record: {type(e).__name__}: {e}") from e
            self.replay_divergences += 1
            return {"__error__": f"replay: the reply could not be rebuilt ({type(e).__name__})"}

    async def query(self, kind: str, **payload: Any) -> Any:
        if self.proc is None:
            raise KernelError(f"no kernel to ask for {kind}")
        cid = uuid.uuid4().hex
        await self._send({"type": kind, "id": cid, **payload})
        while True:
            ev = await self._read(timeout=120)
            if ev is None or ev.get("event") == "__timeout__":
                raise KernelError(f"no reply to {kind}")
            if ev.get("event") == "done" and ev.get("id") == cid:
                if ev.get("error"):
                    raise KernelError(f"{kind}: {ev['error'][:1000]}")
                return ev.get(kind)

    async def digest(self) -> str:
        """The full namespace digest (complete; M16 B4), recorded on the last journal entry so a replay checks it there."""
        d = await self.query("digest")
        if self.journal and self.journal[-1].done:
            self.journal[-1].full_digest = d
        return d

    async def names(self) -> list[str]:
        return await self.query("names")

    async def describe(self, limit: int = 500) -> list[dict[str, Any]]:
        """``[{name, type, size, unit, boot}]`` for the python-state row (M16 §3.4): the model's names (private names and
        the harness objects the bootstrap re-creates left out; the bootstrap's data such as ``OBS`` kept with ``boot``
        True), at most ``limit``, sorted by name; ``describe_total`` counts them all. ``size`` is a length for builtin
        containers and text (``unit`` items / chars / bytes), ``nbytes`` for arrays, else ``sys.getsizeof``."""
        d = await self.query("describe", limit=int(limit))
        self.describe_total = int(d["total"])
        return list(d["entries"])

    # ---- snapshots (M16 §3.4) -----------------------------------------------------------------------------------------
    async def snapshot(self, dest: Path | str, *, max_bytes: int = SNAPSHOT_MAX_BYTES, max_var_bytes: int = SNAPSHOT_MAX_VAR_BYTES,
                       timeout_s: float = 600.0) -> Snapshot:
        """Snapshot the namespace into ``dest`` (between cells). The kernel dills it and streams signed chunks; this side
        writes the bytes, checks count, order and sha256 against the kernel's report, and never unpickles them. Raises
        ``KernelError`` when the kernel cannot (no dill: callers keep journal replay) or the stream fails its checks; if
        the kernel stopped answering it is closed (``proc`` is None: restore it by replay before the next cell)."""
        if self.proc is None:
            raise KernelError("no kernel to snapshot")
        if self.journal and not self.journal[-1].done:
            raise KernelError("a snapshot is taken between cells")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex[:8]}.part")
        cid = uuid.uuid4().hex
        t0 = time.monotonic()
        h, size, seq = hashlib.sha256(), 0, 0
        cap = int(max_bytes) + 4096
        failed: str | None = None
        meta: dict[str, Any] | None = None
        try:
            await self._send({"type": "snapshot", "id": cid, "max_bytes": int(max_bytes), "max_var_bytes": int(max_var_bytes)})
            with open(tmp, "wb") as f:  # noqa: ASYNC230 -- chunk writes between frames, bounded by the snapshot cap
                while True:
                    left = timeout_s - (time.monotonic() - t0)
                    ev = await self._read(timeout=max(left, 0.01))
                    if ev is None or ev.get("event") == "__timeout__":
                        await self.close()
                        raise KernelError("the kernel " + ("exited" if ev is None else f"did not finish the snapshot in {timeout_s:g} s")
                                          + " (it was closed; restore it by replay)")
                    kind = ev.get("event")
                    if kind == "snapshot_chunk" and ev.get("id") == cid:
                        if failed is not None:
                            continue
                        try:
                            b = base64.b64decode(ev["data"], validate=True)
                        except (binascii.Error, ValueError):
                            failed = "a chunk is not base64"
                            continue
                        if ev["seq"] != seq:
                            failed = f"chunk {seq} is missing (got {ev['seq']})"
                            continue
                        size += len(b)
                        if size > cap:
                            failed = f"the stream exceeds {cap} bytes"
                            continue
                        h.update(b)
                        f.write(b)
                        seq += 1
                    elif kind == "done" and ev.get("id") == cid:
                        if ev.get("error"):
                            raise KernelError(str(ev["error"])[:1000])
                        meta = ev.get("snapshot")
                        break
                    elif kind == "host_request":  # background work of an earlier cell: not served between cells
                        with contextlib.suppress(Exception):
                            await self._send({"type": "host_reply", "id": ev["id"], "data": {"__error__": "no host requests during a snapshot"}})
                    else:
                        self.protocol_errors += 1
            if failed is None and not meta:
                failed = "no snapshot report"
            if failed is None and (meta["sha256"] != h.hexdigest() or meta["bytes"] != size or meta["chunks"] != seq):
                failed = "the bytes received do not match the kernel's sha256 / size / chunk count"
            if failed is not None:
                raise KernelError(f"snapshot rejected: {failed}")
            os.replace(tmp, dest)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink()
        assert meta is not None
        snap = Snapshot(path=dest, journal_len=self.journal_offset + len(self.journal), digest=meta["digest"], lost=sorted(meta["lost"]),
                        bytes=size, sha256=meta["sha256"], saved_digest=meta["saved_digest"], lost_reasons=dict(meta["lost"]),
                        bootstrap=[e.to_dict() for e in self._boot], seed=self.seed, python=meta["python"],
                        seconds=round(time.monotonic() - t0, 3))
        if self.journal:
            self.journal[-1].full_digest = snap.digest
        self.last_snapshot = snap
        return snap

    async def _load_blob(self, snap: Snapshot) -> dict[str, Any]:
        """Send the blob to this kernel (it must be bootstrapped) and load it; the kernel's report ``{restored, digest}``."""
        rid = uuid.uuid4().hex
        h = hashlib.sha256()
        with open(snap.path, "rb") as f:  # noqa: ASYNC230 -- chunk reads between sends
            seq = 0
            while True:
                b = f.read(SNAPSHOT_CHUNK)
                if not b:
                    break
                h.update(b)
                await self._send({"type": "restore_chunk", "id": rid, "seq": seq, "data": base64.b64encode(b).decode("ascii")})
                seq += 1
        if h.hexdigest() != snap.sha256:  # changed on disk since it was written: the kernel would refuse it too
            raise KernelError("the snapshot file does not match its sha256")
        await self._send({"type": "restore", "id": rid, "sha256": snap.sha256})
        while True:
            ev = await self._read(timeout=600)
            if ev is None or ev.get("event") == "__timeout__":
                await self.close()
                raise KernelError("the kernel did not load the snapshot")
            if ev.get("event") == "done" and ev.get("id") == rid:
                if ev.get("error") or not isinstance(ev.get("restore"), dict):
                    raise KernelError(str(ev.get("error") or "no restore report")[:1000])
                return ev["restore"]
            if ev.get("event") == "protocol_error" and ev.get("id") == rid:
                raise KernelError(f"restore: {ev.get('error')}")
            self.protocol_errors += ev.get("event") != "protocol_error"

    async def _from_snapshot(self, snap: Snapshot, suffix: list[JournalEntry], live: Handler | None, strict: bool,
                             resolve: Resolver | None = None) -> JournalEntry | None:
        """A fresh kernel, its bootstrap re-run, the blob loaded and digest-checked, then the suffix replayed."""
        if not snap.path.is_file():
            raise KernelError(f"the snapshot file is missing: {snap.path}")
        await self._fresh()
        if snap.python and self.python and snap.python.split(".")[:2] != self.python.split(".")[:2]:
            raise KernelError(f"the snapshot was taken under python {snap.python}; this kernel runs {self.python}")
        boots = [JournalEntry.from_dict(b) for b in snap.bootstrap]
        for b in boots:
            b.bootstrap = b.done = True
        await self._replay_entries(boots, live, strict, resolve)
        boots = list(self.journal)  # the re-executed copies
        rep = await self._load_blob(snap)
        if rep["digest"] != (snap.saved_digest if snap.lost else snap.digest) or (strict and (snap.lost or rep["digest"] != snap.digest)):
            if strict:
                from crucible.world.state.base import SnapshotMismatch

                why = f"names lost: {snap.lost}" if snap.lost else f"digest {rep['digest']} != {snap.digest}"
                raise SnapshotMismatch(f"RLM kernel snapshot restore is not exact ({why})")
            self.replay_divergences += 1
        self.journal, self.journal_offset = [], snap.journal_len
        self._boot = boots
        self.last_snapshot = snap
        return await self._replay_entries(suffix, live, strict, resolve)

    # ---- journal ------------------------------------------------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        d: dict[str, Any] = {"seed": self.seed, "cell_timeout_s": self.cell_timeout_s, "journal": [e.to_dict() for e in self.journal]}
        if self.journal_offset:
            d["journal_offset"] = self.journal_offset
        if self.last_snapshot is not None:
            d["snapshot"] = self.last_snapshot.to_dict()
        return d

    async def _fresh(self) -> None:
        await self.close()
        self.journal, self.journal_offset, self._boot = [], 0, []
        self.replay_divergences = 0
        await self.start()

    async def _replay_entries(self, entries: list[JournalEntry], live: Handler | None, strict: bool,
                              resolve: Resolver | None = None) -> JournalEntry | None:
        """Re-execute completed entries from their record into this kernel (fingerprint- and, where one was recorded,
        full-digest-checked); returns the unfinished last entry, if any, without executing it."""

        async def refuse(data: dict[str, Any]) -> Any:
            if not strict:
                self.replay_divergences += 1
                raise RuntimeError("replay: this request is not in the journal (the cell took another path than before)")
            raise KernelDivergence(f"replay asked for a host request beyond the journal: {data.get('type')!r}")

        from crucible.world.state.base import SnapshotMismatch

        for e in entries:
            if not e.done:
                pending = e.fresh()
                pending.replies = list(e.replies)
                self.journal.append(pending)
                return pending
            copy = e.fresh()
            self.journal.append(copy)
            self._lenient = not strict
            try:
                r = await self.execute(e.code, live or refuse, timeout_s=e.timeout_s, entry=copy, resolve=resolve)
            finally:
                self._lenient = False
            n = self.journal_offset + len(self.journal)
            if r.killed:
                raise KernelDivergence(f"replay: cell {n} did not complete ({r.error})")
            if e.digest is not None and r.digest != e.digest:
                if strict:
                    raise SnapshotMismatch(f"RLM kernel replay diverged after cell {n}: digest {r.digest} != recorded {e.digest}")
                self.replay_divergences += 1
            if e.full_digest is not None:
                d = await self.digest()
                if d != e.full_digest:
                    if strict:
                        raise SnapshotMismatch(f"RLM kernel replay diverged after cell {n}: full digest {d} != recorded {e.full_digest}")
                    self.replay_divergences += 1
        return None

    async def replay(self, journal: list[dict[str, Any]] | list[JournalEntry], *, live: Handler | None = None, strict: bool = True,
                     snapshot: Snapshot | None = None, offset: int = 0, resolve: Resolver | None = None) -> JournalEntry | None:
        """Fresh kernel, then every completed cell from the record (digest-checked). Returns the unfinished last entry, if
        any, *without* executing it: the caller continues it with ``execute(code, handler, entry=...)``.

        ``strict=False`` (the fallback after a divergence): a digest mismatch is counted (``replay_divergences``), not
        raised, and a host request the journal does not hold gets an error reply (that cell fails), never a live action.

        ``snapshot`` (M16 §3.4): restore the blob, check the digest, replay only the cells after it; on any failure a full
        replay, then a lenient one (with ``strict=False`` only the lenient paths run). ``offset``: the absolute index of
        ``journal[0]`` (``snapshot.journal_len`` for a journal that holds only the cells since the snapshot, which can
        only be restored through it). Passing this kernel's own journal (or a slice of it) after a snapshot restore uses
        ``last_snapshot`` and ``journal_offset`` by itself. ``last_restore`` reports the path that ran. ``resolve``
        rebuilds replies journaled as records (``JournalReply``; default ``self.resolve``)."""
        resolve = resolve or self.resolve
        t0 = time.monotonic()
        entries = [e if isinstance(e, JournalEntry) else JournalEntry.from_dict(e) for e in journal]
        if snapshot is None and offset == 0 and self.journal_offset and self.last_snapshot is not None and journal and self.journal \
                and journal[0] is self.journal[0]:
            snapshot, offset = self.last_snapshot, self.journal_offset  # this kernel's own suffix journal
        if snapshot is None:
            if offset:
                raise KernelError("a journal that starts after cell 0 can only be restored with its snapshot")
            prev = self.last_snapshot
            await self._fresh()
            try:
                pending = await self._replay_entries(entries, live, strict, resolve)
            finally:
                self.last_snapshot = prev if prev is not None and prev.journal_len <= len(self.journal) else None
            self._report("full" if strict else "lenient", "journal", len(entries), t0)
            return pending
        return await self._replay_chain(entries, snapshot, offset, live, strict, t0, resolve)

    def _report(self, mode: str, src: str, cells: int, t0: float, *, lost: list[str] | None = None, fallback: list[str] | None = None) -> None:
        self.last_restore = {"mode": mode, "from": src, "cells": cells, "seconds": round(time.monotonic() - t0, 3),
                             "divergences": self.replay_divergences, "lost": list(lost or []), "fallback": list(fallback or [])}

    async def _replay_chain(self, entries: list[JournalEntry], snap: Snapshot, offset: int, live: Handler | None, strict: bool,
                            t0: float, resolve: Resolver | None) -> JournalEntry | None:
        from crucible.world.state.base import SnapshotMismatch

        cut = snap.journal_len - offset
        usable = 0 <= cut <= len(entries)
        suffix = entries[cut:] if usable else []
        full_ok = offset == 0
        fallback: list[str] = [] if usable else [f"the snapshot covers {snap.journal_len} cells; the journal holds {offset}..{offset + len(entries)}"]
        soft = (KernelError, SnapshotMismatch, OSError, ValueError)
        if strict and usable:
            try:
                pending = await self._from_snapshot(snap, suffix, live, True, resolve)
                self._report("snapshot", "snapshot", len(suffix), t0, lost=snap.lost)
                return pending
            except KernelStartError:
                raise
            except soft as e:
                fallback.append(f"snapshot: {type(e).__name__}: {str(e)[:300]}")
        if strict and full_ok:
            try:
                await self._fresh()
                pending = await self._replay_entries(entries, live, True, resolve)
                self.last_snapshot = snap if usable else None
                self._report("full", "journal", len(entries), t0, fallback=fallback)
                return pending
            except KernelStartError:
                raise
            except (SnapshotMismatch, KernelDivergence) as e:
                fallback.append(f"full: {type(e).__name__}: {str(e)[:300]}")
        if usable and (not full_ok or not snap.lost):  # the snapshot's prefix is exact; only the suffix is lenient
            try:
                pending = await self._from_snapshot(snap, suffix, live, False, resolve)
                self._report("lenient", "snapshot", len(suffix), t0, lost=snap.lost, fallback=fallback)
                return pending
            except KernelStartError:
                raise
            except soft as e:
                fallback.append(f"lenient snapshot: {type(e).__name__}: {str(e)[:300]}")
        if not full_ok:
            raise KernelError("the kernel could not be restored (the journal starts at the snapshot): " + "; ".join(fallback))
        await self._fresh()
        pending = await self._replay_entries(entries, live, False, resolve)
        self.last_snapshot = snap if usable else None
        self._report("lenient", "journal", len(entries), t0, fallback=fallback)
        return pending
