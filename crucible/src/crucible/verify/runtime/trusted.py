"""Trusted channels for graders (DESIGN §8.3 layer 7, P26). Standard library only: staged into any verifier image.

A grader runs as root. Every command that executes candidate code goes through ``Trusted.run(..., candidate=True)``:

- privileges are dropped *before* the candidate's first instruction (its ``setup.py``, CMake scripts, import side
  effects): a new session, ``PR_SET_NO_NEW_PRIVS``, no groups, an unprivileged uid never used by the image;
- every invocation gets a **fresh HOME and TMPDIR** (no cache-and-replay across timed repetitions, no memo files);
- the environment is rebuilt from scratch (fixed ``PATH``, nothing inherited, ``PYTHONPATH`` only as given, with the
  grader's own entries first);
- wall and CPU time are measured **by the grader**, around the child, never read from the candidate;
- after the call the uid's whole process cohort is killed (``kill(-1)`` as the uid, then a ``/proc`` sweep), so no
  background job survives into a reference measurement or the next timed round;
- ``/dev/shm`` and SysV IPC objects owned by the uid are removed.

Verdicts never come from a candidate's exit code or stdout: callers read artifacts the candidate wrote into a directory
that is **sealed** (chowned to root, made read-only) before a trusted process compares it.

Without root (developer machines, unit tests) the same calls run unprivileged as the current user and every result is
marked ``trusted=False``; a grader configured with ``require_root`` refuses to grade in that mode.

Ported from our task 03/04 graders (``as_unpriv``, ``kill_unpriv``, ``seal``) and from the Crucible guard (itself
following Karotte ``subprocess.py`` ``_demote`` and ``process_utils``).
"""

from __future__ import annotations

import grp
import hashlib
import json
import os
import pwd
import select
import shutil
import signal
import stat
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
PR_SET_NO_NEW_PRIVS = 38
OUT_CAP = 4 * 1024 * 1024


class InfraError(RuntimeError):
    """The grader or its infrastructure failed (never the candidate): the grade is masked, not scored."""


@dataclass
class RunResult:
    rc: int
    stdout: str
    stderr: str
    wall_s: float  # measured by the grader around the child (never by the candidate)
    cpu_s: float  # user + system time of the child tree (rusage), measured by the grader
    timed_out: bool = False
    candidate: bool = False
    trusted: bool = True  # False when the privilege drop was not available (not root)
    home: str | None = None

    @property
    def log(self) -> str:
        return (self.stdout or "") + (self.stderr or "")


def _uid_taken(uid: int) -> bool:
    for get in (pwd.getpwuid, grp.getgrgid):
        try:
            get(uid)
            return True
        except KeyError:
            pass
    return False


def fresh_uid(start: int = 61000) -> int:
    """An uid/gid no image account or group uses (an image's uid-1000 user may have sudo or own files)."""
    uid = start
    while _uid_taken(uid) or any(_owned_by(uid)):
        uid += 1
    return uid


def _owned_by(uid: int):
    for name in os.listdir("/proc") if os.path.isdir("/proc") else []:
        if name.isdigit():
            try:
                if os.stat(f"/proc/{name}").st_uid == uid:
                    yield int(name)
            except OSError:
                continue


def _no_new_privs() -> None:
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")


def uid_pids(uid: int) -> list[int]:
    pids = []
    if not os.path.isdir("/proc"):
        return pids
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/status") as f:
                st = f.read()
        except OSError:
            continue
        real = next((ln.split()[1] for ln in st.splitlines() if ln.startswith("Uid:")), None)
        state = next((ln.split()[1] for ln in st.splitlines() if ln.startswith("State:")), "")
        if real == str(uid) and state != "Z":
            pids.append(int(name))
    return pids


def kill_cohort(uid: int, timeout_s: float = 20.0) -> dict:
    """Kill every process of ``uid`` (reparented and setsid'd ones included). Refuses uid 0."""
    if uid == 0:
        raise InfraError("refusing to kill the cohort of uid 0")
    deadline = time.monotonic() + timeout_s
    rounds = 0
    while time.monotonic() < deadline:
        rounds += 1
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:  # pragma: no cover - child
            try:
                os.close(r)
                os.setgroups([])
                os.setgid(uid)
                os.setuid(uid)
                for _ in range(100):
                    try:
                        os.kill(-1, signal.SIGKILL)
                    except OSError:
                        break
                os.write(w, b"\x01")
            finally:
                os._exit(0)
        os.close(w)
        os.read(r, 1)
        os.close(r)
        os.waitpid(pid, 0)
        for p in uid_pids(uid):
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if not uid_pids(uid):
            return {"ok": True, "rounds": rounds}
        time.sleep(0.05)
    survivors = uid_pids(uid)
    if survivors:
        raise InfraError(f"candidate processes survive a cohort kill: {survivors[:10]}")
    return {"ok": True, "rounds": rounds}


def _ipc_clean(uid: int) -> None:
    for d in ("/dev/shm", "/dev/mqueue"):
        if os.path.isdir(d):
            for name in os.listdir(d):
                p = os.path.join(d, name)
                try:
                    if os.lstat(p).st_uid == uid:
                        shutil.rmtree(p) if os.path.isdir(p) and not os.path.islink(p) else os.unlink(p)
                except OSError:
                    pass
    for kind, flag in (("/proc/sysvipc/shm", "-m"), ("/proc/sysvipc/sem", "-s"), ("/proc/sysvipc/msg", "-q")):
        try:
            with open(kind) as f:
                rows = f.read().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            cols = row.split()
            # key shmid perms ... uid is column 7 (shm), 4 (sem), 7 (msg) - match on any uid column equal to ours
            if str(uid) in cols[3:9] and shutil.which("ipcrm"):
                subprocess.run(["ipcrm", flag, cols[1]], capture_output=True, check=False)


@dataclass
class Trusted:
    """The grader's handle on trusted execution. ``work`` is a root-owned 0755 scratch root for this grade."""

    work: Path
    uid: int | None = None
    require_root: bool = False
    path: str = SAFE_PATH
    is_root: bool = field(init=False)
    _homes: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.work = Path(self.work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.is_root = hasattr(os, "geteuid") and os.geteuid() == 0
        if self.require_root and not self.is_root:
            raise InfraError("this grader must run as root: it drops privileges for candidate code")
        if self.is_root:
            os.chmod(self.work, 0o755)
            if self.uid is None:
                self.uid = fresh_uid()
        else:
            self.uid = os.getuid() if hasattr(os, "getuid") else None

    def _reachable(self) -> None:
        """The demoted candidate must traverse to its HOME and outputs: a 0700 ancestor (``mkdtemp``, pytest's root temp)
        would turn every candidate write into a spurious policy failure, so it is refused as infra, never widened here."""
        import stat

        for d in self.work.resolve().parents:
            mode = os.stat(d).st_mode
            if not mode & stat.S_IXOTH:
                raise InfraError(f"candidate uid cannot reach the work dir: {d} is mode {oct(mode & 0o777)} (needs o+x); "
                                 "every candidate write would fail as a spurious policy error")

    # ------------------------------------------------------------------ directories
    def udir(self, path: Path) -> Path:
        """A directory the candidate may write (its outputs land here, then get sealed)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.is_root:
            os.lchown(path, self.uid, self.uid)
            os.chmod(path, 0o755)
        return path

    def private(self, path: Path) -> Path:
        """A root-only directory (references, hidden material): 0700, never readable by the candidate."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.is_root:
            os.lchown(path, 0, 0)
        os.chmod(path, 0o700)
        return path

    def give(self, root: Path) -> None:
        """Hand a tree to the candidate uid (a copy of its sources it may build in)."""
        if self.is_root:
            for p in [Path(root), *Path(root).rglob("*")]:
                os.lchown(p, self.uid, self.uid)

    def seal(self, root: Path) -> None:
        """Kill the cohort, then make ``root`` root-owned and read-only: later candidate code cannot alter it."""
        self.kill()
        root = Path(root)
        if not root.exists():
            return
        for p in [root, *root.rglob("*")]:
            if self.is_root:
                os.lchown(p, 0, 0)
            if not p.is_symlink():
                mode = os.lstat(p).st_mode
                os.chmod(p, (mode & 0o755) | (0o755 if stat.S_ISDIR(mode) else 0o444))

    def owned_copy(self, src: Path, dst: Path, *, ignore=None) -> Path:
        """A root-owned, read-only copy made before any candidate code runs (harness, benchmark, consumers, tests)."""
        dst = Path(dst)
        if Path(src).is_dir():
            shutil.copytree(src, dst, symlinks=True, ignore=ignore)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        for p in [dst, *(dst.rglob("*") if dst.is_dir() else [])]:
            if self.is_root:
                os.lchown(p, 0, 0)
            if not p.is_symlink():
                mode = os.lstat(p).st_mode
                os.chmod(p, (mode & 0o755) | (0o755 if stat.S_ISDIR(mode) else 0o444))
        return dst

    def fresh_home(self) -> Path:
        self._homes += 1
        h = self.work / "homes" / f"h{self._homes:04d}"
        if h.exists():
            shutil.rmtree(h, ignore_errors=True)
        return self.udir(h)

    # ------------------------------------------------------------------ processes
    def kill(self) -> None:
        if self.is_root and self.uid:
            kill_cohort(self.uid)
            _ipc_clean(self.uid)

    def _demote(self):
        uid = self.uid

        def demote():  # pragma: no cover - runs in the child
            os.setsid()
            _no_new_privs()
            try:
                with open("/proc/self/oom_score_adj", "w") as f:
                    f.write("1000")
            except OSError:
                pass
            os.setgroups([])
            os.setgid(uid)
            os.setuid(uid)

        return demote

    def run(self, cmd: list, *, cwd: str | Path = "/", env: dict | None = None, timeout: float = 3600, candidate: bool = False,
            pythonpath: list | None = None, stdin: bytes | None = None, pass_fds: tuple = ()) -> RunResult:
        """Run ``cmd``. ``candidate=True``: unprivileged, fresh HOME/TMPDIR, cohort killed afterwards.

        ``pythonpath`` entries are joined in order: put grader-owned code (consumers, tests, harness) *before* the
        candidate's site directory so the candidate cannot shadow them.
        """
        e = {"PATH": self.path, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
        home = None
        if candidate:
            home = str(self.fresh_home())
            e.update({"HOME": home, "TMPDIR": home, "XDG_CACHE_HOME": f"{home}/.cache", "USER": "candidate"})
        else:
            e["HOME"] = os.environ.get("HOME", "/root")
        if pythonpath:
            e["PYTHONPATH"] = ":".join(str(p) for p in pythonpath)
        e.update({k: str(v) for k, v in (env or {}).items()})
        if candidate and self.is_root:
            self._reachable()
        preexec = self._demote() if candidate and self.is_root else (os.setsid if candidate else None)
        t0 = time.monotonic()
        try:
            p = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd), env=e, preexec_fn=preexec,  # noqa: PLW1509
                                 stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, pass_fds=pass_fds)
        except OSError as ex:
            if candidate:  # a missing candidate binary is the candidate's failure
                return RunResult(127, "", f"cannot execute: {ex}", 0.0, 0.0, candidate=True, trusted=self.is_root, home=home)
            raise InfraError(f"cannot execute {cmd[0]}: {ex}") from ex
        if stdin is not None:
            try:
                p.stdin.write(stdin)
                p.stdin.close()
            except BrokenPipeError:
                pass
        on_exit = None
        if candidate:  # the cohort dies the moment the main process exits: a detached job never gets a window to act
            def on_exit():
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                self.kill()
        out, err, timed_out, reaped = self._drain(p, t0, t0 + timeout, on_exit)
        try:
            if candidate:
                os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        if reaped is None:  # the child itself is still running (timeout) or exited as its pipes closed
            _, status, ru = os.wait4(p.pid, 0)
            reaped = (status, ru, time.monotonic() - t0)
        status, ru, wall = reaped
        p.returncode = 0  # reaped by us: keep Popen from waiting on a pid that is gone
        rc = -9 if timed_out else os.waitstatus_to_exitcode(status)
        if candidate:
            self.kill()
            if home and Path(home).parent == self.work / "homes":  # a fresh home dies with its invocation
                shutil.rmtree(home, ignore_errors=True)
        return RunResult(rc, out, err, round(wall, 6), round(ru.ru_utime + ru.ru_stime, 6), timed_out=timed_out, candidate=candidate,
                         trusted=self.is_root or not candidate, home=home)

    def python(self, *, first: list | None = None, last: list | None = None, script: str | Path | None = None, module: str | None = None,
               args: list | None = None, python: str = "python3", **kw) -> RunResult:
        """Run Python code under the candidate's account with an import order the candidate cannot change.

        ``python -I`` ignores ``PYTHONPATH``, the user site and the script directory; a bootstrap then puts grader-owned
        ``first`` entries (tests, consumers, harness) at the front of ``sys.path`` and appends the candidate's ``last``
        entries **after the standard library**, so a candidate ``json.py`` or ``consumer.py`` shadows nothing.
        """
        boot = ("import sys, runpy; sys.path[:0] = {first!r}; sys.path += {last!r}; sys.argv = {argv!r}; "
                + ("runpy.run_module({mod!r}, run_name='__main__', alter_sys=True)" if module else "runpy.run_path({script!r}, run_name='__main__')"))
        argv = [str(script or module), *[str(a) for a in (args or [])]]
        fs, ls = [str(x) for x in (first or [])], [str(x) for x in (last or [])]
        code = boot.format(first=fs, last=ls, argv=argv, mod=module, script=str(script))
        # child interpreters (a test's ``subprocess.run([sys.executable, ...])``) must see the same import order, or they
        # silently import whatever the image has installed (task03 lab golden: the hidden tests' probe and CLI runs
        # imported the image's mdlite, not the candidate's port). ``-I`` makes the parent ignore PYTHONPATH; children
        # read it and run a grader-owned sitecustomize that applies the order (candidate still after the stdlib).
        env = {**(kw.pop("env", None) or {}), "PYTHONPATH": str(self._child_boot(fs, ls))}
        return self.run([python, "-I", "-c", code], env=env, **kw)

    def _child_boot(self, first: list[str], last: list[str]) -> Path:
        key = hashlib.sha256(json.dumps([first, last]).encode()).hexdigest()[:16]
        d = self.work / "_pyboot" / key
        if not (d / "sitecustomize.py").exists():
            d.mkdir(parents=True, exist_ok=True)
            (d / "sitecustomize.py").write_text(
                "import sys\n"
                f"_boot, _first, _last = {str(d)!r}, {first!r}, {last!r}\n"
                "sys.path[:] = [p for p in sys.path if p != _boot and p not in _first and p not in _last]\n"
                "sys.path[:0] = _first\n"
                "sys.path += _last\n")
            if self.is_root:  # root-owned and read-only for the candidate: it cannot rewrite the order its children get
                for x in (d.parent, d, d / "sitecustomize.py"):
                    os.lchown(x, 0, 0)
                os.chmod(d.parent, 0o755)
                os.chmod(d, 0o755)
                os.chmod(d / "sitecustomize.py", 0o644)
        return d

    @staticmethod
    def _drain(p, t0: float, deadline: float, on_exit=None):
        """Read both pipes until they close. The child's exit is observed with ``wait4(WNOHANG)`` (never ``poll``, which
        would reap it and lose its rusage), and its wall time is taken at that moment: a detached background job that
        keeps the pipes open neither inflates the timing nor holds the grader past a short grace period."""
        bufs = {p.stdout.fileno(): bytearray(), p.stderr.fileno(): bytearray()}
        fds = list(bufs)
        timed_out = False
        reaped = None
        drain_until = None
        while fds:
            now = time.monotonic()
            if reaped is None:
                pid, status, ru = os.wait4(p.pid, os.WNOHANG)
                if pid:
                    reaped = (status, ru, now - t0)
                    if on_exit:
                        on_exit()
                    drain_until = now + 1.0  # background jobs holding our pipes get a grace period, not the timeout
            if drain_until is not None and now >= drain_until:
                break
            if now >= deadline:
                timed_out = True
                break
            ready, _, _ = select.select(fds, [], [], min(deadline - now, 0.02))
            for fd in ready:
                chunk = os.read(fd, 65536)
                if not chunk:
                    fds.remove(fd)
                    continue
                b = bufs[fd]
                if len(b) < OUT_CAP:
                    b += chunk[: OUT_CAP - len(b)]
        if timed_out:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                if reaped is None:
                    p.kill()
        p.stdout.close()
        p.stderr.close()
        o, e = bufs.values()
        return o.decode("utf-8", "replace"), e.decode("utf-8", "replace"), timed_out, reaped
