"""One exclusive, non-blocking OS file lock for the whole of a process's use of a path (M10 lab audit R-4: a run directory has
exactly one writer; ``runtime.journal.RunJournal`` takes it on ``runs/<id>/run.lock``).

PORT: Repo2RLEnv ``src/repo2rlenv/locking.py`` ``lock_file`` (27d04ee820, Apache-2.0; the source file has no licence header:
attributed in ``THIRD_PARTY.md``): Windows locks byte zero of the file with ``msvcrt.locking`` (the region may extend past EOF,
so an empty lock file needs no write), and a failure to take it with EACCES / EAGAIN / EDEADLK is "held by another", which
is surfaced as ``BlockingIOError`` exactly as ``flock`` does. The POSIX branch is ``SerialLock``'s ``flock`` (``world/health.py``).
Changed: file-descriptor based (no ``IO`` handle) and non-blocking only; an explicit unlock before the close (an msvcrt
region lock is dropped by the close, but ``LK_UNLCK`` first leaves nothing to chance); release on garbage collection
(``weakref.finalize``: a ``FileLock`` dropped without ``release`` frees the path, so an exception path in a test or a caller
that loses its journal never leaves the run locked for the life of the process); ``process_start_time`` for the owner file.

Semantics both platforms share, and what callers may rely on: the lock belongs to the open file description (POSIX) or the
handle (Windows), so a second ``FileLock`` on the same path is refused even in the same process; the operating system drops
it when its holder dies, so there is no stale-lock reclaim (a process forked while holding the lock closes its copy of the
descriptor at once, ``_drop_in_child``: ``flock`` is shared by a fork, so a surviving child would otherwise keep the lock
after its holder was killed); the path is never unlinked (unlinking a lock file lets a second
opener lock a new inode while the first still holds the old one). Single host only: ``flock`` and ``LockFileEx`` are not
reliable on network filesystems (NFS, SMB), and a run directory on one is unsupported. Stdlib only: ``crucible.core`` must
import without any framework, and ``fcntl`` / ``msvcrt`` are imported inside the functions that use them, so this module
imports on both platforms.
"""

from __future__ import annotations

import contextlib
import errno
import os
import sys
import weakref
from pathlib import Path


def try_lock(fd: int) -> None:
    """Take an exclusive lock on ``fd`` without waiting. ``BlockingIOError`` when another holder has it; any other failure
    is re-raised as it came."""
    if sys.platform != "win32":
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return

    import msvcrt

    os.lseek(fd, 0, os.SEEK_SET)  # msvcrt.locking locks from the current position
    try:
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    except OSError as e:
        if e.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
            raise BlockingIOError(errno.EAGAIN, "another process holds this lock") from e
        raise


def unlock(fd: int) -> None:
    """Release the lock ``try_lock`` took on ``fd``."""
    if sys.platform != "win32":
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)
        return

    import msvcrt

    os.lseek(fd, 0, os.SEEK_SET)
    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


_HELD: set[int] = set()  # the descriptors of the locks this process holds (M10 lab audit R-4 review: see ``_drop_in_child``)


def _release(fd: int) -> None:
    """Unlock, then close: each is best effort (the close frees the lock whatever the unlock did). Module level and taking
    only ``fd``, so the ``weakref.finalize`` that calls it holds no reference to its ``FileLock``. A descriptor that is no
    longer registered was closed already (a forked child dropped its copy, or the lock was released): nothing is done, so a
    finalizer that runs later never unlocks or closes a descriptor number something else has reused."""
    if fd not in _HELD:
        return
    _HELD.discard(fd)
    with contextlib.suppress(OSError):
        unlock(fd)
    with contextlib.suppress(OSError):
        os.close(fd)


def _drop_in_child() -> None:
    """POSIX: after ``fork`` the child holds a duplicate of every lock descriptor, and ``flock`` belongs to the open file
    description both share: a forked child that outlives the process that took the lock (``verify.answers`` grades a math
    answer in a forked process; a policy can make it run for hours) would keep the run locked after its holder died, with no
    stale-lock reclaim. The child closes its duplicates. ``close`` only, never ``LOCK_UN``: unlocking a shared description
    would free the parent's lock. ``O_CLOEXEC`` does nothing here (it acts on exec, not fork)."""
    for fd in tuple(_HELD):
        with contextlib.suppress(OSError):
            os.close(fd)
    _HELD.clear()


if hasattr(os, "register_at_fork"):  # not on Windows: its handles are not inherited by ``spawn``, the only way a child starts there
    os.register_at_fork(after_in_child=_drop_in_child)


class FileLock:
    """An exclusive lock on ``path`` (created, with its parents, when missing; never deleted). ``acquire`` raises
    ``BlockingIOError`` when it is held; ``release`` is idempotent and so is dropping the object."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fd: int | None = None
        self._fin: weakref.finalize | None = None

    def acquire(self) -> None:
        if self._fd is not None:
            raise RuntimeError(f"{self.path} is already locked by this FileLock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o600)
        try:
            try_lock(fd)
        except BaseException:
            os.close(fd)
            raise
        _HELD.add(fd)
        self._fd = fd
        self._fin = weakref.finalize(self, _release, fd)

    def release(self) -> None:
        if self._fin is not None:
            self._fin()  # runs _release once; a second call is a no-op
            self._fin = None
        self._fd = None

    @property
    def held(self) -> bool:
        return self._fd is not None


def process_start_time() -> float | None:
    """When this process started (epoch seconds), or None where it cannot be told. Evidence for the owner file that names a
    run's holder (a pid alone can be reused): best effort, never raises, and never what decides the lock (the OS does)."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
            k32.GetProcessTimes.restype = wintypes.BOOL
            created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
            if not k32.GetProcessTimes(k32.GetCurrentProcess(), ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)):
                return None
            ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime  # 100 ns since 1601-01-01
            return ticks / 1e7 - 11644473600
        if sys.platform.startswith("linux"):
            stat = Path("/proc/self/stat").read_text()
            started = int(stat[stat.rindex(")") + 2:].split()[19])  # field 22: clock ticks after boot (the fields after "comm)" start at 3)
            btime = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime "))
            return btime + started / os.sysconf("SC_CLK_TCK")
    except Exception:  # noqa: BLE001 -- evidence only
        return None
    return None
