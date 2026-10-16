"""Custody of what the policy hands to the verifier (M1; ports Karotte ``save_submission`` / ``untrusted_paths``, MIT,
and Harbor ``trial/artifact_handler`` manifest, Apache-2.0).

Three operations, all over data the policy cannot change any more:

- ``check_snapshot(spec, start, final, store)``: the custody rules of a ``VerifierSpec`` applied to a quiesced
  snapshot. Artifacts may hold only regular files and directories (symlinks only with ``allow_symlinks``) within the
  size, count and depth limits; protected paths must equal the episode start's. Read from the snapshot's tar and
  manifest in the store, never from the policy's box.
- ``collect_artifacts(spec, final, store, dest)``: the artifacts extracted from the snapshot tar into a host directory
  (root-only modes, ``O_EXCL``/``O_NOFOLLOW``, no path leaves ``dest``) plus a Harbor-style ``manifest.json``.
- ``copy_untrusted(source, dest)``: Karotte's no-follow copy of a host path the policy controlled (local backends).

A violation raises or returns ``CustodyViolation`` problems; the harness turns it into ``crucible:policy_misbehavior``
(scored 0, never masked). A missing artifact is not misbehaviour: whether nothing handed in fails is the grader's call.
"""

from __future__ import annotations

import errno
import json
import os
import posixpath
import shutil
import stat
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from crucible.world.state.snippets.fs import match

if TYPE_CHECKING:
    from crucible.core.schema.state import StateRef, WorldRef
    from crucible.core.schema.world import VerifierSpec
    from crucible.core.store import Store

FILE_FLAGS: Final = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_NOCTTY
DIR_FLAGS: Final = os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY
_CHUNK = 1 << 20
# Destination failures the policy can provoke (a full disk, a name collision, an over-long path): its misbehaviour.
_DEST_ERRNOS: Final = frozenset({errno.ENAMETOOLONG, errno.EEXIST, errno.ENOSPC, errno.EDQUOT, errno.EFBIG})


class CustodyViolation(Exception):
    """The policy handed in something the verifier must not touch (Karotte ``StudentMisbehaviorError``)."""

    def __init__(self, problems: list[str] | str):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("; ".join(self.problems[:5]) + (f" (+{len(self.problems) - 5} more)" if len(self.problems) > 5 else ""))


@dataclass
class Limits:
    allow_symlinks: bool = False
    max_file_bytes: int = 256 * 1024**2
    max_total_bytes: int = 1024**3
    max_entries: int = 10_000
    max_depth: int = 32
    total_bytes: int = 0
    entries: int = 0

    @classmethod
    def of(cls, spec: VerifierSpec) -> Limits:
        return cls(spec.allow_symlinks, spec.max_file_bytes, spec.max_total_bytes, spec.max_entries, spec.max_depth)

    def count_entry(self, rel: str) -> None:
        self.entries += 1
        if self.entries > self.max_entries:
            raise CustodyViolation(f"artifacts hold more than {self.max_entries} entries (at {rel})")

    def depth(self, rel: str) -> None:
        if rel.count("/") + 1 > self.max_depth:
            raise CustodyViolation(f"{rel} is nested deeper than {self.max_depth} levels")

    def add_bytes(self, rel: str, n: int, file_bytes: int) -> None:
        self.total_bytes += n
        if file_bytes > self.max_file_bytes:
            raise CustodyViolation(f"{rel} is too large (> {self.max_file_bytes} bytes)")
        if self.total_bytes > self.max_total_bytes:
            raise CustodyViolation(f"artifacts exceed {self.max_total_bytes} bytes in all")


def split_patterns(patterns: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in patterns:
        app, _, glob = p.partition(":")
        out.setdefault(app, []).append(glob)
    return out


def _matches(rel: str, globs: list[str]) -> bool:
    return any(match(rel, g) for g in globs)


def _manifest(ref: StateRef, store: Store) -> dict[str, tuple[str, str]]:
    key = ref.meta.get("manifest")
    if not key:
        raise CustodyViolation(f"{ref.app}: the snapshot carries no manifest, so custody cannot be checked")
    return {rel: (mode, h) for rel, mode, h in store.get_json(key)}


def _members(ref: StateRef, store: Store) -> Iterator[tuple[tarfile.TarFile, tarfile.TarInfo]]:
    if ref.blob is None:
        raise CustodyViolation(f"{ref.app}: the snapshot holds no payload")
    with store.open_blob(ref.blob) as f, tarfile.open(fileobj=f, mode="r|gz") as tf:  # checked against its key (R-22)
        for ti in tf:
            yield tf, ti


def _safe_rel(name: str) -> str | None:
    rel = posixpath.normpath(name)
    if name.startswith("/") or rel == ".." or rel.startswith("../") or "\0" in name:
        return None
    return rel


# ----------------------------------------------------------------------------------------------- snapshot custody
def check_snapshot(spec: VerifierSpec, start: WorldRef | None, final: WorldRef, store: Store) -> list[str]:
    """Every custody problem of ``final`` (empty: the hand-off is clean). ``start`` is the episode-start WorldRef."""
    problems: list[str] = []
    for app, globs in split_patterns(spec.artifacts).items():
        ref = final.apps.get(app)
        if ref is None:
            problems.append(f"{app}: no such app in the snapshot")
            continue
        problems += _check_artifacts(app, globs, ref, store, Limits.of(spec))
    for app, globs in split_patterns(spec.protected).items():
        if start is None or app not in start.apps or app not in final.apps:
            problems.append(f"{app}: protected paths need the episode-start snapshot")
            continue
        try:
            a, b = _manifest(start.apps[app], store), _manifest(final.apps[app], store)
        except CustodyViolation as e:
            problems += e.problems
            continue
        for rel in sorted(set(a) | set(b)):
            if not _matches(rel, globs) or a.get(rel) == b.get(rel):
                continue
            what = "added" if rel not in a else "removed" if rel not in b else "changed"
            problems.append(f"{app}:{rel}: protected path {what}")
    return problems


def _check_artifacts(app: str, globs: list[str], ref: StateRef, store: Store, lim: Limits) -> list[str]:
    problems: list[str] = []
    anomalies = [an for an in ref.meta.get("anomalies", []) if isinstance(an, dict)]
    kinds = {an.get("path"): an.get("kind") for an in anomalies}
    if any(an.get("kind") == "too_many_files" for an in anomalies):
        problems.append(f"{app}: the tree exceeds the snapshot's entry cap")
    for p in ref.meta.get("unpackable", []):  # in the manifest but not in the tar: sockets, devices, oversize, unreadable
        if _matches(p, globs):
            problems.append(f"{app}:{p}: {kinds.get(p) or 'unpackable'}")
    try:
        for _, ti in _members(ref, store):
            rel = _safe_rel(ti.name)
            if rel is None:
                problems.append(f"{app}:{ti.name}: unsafe path in the snapshot")
                continue
            if not _matches(rel, globs):
                continue
            lim.count_entry(rel)
            lim.depth(rel)
            if ti.issym() or ti.islnk():
                if not lim.allow_symlinks or ti.islnk():
                    problems.append(f"{app}:{rel}: is a {'hard link' if ti.islnk() else 'symlink'} (to {ti.linkname!r})")
                elif _safe_rel(posixpath.join(posixpath.dirname(rel), ti.linkname)) is None or ti.linkname.startswith("/"):
                    problems.append(f"{app}:{rel}: symlink leaves the tree (to {ti.linkname!r})")
            elif ti.isfile():
                lim.add_bytes(rel, ti.size, ti.size)
            elif not ti.isdir():
                problems.append(f"{app}:{rel}: is not a regular file ({_tar_kind(ti)})")
    except CustodyViolation as e:
        problems += [f"{app}: {p}" for p in e.problems]
    except (tarfile.TarError, OSError, EOFError) as e:
        problems.append(f"{app}: the snapshot payload is unreadable ({e})")
    return problems


def _tar_kind(ti: tarfile.TarInfo) -> str:
    return {tarfile.FIFOTYPE: "fifo", tarfile.CHRTYPE: "char device", tarfile.BLKTYPE: "block device"}.get(ti.type, f"type {ti.type!r}")


# ----------------------------------------------------------------------------------------------- artifact hand-off
def collect_artifacts(spec: VerifierSpec, final: WorldRef, store: Store, dest: Path) -> dict[str, Any]:
    """Extract the artifacts of ``final`` into ``dest/<app>/...`` and write ``dest/manifest.json`` (Harbor's record of
    every collection attempt). Raises ``CustodyViolation`` on anything ``check_snapshot`` would refuse."""
    dest = Path(dest)
    if os.path.lexists(dest):
        raise FileExistsError(f"{dest} already exists")
    os.makedirs(dest, mode=0o700)
    lim = Limits.of(spec)
    entries: list[dict[str, Any]] = []
    try:
        for app, globs in split_patterns(spec.artifacts).items():
            ref = final.apps[app]
            os.mkdir(dest / app, 0o700)
            got = 0
            for tf, ti in _members(ref, store):
                rel = _safe_rel(ti.name)
                if rel is None or not _matches(rel, globs):
                    continue
                lim.count_entry(rel)
                lim.depth(rel)
                target = dest / app / rel
                if ti.isdir():
                    _mkdirs(dest / app, rel)
                elif ti.isfile():
                    _mkdirs(dest / app, posixpath.dirname(rel))
                    src = tf.extractfile(ti)
                    assert src is not None
                    _write_new(src, target, lim, f"{app}:{rel}")
                    got += 1
                elif ti.issym() and lim.allow_symlinks and not ti.linkname.startswith("/") and \
                        _safe_rel(posixpath.join(posixpath.dirname(rel), ti.linkname)) is not None:
                    _mkdirs(dest / app, posixpath.dirname(rel))
                    with _dest_failure_is_misbehavior(f"{app}:{rel}"):
                        os.symlink(ti.linkname, target)
                else:
                    raise CustodyViolation(f"{app}:{rel}: is not a regular file ({'symlink' if ti.issym() else _tar_kind(ti)})")
            entries.append({"app": app, "patterns": globs, "files": got, "status": "ok" if got else "missing", "snapshot": ref.fingerprint})
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    manifest = {"world_ref": final.fingerprint, "entries": entries, "bytes": lim.total_bytes}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _mkdirs(base: Path, rel: str) -> None:
    """Create ``base/rel`` one component at a time, refusing a symlink at any component (openat walk)."""
    if rel in ("", "."):
        return
    fd = os.open(base, DIR_FLAGS)
    try:
        for name in rel.split("/"):
            try:
                os.mkdir(name, 0o700, dir_fd=fd)
            except FileExistsError:
                pass
            nfd = os.open(name, DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = nfd
    finally:
        os.close(fd)


def _write_new(src: Any, target: Path, lim: Limits, rel: str) -> None:
    n = 0
    with _dest_failure_is_misbehavior(rel):
        out = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with _dest_failure_is_misbehavior(rel):
            while chunk := src.read(_CHUNK):
                n += len(chunk)
                lim.add_bytes(rel, len(chunk), n)
                view = memoryview(chunk)
                while view:
                    view = view[os.write(out, view):]
    finally:
        os.close(out)


@contextmanager
def _dest_failure_is_misbehavior(rel: str) -> Iterator[None]:
    try:
        yield
    except OSError as e:
        if e.errno in _DEST_ERRNOS:
            raise CustodyViolation(f"cannot save {rel}: {e.strerror}") from e
        raise


# ----------------------------------------------------------------------------------------------- host-path copy (Karotte)
@dataclass
class _Copy:
    lim: Limits
    copied: list[str] = field(default_factory=list)


def copy_untrusted(source: Path | str, dest: Path | str, *, limits: Limits | None = None) -> Path | None:
    """Copy a policy-controlled host file or directory to ``dest`` (which must not exist) without trusting it: no
    symlink is followed anywhere on the path, special files and over-limit trees raise ``CustodyViolation``,
    everything is created root-only (0700 dirs, 0600 files). Quiesce the policy first. Returns None when nothing
    was handed in (not misbehaviour)."""
    source = Path(os.path.abspath(source))
    dest = Path(dest)
    if os.path.lexists(dest):
        raise FileExistsError(f"{dest} already exists")
    st = _Copy(limits or Limits())
    try:
        return _copy_source(source, dest, st)
    except BaseException:
        if os.path.lexists(dest):
            if dest.is_dir() and not dest.is_symlink():
                shutil.rmtree(dest, ignore_errors=True)
            else:
                dest.unlink(missing_ok=True)
        raise


def _walk_to_parent(path: Path) -> tuple[int, str]:
    names = list(path.parts[1:])
    if not names:
        raise ValueError("path must not be the filesystem root")
    final = names.pop()
    fd = os.open("/", DIR_FLAGS)
    walked = "/"
    for name in names:
        walked = os.path.join(walked, name)
        try:
            nfd = os.open(name, DIR_FLAGS, dir_fd=fd)
        except OSError as e:
            err = _cannot_open(e, walked, fd, name)
            os.close(fd)
            raise err from e
        os.close(fd)
        fd = nfd
    return fd, final


def _copy_source(source: Path, dest: Path, st: _Copy) -> Path | None:
    parent_fd, final = _walk_to_parent(source)
    try:
        try:
            s = os.lstat(final, dir_fd=parent_fd)
        except FileNotFoundError:
            return None
        except OSError as e:
            raise CustodyViolation(f"cannot stat {source}: {e.strerror}") from e
        if stat.S_ISLNK(s.st_mode):
            raise CustodyViolation(f"{source} is a symlink")
        if stat.S_ISDIR(s.st_mode):
            fd = _open_at(parent_fd, final, DIR_FLAGS, str(source))
            try:
                with _dest_failure_is_misbehavior(str(dest)):
                    os.mkdir(dest, 0o700)
                _copy_dir(fd, dest, 1, st, "")
            finally:
                os.close(fd)
        elif stat.S_ISREG(s.st_mode):
            fd = _open_at(parent_fd, final, FILE_FLAGS, str(source))
            try:
                _copy_file(fd, dest, st, str(source))
            finally:
                os.close(fd)
        else:
            raise CustodyViolation(f"{source} is not a regular file")
        return dest
    finally:
        os.close(parent_fd)


def _copy_dir(dir_fd: int, dest_dir: Path, depth: int, st: _Copy, rel: str) -> None:
    if depth > st.lim.max_depth:
        raise CustodyViolation(f"submission is nested too deep at {rel}")
    names = []
    with os.scandir(dir_fd) as it:
        for entry in it:
            st.lim.count_entry(entry.name)
            names.append(entry.name)
    for name in sorted(names):
        child = f"{rel}/{name}" if rel else name
        out = dest_dir / name
        try:
            s = os.lstat(name, dir_fd=dir_fd)
        except OSError as e:
            raise CustodyViolation(f"cannot stat {child}: {e.strerror}") from e
        if stat.S_ISLNK(s.st_mode):
            if not st.lim.allow_symlinks:
                raise CustodyViolation(f"{child} is a symlink")
            with _dest_failure_is_misbehavior(child):
                os.symlink(os.readlink(name, dir_fd=dir_fd), out)
        elif stat.S_ISDIR(s.st_mode):
            fd = _open_at(dir_fd, name, DIR_FLAGS, child)
            try:
                with _dest_failure_is_misbehavior(child):
                    os.mkdir(out, 0o700)
                _copy_dir(fd, out, depth + 1, st, child)
            finally:
                os.close(fd)
        elif stat.S_ISREG(s.st_mode):
            fd = _open_at(dir_fd, name, FILE_FLAGS, child)
            try:
                _copy_file(fd, out, st, child)
            finally:
                os.close(fd)
        else:
            raise CustodyViolation(f"{child} is not a regular file")


def _copy_file(fd: int, dest: Path, st: _Copy, rel: str) -> None:
    s = os.fstat(fd)
    if not stat.S_ISREG(s.st_mode):  # swapped between lstat and open
        raise CustodyViolation(f"{rel} is not a regular file")
    if s.st_size > st.lim.max_file_bytes:
        raise CustodyViolation(f"{rel} is too large ({s.st_size} > {st.lim.max_file_bytes} bytes)")
    with os.fdopen(os.dup(fd), "rb") as src:
        _write_new(src, dest, st.lim, rel)
    st.copied.append(rel)


def _open_at(dir_fd: int, name: str, flags: int, described: str) -> int:
    try:
        return os.open(name, flags, dir_fd=dir_fd)
    except OSError as e:
        raise _cannot_open(e, described, dir_fd, name) from e


def _cannot_open(e: OSError, described: str, dir_fd: int, name: str) -> CustodyViolation:
    try:  # O_NOFOLLOW|O_DIRECTORY on a symlink is ELOOP on Linux and ENOTDIR on macOS: name the real cause
        is_link = stat.S_ISLNK(os.lstat(name, dir_fd=dir_fd).st_mode)
    except OSError:
        is_link = False
    if e.errno == errno.ELOOP or is_link:
        return CustodyViolation(f"{described} is a symlink")
    if e.errno == errno.ENOENT:
        return CustodyViolation(f"{described} does not exist")
    if e.errno == errno.ENOTDIR:
        return CustodyViolation(f"{described} is not a directory")
    return CustodyViolation(f"cannot open {described}: {e.strerror}")
