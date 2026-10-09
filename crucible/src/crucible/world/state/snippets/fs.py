"""File-tree state canonicalization. STANDALONE: stdlib only; shipped verbatim into containers.

Ops (``main({"op": ..., ...})``):
  manifest  {root, git, include, exclude}            -> {entries, fingerprint, count, bytes, head}
  pack      {root, git, include, exclude, out}       -> manifest + {tar_bytes}; deterministic tar.gz
  unpack    {root, git, include, exclude, tar}       -> manifest after restore (managed files only)
  read      {root, path, max_bytes}                  -> {text, size, truncated}
  list      {root, git, include, exclude, pattern}   -> {paths}
  grep      {root, git, include, exclude, pattern, glob, max_hits} -> {hits: [[path, line, text]]}
  exists    {root, path}                             -> {exists, is_file, is_dir}

Canonical entry per managed file: (relpath, mode, sha256) with mode "100644" | "100755" | "120000"
(git's vocabulary; for symlinks the hash is over the link target). Fingerprint = sha256 over sorted
"rel\\0mode\\0hash\\n" lines (== crucible.core.ids.tree_manifest_hash).
Managed set: with git=True, ``git ls-files -co --exclude-standard`` (tracked + untracked, minus
ignored) that still exist; otherwise every file/symlink under root. ``.git/`` is never managed.

Hostile trees (the policy owns them; this code runs as root): no path is ever followed through a symlink
(every component is opened with O_NOFOLLOW from a directory fd), FIFOs and devices are never opened, files
are hashed while they are packed (no read-twice race), totals are capped, and git never reads a config the
policy can write: with ``git_dir`` set, git runs against a root-owned copy of the repository taken before the
policy started (``--git-dir``/``--work-tree``), and always without system/global config, hooks or fsmonitor.
"""

import errno
import gzip
import hashlib
import os
import re
import stat
import subprocess
import tarfile

_RE_CACHE = {}


def _glob_re(pat):
    if pat in _RE_CACHE:
        return _RE_CACHE[pat]
    i, out = 0, []
    while i < len(pat):
        c = pat[i]
        if pat.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pat.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    rx = re.compile("^" + "".join(out) + "$")
    _RE_CACHE[pat] = rx
    return rx


def match(rel, pat):
    """Glob match with ``**``; a pattern also matches everything under a matching directory."""
    rx = _glob_re(pat.rstrip("/"))
    if rx.match(rel):
        return True
    parts = rel.split("/")
    for k in range(1, len(parts)):
        if rx.match("/".join(parts[:k])):
            return True
    return False


SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_GIT_ENV = {"PATH": SAFE_PATH, "LC_ALL": "C", "HOME": "/nonexistent", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0"}


def _git_bin():
    for d in ("/usr/bin", "/bin", "/usr/local/bin"):
        if os.access(os.path.join(d, "git"), os.X_OK):
            return os.path.join(d, "git")
    return "git"


def git_argv(root, git_dir=None):
    """git with no hooks, no fsmonitor, no external diff/textconv; against a trusted git dir when one is given."""
    argv = [_git_bin(), "-c", "safe.directory=*", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", "-c", "core.quotepath=off"]
    if git_dir:
        return [*argv, f"--git-dir={git_dir}", f"--work-tree={root}", "-C", root]
    return [*argv, "-C", root]


def _git(a, *args, check=True):
    return subprocess.run([*git_argv(a["root"], a.get("git_dir")), *args], capture_output=True, check=check, env=_GIT_ENV)


def _managed(root, git, include, exclude, git_dir=None):
    rels = []
    if git and (git_dir or os.path.isdir(os.path.join(root, ".git")) or _in_git(root)):
        raw = _git({"root": root, "git_dir": git_dir}, "ls-files", "-co", "--exclude-standard", "-z").stdout
        seen = set()
        for r in raw.split(b"\0"):
            if not r:
                continue
            rel = r.decode("utf-8", "surrogateescape")
            if rel in seen:
                continue
            seen.add(rel)
            full = os.path.join(root, rel)
            if os.path.islink(full) or (os.path.lexists(full) and not os.path.isdir(full)):  # a symlink to a dir is an entry
                rels.append(rel)
        rels += _specials(root)  # git never lists FIFOs/sockets/devices: an invisible trap for the grader otherwise
    else:
        for dirpath, dirnames, filenames in os.walk(root):
            reld = os.path.relpath(dirpath, root)
            reld = "" if reld == "." else reld.replace(os.sep, "/")
            keep = []
            for d in sorted(dirnames):
                rd = f"{reld}/{d}" if reld else d
                full = os.path.join(dirpath, d)
                if d == ".git" or any(match(rd, p) for p in exclude):
                    continue
                if os.path.islink(full):
                    rels.append(rd)  # symlinked dir: managed as a symlink, not followed
                    continue
                keep.append(d)
            dirnames[:] = keep
            for f in filenames:
                rels.append(f"{reld}/{f}" if reld else f)
    out = []
    for rel in rels:
        if rel == ".git" or rel.startswith(".git/"):
            continue
        if exclude and any(match(rel, p) for p in exclude):
            continue
        if include and not any(match(rel, p) for p in include):
            continue
        out.append(rel)
    return sorted(set(out))


def _specials(root):
    """Non-regular, non-symlink, non-directory entries anywhere under root (outside .git), found with lstat only."""
    found, stack = [], [""]
    while stack:
        reld = stack.pop()
        try:
            it = os.scandir(os.path.join(root, reld) if reld else root)
        except OSError:
            continue
        with it:
            for e in it:
                rel = f"{reld}/{e.name}" if reld else e.name
                if e.is_symlink():
                    continue
                if e.is_dir(follow_symlinks=False):
                    if rel != ".git":
                        stack.append(rel)
                elif not e.is_file(follow_symlinks=False):
                    found.append(rel)
    return found


def _in_git(root):
    try:
        r = _git({"root": root}, "rev-parse", "--is-inside-work-tree", check=False)
        return r.returncode == 0 and r.stdout.strip() == b"true"
    except FileNotFoundError:
        return False


MAX_FILE_BYTES = 1 << 30  # per file; larger (or sparse-huge) files are recorded, not read
_ANOMALIES = []


def _open_parent(root, rel):
    """dir fd of rel's parent, walking every component with O_NOFOLLOW (Karotte untrusted_paths.walk_to_parent):
    a directory swapped for a symlink mid-walk raises instead of redirecting a root-side read."""
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    parts = rel.split("/")
    try:
        for comp in parts[:-1]:
            nfd = os.open(comp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nfd
    except BaseException:
        os.close(fd)
        raise
    return fd, parts[-1]


def _kind(mode):
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISCHR(mode) or stat.S_ISBLK(mode):
        return "device"
    return "other"


def _entry(root, rel, max_bytes=None):
    max_bytes = MAX_FILE_BYTES if max_bytes is None else max_bytes
    try:
        dfd, name = _open_parent(root, rel)
    except OSError as e:
        _ANOMALIES.append({"path": rel, "kind": "unsafe_path", "detail": e.strerror})
        return (rel, "special:unsafe_path", hashlib.sha256(b"unsafe_path").hexdigest(), 0)
    try:
        st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
        if stat.S_ISLNK(st.st_mode):
            return (rel, "120000", hashlib.sha256(os.readlink(name, dir_fd=dfd).encode("utf-8", "surrogateescape")).hexdigest(), 0)
        if not stat.S_ISREG(st.st_mode):
            k = _kind(st.st_mode)
            _ANOMALIES.append({"path": rel, "kind": k})
            return (rel, "special:" + k, hashlib.sha256(k.encode()).hexdigest(), 0)
        if st.st_size > max_bytes:
            _ANOMALIES.append({"path": rel, "kind": "oversize", "bytes": st.st_size})
            return (rel, "oversize", hashlib.sha256(f"oversize:{st.st_size}".encode()).hexdigest(), 0)
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_NOCTTY, dir_fd=dfd)
        try:
            fst = os.fstat(fd)
            if not stat.S_ISREG(fst.st_mode):  # swapped between stat and open (TOCTOU)
                k = _kind(fst.st_mode)
                _ANOMALIES.append({"path": rel, "kind": k, "detail": "swapped"})
                return (rel, "special:" + k, hashlib.sha256(k.encode()).hexdigest(), 0)
            h = hashlib.sha256()
            read = 0
            while True:
                block = os.read(fd, 1 << 20)
                if not block:
                    break
                read += len(block)
                if read > max_bytes:
                    _ANOMALIES.append({"path": rel, "kind": "oversize", "bytes": read})
                    return (rel, "oversize", hashlib.sha256(b"oversize-growing").hexdigest(), 0)
                h.update(block)
            mode = "100755" if fst.st_mode & 0o111 else "100644"
            return (rel, mode, h.hexdigest(), read)
        finally:
            os.close(fd)
    except OSError as e:
        _ANOMALIES.append({"path": rel, "kind": "unreadable", "detail": e.strerror})
        return (rel, "special:unreadable", hashlib.sha256(b"unreadable").hexdigest(), 0)
    finally:
        os.close(dfd)


def _fingerprint(entries):
    outer = hashlib.sha256()
    for rel, mode, h in sorted((e[0], e[1], e[2]) for e in entries):
        outer.update(f"{rel}\0{mode}\0{h}\n".encode("utf-8", "surrogateescape"))
    return outer.hexdigest()


def _head(a):
    if not a.get("git"):
        return None
    r = _git(a, "rev-parse", "HEAD", check=False)
    return r.stdout.decode().strip() if r.returncode == 0 else None


def _rels(a):
    return _managed(a["root"], a.get("git", False), a.get("include") or [], a.get("exclude") or [], a.get("git_dir"))


def manifest(a):
    root = a["root"]
    if not os.path.isdir(root):
        return {"entries": [], "fingerprint": _fingerprint([]), "count": 0, "bytes": 0, "head": None, "missing_root": True}
    del _ANOMALIES[:]
    entries = [_entry(root, r, a.get("max_file_bytes")) for r in _capped(_rels(a), a)]
    return {
        "entries": [[e[0], e[1], e[2]] for e in entries],
        "fingerprint": _fingerprint(entries),
        "count": len(entries),
        "bytes": sum(e[3] for e in entries),
        "head": _head(a),
        "anomalies": list(_ANOMALIES),
    }


MAX_ENTRIES = 500_000


def _capped(rels, a):
    cap = int(a.get("max_entries", MAX_ENTRIES))
    if len(rels) > cap:
        _ANOMALIES.append({"path": "", "kind": "too_many_files", "count": len(rels), "cap": cap})
        return rels[:cap]
    return rels


class _HashingReader:
    """Feeds tarfile exactly ``size`` bytes from an fd and hashes what was packed (the manifest is what the tar holds)."""

    def __init__(self, fd, size):
        self.fd, self.left, self.h, self.n = fd, size, hashlib.sha256(), 0

    def read(self, n=-1):
        if self.left <= 0:
            return b""
        want = self.left if n is None or n < 0 else min(n, self.left)
        data = os.read(self.fd, want)
        if not data:  # shrank under us: pad so the tar stays well formed; the hash records what was packed
            data = b"\0" * want
        self.left -= len(data)
        self.n += len(data)
        self.h.update(data)
        return data


def pack(a):
    """Manifest + deterministic tar.gz in one pass: each file is opened once (no-follow), streamed and hashed."""
    root, out = a["root"], a["out"]
    if not os.path.isdir(root):
        m = manifest(a)
        with open(out, "wb") as f, gzip.GzipFile(filename="", fileobj=f, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT):
            pass
        m["tar_bytes"] = os.path.getsize(out)
        return m
    del _ANOMALIES[:]
    max_file = MAX_FILE_BYTES if a.get("max_file_bytes") is None else int(a["max_file_bytes"])
    max_total = int(a.get("max_total_bytes", 16 << 30))
    entries, unpackable, total = [], [], 0
    with open(out, "wb") as f, gzip.GzipFile(filename="", fileobj=f, mode="wb", mtime=0, compresslevel=6) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for rel in _capped(_rels(a), a):
            ti = tarfile.TarInfo(rel)
            ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname = 0, 0, 0, "", ""
            try:
                dfd, name = _open_parent(root, rel)
            except OSError as e:
                _ANOMALIES.append({"path": rel, "kind": "unsafe_path", "detail": e.strerror})
                entries.append((rel, "special:unsafe_path", hashlib.sha256(b"unsafe_path").hexdigest(), 0))
                unpackable.append(rel)
                continue
            try:
                st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
                if stat.S_ISLNK(st.st_mode):
                    target = os.readlink(name, dir_fd=dfd)
                    ti.type, ti.linkname, ti.mode = tarfile.SYMTYPE, target, 0o777
                    tf.addfile(ti)
                    entries.append((rel, "120000", hashlib.sha256(target.encode("utf-8", "surrogateescape")).hexdigest(), 0))
                    continue
                if not stat.S_ISREG(st.st_mode):
                    k = _kind(st.st_mode)
                    _ANOMALIES.append({"path": rel, "kind": k})
                    entries.append((rel, "special:" + k, hashlib.sha256(k.encode()).hexdigest(), 0))
                    if k == "fifo":
                        ti.type, ti.mode = tarfile.FIFOTYPE, 0o644
                        tf.addfile(ti)
                    else:
                        unpackable.append(rel)
                    continue
                if st.st_size > max_file or total + st.st_size > max_total:
                    kind = "oversize" if st.st_size > max_file else "total_cap"
                    _ANOMALIES.append({"path": rel, "kind": kind, "bytes": st.st_size})
                    entries.append((rel, "oversize", hashlib.sha256(f"oversize:{st.st_size}".encode()).hexdigest(), 0))
                    unpackable.append(rel)
                    continue
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_NOCTTY, dir_fd=dfd)
                try:
                    fst = os.fstat(fd)
                    if not stat.S_ISREG(fst.st_mode):  # swapped between stat and open
                        k = _kind(fst.st_mode)
                        _ANOMALIES.append({"path": rel, "kind": k, "detail": "swapped"})
                        entries.append((rel, "special:" + k, hashlib.sha256(k.encode()).hexdigest(), 0))
                        unpackable.append(rel)
                        continue
                    mode = "100755" if fst.st_mode & 0o111 else "100644"
                    ti.mode, ti.size = (0o755 if mode == "100755" else 0o644), fst.st_size
                    reader = _HashingReader(fd, fst.st_size)
                    tf.addfile(ti, reader)
                    total += reader.n
                    entries.append((rel, mode, reader.h.hexdigest(), reader.n))
                finally:
                    os.close(fd)
            except OSError as e:
                _ANOMALIES.append({"path": rel, "kind": "unreadable", "detail": e.strerror})
                entries.append((rel, "special:unreadable", hashlib.sha256(b"unreadable").hexdigest(), 0))
                unpackable.append(rel)
            finally:
                os.close(dfd)
    m = {"entries": [[e[0], e[1], e[2]] for e in entries], "fingerprint": _fingerprint(entries), "count": len(entries),
         "bytes": sum(e[3] for e in entries), "head": _head(a), "anomalies": list(_ANOMALIES), "tar_bytes": os.path.getsize(out)}
    if unpackable:
        m["unpackable"] = unpackable
    return m


def _walk_dirs(root_fd, parts, owner):
    """dir fd for root/parts..., creating directories and replacing anything in the way that is not a real
    directory (a symlink planted by the policy is unlinked, never followed)."""
    fd = os.dup(root_fd)
    try:
        for comp in parts:
            try:
                nfd = os.open(comp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            except FileNotFoundError:
                os.mkdir(comp, 0o755, dir_fd=fd)
                if owner is not None:
                    os.chown(comp, owner, owner, dir_fd=fd, follow_symlinks=False)
                nfd = os.open(comp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            except OSError as e:
                if e.errno not in (errno.ELOOP, errno.ENOTDIR):
                    raise
                os.unlink(comp, dir_fd=fd)  # a symlink (ELOOP) or a file (ENOTDIR) where a directory belongs
                os.mkdir(comp, 0o755, dir_fd=fd)
                if owner is not None:
                    os.chown(comp, owner, owner, dir_fd=fd, follow_symlinks=False)
                nfd = os.open(comp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nfd
    except BaseException:
        os.close(fd)
        raise
    return fd


def _rmtree_at(name, pfd):
    """rm -rf of pfd/name through directory fds only (shutil.rmtree(dir_fd=) needs 3.11; images ship older Pythons)."""
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=pfd)
    try:
        for e in os.scandir(fd):
            if e.is_dir(follow_symlinks=False):
                _rmtree_at(e.name, fd)
            else:
                os.unlink(e.name, dir_fd=fd)
    finally:
        os.close(fd)
    os.rmdir(name, dir_fd=pfd)


def _clear(name, pfd):
    try:
        st = os.stat(name, dir_fd=pfd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(st.st_mode):
        _rmtree_at(name, pfd)
    else:
        os.unlink(name, dir_fd=pfd)


def _open_dirs(root_fd, parts):
    """dir fd for root/parts... without creating anything and without following symlinks (OSError otherwise)."""
    fd = os.dup(root_fd)
    try:
        for comp in parts:
            nfd = os.open(comp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nfd
    except BaseException:
        os.close(fd)
        raise
    return fd


def unpack(a):
    """Make the managed set under root exactly the tar's, without ever following a path the policy could have
    redirected. Files are created O_EXCL|O_NOFOLLOW from directory fds; ``owner`` (uid) re-owns what is written."""
    root, tar = a["root"], a["tar"]
    owner = a.get("owner")
    os.makedirs(root, exist_ok=True)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with tarfile.open(tar, "r:gz") as tf:
            members = tf.getmembers()
            want = {m.name for m in members}
            for m in members:
                if m.name.startswith("/") or ".." in m.name.split("/") or not m.name or "\0" in m.name:
                    raise ValueError(f"unsafe tar member {m.name!r}")
            for rel in _rels(a):
                if rel in want:
                    continue
                try:
                    pfd = _open_dirs(root_fd, rel.split("/")[:-1])
                except OSError:
                    continue  # its parent is gone or was not a directory: nothing of it is reachable under root
                try:
                    _clear(rel.split("/")[-1], pfd)
                finally:
                    os.close(pfd)
            for m in members:
                parts = m.name.split("/")
                pfd = _walk_dirs(root_fd, parts[:-1], owner)
                name = parts[-1]
                try:
                    _clear(name, pfd)
                    if m.issym():
                        os.symlink(m.linkname, name, dir_fd=pfd)
                    elif m.isfifo():
                        os.mkfifo(name, 0o644, dir_fd=pfd)
                    else:
                        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, m.mode & 0o777, dir_fd=pfd)
                        try:
                            src = tf.extractfile(m)
                            while chunk := src.read(1 << 20):
                                os.write(fd, chunk)
                            os.fchmod(fd, m.mode & 0o777)
                        finally:
                            os.close(fd)
                    if owner is not None:
                        os.chown(name, owner, owner, dir_fd=pfd, follow_symlinks=False)
                finally:
                    os.close(pfd)
    finally:
        os.close(root_fd)
    return manifest(a)


def _open_regular(root, rel):
    """fd of a regular file at root/rel reached without following any symlink; FIFOs/devices refused."""
    dfd, name = _open_parent(root, rel)
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_NOCTTY, dir_fd=dfd)
    finally:
        os.close(dfd)
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        os.close(fd)
        raise OSError(f"{rel}: not a regular file ({_kind(st.st_mode)})")
    return fd, st


def read(a):
    max_bytes = int(a.get("max_bytes", 200000))
    fd, st = _open_regular(a["root"], a["path"])
    with os.fdopen(fd, "rb") as f:
        data = f.read(max_bytes + 1)
    return {"text": data[:max_bytes].decode("utf-8", "replace"), "size": st.st_size, "truncated": len(data) > max_bytes}


def list_(a):
    rels = _rels(a)
    pat = a.get("pattern")
    return {"paths": [r for r in rels if not pat or match(r, pat)]}


def grep(a):
    rx = re.compile(a["pattern"])
    rels = _rels(a)
    g = a.get("glob")
    hits, max_hits = [], int(a.get("max_hits", 200))
    for rel in rels:
        if g and not match(rel, g):
            continue
        try:
            fd, _st = _open_regular(a["root"], rel)
            with os.fdopen(fd, encoding="utf-8") as f:
                for n, line in enumerate(f, 1):
                    if rx.search(line):
                        hits.append([rel, n, line.rstrip("\n")[:400]])
                        if len(hits) >= max_hits:
                            return {"hits": hits, "truncated": True}
        except (UnicodeDecodeError, OSError):
            continue
    return {"hits": hits, "truncated": False}


def exists(a):
    try:
        dfd, name = _open_parent(a["root"], a["path"])
    except OSError:
        return {"exists": False, "is_file": False, "is_dir": False}
    try:
        st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
    except OSError:
        return {"exists": False, "is_file": False, "is_dir": False}
    finally:
        os.close(dfd)
    return {"exists": True, "is_file": stat.S_ISREG(st.st_mode), "is_dir": stat.S_ISDIR(st.st_mode), "is_symlink": stat.S_ISLNK(st.st_mode)}


OPS = {"manifest": manifest, "pack": pack, "unpack": unpack, "read": read, "list": list_, "grep": grep, "exists": exists}


def main(a):
    return OPS[a["op"]](a)
