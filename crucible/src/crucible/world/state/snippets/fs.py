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
"""

import gzip
import hashlib
import io
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


def _managed(root, git, include, exclude):
    rels = []
    if git and (os.path.isdir(os.path.join(root, ".git")) or _in_git(root)):
        raw = subprocess.run(
            ["git", "-C", root, "-c", "core.quotepath=off", "ls-files", "-co", "--exclude-standard", "-z"],
            capture_output=True,
            check=True,
        ).stdout
        seen = set()
        for r in raw.split(b"\0"):
            if not r:
                continue
            rel = r.decode("utf-8", "surrogateescape")
            if rel in seen:
                continue
            seen.add(rel)
            if os.path.lexists(os.path.join(root, rel)) and not os.path.isdir(os.path.join(root, rel)):
                rels.append(rel)
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


def _in_git(root):
    try:
        r = subprocess.run(["git", "-C", root, "rev-parse", "--is-inside-work-tree"], capture_output=True)
        return r.returncode == 0 and r.stdout.strip() == b"true"
    except FileNotFoundError:
        return False


def _entry(root, rel):
    full = os.path.join(root, rel)
    st = os.lstat(full)
    if stat.S_ISLNK(st.st_mode):
        return (rel, "120000", hashlib.sha256(os.readlink(full).encode("utf-8", "surrogateescape")).hexdigest(), 0)
    h = hashlib.sha256()
    with open(full, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    mode = "100755" if st.st_mode & 0o111 else "100644"
    return (rel, mode, h.hexdigest(), st.st_size)


def _fingerprint(entries):
    outer = hashlib.sha256()
    for rel, mode, h in sorted((e[0], e[1], e[2]) for e in entries):
        outer.update(f"{rel}\0{mode}\0{h}\n".encode("utf-8", "surrogateescape"))
    return outer.hexdigest()


def _head(root, git):
    if not git:
        return None
    r = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True)
    return r.stdout.decode().strip() if r.returncode == 0 else None


def manifest(a):
    root = a["root"]
    if not os.path.isdir(root):
        return {"entries": [], "fingerprint": _fingerprint([]), "count": 0, "bytes": 0, "head": None, "missing_root": True}
    rels = _managed(root, a.get("git", False), a.get("include") or [], a.get("exclude") or [])
    entries = [_entry(root, r) for r in rels]
    return {
        "entries": [[e[0], e[1], e[2]] for e in entries],
        "fingerprint": _fingerprint(entries),
        "count": len(entries),
        "bytes": sum(e[3] for e in entries),
        "head": _head(root, a.get("git", False)),
    }


def pack(a):
    m = manifest(a)
    root, out = a["root"], a["out"]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for rel, mode, _h in m["entries"]:
            full = os.path.join(root, rel)
            ti = tarfile.TarInfo(rel)
            ti.mtime = 0
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            if mode == "120000":
                ti.type = tarfile.SYMTYPE
                ti.linkname = os.readlink(full)
                ti.mode = 0o777
                tf.addfile(ti)
            else:
                ti.mode = 0o755 if mode == "100755" else 0o644
                with open(full, "rb") as f:
                    data = f.read()
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
    raw = buf.getvalue()
    with open(out, "wb") as f:
        with gzip.GzipFile(filename="", fileobj=f, mode="wb", mtime=0, compresslevel=6) as gz:
            gz.write(raw)
    m["tar_bytes"] = os.path.getsize(out)
    return m


def unpack(a):
    root, tar = a["root"], a["tar"]
    os.makedirs(root, exist_ok=True)
    with tarfile.open(tar, "r:gz") as tf:
        members = tf.getmembers()
        want = {m.name for m in members}
        for m in members:
            if m.name.startswith("/") or ".." in m.name.split("/"):
                raise ValueError(f"unsafe tar member {m.name!r}")
        current = _managed(root, a.get("git", False), a.get("include") or [], a.get("exclude") or [])
        for rel in current:
            if rel not in want:
                os.unlink(os.path.join(root, rel))
        for m in members:
            dest = os.path.join(root, m.name)
            os.makedirs(os.path.dirname(dest) or root, exist_ok=True)
            if os.path.lexists(dest) and (os.path.islink(dest) or not os.path.isdir(dest)):
                os.unlink(dest)
            if m.issym():
                os.symlink(m.linkname, dest)
            else:
                src = tf.extractfile(m)
                with open(dest, "wb") as f:
                    f.write(src.read())
                os.chmod(dest, m.mode)
    return manifest(a)


def read(a):
    full = os.path.join(a["root"], a["path"])
    max_bytes = int(a.get("max_bytes", 200000))
    with open(full, "rb") as f:
        data = f.read(max_bytes + 1)
    return {"text": data[:max_bytes].decode("utf-8", "replace"), "size": os.path.getsize(full), "truncated": len(data) > max_bytes}


def list_(a):
    rels = _managed(a["root"], a.get("git", False), a.get("include") or [], a.get("exclude") or [])
    pat = a.get("pattern")
    return {"paths": [r for r in rels if not pat or match(r, pat)]}


def grep(a):
    rx = re.compile(a["pattern"])
    rels = _managed(a["root"], a.get("git", False), a.get("include") or [], a.get("exclude") or [])
    g = a.get("glob")
    hits, max_hits = [], int(a.get("max_hits", 200))
    for rel in rels:
        if g and not match(rel, g):
            continue
        full = os.path.join(a["root"], rel)
        if os.path.islink(full):
            continue
        try:
            with open(full, encoding="utf-8") as f:
                for n, line in enumerate(f, 1):
                    if rx.search(line):
                        hits.append([rel, n, line.rstrip("\n")[:400]])
                        if len(hits) >= max_hits:
                            return {"hits": hits, "truncated": True}
        except (UnicodeDecodeError, OSError):
            continue
    return {"hits": hits, "truncated": False}


def exists(a):
    full = os.path.join(a["root"], a["path"])
    return {"exists": os.path.lexists(full), "is_file": os.path.isfile(full), "is_dir": os.path.isdir(full)}


OPS = {"manifest": manifest, "pack": pack, "unpack": unpack, "read": read, "list": list_, "grep": grep, "exists": exists}


def main(a):
    return OPS[a["op"]](a)
