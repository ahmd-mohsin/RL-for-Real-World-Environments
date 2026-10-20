"""Kernel lane, S1: real GPU-migration commits mined from open-source repositories (swarm-pipeline/06 §3).

A *migration commit* removes a legacy GPU construct from a CUDA source and adds its modern form in the same file. Kinds
(``KINDS``), each a pair of line patterns (legacy, modern) checked on the commit's removed and added lines:

| kind | legacy (removed) | modern (added) |
|---|---|---|
| ``texref`` | ``texture<..> name``, ``cudaBindTexture*``, ``cudaUnbindTexture`` | ``cudaTextureObject_t``, ``cudaCreateTextureObject`` |
| ``surfref`` | ``surface<..> name``, ``cudaBindSurfaceToArray`` | ``cudaSurfaceObject_t``, ``cudaCreateSurfaceObject`` |
| ``warp_sync`` | ``__shfl``/``__shfl_up``/``__shfl_down``/``__shfl_xor``/``__any``/``__all``/``__ballot`` without ``_sync`` | the ``*_sync`` form |
| ``volatile_warp`` | a ``volatile`` shared-memory pointer in a warp reduction | ``__syncwarp`` or a ``*_sync`` shuffle |
| ``thread_api`` | ``cudaThreadSynchronize``/``Exit``/``SetLimit``/``GetLimit``/``SetCacheConfig`` | the ``cudaDevice*`` form |
| ``lib_api`` | removed cuSPARSE legacy calls (``cusparse?csrmv``, ``csrmm``, ``csr2csc``, ``gemmi``) | ``cusparseSpMV``/``SpMM``/``Csr2cscEx2`` |
| ``arch`` | ``sm_2x``/``sm_3x``/``compute_3x`` targets in a build file | a newer target |

Ground truth only (V1): the oracle is the commit itself; nothing is synthesized here. Licence is read from the clone's
licence file and only permissive licences pass (``PERMISSIVE``). Standard library + git.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CUDA_SRC = (".cu", ".cuh")
CXX_SRC = (".h", ".hpp", ".hh", ".hxx", ".inl", ".cpp", ".cc", ".cxx", ".c")
BUILD_FILES = ("CMakeLists.txt", "Makefile", "makefile", "GNUmakefile")
BUILD_SUFFIX = (".cmake", ".mk", ".make")


def _rx(*ps: str) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in ps))


KINDS: dict[str, tuple[re.Pattern[str], re.Pattern[str]]] = {
    "texref": (_rx(r"\btexture\s*<[^;{}()]*>\s*[A-Za-z_]\w*\s*[;,=]", r"\bcuda(?:Bind|Unbind)Texture\w*\s*\(", r"\btextureReference\b",
                   r"\bcudaGetTextureReference\s*\("),
               _rx(r"\bcudaTextureObject_t\b", r"\bcudaCreateTextureObject\s*\(", r"\bcudaResourceDesc\b")),
    "surfref": (_rx(r"\bsurface\s*<[^;{}()]*>\s*[A-Za-z_]\w*\s*[;,=]", r"\bcudaBindSurfaceToArray\s*\(", r"\bsurfaceReference\b"),
                _rx(r"\bcudaSurfaceObject_t\b", r"\bcudaCreateSurfaceObject\s*\(")),
    "warp_sync": (_rx(r"\b__shfl(?:_up|_down|_xor)?\s*\(", r"\b__any\s*\(", r"\b__all\s*\(", r"\b__ballot\s*\("),
                  _rx(r"\b__shfl(?:_up|_down|_xor)?_sync\s*\(", r"\b__any_sync\s*\(", r"\b__all_sync\s*\(", r"\b__ballot_sync\s*\(",
                      r"\bcoalesced_group\b", r"\btiled_partition\b")),
    "volatile_warp": (_rx(r"\bvolatile\s+[\w:<>]+\s*\*\s*\w+\s*=\s*\w*(?:sdata|smem|shared|s_\w*|buf\w*)", r"\bvolatile\s+__shared__\b"),
                      _rx(r"\b__syncwarp\s*\(", r"\b__shfl(?:_up|_down|_xor)?_sync\s*\(", r"\bcg::reduce\b")),
    "thread_api": (_rx(r"\bcudaThread(?:Synchronize|Exit|SetLimit|GetLimit|SetCacheConfig|GetCacheConfig)\s*\("),
                   _rx(r"\bcudaDevice(?:Synchronize|Reset|SetLimit|GetLimit|SetCacheConfig|GetCacheConfig)\s*\(")),
    "lib_api": (_rx(r"\bcusparse[SDCZ](?:csrmv|csrmm2?|csr2csc|gemmi|csrsv_analysis|csrsv_solve|hybmv)\s*\("),
                _rx(r"\bcusparseSpMV\s*\(", r"\bcusparseSpMM\s*\(", r"\bcusparseCsr2cscEx2\s*\(", r"\bcusparseSpSV_\w+\s*\(")),
    "arch": (_rx(r"\b(?:sm|compute)_(?:1\d|2\d|3\d)\b"), _rx(r"\b(?:sm|compute)_(?:[5-9]\d|1\d\d)\b")),
}
SOURCE_KINDS = tuple(k for k in KINDS if k != "arch")
COMPLETE_KINDS = ("texref", "surfref", "warp_sync", "lib_api")  # a complete removal counts (texture -> plain loads)

PERMISSIVE = {
    "MIT": (r"Permission is hereby granted, free of charge",),
    "BSD-3-Clause": (r"Neither the name of", r"Redistribution and use in source and binary forms"),
    "BSD-2-Clause": (r"Redistribution and use in source and binary forms",),
    "Apache-2.0": (r"Apache License,?\s+Version 2\.0", r"www\.apache\.org/licenses/LICENSE-2\.0"),
    "ISC": (r"Permission to use, copy, modify, and/or distribute this software for any purpose",),
    "Zlib": (r"This software is provided 'as-is', without any express or implied",),
    "Unlicense": (r"This is free and unencumbered software released into the public domain",),
    "BSL-1.0": (r"Boost Software License - Version 1\.0",),
}
COPYLEFT = (r"GNU (?:LESSER |LIBRARY |AFFERO )?GENERAL PUBLIC LICENSE", r"Mozilla Public License", r"Eclipse Public License",
            r"non-?commercial", r"Creative Commons")
LICENCE_FILES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "LICENCE", "COPYING", "COPYING.txt", "License.txt", "license.txt", "LICENSE.TXT",
                 "license", "LICENSE.rst")


def licence_of(text: str) -> str | None:
    """The permissive licence ``text`` states, or ``None`` (unknown, copyleft or non-commercial: refused)."""
    head = text[:20000]
    if any(re.search(p, head, re.IGNORECASE) for p in COPYLEFT):
        return None
    if re.search(PERMISSIVE["Apache-2.0"][0], head) or re.search(PERMISSIVE["Apache-2.0"][1], head):
        return "Apache-2.0"
    if re.search(PERMISSIVE["BSD-2-Clause"][0], head):
        return "BSD-3-Clause" if re.search(PERMISSIVE["BSD-3-Clause"][0], head) else "BSD-2-Clause"
    for name, pats in PERMISSIVE.items():
        if all(re.search(p, head) for p in pats):
            return name
    return None


def repo_licence(repo: Path, ref: str = "HEAD") -> tuple[str | None, str | None]:
    """``(licence, file)`` read from the tree at ``ref`` (the licence at B is the one the oracle was published under)."""
    names = set(_git(repo, "ls-tree", "--name-only", ref).split())
    for f in LICENCE_FILES:
        if f in names:
            return licence_of(_git(repo, "show", f"{ref}:{f}")), f
    return None, None


def licence_for(repo: Path, sha: str, files: Sequence[str]) -> tuple[str | None, str]:
    """The licence of a commit's migrated files: the repository's licence file at the commit, else at HEAD (a licence
    added later covers the history its authors published), else the migrated files' own header comments."""
    lic, f = repo_licence(repo, sha)
    if lic:
        return lic, f"{f}@commit"
    lic, f = repo_licence(repo, "HEAD")
    if lic:
        return lic, f"{f}@HEAD"
    heads = []
    for path in files[:6]:
        try:
            heads.append(licence_of(_git(repo, "show", f"{sha}:{path}")[:6000]))
        except RuntimeError:
            heads.append(None)
    if heads and all(h is not None for h in heads) and len(set(heads)) == 1:
        return heads[0], "file_header"
    return None, "unknown"


def _show(repo: Path, ref: str, path: str) -> str | None:
    try:
        return _git(repo, "show", f"{ref}:{path}")
    except RuntimeError:
        return None


def _git(repo: Path, *args: str, check: bool = True) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, errors="replace", check=False)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])}: {p.stderr.strip()[:300]}")
    return p.stdout


def is_source(path: str) -> bool:
    return path.endswith(CUDA_SRC + CXX_SRC)


def is_build(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name in BUILD_FILES or name.endswith(BUILD_SUFFIX)


def strip_c_light(text: str) -> str:
    """Comments out (block and line), strings kept: enough to find ``main`` and constructs in code."""
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL))


def _strip_comment(line: str) -> str:
    return re.sub(r"//.*$", "", line)


@dataclass
class FileChange:
    path: str
    removed: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    status: str = "M"

    def kinds(self, kinds: Mapping[str, tuple[re.Pattern[str], re.Pattern[str]]] | None = None) -> dict[str, dict[str, int]]:
        """kind -> {legacy removed, modern added, legacy still added} for this file's change (``kinds``: another kind
        table with the same shape, e.g. the CUDA 13 kinds of ``kernel_debug_migrate``)."""
        out: dict[str, dict[str, int]] = {}
        src, bld = is_source(self.path), is_build(self.path)
        for k, (old, new) in (kinds or KINDS).items():
            if (k == "arch") != bld or not (src or bld):
                continue
            rem = sum(1 for ln in self.removed if old.search(_strip_comment(ln)))
            add_new = sum(1 for ln in self.added if new.search(_strip_comment(ln)))
            add_old = sum(1 for ln in self.added if old.search(_strip_comment(ln)) and not new.search(_strip_comment(ln)))
            if rem:
                out[k] = {"legacy_removed": rem, "modern_added": add_new, "legacy_added": add_old}
        return out


@dataclass
class MigrationCommit:
    repo: str
    sha: str
    parent: str
    date: str
    subject: str
    kinds: dict[str, dict[str, int]]
    migrated_files: list[str]
    changed_files: list[str]
    lines_added: int
    lines_removed: int
    licence: str | None = None
    licence_source: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def parse_patch(text: str) -> list[FileChange]:
    files: list[FileChange] = []
    cur: FileChange | None = None
    for ln in text.splitlines():
        if ln.startswith("diff --git "):
            m = re.match(r"diff --git a/(.+?) b/(.+)$", ln)
            cur = FileChange(path=m.group(2) if m else ln.split()[-1][2:])
            files.append(cur)
        elif cur is None:
            continue
        elif ln.startswith("new file mode"):
            cur.status = "A"
        elif ln.startswith("deleted file mode"):
            cur.status = "D"
        elif ln.startswith(("+++ ", "--- ")):
            continue
        elif ln.startswith("+"):
            cur.added.append(ln[1:])
        elif ln.startswith("-"):
            cur.removed.append(ln[1:])
    return files


def legacy_count(text: str | None, kind: str, kinds: Mapping[str, tuple[re.Pattern[str], re.Pattern[str]]] | None = None) -> int:
    if not text:
        return 0
    return sum(1 for ln in text.splitlines() if (kinds or KINDS)[kind][0].search(_strip_comment(ln)))


def classify(files: Sequence[FileChange], texts: Callable[[str], tuple[str | None, str | None]] | None = None, *,
             kinds: Mapping[str, tuple[re.Pattern[str], re.Pattern[str]]] | None = None, complete: Sequence[str] = COMPLETE_KINDS
             ) -> tuple[dict[str, dict[str, int]], list[str]]:
    """Kinds of a commit and the files it *migrates*: legacy removed and the modern form added in the same file, the file
    kept (not deleted) and no new legacy use of that kind added. A file that only loses legacy lines (dead code removal) or
    only gains modern ones (new code) is not a migration. With ``texts(path) -> (before, after)``, a *complete* removal
    (the file had the construct before and has none after, e.g. a texture reference replaced by plain loads) also counts
    for the source kinds."""
    found: dict[str, dict[str, int]] = {}
    migrated: list[str] = []
    for f in files:
        if f.status == "D":
            continue
        for k, c in f.kinds(kinds).items():
            real = c["modern_added"] > 0 and c["legacy_added"] == 0
            if not real and k in complete and texts is not None and c["legacy_added"] == 0:
                before, after = texts(f.path)
                real = legacy_count(before, k, kinds) > 0 and legacy_count(after, k, kinds) == 0
            if k == "arch":
                real = c["modern_added"] > 0
            if not real:
                continue
            agg = found.setdefault(k, {"legacy_removed": 0, "modern_added": 0, "files": 0})
            agg["legacy_removed"] += c["legacy_removed"]
            agg["modern_added"] += c["modern_added"]
            agg["files"] += 1
            if f.path not in migrated:
                migrated.append(f.path)
    return found, migrated


# a coarse POSIX-ERE prefilter for ``git log -G`` (no ``\\b``, no non-capturing groups); ``classify`` decides
PICKAXE = r"texture *<|cudaBindTexture|cudaUnbindTexture|textureReference|surface *<|cudaBindSurface|__shfl|__any *[(]|__all *[(]|__ballot *[(]|volatile|cudaThread|cusparse[SDCZ]"


def scan_repo(repo: Path, name: str, *, max_files: int = 60, since: str | None = None, kinds: Mapping[str, Any] | None = None,
              pickaxe: str = PICKAXE, source_kinds: Sequence[str] = SOURCE_KINDS, complete: Sequence[str] = COMPLETE_KINDS) -> list[MigrationCommit]:
    """Every non-merge commit of ``repo`` that migrates at least one CUDA source (``classify``); commits touching more than
    ``max_files`` source files are kept but marked by their size (the window builder splits or drops them)."""
    args = ["log", "--no-merges", "--format=%H", f"-G{pickaxe}"]
    if since:
        args.append(f"--since={since}")
    args += ["--", *[f"*{s}" for s in CUDA_SRC + CXX_SRC]]
    shas = _git(repo, *args).split()
    out: list[MigrationCommit] = []
    for sha in shas:
        meta = _git(repo, "show", "-s", "--format=%P%x00%cI%x00%s", sha).rstrip("\n").split("\x00")
        parents = meta[0].split()
        if len(parents) != 1:
            continue
        patch = _git(repo, "show", "--format=", "--no-color", "--no-ext-diff", "-U0", "--no-renames", sha)
        files = parse_patch(patch)

        def texts(path: str, sha: str = sha, parent: str = parents[0]) -> tuple[str | None, str | None]:
            return _show(repo, parent, path), _show(repo, sha, path)

        found, migrated = classify(files, texts, kinds=kinds, complete=complete)
        if not any(k in found for k in source_kinds):
            continue
        changed = [f.path for f in files]
        if sum(1 for p in changed if is_source(p)) > max_files:
            found["_oversized"] = {"files": len(changed)}
        lic, lic_src = licence_for(repo, sha, migrated)
        out.append(MigrationCommit(repo=name, sha=sha, parent=parents[0], date=meta[1], subject=meta[2][:200], kinds=found, migrated_files=migrated,
                                   changed_files=changed, lines_added=sum(len(f.added) for f in files),
                                   lines_removed=sum(len(f.removed) for f in files), licence=lic, licence_source=lic_src))
    return out


def clone(full_name: str, dest: Path, *, blob_limit: str = "512k") -> Path:
    """A partial clone (blobs over ``blob_limit`` fetched on demand): sources are small, data files are not."""
    if (dest / ".git").exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(["git", "clone", "-q", f"--filter=blob:limit={blob_limit}", f"https://github.com/{full_name}", str(dest)],
                       capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"clone {full_name}: {p.stderr.strip()[:300]}")
    return dest


# ---- includes and units --------------------------------------------------------------------------------------------------
INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)


def includes_of(repo: Path, ref: str, path: str) -> list[str]:
    """Repository files ``path`` includes at ``ref`` (resolved relative to the file, then by basename anywhere)."""
    try:
        text = _git(repo, "show", f"{ref}:{path}")
    except RuntimeError:
        return []
    tree = _tree(repo, ref)
    out = []
    base = path.rsplit("/", 1)[0] if "/" in path else ""
    for inc in INCLUDE.findall(text):
        cand = f"{base}/{inc}" if base else inc
        cand = _norm(cand)
        if cand in tree:
            out.append(cand)
            continue
        hits = [p for p in tree if p == inc or p.endswith("/" + inc)]
        if len(hits) == 1:
            out.append(hits[0])
    return out


_TREES: dict[tuple[str, str], frozenset[str]] = {}


def _tree(repo: Path, ref: str) -> frozenset[str]:
    key = (str(repo), ref)
    if key not in _TREES:
        _TREES[key] = frozenset(_git(repo, "ls-tree", "-r", "--name-only", ref).split("\n")) - {""}
    return _TREES[key]


def _norm(p: str) -> str:
    parts: list[str] = []
    for s in p.split("/"):
        if s in ("", "."):
            continue
        if s == ".." and parts:
            parts.pop()
        else:
            parts.append(s)
    return "/".join(parts)


def dependency_edges(repo: Path, ref: str, files: Sequence[str]) -> dict[str, list[str]]:
    """file -> migrated files it includes (directly or through non-migrated headers, depth 3): the *candidate* edges. They
    are claims until S6 executes them (a dependent unit fails to build or match without its prerequisite)."""
    mig = set(files)
    out: dict[str, list[str]] = {}
    for f in files:
        seen: set[str] = set()
        frontier = [f]
        for _ in range(3):
            nxt = []
            for g in frontier:
                for inc in includes_of(repo, ref, g):
                    if inc not in seen:
                        seen.add(inc)
                        nxt.append(inc)
            frontier = nxt
        deps = sorted((seen & mig) - {f})
        if deps:
            out[f] = deps
    return out


def module_of(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else "."


def rung_estimate(c: MigrationCommit, edges: Mapping[str, Sequence[str]], consumers: Sequence[str] = ()) -> str:
    """The rung a window *could* reach from its scope alone (§2); gates decide the final rung (L4 needs calibrated
    sanitizer and perf gates; L5 is the benchmark seed)."""
    n = len(c.migrated_files)
    if consumers and n >= 1:
        return "L3"
    if n >= 2 and edges:
        return "L2"
    if n >= 2:
        return "L2?"  # several files, no claimed edge yet
    return "L1"


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]
