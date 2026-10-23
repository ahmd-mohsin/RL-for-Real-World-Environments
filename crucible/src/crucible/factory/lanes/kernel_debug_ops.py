"""Kernel debugging lane, which ops a CUDA change reaches (swarm-pipeline/10 §3). Deterministic, from the tree at B.

ggml's CUDA backend dispatches each graph op in ``ggml_cuda_compute_forward`` (``case GGML_OP_X: ggml_cuda_op_x(ctx,
dst); break;``, with nested ``GGML_UNARY_OP_*`` / ``GGML_GLU_OP_*`` switches). ``op_map`` reads that switch, builds the
name-level reference graph of every function defined under the backend directory (calls, ``<<<...>>>`` launches, kernels passed
as template arguments or pointers),
and returns, per op, the functions reachable from its entry. A change reaches an op when one of the functions it changed
is reachable from the op's entry (``ops_for``); a change whose functions are not found reaches the ops whose reachable
functions live in its files. The op names are the harness's (``ggml_op_desc``: ``MUL_MAT``, ``FLASH_ATTN_EXT``, unary and
GLU ops by their own names), so they select test cases directly. This only *proposes* the case pool; which cases a fault
breaks is executed on the GPU (S1 executed, ``kernel_debug_plan``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from crucible.factory.langs.cuda import code_mask, find_funcs, kernels_of

DISPATCH = "ggml_cuda_compute_forward"


@dataclass(frozen=True)
class Layout:
    """Where a ggml tree keeps its CUDA backend: the backend directory and the file holding the op dispatch (inside the
    directory in ggml-org's layout, next to it in the 2024 layout forks keep)."""

    cuda_dir: str = "ggml/src/ggml-cuda/"
    dispatch_file: str = "ggml/src/ggml-cuda/ggml-cuda.cu"

    def is_backend(self, path: str) -> bool:
        return path.startswith(self.cuda_dir) or path == self.dispatch_file

    def paths(self) -> list[str]:
        return sorted({self.cuda_dir.rstrip("/"), self.dispatch_file})


GGML_ORG = Layout()
GGML_2024 = Layout(dispatch_file="ggml/src/ggml-cuda.cu")


def layout_of(tree_files: Iterable[str]) -> Layout:
    """``GGML_2024`` when the dispatch file sits next to the backend directory, else ggml-org's layout."""
    files = set(tree_files)
    return GGML_2024 if GGML_2024.dispatch_file in files and GGML_ORG.dispatch_file not in files else GGML_ORG
LABEL = re.compile(r"case\s+GGML_(?:(UNARY|GLU)_)?OP_([A-Z0-9_]+)\s*:")
CALL = re.compile(r"\b([A-Za-z_]\w{3,})\s*(?:<[^;(){}<>]*(?:<[^;(){}<>]*>[^;(){}<>]*)*>)?\s*(?:<<<|\()")
MACRO = re.compile(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)\([^)\n]*\)((?:[^\n]*\\\n)*[^\n]*)", re.MULTILINE)
IDENT = re.compile(r"\b[A-Za-z_]\w{3,}\b")
SKIP_CALLS = {"switch", "return", "sizeof", "while", "static_cast", "reinterpret_cast", "const_cast", "dynamic_cast", "GGML_ASSERT", "GGML_ABORT",
              "CUDA_CHECK", "GGML_UNUSED", "decltype", "static_assert"}


def backend_files(tree: Path, layout: Layout = GGML_ORG) -> list[str]:
    base = Path(tree) / layout.cuda_dir
    out = sorted(p.relative_to(tree).as_posix() for p in base.rglob("*") if p.is_file() and p.suffix in (".cu", ".cuh", ".h", ".cpp")
                 and "template-instances" not in p.parts) if base.is_dir() else []
    if (Path(tree) / layout.dispatch_file).is_file() and layout.dispatch_file not in out:
        out.append(layout.dispatch_file)
    return out


def _read(tree: Path, rel: str) -> str:
    try:
        return (Path(tree) / rel).read_text(errors="replace")
    except OSError:
        return ""


def definitions(tree: Path, files: Sequence[str]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """``(name -> files defining it, name -> names it calls)`` over ``files`` (leaf names; overloads merge)."""
    where: dict[str, set[str]] = {}
    calls: dict[str, set[str]] = {}
    for f in files:
        text = _read(tree, f)
        masked = code_mask(text)
        for name, a, b in find_funcs(text):
            leaf = name.rsplit("::", 1)[-1]
            where.setdefault(leaf, set()).add(f)
            body = masked[a:b]
            # every reference, not only calls: kernels reach their launchers as template arguments and function pointers
            calls.setdefault(leaf, set()).update(c for c in IDENT.findall(body) if c not in SKIP_CALLS and c != leaf)
        # function-like macros are nodes too: dispatch tables call kernels through them (``FATTN_VEC_CASE(...)``)
        for m in MACRO.finditer(text):
            where.setdefault(m.group(1), set()).add(f)
            calls.setdefault(m.group(1), set()).update(c for c in IDENT.findall(m.group(2)) if c not in SKIP_CALLS and c != m.group(1))
    known = set(where)
    return where, {k: v & known for k, v in calls.items()}


def dispatch_entries(text: str) -> dict[str, set[str]]:
    """op name -> functions its ``case`` block calls, from the dispatch function's body."""
    masked = code_mask(text)
    m = re.search(rf"\b{DISPATCH}\s*\([^)]*\)\s*\{{", masked)
    if not m:
        return {}
    depth, end = 0, len(masked)
    for j in range(m.end() - 1, len(masked)):
        if masked[j] == "{":
            depth += 1
        elif masked[j] == "}":
            depth -= 1
            if depth == 0:
                end = j
                break
    body = masked[m.end():end]
    out: dict[str, set[str]] = {}
    pending: list[str] = []
    for ln in body.splitlines():
        labels = LABEL.findall(ln)
        if labels:
            pending += [name for _kind, name in labels]  # the outer UNARY/GLU labels are skipped below
        called = [c for c in CALL.findall(ln) if c.startswith("ggml_")]
        if called and pending:
            for p in pending:
                if p in ("UNARY", "GLU"):
                    continue
                out.setdefault(p, set()).update(c for c in called if c not in SKIP_CALLS)
        if re.search(r"\bbreak\s*;|\breturn\b", ln):
            pending = []
    return out


def reachable(entries: Iterable[str], calls: Mapping[str, set[str]], *, depth: int = 8) -> set[str]:
    seen = set(entries)
    frontier = list(seen)
    for _ in range(depth):
        nxt = []
        for f in frontier:
            for g in calls.get(f, ()):
                if g not in seen:
                    seen.add(g)
                    nxt.append(g)
        frontier = nxt
    return seen


def op_map(tree: Path, layout: Layout = GGML_ORG) -> dict[str, dict[str, set[str]]]:
    """op -> {"functions": reachable names, "files": files defining them} at ``tree``."""
    files = backend_files(tree, layout)
    where, calls = definitions(tree, files)
    entries = dispatch_entries(_read(tree, layout.dispatch_file))
    out = {}
    for op, ent in entries.items():
        fns = reachable({e for e in ent if e in where}, calls)
        out[op] = {"functions": fns, "files": {f for n in fns for f in where.get(n, ())}}
    return out


def fault_kernels(tree: Path, names: Sequence[str], layout: Layout = GGML_ORG) -> list[str]:
    """The backend's ``__global__`` kernels a change touches at ``tree``: a kernel it changed, a kernel that reaches a
    device function it changed, or a kernel a host function it changed reaches (its launcher). The GPU calibration runs a
    milestone's sanitizer checks on cases whose processes launch one of these (CUPTI), not on whichever cases come first:
    a race in the transposed copy path is invisible to racecheck on a contiguous copy."""
    files = backend_files(tree, layout)
    where, calls = definitions(tree, files)
    ks = set(kernels_of([_read(tree, f) for f in files]))
    targets = {n.rsplit("::", 1)[-1] for n in names} & set(where)
    below = reachable(targets, calls)
    return sorted(k for k in ks if k in below or targets & reachable({k}, calls))


def ops_for(omap: Mapping[str, Mapping[str, set[str]]], files: Sequence[str], names: Sequence[str], *, limit: int = 12) -> list[str]:
    """The ops a change reaches: through the functions it changed, else through its files. More than ``limit`` ops (a
    change to shared plumbing every op reaches) is returned as is; the caller treats it as too broad to target."""
    names = set(names)
    by_name = sorted(op for op, v in omap.items() if names & set(v["functions"]))
    if by_name:
        return by_name
    return sorted(op for op, v in omap.items() if set(files) & set(v["files"]))
