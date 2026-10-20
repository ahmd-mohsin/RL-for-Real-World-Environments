"""Kernel debugging lane, S1: real kernel fixes and real kernel speed-ups mined from a CUDA backend's history
(swarm-pipeline/10 §3). Ground truth only (V1): every fault is a commit that fixed (or sped up) real kernels; nothing is
synthesized. Standard library + git.

**What a change must be.** A non-merge commit dated on or after ``since`` (the target student's release cut) that
changes CUDA backend sources (``kernel_debug_ops.Layout``) and nothing else but the repository's op test (``kernel_harness.HARNESS``,
where a fix adds its own failing cases). Refused, with the reason recorded:

- platform-only work the grading GPUs cannot observe (HIP, MUSA, ROCm, Windows/MSVC, pre-Hopper architectures named in
  the subject: Volta, Turing, Pascal, ``sm_5x``-``sm_8x``), warnings, build and CI fixes, reverts;
- changes that touch any other code (a CPU-backend change moves the reference the grader compares with);
- changes whose functions reach no op the harness tests (``kernel_debug_ops``), or reach so many ops (shared plumbing)
  that no milestone could own them.

**Kinds.** ``bug`` (message says fix/race/overflow/...) and ``perf`` (message says faster/optimize/speed up/fast path,
and does not say fix). The fault class (``CLASSES``) is read from the subject, body and diff, and fixes the detectors
the grader runs for the fault: golden comparison always; ``racecheck`` for races and shared-memory hazards,
``synccheck`` for divergent barriers, ``memcheck`` for out-of-bounds access; repeated seeds for nondeterministic faults;
the perf bar for speed-ups.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crucible.factory.lanes import kernel_debug_ops as ko
from crucible.factory.lanes.kernel_harness import HARNESS
from crucible.factory.lanes.kernel_mine import licence_for
from crucible.factory.langs import cuda

SINCE = "2026-05-01"  # Qwen/Qwen3.6-35B-A3B was released 2026-04-15: changes after this cut are unseen by it
CUDA_DIR = "ggml/src/ggml-cuda/"
BUG = re.compile(r"\b(fix(?:es|ed)?|bugs?|races?|hazards?|overflows?|out[- ]of[- ]bounds|oob|corrupt\w*|wrong|incorrect|memory fault|illegal|crash|nan|"
                 r"truncation|divergent|data[- ]races?)\b", re.IGNORECASE)
PERF = re.compile(r"\b(faster|speed[- ]?up|speeds? up|optimi[sz]e\w*|perf(?:ormance)?|fast(?:er)? path|throughput|branchless|fastdiv|prefetch|"
                  r"use all sms)\b", re.IGNORECASE)
ARCH_ONLY = re.compile(r"\b(volta|turing|pascal|maxwell|kepler|p100|v100|t4|sm_?[5-8]\d|sm_?[5-8]x|jetson|windows|msvc)\b", re.IGNORECASE)
VENDOR = re.compile(r"\b(hip|musa|rocm|rdna\w*|cdna\w*|gfx\d+|amd|mtt|vulkan|sycl|metal|opencl|cann)\b", re.IGNORECASE)
MULTI_GPU = ("allreduce.cu", "nccl")  # code that only runs with several GPUs: one grading GPU never reaches it


def platform_only(subject: str) -> bool:
    """Work the grading GPUs (sm_90, sm_120) cannot observe: an older architecture or Windows named in the subject, or
    another vendor's backend named without CUDA (``CUDA/HIP: fix ...`` is kept)."""
    return bool(ARCH_ONLY.search(subject) or (VENDOR.search(subject) and not re.search(r"\bcuda\b", subject, re.IGNORECASE)))


NOT_BEHAVIOUR = re.compile(r"\b(warnings?|typos?|docs?|comments?|ci|build|compil(?:e|er|ation)|link(?:er|ing)?|cmake|revert|chores?|format|lint|"
                           r"guard|version|deprecat\w*|debug env|env vars?|log(?:ging)?|stubs?|clash)\b|-W[\w-]+", re.IGNORECASE)
NEVER = r"(?!x)x"
CLASSES: dict[str, dict[str, Any]] = {  # class -> how it is recognised (message or diff) and what detects it on the GPU
    "race": {"msg": r"\brace|hazard|data[- ]race|WAR\b|RAW\b|reus\w+ smem|shared[- ]memory race", "diff": r"^\+.*__syncthreads\s*\(|^\+.*__syncwarp\s*\(",
             "detectors": ["golden_repeat", "racecheck"], "text": "a data race or missing barrier"},
    "divergent_barrier": {"msg": r"divergent barrier|barrier divergence", "diff": NEVER, "detectors": ["golden_repeat", "synccheck"],
                          "text": "a barrier reached by only part of a block"},
    "bounds": {"msg": r"out[- ]of[- ]bounds|\boob\b|memory fault|illegal (?:memory|address)|overrun", "diff": NEVER, "detectors": ["golden", "memcheck"],
               "text": "an out-of-bounds memory access"},
    "overflow": {"msg": r"overflow|truncation|int64|64[- ]bit|integer", "diff": r"^\+.*\(\s*(?:int64_t|size_t|long long)\s*\)", "detectors": ["golden"],
                 "text": "an integer overflow or truncation in index arithmetic"},
    "grid_limit": {"msg": r"grid[- ]?dim|grid-?y|65535|grid limit|grid overflow|exceed", "diff": r"^\+.*(?:gridDim|65535|MAX_GRID)", "detectors": ["golden"],
                   "text": "a launch grid that exceeds the hardware limit"},
    "indexing": {"msg": r"stride|non-?contiguous|transpos\w*|index\w*|offset|view|permut\w*|padding", "diff": NEVER, "detectors": ["golden"],
                 "text": "wrong indexing or strides"},
    "launch_config": {"msg": r"thread/block|block count|thread count|launch|num_blocks|block size", "diff": NEVER, "detectors": ["golden"],
                      "text": "a wrong launch configuration"},
    "stream_async": {"msg": r"stream|async|in-place|in place|pdl|graph", "diff": r"^\+.*cudaStream|^\+.*cudaMemcpyAsync", "detectors": ["golden_repeat"],
                     "text": "stream or asynchronous-copy misuse"},
    "precision": {"msg": r"round\w*|precision|accumulat\w*|\bnan\b|\binf\b|denormal|fp16|bf16", "diff": NEVER, "detectors": ["golden"],
                  "text": "a numerical precision error"},
}
CLASS_ORDER = ("race", "divergent_barrier", "bounds", "grid_limit", "overflow", "launch_config", "stream_async", "indexing", "precision")


def _git(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, errors="replace", check=False)
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])}: {p.stderr.strip()[:300]}")
    return p.stdout


def show(repo: Path, ref: str, path: str) -> str | None:
    p = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{path}"], capture_output=True, text=True, errors="replace", check=False)
    return p.stdout if p.returncode == 0 else None


@dataclass
class KernelChange:
    repo: str
    sha: str
    parent: str
    date: str  # committer date (when it landed on the branch)
    author_date: str
    subject: str
    body: str
    files: list[str]
    kind: str  # bug | perf
    fault_class: str
    detectors: list[str]
    functions: list[str] = field(default_factory=list)
    kernels: list[str] = field(default_factory=list)
    ops: list[str] = field(default_factory=list)
    adds_cases: bool = False
    licence: str | None = None
    licence_source: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def classify(subject: str, body: str, diff: str, kind: str) -> tuple[str, list[str]]:
    """``(fault class, detectors)`` of a change (``CLASSES``, first match in ``CLASS_ORDER``); a perf change is
    ``perf`` with the perf detectors."""
    if kind == "perf":
        return "perf", ["golden", "perf_bar"]
    text = f"{subject}\n{body}"
    for c in CLASS_ORDER:  # the message names the fault first; the diff decides only when the message does not
        if re.search(CLASSES[c]["msg"], text, re.IGNORECASE):
            return c, list(CLASSES[c]["detectors"])
    for c in CLASS_ORDER:
        if re.search(CLASSES[c]["diff"], diff, re.MULTILINE):
            return c, list(CLASSES[c]["detectors"])
    return "logic", ["golden"]


def kind_of(subject: str, body: str) -> str | None:
    """``bug``, ``perf`` or None (neither, or platform/build-only work) from the commit message."""
    if platform_only(subject) or re.match(r"\s*revert\b", subject, re.IGNORECASE):
        return None
    if BUG.search(subject) and not NOT_BEHAVIOUR.search(re.sub(BUG, "", subject)):
        return "bug"
    if BUG.search(subject):
        return None  # "fix warnings", "fix build": not a behavioural fault
    if PERF.search(subject) and not NOT_BEHAVIOUR.search(subject):
        return "perf"
    return None


def backend_tree(repo: Path, sha: str, cache: Path, layout: ko.Layout = ko.GGML_ORG) -> Path:
    """The CUDA backend of ``sha`` exported under ``cache/<sha>`` (what ``kernel_debug_ops.op_map`` reads)."""
    import time

    from crucible.factory.lanes import kernel_world as kw

    dest = Path(cache) / sha[:12]
    for attempt in range(4):  # a partial clone fetches large blobs on demand: a network blip is retried, then raised
        if (dest / layout.cuda_dir).is_dir():
            break
        try:
            kw.export_tree(repo, sha, dest, paths=layout.paths())
        except RuntimeError:
            if attempt == 3:
                raise
            time.sleep(20 * (attempt + 1))
    return dest


def layout_at(repo: Path, ref: str = "HEAD") -> ko.Layout:
    return ko.layout_of(_git(repo, "ls-tree", "--name-only", ref, "ggml/src/").split())


def mine(repo: Path, name: str, *, since: str = SINCE, until: str = "HEAD", layout: ko.Layout | None = None, cache: Path | None = None,
         max_ops: int = 10) -> tuple[list[KernelChange], list[dict[str, Any]]]:
    """Every change of ``repo`` (module doc) as a ``KernelChange``, and the drops with their reasons. ``cache``: where the
    backend tree of each change is exported for its op map (``kernel_debug_ops.op_map`` at the change itself, so a
    function renamed later still maps); without it, ops stay empty."""
    layout = layout or layout_at(repo, until)
    log = _git(repo, "log", "--no-merges", f"--since={since}", "--format=%H%x1f%P%x1f%cI%x1f%aI%x1f%s%x1e", until, "--", *layout.paths())
    out: list[KernelChange] = []
    drops: list[dict[str, Any]] = []
    for rec in log.split("\x1e"):
        rec = rec.strip()
        if not rec:
            continue
        sha, parents, cdate, adate, subject = (rec.split("\x1f") + [""] * 5)[:5]
        par = parents.split()
        row = {"repo": name, "sha": sha, "date": cdate, "subject": subject[:160]}
        if len(par) != 1:
            drops.append({**row, "reason": "merge"})
            continue
        body = _git(repo, "show", "-s", "--format=%b", sha).strip()[:4000]
        kind = kind_of(subject, body)
        if kind is None:
            reason = ("platform-specific (not observable on sm_90/sm_120)" if platform_only(subject) else
                      "revert" if re.match(r"\s*revert\b", subject, re.IGNORECASE) else
                      "not a behavioural fix or speed-up (warnings, build, docs, features)")
            drops.append({**row, "reason": reason})
            continue
        files = [f for f in _git(repo, "show", "--format=", "--name-only", "--no-renames", sha).splitlines() if f]
        other = [f for f in files if not layout.is_backend(f) and f != HARNESS]
        if other:
            drops.append({**row, "kind": kind, "reason": f"touches code outside the CUDA backend ({', '.join(other[:3])})"})
            continue
        src = [f for f in files if layout.is_backend(f) and cuda.PROFILE.is_source(f)]
        if not src:
            drops.append({**row, "kind": kind, "reason": "no CUDA source changed"})
            continue
        diff = _git(repo, "show", "--format=", "--no-color", "-U0", sha, "--", *src)
        cls, det = classify(subject, body, diff, kind)
        if all(any(m in f for m in MULTI_GPU) for f in src):
            drops.append({**row, "kind": kind, "reason": "multi-GPU code only (one grading GPU never runs it)"})
            continue
        names: set[str] = set()
        reach: set[str] = set()  # functions that existed before: a function the change adds is reached only through its changed callers
        kernels: set[str] = set()
        for f in src:
            before, after = show(repo, par[0], f), show(repo, sha, f)
            changed = set(cuda.changed_names(before, after))
            names |= changed
            reach |= changed & {n.rsplit("::", 1)[-1] for n, _a, _b in cuda.find_funcs(before or "")}
            kernels |= set(cuda.PROFILE.kernels(after or "")) & changed
        omap = ko.op_map(backend_tree(repo, sha, cache, layout), layout) if cache is not None else {}
        ops = ko.ops_for(omap, src, sorted(reach)) if omap else []
        lic, lic_src = licence_for(repo, sha, src)
        ch = KernelChange(repo=name, sha=sha, parent=par[0], date=cdate, author_date=adate, subject=subject, body=body, files=files, kind=kind,
                          fault_class=cls, detectors=det, functions=sorted(names), kernels=sorted(kernels), ops=ops, adds_cases=HARNESS in files,
                          licence=lic, licence_source=lic_src)
        if not lic:
            drops.append({**row, "kind": kind, "reason": "licence not permissive or unknown"})
            continue
        if cache is not None and not ops:
            drops.append({**row, "kind": kind, "reason": "reaches no op the harness tests", "functions": sorted(names)[:8]})
            continue
        if cache is not None and len(ops) > max_ops:
            drops.append({**row, "kind": kind, "reason": f"reaches {len(ops)} ops (shared plumbing: no milestone can own it)", "functions": sorted(names)[:8]})
            continue
        out.append(ch)
    return out[::-1], drops  # oldest first


# ---- groups --------------------------------------------------------------------------------------------------------------------
def group_changes(changes: Sequence[KernelChange], *, kind: str, max_size: int = 4, min_size: int = 2, max_span_days: int = 120,
                  since: str = SINCE) -> list[list[KernelChange]]:
    """Groups of changes of one kind connected by a shared op or a shared source file (where faults can interact: one
    op's test cases observe both). Seed-and-absorb in date order: each unused change seeds a group that absorbs the
    later unused changes connected to it, within ``max_span_days`` of the seed (older faults revert badly at a much newer
    B), up to ``max_size``; groups below ``min_size`` dissolve. Changes after ``since`` (the student's cut) are grouped
    first and only with each other, so a post-release group is never diluted by a fallback change. The connection is a
    *claim*: which cases each fault breaks, and so the executed edges, are measured on the GPU."""
    pool = sorted((c for c in changes if c.kind == kind), key=lambda c: c.date)
    groups: list[list[KernelChange]] = []
    for part in ([c for c in pool if c.date[:10] >= since], [c for c in pool if c.date[:10] < since]):
        used: set[str] = set()
        for i, seed in enumerate(part):
            if seed.sha in used:
                continue
            g = [seed]
            for c in part[i + 1:]:
                if len(g) >= max_size:
                    break
                if c.sha in used or _days(seed.date, c.date) > max_span_days or not _connected(g, c):
                    continue
                g.append(c)
            if len(g) >= min_size:
                groups.append(g)
                used |= {c.sha for c in g}
    return groups


def _connected(group: Sequence[KernelChange], c: KernelChange) -> bool:
    ops = {o for g in group for o in g.ops}
    files = {f for g in group for f in g.files if f != HARNESS}
    return bool(ops & set(c.ops) or files & {f for f in c.files if f != HARNESS})


def _days(a: str, b: str) -> int:
    from datetime import datetime

    return abs((datetime.fromisoformat(b) - datetime.fromisoformat(a)).days)


def interactions(group: Sequence[KernelChange]) -> list[dict[str, Any]]:
    """The claimed interaction edges of a group: pairs sharing an op (the same op's cases observe both) or a file."""
    out = []
    for i, a in enumerate(group):
        for b in group[i + 1:]:
            ops = sorted(set(a.ops) & set(b.ops))
            files = sorted({f for f in a.files if f != HARNESS} & {f for f in b.files if f != HARNESS})
            if ops or files:
                out.append({"a": a.sha, "b": b.sha, "ops": ops, "files": files, "evidence": "claimed"})
    return out


def as_changes(rows: Sequence[Mapping[str, Any]]) -> list[KernelChange]:
    return [KernelChange(**{k: v for k, v in r.items() if k in KernelChange.__dataclass_fields__}) for r in rows]
