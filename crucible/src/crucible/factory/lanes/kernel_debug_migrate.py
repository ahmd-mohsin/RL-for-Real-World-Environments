"""Kernel debugging lane, the migration family (swarm-pipeline/10 §3.3; intent KD3): several real toolkit-migration
commits of one repository reverted together at B, so the start breaks several programs and APIs at once (task04-scale
breadth), regrouped from the kernel-migration family's single-commit windows (swarm-pipeline/06).

Two generations:

- **CUDA 12 -> 13** (``KINDS13``; World: CUDA 13, original: the start built with CUDA 12.8). CUDA 13 and CCCL 3 removed
  the ``cub::Sum/Max/Min/...`` functors (``cccl3_functors``), the device-number forms of ``cudaMemPrefetchAsync`` and
  ``cudaMemAdvise`` (``memloc_api``: a ``cudaMemLocation`` now), and ``cudaDeviceProp`` fields such as ``clockRate``,
  ``memoryClockRate``, ``computeMode`` (``devprop_removed``: ``cudaDeviceGetAttribute`` now);
- **CUDA 11 -> 12** (``kernel_mine.KINDS``; World: CUDA 12.8, original: CUDA 11.8 PTX through driver JIT): texture and
  surface references, mask-less warp intrinsics, ``cudaThread*``.

**Ground truth.** Commits are mined by ``kernel_mine.scan_repo`` with the generation's kind table (removed construct out,
modern form in, same file). A group is 2-4 such commits whose migrated programs differ (breadth) and, where the history
allows, whose kinds differ. A' = B (the newest commit of the group) with every group commit's source changes reversed
(``kernel_debug.synthesize``'s chain on ``KernelChange`` shims); the window is ``kernel.windows_from_commit`` on
(A', B) with its per-component windows merged into one task (``merge_windows``): one compile milestone per migrated
unit, one run milestone per program, the run milestones ``requires`` their units (executed at compile level by
``kernel_gates.compile_gates``: the program fails to build with its unit's files back at the start).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from crucible.factory.lanes import kernel as K
from crucible.factory.lanes import kernel_mine as km


def _rx(*ps: str) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in ps))


# the removed constructs as the kit grader's static scan sees them (``kernel_plan.LEGACY``: one source of truth)
CCCL3_LEGACY = K.kp.LEGACY["cccl3_functors"]
MEMLOC_LEGACY = K.kp.LEGACY["memloc_api"]
DEVPROP_LEGACY = K.kp.LEGACY["devprop_removed"]
KINDS13: dict[str, tuple[re.Pattern[str], re.Pattern[str]]] = {
    "cccl3_functors": (_rx(CCCL3_LEGACY), _rx(r"\bcuda::std::(?:plus|minimum|maximum|equal_to|not_equal_to)\b", r"\bcuda::(?:maximum|minimum)\b",
                                              r"\bstruct\s+\w+\s*\{", r"\boperator\s*\(\s*\)")),
    "memloc_api": (_rx(MEMLOC_LEGACY), _rx(r"\bcudaMemLocation\b", r"\bcudaMemLocationType\w*\b")),
    "devprop_removed": (_rx(DEVPROP_LEGACY), _rx(r"\bcudaDeviceGetAttribute\s*\(", r"\bcudaDevAttr\w+\b")),
}
PICKAXE13 = r"cub::Sum|cub::Max|cub::Min|cub::Equality|cub::Inequality|cudaMemPrefetchAsync|cudaMemAdvise|clockRate|computeMode|deviceOverlap|kernelExecTimeoutEnabled|maxTexture1DLinear"
CUDA13_FROM = "nvidia/cuda:13.0.2-devel-ubuntu24.04"
CUDA13_TAG = "crucible-lh-k:base-13.0.2"
GENERATIONS = {
    "cuda13": {"kinds": KINDS13, "pickaxe": PICKAXE13, "complete": tuple(KINDS13), "world": CUDA13_TAG, "legacy": "crucible-lh-k:base-12.8.1",
               "legacy_arches": ("80", "86", "70", "75", "60"), "toolkit": "CUDA 13", "from": "CUDA 12"},
    "cuda12": {"kinds": km.KINDS, "pickaxe": km.PICKAXE, "complete": km.COMPLETE_KINDS, "world": "crucible-lh-k:base-12.8.1", "legacy": "crucible-k-legacy:11.8.0",
               "legacy_arches": ("60", "61", "70", "75", "80", "86"), "toolkit": "CUDA 12.8", "from": "CUDA 11"},
}


def scan(repo: Path, name: str, generation: str, *, since: str | None = None) -> list[km.MigrationCommit]:
    g = GENERATIONS[generation]
    return km.scan_repo(repo, name, since=since, kinds=g["kinds"], pickaxe=g["pickaxe"], source_kinds=tuple(g["kinds"]) if generation == "cuda13" else km.SOURCE_KINDS,
                        complete=g["complete"])


OTHER_BACKENDS = re.compile(r"(^|/)[^/]*-(hip|sycl|omp|opencl)(/|$)|(^|/)(hip|sycl|opencl)/", re.IGNORECASE)


def program_key(path: str) -> str:
    """The program a migrated file belongs to: its benchmark directory in a deep tree (``src/topk-cuda``), the file
    itself in a flat chapter tree (``texturing/tex1dfetch``)."""
    parts = path.split("/")
    return "/".join(parts[:2]) if len(parts) > 2 else path.rsplit(".", 1)[0]


def cuda_only(c: km.MigrationCommit) -> bool:
    """The commit migrates CUDA code, not another backend's port of the same benchmark (HeCBench's ``-hip`` twins)."""
    return bool(c.migrated_files) and not all(OTHER_BACKENDS.search(f) for f in c.migrated_files)


def group_commits(commits: Sequence[km.MigrationCommit], *, min_size: int = 2, max_size: int = 4, hard: Sequence[str] = ()) -> list[list[km.MigrationCommit]]:
    """Groups of commits migrating different files (they revert independently) of different programs, a new kind
    first when one is available (breadth: programs break for different reasons), seed-and-absorb in date order."""
    pool = sorted((c for c in commits if c.licence and "_oversized" not in c.kinds and cuda_only(c) and (not hard or set(c.kinds) & set(hard))),
                  key=lambda c: c.date)
    groups: list[list[km.MigrationCommit]] = []
    used: set[str] = set()
    for i, seed in enumerate(pool):
        if seed.sha in used:
            continue
        g = [seed]
        files = set(seed.changed_files)
        kinds = set(seed.kinds)
        rest = sorted((c for c in pool[i + 1:] if c.sha not in used), key=lambda c: (not (set(c.kinds) - kinds), c.date))
        for c in rest:
            if len(g) >= max_size:
                break
            if set(c.changed_files) & files:
                continue
            g.append(c)
            files |= set(c.changed_files)
            kinds |= set(c.kinds)
        if len(g) >= min_size:
            groups.append(sorted(g, key=lambda c: c.date))
            used |= {c.sha for c in g}
    return groups


def pick_components(ws: Sequence[Mapping[str, Any]], commits: Sequence[km.MigrationCommit], *, cap: int = 6) -> list[Mapping[str, Any]]:
    """At most ``cap`` component windows, round-robin over the group's commits (every commit keeps a program) and then
    over kinds, smallest programs first within a commit (cheaper GPU runs)."""
    owner = {f: c.sha for c in commits for f in c.migrated_files}
    by_commit: dict[str, list[Mapping[str, Any]]] = {}
    for w in ws:
        sha = next((owner[f] for f in w["migrated_files"] if f in owner), "")
        by_commit.setdefault(sha, []).append(w)
    for v in by_commit.values():
        v.sort(key=lambda w: (len(w["programs"]), len(w["milestones"])))
    out: list[Mapping[str, Any]] = []
    while len(out) < cap and any(by_commit.values()):
        for sha in [c.sha for c in commits] + [""]:
            if by_commit.get(sha) and len(out) < cap:
                out.append(by_commit[sha].pop(0))
    return out


def merge_windows(ws: Sequence[Mapping[str, Any]], wid: str) -> dict[str, Any]:
    """The component windows of one (A', B) pair as one window: programs and oracle files united, milestones renumbered
    (``c01..`` units, ``r01..`` programs) with their ``requires`` remapped."""
    progs: dict[str, Any] = {}
    oracle: dict[str, Any] = {}
    kinds: dict[str, Any] = {}
    comp, runs, files = [], [], []
    for w in ws:
        progs.update(w["programs"])
        oracle.update(w["oracle_files"])
        for k, v in (w.get("kinds") or {}).items():
            kinds.setdefault(k, v)
        files += w["migrated_files"]
        ren = {}
        for m in w["milestones"]:
            if m["kind"] == "compile":
                new = f"c{len(comp) + 1:02d}"
                ren[m["id"]] = new
                comp.append({**m, "id": new})
        for m in w["milestones"]:
            if m["kind"] == "run":
                runs.append({**m, "id": f"r{len(runs) + 1:02d}", "requires": [ren[r] for r in m.get("requires") or [] if r in ren]})
    base = dict(ws[0])
    base.update(id=wid, programs=progs, oracle_files=oracle, kinds=kinds, migrated_files=sorted(set(files)), milestones=comp + runs)
    return base


def migration_requirements(window: Mapping[str, Any], generation: str) -> list[dict[str, Any]]:
    """``kernel.requirement_texts`` with the generation's toolkit named (CUDA 13 instead of 12.8 for ``cuda13``)."""
    reqs = K.requirement_texts(window)
    if generation == "cuda13":
        for r in reqs:
            r["text"] = r["text"].replace("CUDA 12.8", "CUDA 13")
    return reqs


def lane_sentences(generation: str) -> tuple[str, ...]:
    if generation == "cuda13":
        return (f"Only CUDA 13 is installed; build for this machine's GPU, `sm_{K.ARCH_TOKEN}`.", *K.LANE_SENTENCES[1:])
    return K.LANE_SENTENCES


def compose(opening: str, window: Mapping[str, Any], generation: str) -> tuple[str, dict[str, list[int]]]:
    """``kernel.compose_prompt`` with the generation's toolkit (the requirement list and lane sentences are fixed text)."""
    reqs = migration_requirements(window, generation)
    text = opening.strip() + "\n\nWhat needs to work:\n"
    spans: dict[str, list[int]] = {}
    for r in reqs:
        line = f"- {r['id']}: "
        spans[r["id"]] = [len(text) + len(line), len(text) + len(line) + len(r["text"])]
        text += line + r["text"] + "\n"
    text += "\n" + " ".join(lane_sentences(generation)) + "\n\n" + K.EPILOGUE + "\n"
    return text, spans


def save_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
