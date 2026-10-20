"""Kernel lane, the deterministic CPU gates (swarm-pipeline/06 §6), executed with the grader's own compiler commands.

| state | executed here | passes when |
|---|---|---|
| S3 | the start at sm_90 and sm_120 (CUDA 12.8) | every compile milestone's unit fails to compile, or its files hold a graded removed construct |
| S3 | the original in the legacy image (CUDA 11.8, PTX ``compute_60``, falling back to 61/70/75/80/86 for code that needs more) | every program builds |
| S4 | the oracle at sm_90 and sm_120 | every unit and program builds; no graded construct is left in the window's files |
| S6 | the oracle with one compile milestone's files back at the start | each run milestone that requires it fails to build (the edge executes) |
| S8 | revert mutants (the same trees) and fault-class mutants | a revert fails its milestone; a fault mutant still builds (it is left for the GPU to catch) |
| S8 | floors: the start, a claims-only patch, an unrelated file | every compile milestone fails |

GPU-side gates (golden outputs, launches, sanitizer, perf, flake, alternatives, second verifier, hackers) are staged by
``scripts/lh_kernel.py stage-gpu``; their status here is ``pending_gpu``.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from crucible.factory.lanes import kernel as K
from crucible.factory.lanes import kernel_world as kw

LEGACY_ARCHES = ("60", "61", "70", "75", "80", "86")
NEWER_ENUM = re.compile(r'identifier "((?:CUDA_ERROR|cudaError|CUDA_|cudaDevAttr|cudaLimit)\w*)" is undefined')


def legacy_repair(errors: Sequence[str]) -> list[str]:
    """The original's only legacy-toolkit failures are enum names newer than CUDA 11.8 (an error-string table): each gets
    a distinct out-of-range value, so the original builds unchanged otherwise. Any other error: no repair (``[]``)."""
    names = []
    for e in errors:
        m = NEWER_ENUM.search(e)
        if not m and re.search(r"errors? detected in the compilation", e):
            continue
        if not m:
            if " error" in e or e.startswith("error"):
                return []
            continue
        names.append(m.group(1))
    return [f"{n}={990001 + i}" for i, n in enumerate(dict.fromkeys(names))]


def overlay(tree_a: Path, files: Mapping[str, str | None], dest: Path) -> Path:
    """A copy of the start with ``files`` replaced (``None`` = deleted)."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(tree_a, dest, symlinks=True)
    for f, text in files.items():
        p = dest / f
        if text is None:
            p.unlink(missing_ok=True)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            K.write_raw(p, text)
    return dest


def jobs(window: Mapping[str, Any], milestones: Sequence[Mapping[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """Build jobs: ``c:<mid>`` compiles a compile milestone's unit, ``p:<pid>`` builds a program."""
    out: dict[str, dict[str, Any]] = {}
    for m in milestones if milestones is not None else window["milestones"]:
        if m.get("kind") == "compile":
            out[f"c_{m['id']}"] = {**window["programs"][m["program"]], "_tu": m["tu"]}
        else:
            out[f"p_{m['program']}"] = dict(window["programs"][m["program"]])
    return out


def _ok(res: Mapping[str, kw.BuildResult], key: str) -> bool:
    r = res.get(key)
    return bool(r and r.ok)


def compile_gates(window: dict[str, Any], tree_a: Path, *, arches: Sequence[str] = kw.ARCHES, faults: bool = True, base_image: str | None = None,
                  legacy_image: str | None = None, legacy_arches: Sequence[str] = LEGACY_ARCHES) -> dict[str, Any]:
    """Run every CPU gate of one window and record the evidence in ``window['gates']`` (also returned). Updates the
    milestones' ``requires_evidence``, the window's ``graded_kinds``, ``rung``, ``l4_candidate``, ``mutants`` and the
    legacy PTX arch per program. ``base_image`` / ``legacy_image``: the World's toolkit and the original's (default CUDA
    12.8 and 11.8; the CUDA 13 migrations of ``kernel_debug_migrate`` pass 13.x and 12.8)."""
    g: dict[str, Any] = {}
    ms = window["milestones"]
    J = jobs(window)
    with tempfile.TemporaryDirectory(prefix="kg-") as td:
        work = Path(td)
        tree_b = overlay(tree_a, window["oracle_files"], work / "oracle")
        # S3 / S4 static: which kinds the oracle removes completely from the window's files
        files = sorted({f for m in ms for f in m["files"]})
        kinds = [k for k in window.get("kinds") or {} if k in K.kp.LEGACY]
        start_hits = kw.static_scan(tree_a, files, kinds)
        oracle_hits = kw.static_scan(tree_b, files, kinds)
        graded = [k for k in kinds if any(h["kind"] == k for h in start_hits) and not any(h["kind"] == k for h in oracle_hits)]
        window["graded_kinds"] = graded
        g["static"] = {"start_hits": len(start_hits), "oracle_hits": [h for h in oracle_hits if h["kind"] in graded or h["kind"] == "jit_escape"][:10],
                       "graded_kinds": graded, "ungraded_kinds": sorted(set(kinds) - set(graded))}
        # S3: the start, S4: the oracle (each arch)
        g["start"], g["oracle"] = {}, {}
        for arch in arches:
            rs = kw.compile_many(tree_a, J, arch=arch, image=base_image)
            ro = kw.compile_many(tree_b, J, arch=arch, image=base_image)
            g["start"][arch] = {k: v.to_json() for k, v in rs.items()}
            g["oracle"][arch] = {k: v.to_json() for k, v in ro.items()}
        a0 = arches[0]
        per_m = {}
        for m in ms:
            key = f"c_{m['id']}" if m.get("kind") == "compile" else f"p_{m['program']}"
            start_fail = all(not g["start"][a][key]["ok"] for a in arches)
            static_start = any(h["file"] in m["files"] and h["kind"] in graded for h in start_hits)
            oracle_ok = all(g["oracle"][a][key]["ok"] for a in arches) and not any(h["file"] in m["files"] and h["kind"] in graded for h in oracle_hits)
            per_m[m["id"]] = {"start_breaks": start_fail, "start_static": static_start, "needed": start_fail or static_start, "oracle_ok": oracle_ok}
        g["milestones"] = per_m
        g["S3_needed"] = all(v["needed"] for v in per_m.values())
        g["S3_breaks"] = any(v["start_breaks"] for v in per_m.values())
        # why the start needs the migration: it does not build ("breaks"); it builds but holds warp code whose meaning changed
        # under independent thread scheduling ("semantic": __any/__all/__ballot, volatile warp code); or only a removed name
        # is left in dead code ("static_only": not counted as a task)
        semantic = any(re.search(r"__(any|all|ballot)\s*\(", h["match"]) or h["kind"] == "volatile_warp" for h in start_hits)
        g["needs"] = "breaks" if g["S3_breaks"] else ("semantic" if semantic else "static_only")
        g["S4"] = all(v["oracle_ok"] for v in per_m.values())
        # the original (legacy toolkit, PTX for driver JIT)
        progs = {k: v for k, v in J.items() if k.startswith("p_")}
        legacy: dict[str, str | None] = {}
        todo = dict(progs)
        for la in legacy_arches:
            if not todo:
                break
            rl = kw.compile_many(tree_a, todo, legacy=True, legacy_arch=la, image=legacy_image)
            retry = {}
            for k, r in rl.items():
                if r.ok:
                    legacy[k[2:]] = la
                elif (defs := legacy_repair(r.errors)):  # enum names newer than CUDA 11.8 in error-string tables only
                    retry[k] = {**todo[k], "defines": [*(todo[k].get("defines") or []), *defs]}
            if retry:
                rr = kw.compile_many(tree_a, retry, legacy=True, legacy_arch=la, image=legacy_image)
                for k, r in rr.items():
                    if r.ok:
                        legacy[k[2:]] = la
                        window["programs"][k[2:]]["legacy_defines"] = [d for d in retry[k]["defines"] if d not in (todo[k].get("defines") or [])]
                        rl[k] = r
            todo = {k: v for k, v in todo.items() if not rl[k].ok}
            if la == legacy_arches[0]:
                g["legacy_errors_60"] = {k: rl[k].errors[:3] for k in todo}
        for k in todo:
            legacy[k[2:]] = None
        g["legacy"] = legacy
        g["S3_original"] = all(v is not None for v in legacy.values())
        for pid, la in legacy.items():
            if la and pid in window["programs"]:
                window["programs"][pid]["legacy_arch"] = la
        # S6 + S8 reverts: the oracle with one compile milestone's files back at the start
        evidence: dict[str, dict[str, str]] = {m["id"]: {} for m in ms}
        mutants = []
        for c in [m for m in ms if m.get("kind") == "compile"]:
            back = {f: (K.read_raw(tree_a / f) if (tree_a / f).is_file() else None) for f in c["files"]}
            t = overlay(tree_b, back, work / f"revert_{c['id']}")
            rr = kw.compile_many(t, J, arch=a0, image=base_image)
            killed = not _ok(rr, f"c_{c['id']}")
            mutants.append({"id": f"revert-{c['id']}", "kind": "revert", "milestone": c["id"], "files": c["files"], "compile_killed": killed})
            for r in [m for m in ms if m.get("kind") == "run" and c["id"] in (m.get("requires") or [])]:
                fails_alone = not _ok(rr, f"p_{r['program']}")
                passes_with = bool(g["oracle"][a0].get(f"p_{r['program']}", {}).get("ok"))
                evidence[r["id"]][c["id"]] = "executed" if fails_alone and passes_with else "claimed"
        for m in ms:
            m["requires_evidence"] = evidence[m["id"]]
        # S8 fault-class mutants: must still build (the GPU decides)
        fm = []
        if faults:
            owner = {f: m["id"] for m in ms if m.get("kind") == "compile" for f in m["files"]}
            for mu in K.fault_mutants(window["oracle_files"], lambda f: owner.get(f)):
                t = overlay(tree_b, {mu["file"]: mu["content"]}, work / "fault")
                rf = kw.compile_many(t, jobs(window, [m for m in ms if mu["milestone"] == m["id"] or mu["milestone"] in (m.get("requires") or [])]), arch=a0, image=base_image)
                builds = all(r.ok for r in rf.values())
                fm.append({**{k: mu[k] for k in ("id", "kind", "file", "milestone", "note")}, "builds": builds, "gpu": "pending" if builds else "compile_killed",
                           "content_sha256": K.kp_hash(mu["content"])})
                mutants.append({**mu, "builds": builds})
        g["fault_mutants"] = fm
        window["mutants"] = [{k: v for k, v in mu.items() if k != "content"} | ({"content": mu["content"]} if mu.get("kind") == "fault" else {}) for mu in mutants]
        # S8 floors (compile level): the start fails every compile milestone it must
        g["floors"] = {"no_op": {"compile_milestones_failed": sum(1 for m in ms if m.get("kind") == "compile" and not g["start"][a0][f"c_{m['id']}"]["ok"]),
                                 "compile_milestones": sum(1 for m in ms if m.get("kind") == "compile")}}
    window["rung"] = K.rung_of(window)
    window["l4_candidate"] = K.l4_candidate(window)
    g["cpu_pass"] = bool(g["S3_needed"] and g["needs"] != "static_only" and g["S4"] and g["S3_original"] and graded)
    window["gates"] = g
    return g
