#!/usr/bin/env python3
"""The kernel debugging lane's funnel on CPU (swarm-pipeline/10): mine, build, CPU gates, author, stage the GPU gates, report.

    python scripts/lh_kernel_debug.py mine                 # S1: real kernel fixes and speed-ups since the cut, per clone
    python scripts/lh_kernel_debug.py build                # groups -> A'..B' records (train: ggml-org/llama.cpp, held-out: ik_llama.cpp)
    python scripts/lh_kernel_debug.py gates [--ids ...]    # CPU gates in Docker (compile units, harness on the CPU, World, floors, mutants)
    python scripts/lh_kernel_debug.py author               # the builder's symptom reports (prompts/<id>.json): lint + ledger + intent
    python scripts/lh_kernel_debug.py stage-gpu            # the persistent GPU bundle (data/kernel_debug/gpu_bundle)
    python scripts/lh_kernel_debug.py ingest-gpu           # the bundle's calibration and GPU gates back into the records
    python scripts/lh_kernel_debug.py report               # funnel per family, candidates, pending GPU gates

Everything lives under ``data/kernel_debug`` (clones, derived clones with the synthetic refs, the record store, Docker
build dirs, prompts, the GPU bundle). No model is called; no GPU is used. Containers are ``crucible-lh-k-*`` with at
most two CPUs; images are ``crucible-lh-k:*``.
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1] / "data" / "kernel_debug"
# training supply after the student's release cut (2026-05-01) is 17 bug fixes and 4 speed-ups in ggml-org/llama.cpp; the
# training side also mines back to FALLBACK_SINCE and every record states its contamination (post-release | fallback)
FALLBACK_SINCE = "2025-09-01"
REPOS = {"ggml-org_llama.cpp": ("ggml-org/llama.cpp", "train"), "ikawrakow_ik_llama.cpp": ("ikawrakow/ik_llama.cpp", "heldout")}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jdump(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
    tmp.replace(p)


def funnel():
    from crucible.factory.generators.base import Funnel

    return Funnel(ROOT / "funnel.jsonl")


# ---- mine ------------------------------------------------------------------------------------------------------------------------
def cmd_mine(a: argparse.Namespace) -> None:
    from crucible.factory.lanes import kernel_debug as kd
    from crucible.factory.lanes import kernel_debug_mine as kdm
    from crucible.factory.sources.lease import EvalLease

    lease = EvalLease.default(extra_repos=kd.SEALED_REPOS)
    fn = funnel()
    for name, (full, split) in REPOS.items():
        if a.repos and name not in a.repos:
            continue
        d = ROOT / "clones" / name
        if split == "train":
            lease.guard_repo(full)  # a training source must not be sealed
        since = a.since or (kdm.SINCE if split == "heldout" else FALLBACK_SINCE)
        ch, drops = kdm.mine(d, name, since=since, cache=ROOT / "cache" / "backend" / name)
        for c in ch:
            fn.note("kernel_debug", f"{name}:{c.sha[:10]}", "S1_source", True, "", kind=c.kind, cls=c.fault_class, split=split)
        for x in drops:
            fn.note("kernel_debug", f"{name}:{x['sha'][:10]}", "S1_source", False, x["reason"], kind=x.get("kind"), split=split)
        jdump(ROOT / "mine" / f"{name}.json", {"repo": full, "split": split, "since": since, "at": now(), "changes": [c.to_json() for c in ch], "drops": drops})
        print(f"{name}: {len(ch)} kept ({collections.Counter(c.kind for c in ch)}), {len(drops)} dropped", flush=True)


# ---- build ---------------------------------------------------------------------------------------------------------------------------
def hunks(patch: str) -> list[str]:
    """``patch`` split into one patch per hunk (file headers repeated): the partial-fix mutants."""
    out, header, cur = [], [], []
    for ln in patch.splitlines(keepends=True):
        if ln.startswith("diff --git "):
            if cur:
                out.append("".join(header + cur))
            header, cur = [ln], []
        elif ln.startswith("@@"):
            if cur:
                out.append("".join(header + cur))
            cur = [ln]
        elif cur:
            cur.append(ln)
        else:
            header.append(ln)
    if cur:
        out.append("".join(header + cur))
    return out


def supports_floor(text: str, op: str) -> str | None:
    """The dispatch file with the CUDA backend claiming not to support ``op`` (the host-fallback floor), or None."""
    import re

    m = re.search(r"((?:GGML_CALL\s+)?static\s+bool\s+ggml_backend_cuda(?:_device)?_supports_op\s*\([^)]*\)\s*\{)", text)
    if not m or not re.fullmatch(r"[A-Z0-9_]+", op):
        return None
    return text[:m.end()] + f"\n    if (op->op == GGML_OP_{op}) {{ return false; }}  // crucible floor: claim no support\n" + text[m.end():]


def cmd_build(a: argparse.Namespace) -> None:
    from crucible.factory.generators import base as gb
    from crucible.factory.lanes import kernel_debug as kd
    from crucible.factory.lanes import kernel_debug_mine as kdm
    from crucible.factory.lanes import kernel_harness as kh

    fn = funnel()
    store = ROOT / "store"
    for name, (full, split) in REPOS.items():
        if a.repos and name not in a.repos:
            continue
        mined = json.loads((ROOT / "mine" / f"{name}.json").read_text())
        changes = kdm.as_changes(mined["changes"])
        if a.unused_only:  # more supply: only changes no stored record uses, so existing records and their faults stay as they are
            used = {f["sha"] for rec in kd.records(store) for f in rec["faults"]}
            changes = [c for c in changes if c.sha not in used]
        clone = ROOT / "clones" / name
        derived = gb.derived_clone(clone, ROOT / "derived" / name)
        layout = kdm.layout_at(clone)
        for kind in ("bug", "perf"):
            groups = kdm.group_changes(changes, kind=kind, max_size=4 if kind == "bug" else 3)
            for g in groups:
                rid = kd.record_id(kind, full, g)
                if a.ids and rid not in a.ids:
                    continue
                if (store / split / rid / "record.json").exists() and not a.redo:
                    print(rid, "exists")
                    continue
                syn = kd.synthesize(derived, g, layout, rid=rid)
                fn.note("kernel_debug", rid, "S3_world", syn.ok, syn.reason, family=kd.FAMILIES[kind], split=split, dropped=syn.dropped)
                if not syn.ok:
                    print(rid, "synthesis:", syn.reason, syn.dropped)
                    continue
                faults = syn.faults
                d = store / split / rid
                tree_a, tree_b = kd.export(derived, syn.a, d / "a"), kd.export(derived, syn.b, d / "b")
                paths = layout.paths()
                oracle = kd.diff_patch(derived, syn.a, syn.b, paths)
                ms = kd.milestones_of(faults, {}, {})
                by_sha = {m["fault"]: m for m in ms}
                mutants, floors = [], []
                for f in faults:
                    fwd = gb.full_patch(derived, f.parent, f.sha, kd.backend_paths(f, layout, with_harness=False))
                    if syn.revert_alone.get(f.sha):
                        trees, _bad = gb.apply_chain(derived, syn.b, [(fwd, True)])
                        p = gb.git(derived, "diff", "--full-index", "--binary", gb.tree_of(derived, syn.a), trees[-1], "--", *paths).stdout.decode("utf-8", "replace")
                        mutants.append({"id": f"revert-{by_sha[f.sha]['id']}", "kind": "revert", "milestone": by_sha[f.sha]["id"], "patch": p,
                                        "note": f"B with only {f.sha[:10]} reverted: its milestone must fail"})
                    hs = hunks(fwd.decode("utf-8", "replace"))
                    if len(hs) >= 2:
                        trees, bad = gb.apply_chain(derived, syn.b, [(hs[0].encode(), True)])
                        if bad is None:
                            p = gb.git(derived, "diff", "--full-index", "--binary", gb.tree_of(derived, syn.a), trees[-1], "--", *paths).stdout.decode("utf-8", "replace")
                            mutants.append({"id": f"partial-{by_sha[f.sha]['id']}", "kind": "partial_fix", "milestone": by_sha[f.sha]["id"], "patch": p,
                                            "note": f"B with the first of {len(hs)} hunks of {f.sha[:10]} reverted: a half fix must still fail"})
                dtext = (tree_a / layout.dispatch_file).read_text(errors="replace") if (tree_a / layout.dispatch_file).is_file() else ""
                for op in sorted({o for m in ms for o in m["ops"]})[:2]:
                    ft = supports_floor(dtext, op)
                    if ft is not None:
                        from crucible.factory.lanes import kernel as K

                        floors.append({"id": f"no_support_{op.lower()}", "patch": K.oracle_patch_of(tree_a, {layout.dispatch_file: ft}),
                                       "note": f"the CUDA backend claims not to support {op} (the harness would skip it): must score 0"})
                harness_b = kh.harness_file(tree_b).read_text(errors="replace") if kh.harness_file(tree_b).is_file() else ""
                rec = {"id": rid, "family": kd.FAMILIES[kind], "intent_id": kd.INTENTS[kind], "kind": kind, "repo": full, "source_clone": name, "split": split,
                       "licence": faults[-1].licence, "b": syn.b, "a": syn.a, "b_prime": syn.b_prime, "chain": syn.chain, "b_date": faults[-1].date,
                       "cutoff": kdm.SINCE, "contamination": kd.contamination(faults), "fault_dates": {f.sha: f.date for f in faults}, "dropped": syn.dropped,
                       "fix_alone_applies": syn.alone, "revert_alone_applies": syn.revert_alone,
                       "layout": {"cuda_dir": layout.cuda_dir, "dispatch_file": layout.dispatch_file},
                       "harness": {"file": kh.HARNESS, "dialect": kh.dialect_of(harness_b)}, "faults": [f.to_json() for f in faults], "milestones": ms,
                       "interactions": kdm.interactions(faults), "oracle_patch": oracle, "mutants": mutants, "floor_patches": floors,
                       "tree_a": str((d / "a").relative_to(store)), "tree_b": str((d / "b").relative_to(store)), "store": str(store), "built_at": now(),
                       "world": {"image": "crucible-lh-k:base-12.8.1", "base_dir": str(d / "a")}}
                kd.save(store, rec)
                print(rid, kind, split, [f.sha[:9] for f in faults], f"mutants {len(mutants)} floors {len(floors)}", flush=True)
                del tree_a


# ---- CPU gates -------------------------------------------------------------------------------------------------------------------------
def files_at(derived: Path, rev: str, files: list[str]) -> dict[str, str | None]:
    from crucible.factory.generators import base as gb

    return {f: gb.show(derived, rev, f) for f in files}


def cmd_gates(a: argparse.Namespace) -> None:
    from crucible.factory.lanes import kernel_debug as kd
    from crucible.factory.lanes import kernel_debug_ops as ko
    from crucible.factory.lanes import kernel_debug_world as kdw
    from crucible.factory.lanes import kernel_gates as kg
    from crucible.factory.lanes import kernel_world as kw

    kdw.ensure_image()
    fn = funnel()
    store = ROOT / "store"
    todo = [r for r in kd.records(store) if (not a.ids or r["id"] in a.ids) and (a.redo or "cpu" not in (r.get("gates") or {}))]
    print(f"{len(todo)} records to gate", flush=True)
    for rec in todo:
        t0 = time.time()
        g: dict[str, Any] = {"at": now()}
        layout = ko.Layout(**rec["layout"])
        tree_a, tree_b = store / rec["tree_a"], store / rec["tree_b"]
        files = sorted({f for m in rec["milestones"] for f in m["files"]})
        work = ROOT / "work" / rec["id"]
        try:
            # S3/S4 compile: every unit a fault touches, at A' (the start must still build: faults are behavioural) and at B
            bld = ROOT / "build" / rec["source_clone"] / rec["b"][:10]
            cfg = kdw.configure(tree_b, bld)
            g["configure_b"] = cfg
            units = kd.units_for(tree_b, files, layout)
            g["units"] = units
            if cfg["ok"]:
                g["compile_b"] = {u: r.to_json() for u, r in kdw.compile_units(tree_b, bld, units).items()}
                # a unit a fault adds (a new template instance) does not exist at A': the start builds without it
                g["compile_a"] = {u: r.to_json() for u, r in kdw.compile_units(tree_a, bld, [u for u in units if (tree_a / u).is_file()]).items()}
            ok_b = cfg["ok"] and all(v["ok"] for v in (g.get("compile_b") or {}).values()) and bool(units)
            ok_a = cfg["ok"] and all(v["ok"] for v in (g.get("compile_a") or {}).values()) and bool(units)
            g["S4_oracle_builds"], g["S3_start_builds"] = ok_b, ok_a
            # S8 mutants must build (a mutant killed by the compiler tests nothing on the GPU)
            mres = {}
            for mu in rec.get("mutants") or []:
                mt = work / "mutant"
                derived = ROOT / "derived" / rec["source_clone"]
                kg.overlay(tree_a, {}, mt)
                ok_apply = apply_patch(mt, mu["patch"])
                mfiles = [f for f in files if f.endswith((".cu", ".cuh", ".h"))]
                mres[mu["id"]] = {"applies": ok_apply}
                if ok_apply and cfg["ok"]:
                    mu_units = kd.units_for(mt, [f for f in mfiles if f in patch_files(mu["patch"])], layout)
                    rr = kdw.compile_units(mt, bld, mu_units, arch="90")  # one arch: the mutant must build, the GPU decides the rest
                    mres[mu["id"]]["builds"] = all(r.ok for r in rr.values()) and bool(rr)
                del derived
            g["mutants"] = mres
            # S8 floors apply to the start tree
            g["floors"] = {f["id"]: {"applies": apply_check(tree_a, f["patch"], work / "floor")} for f in rec.get("floor_patches") or []}
            harness_gates(rec, g, layout, tree_a, tree_b)
            # S3 World: one-commit workspace, history probe, leak probe (no B-only line in the image), the start configures
            ws = work / "ws" / rec["repo"].split("/")[-1]
            kw.workspace(tree_a, ws)
            g["history"] = kw.history_probe(ws)
            shutil.rmtree(work / "ws", ignore_errors=True)
            needles = [ln[1:].strip() for ln in rec["oracle_patch"].splitlines() if ln.startswith("+") and not ln.startswith("+++") and len(ln.strip()) >= 30]
            g["leak"] = kdw.leak_probe(needles[:40])
            cfg_a = kdw.configure(tree_a, work / "cfg_a")
            g["configure_a"] = {k: v for k, v in cfg_a.items() if k != "log"}
            shutil.rmtree(work / "cfg_a", ignore_errors=True)
        except Exception as e:  # noqa: BLE001 - a crashed gate run is a failed gate with its reason, never a pass
            g["error"] = f"{type(e).__name__}: {e}"[:600]
        verdict(rec, g)
        oom = [k for side in ("compile_a", "compile_b") for k, v in (g.get(side) or {}).items() if v.get("rc") in (137, -9)]
        if oom:
            g["infra"] = f"units killed for memory: {oom[:4]} (re-run; not a gate verdict)"
        g["seconds"] = round(time.time() - t0)
        rec.setdefault("gates", {})["cpu"] = g
        kd.save(store, rec)
        for st, ok in (("S4_compile", g.get("S4_oracle_builds")), ("S3_compile", g.get("S3_start_builds")), ("S5_harness_cpu", g.get("S5_harness_cpu")),
                       ("S3_world", bool((g.get("history") or {}).get("ok") and (g.get("leak") or {}).get("ok") and (g.get("configure_a") or {}).get("ok"))),
                       ("S8_cpu", all(v.get("builds") for v in (g.get("mutants") or {}).values()) and bool(g.get("mutants")))):
            fn.note("kernel_debug", rec["id"], st, bool(ok), g.get("error", "") if not ok else "", family=rec["family"], split=rec["split"])
        print(rec["id"], "cpu_pass", g["cpu_pass"], {k: g.get(k) for k in ("S4_oracle_builds", "S3_start_builds", "pools")}, g.get("error", ""), f"{g['seconds']}s", flush=True)
    kdw.cleanup()


def harness_gates(rec: dict[str, Any], g: dict[str, Any], layout: Any, tree_a: Path, tree_b: Path) -> None:
    """S5 (CPU half): the grader's harness, patched at B, builds on the CPU backend; each milestone's case pool is listed
    from it, the seeding is executed, and the fixes' own cases are B's list minus A''s list (A''s harness rebuilt in the
    same directory). A listing that returns nothing is a failed gate, never an empty pool."""
    from crucible.factory.lanes import kernel_debug_world as kdw
    from crucible.factory.lanes import kernel_harness as kh

    hb = kh.harness_file(tree_b).read_text(errors="replace")
    ha = kh.harness_file(tree_a).read_text(errors="replace")
    try:
        pb, pa = kh.patch_harness(hb, need_perf=rec["kind"] == "perf", header=kh.ggml_header(tree_b)), kh.patch_harness(ha, header=kh.ggml_header(tree_a))
        g["harness_patch"] = {"ok": True, "dialect": kh.dialect_of(hb), "repairs": kh.repair_harness(hb, kh.ggml_header(tree_b))[1]}
    except kh.HarnessPatchError as e:
        g["harness_patch"] = {"ok": False, "error": str(e)}
        return
    for k in ("harness_runs", "harness_list", "seeded", "pools", "harness_build_a"):
        g.pop(k, None)
    hbld = ROOT / "cpu_harness" / rec["id"]
    rt = kdw.CPU_GATE.get(layout.dispatch_file) or {}
    if rt.get("platform"):
        kdw.ensure_amd64_image()
    hbuild = kdw.cpu_harness_build(tree_b, hbld, pb, extra=rt.get("extra", ()), image=rt.get("image", kdw.IMAGE), platform=rt.get("platform"),
                                   ccache=ROOT / "ccache" / (rt.get("platform") or "native").replace("/", "-"))
    g["harness_build_b"] = hbuild
    if not hbuild["ok"]:
        return
    if not rt.get("runnable_here", True):
        # built for x86_64 under emulation, which has no AVX: the listing and the seeding run on the x86 GPU host
        g["harness_runs"] = "pending_x86_host (the repository's CPU backend needs AVX2; this machine emulates x86_64 without AVX)"
        return
    pools_b = list_cases(kdw, hbld, rec, kh.dialect_of(hb))
    if not pools_b:
        g["harness_list"] = "the B harness listed no case for these ops"
        return
    for m in rec["milestones"]:
        m["case_pool"] = sorted({c for op in m["ops"] for c in pools_b.get(op, [])})
        if rec["kind"] == "perf":
            m["perf_pool"] = sorted({c for op in m["ops"] for c in pools_b.get(f"perf:{op}", [])})
    g["seeded"] = seeded_check(kdw, hbld, rec)  # on B's harness, the grader's
    (hbld / "src" / kh.HARNESS).write_text(pa)  # then A''s harness: the agent's own cases
    habuild = rebuild(kdw, hbld)
    g["harness_build_a"] = habuild
    pools_a = list_cases(kdw, hbld, rec, kh.dialect_of(hb)) if habuild["ok"] else {}
    if not pools_a:
        g["harness_list"] = "the start's harness listed no case (its build or listing failed)"
        return
    for m in rec["milestones"]:
        m["cases_added"] = sorted(set(m["case_pool"]) - {c for op in m["ops"] for c in pools_a.get(op, [])})
    g["pools"] = {m["id"]: {"pool": len(m["case_pool"]), "added": len(m["cases_added"]), "perf_pool": len(m.get("perf_pool") or [])} for m in rec["milestones"]}


def verdict(rec: dict[str, Any], g: dict[str, Any]) -> None:
    """``cpu_structural_pass`` (everything but the harness runs) and ``cpu_pass`` (every CPU gate) from the gate evidence."""
    mut_ok = all(v.get("builds") for v in (g.get("mutants") or {}).values()) if g.get("mutants") else False
    pending_host = bool(g.get("harness_runs"))
    pools_ok = bool(g.get("pools")) and all(v["pool"] > 0 for v in g["pools"].values()) and (rec["kind"] != "perf" or all(v["perf_pool"] > 0 for v in g["pools"].values()))
    g["S5_harness_cpu"] = bool(pools_ok and (g.get("seeded") or {}).get("ok"))
    structural = bool(not g.get("error") and g.get("S4_oracle_builds") and g.get("S3_start_builds") and (g.get("harness_patch") or {}).get("ok")
                      and (g.get("harness_build_b") or {}).get("ok") and (g.get("history") or {}).get("ok") and (g.get("leak") or {}).get("ok")
                      and (g.get("configure_a") or {}).get("ok") and mut_ok and all(v["applies"] for v in (g.get("floors") or {}).values()))
    g["cpu_structural_pass"] = structural
    g["cpu_pass"] = bool(structural and pools_ok and (g.get("seeded") or {}).get("ok"))
    if structural and pending_host:
        g["cpu_pending_host"] = ["S5 case pools (listing)", "S5 seeding check"]


def cmd_repool(a: argparse.Namespace) -> None:
    """Re-run only the harness gates of gated records (the case listing, the seeding, the fixes' own cases) and the
    verdict, keeping the compile, World, floor and mutant evidence."""
    from crucible.factory.lanes import kernel_debug as kd
    from crucible.factory.lanes import kernel_debug_ops as ko

    store = ROOT / "store"
    for rec in kd.records(store):
        if a.ids and rec["id"] not in a.ids or "cpu" not in (rec.get("gates") or {}):
            continue
        g = rec["gates"]["cpu"]
        harness_gates(rec, g, ko.Layout(**rec["layout"]), store / rec["tree_a"], store / rec["tree_b"])
        verdict(rec, g)
        g["repooled_at"] = now()
        kd.save(store, rec)
        print(rec["id"], "cpu_pass", g["cpu_pass"], "structural", g.get("cpu_structural_pass"), g.get("pools"), g.get("harness_list", ""), flush=True)


def patch_files(patch: str) -> set[str]:
    import re

    return set(re.findall(r"^diff --git a/(\S+) b/", patch, re.MULTILINE))


def apply_patch(tree: Path, patch: str) -> bool:
    import subprocess

    if not patch.strip():
        return True
    p = subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], cwd=tree, input=patch, text=True, capture_output=True, check=False)
    return p.returncode == 0


def apply_check(tree: Path, patch: str, scratch: Path) -> bool:
    import subprocess

    if not patch.strip():
        return True
    p = subprocess.run(["git", "apply", "--check", "--whitespace=nowarn", "-"], cwd=tree, input=patch, text=True, capture_output=True, check=False)
    del scratch
    return p.returncode == 0


def rebuild(kdw: Any, hbld: Path) -> dict[str, Any]:
    return kdw.cpu_harness_rebuild(hbld)


def list_cases(kdw: Any, hbld: Path, rec: dict[str, Any], dialect: str | None) -> dict[str, list[str]]:
    """op -> the harness's test-mode case names for the record's ops, and ``perf:<op>`` -> its perf-mode case names
    (a perf record), from ``CRUCIBLE_LIST`` runs of the built harness (nothing is computed)."""
    ops = sorted({o for m in rec["milestones"] for o in m["ops"]})
    runs = [{"name": "list", "mode": "test", "list": True, "timeout_s": 600}]
    if rec["kind"] == "perf":
        runs.append({"name": "listperf", "mode": "perf", "list": True, "timeout_s": 600})
    res = {r["name"]: r for r in kdw.cpu_harness_runs(hbld, runs, timeout=3600)}
    out: dict[str, list[str]] = {}
    for name, prefix in (("list", ""), ("listperf", "perf:")):
        for case in (res.get(name) or {}).get("listed") or []:
            op = case.split("(", 1)[0]
            if op in ops:
                out.setdefault(prefix + op, []).append(case)
    del dialect
    return {k: sorted(set(v)) for k, v in out.items()}


def seeded_check(kdw: Any, hbld: Path, rec: dict[str, Any]) -> dict[str, Any]:
    """The grader's seeding executed on the CPU backend: the same seed twice gives identical dumps, another seed gives
    different inputs; up to 6 pool cases per record."""
    from crucible.factory.lanes import kernel_harness as kh

    pool = [c for m in rec["milestones"] for c in (m.get("case_pool") or [])]
    cases = sorted(set(pool), key=lambda c: (len(c), c))[:6]
    if not cases:
        return {"ok": False, "why": "no case to run"}
    runs = [{"name": n, "mode": "test", "seed": s, "cases": cases, "dump": True, "timeout_s": 1200} for n, s in (("s1", 11), ("s1b", 11), ("s2", 12))]
    res = {r["name"]: r for r in kdw.cpu_harness_runs(hbld, runs, timeout=3600)}
    if any(r.get("rc") != 0 or not r.get("dump") for r in res.values()):  # a run killed by the shared machine: once more before judging
        res = {r["name"]: r for r in kdw.cpu_harness_runs(hbld, runs, timeout=3600)}
    dumps = {k: kh.read_dump(Path(v["dump"]).read_bytes()) if v.get("dump") else [] for k, v in res.items()}
    same = bool(dumps["s1"]) and [(x["case"], x["out"], x["ref"]) for x in dumps["s1"]] == [(x["case"], x["out"], x["ref"]) for x in dumps["s1b"]]
    differ = bool(dumps["s2"]) and any(x["ref"] != y["ref"] for x, y in zip(dumps["s1"], dumps["s2"], strict=False))
    cpu_ok = all(st in ("OK",) for st in res["s1"]["cases"].values()) and len(res["s1"]["cases"]) == len(cases)
    return {"ok": same and differ and cpu_ok, "cases": cases, "same_seed_identical": same, "new_seed_differs": differ, "cpu_reference_ok": cpu_ok,
            "records": len(dumps["s1"])}


# ---- author ------------------------------------------------------------------------------------------------------------------------------
def cmd_author(a: argparse.Namespace) -> None:
    from crucible.factory.lanes import kernel_debug as kd

    store = ROOT / "store"
    fn = funnel()
    for rec in kd.records(store):
        if a.ids and rec["id"] not in a.ids:
            continue
        p = ROOT / "prompts" / f"{rec['id']}.json"
        if not p.exists():
            print(rec["id"], "no prompt yet")
            continue
        res = kd.author(rec, json.loads(p.read_text()))
        rec["intent"] = kd.task_intent(rec)
        rec["fairness"] = kd.fairness(rec)
        res["findings"] += [f"fairness: {f}" for f in rec["fairness"]["findings"]]
        kd.save(store, rec)
        fn.note("kernel_debug", rec["id"], "S7", not res["findings"], "; ".join(res["findings"])[:300], family=rec["family"], split=rec["split"])
        print(rec["id"], "lint", res["findings"] or "clean", flush=True)


# ---- stage the GPU gates -------------------------------------------------------------------------------------------------------------------
GPU_PENDING = ["S1_executed real-fix check (B passes, each single revert fails its milestone's cases)", "case calibration (milestone cases = cases each revert breaks)",
               "S6 executed edges (cases broken by two single reverts; prober A' + one fix)", "S5 oracle reward 1 on the GPU", "S8 floors 0 and mutants fail (GPU)",
               "sanitizer checks (racecheck/synccheck/memcheck) clean on B, dirty on the revert", "kernel-launch evidence", "perf bars + noise floor (perf family)",
               "S10 flake: 60 oracle grades at concurrency 1 and 4", "S12 hackers, S13 student and strong solver, S14/S15 audit and targeting"]


def alone_probers(rec: dict[str, Any]) -> dict[str, str]:
    """milestone -> the patch A' -> A' + its fix alone, for each milestone whose single revert does not apply at B (a
    later fix rewrote its lines): the GPU calibration owns such a milestone's cases by what its fix alone repairs (A'
    fails, A' + the fix passes, B passes) instead of what its revert breaks."""
    from crucible.factory.generators import base as gb
    from crucible.factory.lanes import kernel_debug as kd
    from crucible.factory.lanes import kernel_debug_mine as kdm
    from crucible.factory.lanes import kernel_debug_ops as ko

    have = {mu["milestone"] for mu in rec.get("mutants") or [] if mu["kind"] == "revert"}
    need = [m for m in rec["milestones"] if m["id"] not in have and (rec.get("fix_alone_applies") or {}).get(m["fault"])]
    if not need:
        return {}
    derived, layout = ROOT / "derived" / rec["source_clone"], ko.Layout(**rec["layout"])
    by_sha = {f.sha: f for f in kdm.as_changes(rec["faults"])}
    out = {}
    for m in need:
        f = by_sha[m["fault"]]
        trees, bad = gb.apply_chain(derived, rec["a"], [(gb.full_patch(derived, f.parent, f.sha, kd.backend_paths(f, layout, with_harness=False)), False)])
        if bad is None:
            out[m["id"]] = gb.git(derived, "diff", "--full-index", "--binary", gb.tree_of(derived, rec["a"]), trees[-1], "--", *layout.paths()).stdout.decode("utf-8", "replace")
    return out


def cmd_stage_gpu(a: argparse.Namespace) -> None:
    import tarfile

    from crucible.factory.lanes import kernel_debug as kd
    from crucible.factory.lanes import kernel_world as kw
    from crucible.factory.longtask import status_claim_patch
    from crucible.verify.stage import bundle

    out = Path(a.out) if a.out else ROOT / "gpu_bundle"
    out.mkdir(parents=True, exist_ok=True)
    src = Path(kd.__file__).parent
    bundle(out / "kit", {"crucible.factory.lanes.kernel_debug_plan": src / "kernel_debug_plan.py", "crucible.factory.lanes.kernel_harness": src / "kernel_harness.py",
                         "crucible.factory.lanes.kernel_plan": src / "kernel_plan.py"})
    kw.write_hooks(out / "hooks")
    index = []
    for rec in kd.records(ROOT / "store"):
        gc = (rec.get("gates") or {}).get("cpu", {})
        if not (gc.get("cpu_pass") or (gc.get("cpu_structural_pass") and gc.get("cpu_pending_host"))):  # pending-host: the GPU host lists the pools
            continue
        d = out / "tasks" / rec["id"]
        d.mkdir(parents=True, exist_ok=True)
        for side in ("a", "b"):
            tgt = d / f"{side}.tar.gz"
            if not tgt.exists() or a.redo:
                with tarfile.open(tgt, "w:gz") as t:
                    t.add(ROOT / "store" / rec[f"tree_{side}"], arcname=side)
        (d / "oracle.patch").write_text(rec["oracle_patch"])
        jdump(d / "kdspec.json", kd.kdspec(rec))
        floors = {"no_op": "", "claims_only": status_claim_patch(kd.KernelDebugTask(rec))}
        for f in rec.get("floor_patches") or []:
            floors[f["id"]] = f["patch"]
        for k, v in floors.items():
            (d / f"floor_{k}.patch").write_text(v)
        for mu in rec.get("mutants") or []:
            (d / f"mutant_{mu['id']}.patch").write_text(mu["patch"])
        probers = alone_probers(rec)
        for mid, p in probers.items():
            (d / f"prober_alone-{mid}.patch").write_text(p)
        jdump(d / "milestones.json", [{k: m.get(k) for k in ("id", "kind", "class", "ops", "checks", "repeat", "cases_added")} for m in rec["milestones"]])
        index.append({"id": rec["id"], "family": rec["family"], "split": rec["split"], "kind": rec["kind"], "milestones": [m["id"] for m in rec["milestones"]],
                      "cpu_pending_host": gc.get("cpu_pending_host") or [], "probers": {f"alone-{k}": k for k in probers},
                      "mutants": {mu["id"]: mu["milestone"] for mu in rec.get("mutants") or []}})
    jdump(out / "index.json", index)
    shutil.copy(Path(__file__).with_name("lh_kernel_debug_gpu.py"), out / "run_gpu.py")
    (out / "Dockerfile").write_text(kw.dockerfile(kw.BASE_FROM, extra=("ccache",)))  # ccache: the gate runs only
    (out / "run_gpu.sh").write_text(RUNNER)
    jdump(out / "PENDING.json", {"gates_pending_gpu": GPU_PENDING, "tasks": [i["id"] for i in index], "staged_at": now()})
    print(f"staged {len(index)} tasks in {out}")


RUNNER = r"""#!/usr/bin/env bash
# Kernel debugging lane, GPU gates (swarm-pipeline/10 §6). From this bundle's directory on a CUDA host:
#   bash run_gpu.sh docker      # lab RTX 5090 (Docker with --runtime=nvidia)
#   bash run_gpu.sh apptainer   # Marlowe H100 (apptainer --nv --fakeroot; the image is built from Dockerfile)
# Shared-machine rules: containers are crucible-lh-k-gpu-*, only those are removed; nothing is pruned; the runner waits
# for an idle GPU before every timed step.
set -euo pipefail
MODE=${1:-docker}
ARCH=${CRUCIBLE_CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')}
echo "arch sm_$ARCH on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python3 run_gpu.py --arch "$ARCH" --mode "$MODE" "${@:2}"
"""


# ---- ingest the GPU results ----------------------------------------------------------------------------------------------------------------
def cmd_ingest_gpu(a: argparse.Namespace) -> None:
    """Write a bundle's calibration and GPU gates back to the records: each milestone's calibrated cases, its executed
    ``requires`` edges (``requires_evidence: executed``), sanitizer checks kept or dropped, perf cases and bars per GPU
    model; milestones the calibration dropped leave the record with their requirements (the prompt is re-composed by
    ``author``); ``gates.gpu`` holds S5/S8/S10 per GPU model."""
    from crucible.factory.lanes import kernel_debug as kd

    bundle = Path(a.bundle) if a.bundle else ROOT / "gpu_bundle"
    store = ROOT / "store"
    for rec in kd.records(store):
        res_p, cal_p = bundle / "results" / f"{rec['id']}.json", bundle / "tasks" / rec["id"] / "kdspec.calibrated.json"
        if not res_p.exists():
            continue
        res = json.loads(res_p.read_text())
        if res.get("cpu_host"):  # the CPU gates the GPU host ran (x86 AVX2): the pools, the seeding, the fixes' own cases
            g, host = rec["gates"]["cpu"], res["cpu_host"]
            pooled = {m["id"]: m for m in json.loads((bundle / "tasks" / rec["id"] / "kdspec.json").read_text())["milestones"]}
            added = {m["id"]: m.get("cases_added") or [] for m in json.loads((bundle / "tasks" / rec["id"] / "milestones.json").read_text())}
            for m in rec["milestones"]:
                if host.get("ok"):
                    m["case_pool"], m["cases_added"] = pooled[m["id"]].get("case_pool") or [], added.get(m["id"]) or []
            g.pop("harness_runs", None)
            g.pop("cpu_pending_host", None)
            g.update(host_cpu=host, harness_runs_on="x86 GPU host", seeded=host.get("seeded"), pools=host.get("pools") if host.get("ok") else None,
                     harness_build_a={"ok": bool((host.get("build_a") or {}).get("ok"))})
            verdict(rec, g)
        gpu = res.get("gpu", "unknown")
        summary = {k: res.get(k) for k in ("S5_oracle", "S8_floors", "S8_mutants_fail_their_milestone", "S8_mutants_excluded", "S10", "flake", "calibration", "at",
                                            "arch", "error")}
        summary["reference_ok"] = (res.get("reference") or {}).get("ok")
        summary["gradable"] = bool((res.get("calibration") or {}).get("gradable"))
        summary["pass"] = bool(summary["gradable"] and res.get("S5_oracle") and res.get("S8_floors") and res.get("S10")
                               and all((res.get("S8_mutants_fail_their_milestone") or {}).values()))
        rec.setdefault("gates", {}).setdefault("gpu", {})[gpu] = summary
        if not cal_p.exists() or not summary["gradable"]:
            kd.save(store, rec)
            print(rec["id"], "not gradable on", gpu, res.get("error") or (res.get("reference") or {}).get("ok"), flush=True)
            continue
        cal = json.loads(cal_p.read_text())
        kept = {m["id"]: m for m in cal["milestones"]}
        ms = []
        for m in rec["milestones"]:
            c = kept.get(m["id"])
            if c is None:
                continue
            m.update(cases=c.get("cases") or [], checks=c["checks"], tests=[f"{m['id']}:{x}" for x in c["checks"]], perf_cases=c.get("perf_cases") or [],
                     requires=list(c.get("requires") or []), requires_evidence={q: "executed" for q in c.get("requires") or []})
            ms.append(m)
        rec["dropped_milestones"] = sorted({m["id"] for m in rec["milestones"]} - {m["id"] for m in ms} | set(rec.get("dropped_milestones") or []))
        rec["milestones"] = ms
        rec["perf_bars"] = {**(rec.get("perf_bars") or {}), **(cal.get("perf_bars") or {})}
        kd.save(store, rec)
        print(rec["id"], "milestones", [m["id"] for m in ms], "edges", [(m["id"], m["requires"]) for m in ms if m["requires"]], flush=True)


# ---- report ------------------------------------------------------------------------------------------------------------------------------
def migration_family(fam: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """The migration family's funnel and rows (``lh_kernel_debug_migrate.py``'s store)."""
    mig = ROOT / "migrate"
    mined = [json.loads(p.read_text()) for p in sorted((mig / "mine").glob("*.json"))] if (mig / "mine").is_dir() else []
    ws = [json.loads(p.read_text()) for p in sorted((mig / "store" / "windows").glob("*.json"))] if (mig / "store" / "windows").is_dir() else []
    f = fam.setdefault("kernel-migration:train", {"mined_commits": 0, "real_changes": 0, "drop_reasons": collections.Counter()})
    f["mined_commits"] = sum(len(m["commits"]) for m in mined)
    f["real_changes"] = f["mined_commits"]
    f["by_generation"] = {m["generation"]: len(m["commits"]) for m in mined}
    f["grouped"] = len(ws)
    ok = [w for w in ws if (w.get("gates") or {}).get("cpu_pass") and ((w.get("gates") or {}).get("kit_compile") or {}).get("ok")]
    f["cpu_pass"] = len(ok)
    f["authored_clean"] = sum(1 for w in ok if w.get("prompt") and not (w.get("lint") or {}).get("prompt") and not (w.get("lint") or {}).get("ledger"))
    for w in ws:
        g = w.get("gates") or {}
        gpu = next(iter((g.get("gpu") or {}).items()), (None, None))
        failed, why = (gpu[1].get("failed_gate"), gpu[1].get("why")) if gpu[1] else ("pending", "not run on a GPU")
        if g.get("cpu_pass") and (g.get("kit_compile") or {}).get("ok"):
            f.setdefault("gpu_drops", collections.Counter())
            if failed is None:
                f["gpu_pass"] = f.get("gpu_pass", 0) + 1
            else:
                f["gpu_drops"][failed] += 1
        rows.append({"id": w["id"], "family": "kernel-migration", "gpu": gpu[0], "gpu_failed_gate": failed, "gpu_why": why, "gpu_pass": failed is None,
                     "flake": (gpu[1] or {}).get("flake"), "split": w.get("split", "train"), "repo": (w.get("source") or {}).get("full_name"),
                     "generation": w["generation"], "commits": len((w.get("source") or {}).get("commits") or []), "programs": len(w["programs"]),
                     "milestones": len(w["milestones"]), "kinds": g.get("graded_kinds") or sorted(w.get("kinds") or {}), "rung": w.get("rung"),
                     "executed_cross_edges": len(g.get("executed_cross_edges") or []),
                     "executed_edges": sum(1 for m in w["milestones"] for v in (m.get("requires_evidence") or {}).values() if v == "executed"),
                     "cpu_pass": bool(g.get("cpu_pass") and (g.get("kit_compile") or {}).get("ok")), "authored": bool(w.get("prompt")),
                     "contamination": w.get("contamination"), "commit_dates": sorted(v[:10] for v in (w.get("commit_dates") or {}).values()),
                     "fault_mutants_pending_gpu": sum(1 for x in g.get("fault_mutants") or [] if x.get("builds")),
                     "gpu_pending": ["run milestones: golden vs the original on fresh seeds, launches, binary check", "fault-class mutants", "S10 flake",
                                     "S12-S15"]})


def sealing_report(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """``factory.sealing.seal`` on the training vs held-out records: repositories must be disjoint; the time split and the
    prompt/diff contamination scan are reported (the held-out kernels share the training period: a time split is not
    possible for this source, so ``time_violations`` is expected and stated, not hidden)."""
    from crucible.factory.lanes import kernel_debug as kd
    from crucible.factory.sealing import seal

    tr = [r for r in recs if r["split"] == "train"]
    ho = [r for r in recs if r["split"] == "heldout"]
    if not tr or not ho:
        return {"skipped": "one split is empty"}
    rep = seal(tr, ho, repo=lambda r: r["repo"], date=lambda r: r.get("b_date"), prompt=lambda r: r.get("prompt") or "", diff=lambda r: r.get("oracle_patch") or "",
               wid=lambda r: r["id"], model_cutoff="2026-04-15", template=[kd.EPILOGUE, " ".join(kd.LANE_SENTENCES), kd.HEADING])
    d = rep.to_dict()
    parts = collections.Counter(str(h.get("source") or h.get("corpus") or h).split(":")[-1][:12] for h in d["contamination"])
    return {"repos_disjoint": not d["shared_repos"], "shared_repos": d["shared_repos"], "time_violations": len(d["time_violations"]),
            "contamination_hits": len(d["contamination"]), "contamination_by_part": dict(parts), "contamination_examples": d["contamination"][:3],
            "cutoff_violations": len(d["cutoff_violations"]), "counts": d["counts"]}


GPU_GATES = ("gpu_reference", "S1_executed", "calibration", "S5_oracle", "S8_floors", "S8_mutants", "S10_flake")


def gpu_verdict(g: dict[str, Any]) -> tuple[str | None, str]:
    """(the first GPU gate a record failed, or None when it passed all; why) from one GPU model's ingested summary."""
    cal = g.get("calibration") or {}
    if not g.get("reference_ok"):
        return "gpu_reference", g.get("error") or "the reference build failed"
    if not g.get("gradable"):
        why = cal.get("error") or ""
        if why.startswith("not real on this GPU"):
            return "S1_executed", why
        kept = cal.get("kept_milestones") or []
        return "calibration", why or f"{len(kept)} milestone(s) kept after calibration (owned {cal.get('owned')}, unstable {list(cal.get('unstable_cases') or {})})"
    if not g.get("S5_oracle"):
        return "S5_oracle", "the oracle does not score 1 on the calibrated spec"
    if not g.get("S8_floors"):
        return "S8_floors", "a floor scores above 0"
    if not all((g.get("S8_mutants_fail_their_milestone") or {}).values()):
        return "S8_mutants", "a mutant passes its milestone: " + ", ".join(k for k, v in (g.get("S8_mutants_fail_their_milestone") or {}).items() if not v)
    if not g.get("S10"):
        return "S10_flake", f"flake {g.get('flake')}"
    return None, ""


def cmd_report(a: argparse.Namespace) -> None:
    from crucible.factory.lanes import kernel_debug as kd

    recs = kd.records(ROOT / "store")
    mined = {n: json.loads((ROOT / "mine" / f"{n}.json").read_text()) for n in REPOS if (ROOT / "mine" / f"{n}.json").exists()}
    fam: dict[str, Any] = {}
    unclassified: dict[str, Any] = {}
    for m in mined.values():
        u = unclassified.setdefault(m["split"], collections.Counter())
        for d in m["drops"]:
            if d.get("kind") is None:
                u[d["reason"].split(" (")[0]] += 1
        for k in ("bug", "perf"):
            f = fam.setdefault(f"{kd.FAMILIES[k]}:{m['split']}", {"mined_commits": 0, "real_changes": 0, "drop_reasons": collections.Counter()})
            kept = sum(1 for c in m["changes"] if c["kind"] == k)
            dropped = [d for d in m["drops"] if d.get("kind") == k]
            f["real_changes"] += kept
            f["mined_commits"] += kept + len(dropped)
            for d in dropped:
                f["drop_reasons"][d["reason"].split(" (")[0]] += 1
    rows = []
    for r in recs:
        g = (r.get("gates") or {}).get("cpu") or {}
        key = f"{r['family']}:{r['split']}"
        f = fam.setdefault(key, {"mined_commits": 0, "real_changes": 0, "drop_reasons": collections.Counter()})
        f["grouped"] = f.get("grouped", 0) + 1
        f["cpu_pass"] = f.get("cpu_pass", 0) + int(bool(g.get("cpu_pass")))
        f["cpu_structural_pass"] = f.get("cpu_structural_pass", 0) + int(bool(g.get("cpu_structural_pass")))
        f["authored_clean"] = f.get("authored_clean", 0) + int(bool(r.get("prompt")) and not (r.get("lint") or {}).get("findings"))
        gpu = next(iter(((r.get("gates") or {}).get("gpu") or {}).items()), (None, None))
        failed, why = gpu_verdict(gpu[1]) if gpu[1] else ("pending", "not run on a GPU")
        host = ((g.get("host_cpu") or {}).get("ok"))
        if g.get("cpu_pass") or host:
            f.setdefault("gpu_drops", collections.Counter())
            if failed is None:
                f["gpu_pass"] = f.get("gpu_pass", 0) + 1
            else:
                f["gpu_drops"][failed] += 1
        rows.append({"id": r["id"], "family": r["family"], "split": r["split"], "repo": r["repo"], "faults": len(r["milestones"]),
                     "gpu": gpu[0], "gpu_failed_gate": failed, "gpu_why": why, "gpu_pass": failed is None,
                     "gpu_kept_milestones": ((gpu[1] or {}).get("calibration") or {}).get("kept_milestones"),
                     "gpu_executed_edges": {k: v for k, v in (((gpu[1] or {}).get("calibration") or {}).get("executed_edges") or {}).items() if v},
                     "gpu_mutants_excluded": (gpu[1] or {}).get("S8_mutants_excluded"), "flake": (gpu[1] or {}).get("flake"),
                     "classes": [m["class"] for m in r["milestones"]], "ops": sorted({o for m in r["milestones"] for o in m["ops"]}),
                     "claimed_edges": [(m["id"], q) for m in r["milestones"] for q in m.get("claimed_requires") or []], "cpu_pass": bool(g.get("cpu_pass")),
                     "cpu_structural_pass": bool(g.get("cpu_structural_pass")), "cpu_pending_host": g.get("cpu_pending_host"), "pools": g.get("pools"), "authored": bool(r.get("prompt")), "lint": (r.get("lint") or {}).get("findings"), "contamination": r.get("contamination"),
                     "cutoff": r.get("cutoff"), "fault_dates": sorted(v[:10] for v in (r.get("fault_dates") or {}).values()), "gpu_pending": GPU_PENDING})
    migration_family(fam, rows)
    for f in fam.values():
        f["drop_reasons"] = dict(f["drop_reasons"])
        if "gpu_drops" in f:
            f["gpu_drops"] = dict(f["gpu_drops"])
    rep = {"at": now(), "families": fam, "unclassified_drops": {k: dict(v) for k, v in unclassified.items()}, "tasks": rows, "gpu_pending": GPU_PENDING,
           "sealing": sealing_report(recs)}
    jdump(Path(a.out), rep)
    print(json.dumps({"families": fam, "cpu_pass": sum(1 for x in rows if x["cpu_pass"])}, indent=1, default=str))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("mine")
    s.add_argument("--repos", nargs="*")
    s.add_argument("--since", help="default: the student's cut for held-out sources, the marked fallback for training sources")
    s = sub.add_parser("build")
    s.add_argument("--repos", nargs="*")
    s.add_argument("--ids", nargs="*")
    s.add_argument("--redo", action="store_true")
    s.add_argument("--unused-only", action="store_true", help="group only changes no stored record uses (adding supply)")
    s = sub.add_parser("gates")
    s.add_argument("--ids", nargs="*")
    s.add_argument("--redo", action="store_true")
    s = sub.add_parser("author")
    s.add_argument("--ids", nargs="*")
    s = sub.add_parser("repool")
    s.add_argument("--ids", nargs="*")
    s = sub.add_parser("stage-gpu")
    s.add_argument("--out")
    s.add_argument("--redo", action="store_true")
    s = sub.add_parser("ingest-gpu")
    s.add_argument("--bundle")
    s = sub.add_parser("report")
    s.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "docs" / "research" / "swarm-pipeline" / "runs" / "kernel_debug_funnel.json"))
    a = ap.parse_args(argv)
    {"mine": cmd_mine, "build": cmd_build, "gates": cmd_gates, "author": cmd_author, "repool": cmd_repool, "stage-gpu": cmd_stage_gpu, "ingest-gpu": cmd_ingest_gpu, "report": cmd_report}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
