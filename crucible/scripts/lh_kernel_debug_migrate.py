#!/usr/bin/env python3
"""The kernel debugging lane's migration family on CPU (swarm-pipeline/10 §3.3): mine, group, build, CPU gates, author.

    python scripts/lh_kernel_debug_migrate.py mine       # CUDA 12->13 (HeCBench), CUDA 11->12 (cudahandbook) migration commits
    python scripts/lh_kernel_debug_migrate.py build      # groups of 2-4 commits -> A'..B -> one merged multi-program window each
    python scripts/lh_kernel_debug_migrate.py gates      # kernel_gates.compile_gates with the generation's toolkits, then the kit grader on CPU
    python scripts/lh_kernel_debug_migrate.py author     # the builder's openings (prompts/<id>.json) -> prompt, ledger, lint, intent
    python scripts/lh_kernel_debug_migrate.py stage-gpu  # the GPU bundle (data/kernel_debug/gpu_bundle/migration)

State lives under ``data/kernel_debug/migrate`` (windows in ``store/windows``, trees in ``store/trees``). Containers are
``crucible-k-*`` (the kernel lane's compile helpers) and ``crucible-lh-k-*``; no model, no GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1] / "data" / "kernel_debug"
MIG = ROOT / "migrate"
SOURCES = {"zjin-lcf_HeCBench": ("zjin-lcf/HeCBench", "cuda13", "2025-08-01"), "ArchaeaSoftware_cudahandbook": ("ArchaeaSoftware/cudahandbook", "cuda12", None)}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jdump(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
    tmp.replace(p)


def windows() -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((MIG / "store" / "windows").glob("*.json"))]


def cmd_mine(a: argparse.Namespace) -> None:
    from crucible.factory.lanes import kernel_debug_migrate as kdm

    for name, (full, gen, since) in SOURCES.items():
        cs = kdm.scan(ROOT / "clones" / name, name, gen, since=since)
        rows = [{**c.to_json(), "full_name": full, "generation": gen} for c in cs]
        jdump(MIG / "mine" / f"{name}.json", {"repo": full, "generation": gen, "since": since, "at": now(), "commits": rows})
        print(f"{name} ({gen}): {len(cs)} migration commits, kinds {collections.Counter(k for c in cs for k in c.kinds if not k.startswith('_'))}", flush=True)


def commit_of(row: dict[str, Any]) -> Any:
    from crucible.factory.lanes import kernel_mine as km

    return km.MigrationCommit(**{k: v for k, v in row.items() if k in km.MigrationCommit.__dataclass_fields__})


def include_dirs(derived: Path, rev: str, files: list[str]) -> set[str]:
    """Directories the Makefiles of the migrated programs pass with ``-I`` (HeCBench's ``scan3`` includes ``../scan2-cuda``)."""
    import re

    from crucible.factory.generators import base as gb
    from crucible.factory.lanes import kernel_debug_migrate as kdm
    from crucible.factory.lanes import kernel_mine as km

    out = set()
    for d in {kdm.program_key(f) for f in files}:
        mk = gb.show(derived, rev, f"{d}/Makefile") or ""
        for inc in re.findall(r"-I\s*([^\s$]+)", mk):
            if not inc.startswith("/"):
                out.add(km._norm(f"{d}/{inc}"))
    return out


def cmd_build(a: argparse.Namespace) -> None:
    from crucible.factory.generators import base as gb
    from crucible.factory.lanes import kernel as K
    from crucible.factory.lanes import kernel_debug as kd
    from crucible.factory.lanes import kernel_debug_migrate as kdm
    from crucible.factory.lanes import kernel_mine as km
    from crucible.factory.lanes import kernel_world as kw

    store = MIG / "store"
    for name, (full, gen, _since) in SOURCES.items():
        mined = json.loads((MIG / "mine" / f"{name}.json").read_text())
        commits = [commit_of(r) for r in mined["commits"]]
        hard = K.HARD_KINDS if gen == "cuda12" else ()
        groups = kdm.group_commits(commits, max_size=a.max_size, hard=hard)
        clone = ROOT / "clones" / name
        derived = gb.derived_clone(clone, ROOT / "derived" / name)
        print(f"{name}: {len(groups)} groups", flush=True)
        for grp in groups:
            h = gb.out(gb.git(derived, "rev-parse", grp[-1].sha))[:7]
            wid = f"km-{gen}-{name.split('_', 1)[-1].lower()[:12]}-{h}-{len(grp)}"
            if a.ids and wid not in a.ids:
                continue
            if (store / "windows" / f"{wid}.json").exists() and not a.redo:
                continue

            def paths_of(c: Any, _with_harness: bool) -> list[str]:
                return [f for f in c.changed_files if km.is_source(f)]

            syn = kd.synthesize(derived, grp, None, rid=wid, paths_of=paths_of)
            if not syn.ok:
                print(wid, "synthesis:", syn.reason, syn.dropped)
                continue
            grp = syn.faults
            files = sorted({f for c in grp for f in c.migrated_files})
            sub = kw.subtree_paths(derived, syn.b, files)
            if sub:  # a large tree exported in parts: the include directories the programs' build files name come too
                sub = sorted(set(sub) | include_dirs(derived, syn.b, files))
            ta, tb = store / "trees" / f"{wid}-a", store / "trees" / f"{wid}-b"
            kw.export_tree(derived, syn.a, ta, paths=sub)
            kw.export_tree(derived, syn.b, tb, paths=sub)
            kinds: dict[str, Any] = {}
            for c in grp:
                for k, v in c.kinds.items():
                    if not k.startswith("_"):
                        kinds.setdefault(k, v)
            commit = {"repo": name, "sha": syn.b, "parent": syn.a, "date": grp[-1].date, "subject": f"{len(grp)} migration commits", "kinds": kinds,
                      "migrated_files": files, "changed_files": sorted({f for c in grp for f in c.changed_files}), "licence": grp[-1].licence}
            ws, drops = K.windows_from_commit(commit, ta, tb, max_programs=4)
            if len(ws) < 2:
                print(wid, f"{len(ws)} buildable program group(s): not multi-program", [d["reason"] for d in drops][:3])
                continue
            ws = kdm.pick_components(ws, grp, cap=a.cap)
            w = kdm.merge_windows(ws, wid)
            w.update(generation=gen, split="train" if gen == "cuda12" or a.heldout_after is None or grp[-1].date[:10] <= a.heldout_after else "heldout",
                     tree_a=str(ta.relative_to(store)), source={"commits": [c.sha for c in grp], "full_name": full}, family=kd.FAMILIES["migration"],
                     intent_id=kd.INTENTS["migration"], commit_dates={c.sha: c.date for c in grp}, cutoff="2026-05-01",
                     contamination="post-release" if all(c.date[:10] >= "2026-05-01" for c in grp) else "pre-release",
                     message="\n\n".join(f"{c.subject}" for c in grp), chain=syn.chain, b_prime=syn.b_prime, built_at=now())
            w["oracle_patch"] = K.oracle_patch_of(ta, w["oracle_files"])
            w["oracle_new_names"] = K.oracle_new_names(w)
            w["locate"] = K.locate_lines(ta, w)
            jdump(store / "windows" / f"{wid}.json", w)
            print(wid, [c.sha[:9] for c in grp], "programs", len(w["programs"]), "milestones", len(w["milestones"]), "kinds", sorted(kinds), flush=True)


def cmd_gates(a: argparse.Namespace) -> None:
    from crucible.factory.lanes import kernel as K
    from crucible.factory.lanes import kernel_debug_migrate as kdm
    from crucible.factory.lanes import kernel_gates as kg
    from crucible.factory.lanes import kernel_world as kw
    from crucible.factory.longtask import status_claim_patch

    store = MIG / "store"
    for w in windows():
        if a.ids and w["id"] not in a.ids:
            continue
        if "gates" in w and not a.redo:
            continue
        g = kdm.GENERATIONS[w["generation"]]
        t0 = time.time()
        try:
            kg.compile_gates(w, store / w["tree_a"], base_image=g["world"], legacy_image=g["legacy"], legacy_arches=g["legacy_arches"])
            # the kit grader in the World image, compile milestones only: oracle passes all, floors none, each revert fails its milestone
            be = kw.DockerBackend(store, gpu=False, image=g["world"])
            t = K.KernelTask(w, be)
            comp = [m["id"] for m in w["milestones"] if m.get("kind") == "compile"]
            cases = {"oracle": t.oracle_patch(), "no_op": "", "claims_only": status_claim_patch(t)}
            for m in comp:
                back = {f: K.read_raw(store / w["tree_a"] / f) if (store / w["tree_a"] / f).is_file() else None
                        for f in next(x for x in w["milestones"] if x["id"] == m)["files"]}
                cases[f"revert-{m}"] = K.oracle_patch_of(store / w["tree_a"], {**w["oracle_files"], **back})
            res = {}
            for name, patch in cases.items():
                rep = asyncio.run(t.grade_patch(patch))
                res[name] = rep.meta.get("compile_milestones") or {}
            ok = (bool(res["oracle"]) and all(res["oracle"].values()) and not any(res["no_op"].values()) and not any(res["claims_only"].values())
                  and all(res[f"revert-{m}"].get(m) is False for m in comp))
            w["gates"]["kit_compile"] = {"ok": ok, "cases": res}
            w["rung"] = K.rung_of(w)
            w["kspec"] = K.kspec(w)
        except Exception as e:  # noqa: BLE001 - a crashed gate run is a failed gate with its reason, never a pass
            w["gates"] = {**(w.get("gates") or {}), "error": f"{type(e).__name__}: {e}"[:500], "cpu_pass": False}
        w["gates"]["seconds"] = round(time.time() - t0)
        w["gates"]["at"] = now()
        edges = K.cross_edges(w)
        w["gates"]["executed_cross_edges"] = edges
        jdump(store / "windows" / f"{w['id']}.json", w)
        gg = w["gates"]
        print(w["id"], "cpu_pass", gg.get("cpu_pass"), "kit", (gg.get("kit_compile") or {}).get("ok"), "rung", w.get("rung"), "edges", len(edges), gg.get("error", ""),
              f"{gg['seconds']}s", flush=True)
    kw.cleanup()


def cmd_author(a: argparse.Namespace) -> None:
    from crucible.factory.lanes import kernel as K
    from crucible.factory.lanes import kernel_debug as kd
    from crucible.factory.lanes import kernel_debug_migrate as kdm

    store = MIG / "store"
    for w in windows():
        if a.ids and w["id"] not in a.ids:
            continue
        p = ROOT / "prompts" / f"{w['id']}.json"
        if not p.exists():
            print(w["id"], "no prompt yet")
            continue
        au = json.loads(p.read_text())
        w["opening"] = au["opening"]
        w["prompt"], w["spans"] = kdm.compose(au["opening"], w, w["generation"])
        w["requirements"] = kdm.migration_requirements(w, w["generation"])
        _led, led_problems = K.ledger_of(w)
        lint = K.lint_prompt(w["prompt"], w) + [f for f in kd.lint_prompt(au["opening"], {"milestones": []}) if "hidden" not in f]
        w["lint"] = {"prompt": lint, "ledger": led_problems}
        w["prompt_provenance"] = {"builder": au.get("builder"), "kind": "maintainer request", "states_only": "goal and symptom (build failure on the new toolkit)"}
        w["kspec"] = K.kspec(w)
        w["intent"] = migration_intent(w)
        jdump(store / "windows" / f"{w['id']}.json", w)
        print(w["id"], "lint", lint or "clean", "ledger", led_problems[:3] or "clean", flush=True)


def cmd_stage_gpu(a: argparse.Namespace) -> None:
    """The migration windows' GPU bundle: the kernel lane's bundle (``lh_kernel.py stage-gpu``: base tree, oracle, floors,
    fault mutants, kspec, the kit with ``kernel_plan``, ``run_gpu.sh`` + ``stage_gpu.py``) with per-window images
    (``images.json``: CUDA 13 World with the CUDA 12.8 original for the CUDA 13 generation) and their Dockerfiles."""
    import argparse as ap

    from crucible.factory.lanes import kernel_debug_migrate as kdm
    from crucible.factory.lanes import kernel_world as kw

    sys.path.insert(0, str(Path(__file__).parent))
    import lh_kernel

    out = Path(a.out) if a.out else ROOT / "gpu_bundle" / "migration"
    lh_kernel.cmd_stage_gpu(ap.Namespace(store=str(MIG / "store"), out=str(out)))
    # on the GPU host every image carries this lane's tag (shared machine: only crucible-lh-k images are created and
    # removed); the 12.8 image is the main bundle's (same Dockerfile, so whichever bundle runs first builds it once)
    base, legacy = "crucible-lh-k:base-12.8.1", "crucible-lh-k:legacy-11.8.0"
    images = {base: "Dockerfile.base", legacy: "Dockerfile.legacy", kdm.CUDA13_TAG: "Dockerfile.cuda13"}
    (out / "Dockerfile.base").write_text(kw.dockerfile(kw.BASE_FROM, extra=("ccache",)))
    (out / "Dockerfile.cuda13").write_text(kw.dockerfile(kdm.CUDA13_FROM))
    jdump(out / "images.json", images)
    for w in windows():
        d = out / "windows" / w["id"]
        if d.is_dir():
            jdump(d / "images.json", {"base": kdm.CUDA13_TAG if w["generation"] == "cuda13" else base, "legacy": base if w["generation"] == "cuda13" else legacy})
    print(f"migration bundle in {out}")


def migration_intent(w: dict[str, Any]) -> dict[str, Any]:
    from crucible.factory.lanes import kernel as K
    from crucible.factory.lanes import kernel_debug as kd

    edges = K.cross_edges(w)
    progs = sorted({p["main"] for p in w["programs"].values()})
    skills = ["toolkit_migration", "kernel_fault_localization"] + (["interacting_faults"] if edges else [])
    return {"intent": "KD3", "family": kd.FAMILIES["migration"], "skills": skills, "generation": w["generation"], "kinds": sorted(w.get("graded_kinds") or w.get("kinds") or {}),
            "programs": progs, "why_long": (f"{len(progs)} programs broken by {len(w.get('kinds') or {})} different removed APIs across {len(w['migrated_files'])} files; "
                                           f"{len(edges)} executed compile edges (a program builds only after its migrated headers do)"),
            "milestones": [{"id": m["id"], "kind": m["kind"], "files": m["files"]} for m in w["milestones"]],
            "expected_failures": {k: kd.SKILLS[k]["failure_mode"] for k in skills}, "trace_evidence": {k: kd.SKILLS[k]["evidence"] for k in skills},
            "attribution_rule": "as KD1: >= 70% of failures with their trace evidence; locate counterfactual (removed-construct lines named)",
            "targets": {"student": kd.STUDENT, "train_band": [0.10, 0.60], "benchmark_max": 0.30, "fairness": "a strong solver passes (S13)"}}


def migration_gpu_verdict(res: dict[str, Any]) -> tuple[str | None, str]:
    """(the first GPU gate a window failed, or None; why) from ``stage_gpu.py``'s result."""
    if not res.get("originals") or not all(v for k, v in res["originals"].items() if not k.startswith("_")):
        return "gpu_reference", f"an original does not build with the older toolkit: {res.get('originals')}"
    if not res.get("S5_oracle"):
        return "S5_oracle", f"oracle reward {(res.get('oracle') or {}).get('reward')}, gates {(res.get('oracle') or {}).get('gates')}"
    if not res.get("S8_floors"):
        return "S8_floors", "a floor scores above 0"
    if not all((res.get("S8_mutants_killed") or {}).values()):
        return "S8_mutants", "a fault mutant passes every milestone: " + ", ".join(k for k, v in res["S8_mutants_killed"].items() if not v)
    if (res.get("flake") or {}).get("cp_upper", 1.0) > 0.05:
        return "S10_flake", f"flake {res.get('flake')}"
    return None, ""


def cmd_ingest_gpu(a: argparse.Namespace) -> None:
    """The migration bundle's GPU results (``results/<window>.json``) into the windows: ``gates.gpu[<GPU>]``."""
    bundle = Path(a.bundle) if a.bundle else ROOT / "gpu_bundle" / "migration"
    for w in windows():
        p = bundle / "results" / f"{w['id']}.json"
        if not p.exists():
            continue
        res = json.loads(p.read_text())
        failed, why = migration_gpu_verdict(res)
        w.setdefault("gates", {}).setdefault("gpu", {})[res.get("gpu") or "unknown"] = {
            "at": res.get("at"), "arch": res.get("arch"), "originals": res.get("originals"), "S5_oracle": res.get("S5_oracle"), "S8_floors": res.get("S8_floors"),
            "S8_mutants_killed": res.get("S8_mutants_killed"), "flake": res.get("flake"), "failed_gate": failed, "why": why, "pass": failed is None}
        jdump(MIG / "store" / "windows" / f"{w['id']}.json", w)
        print(w["id"], "gpu", "pass" if failed is None else f"{failed}: {why}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("mine")
    s = sub.add_parser("build")
    s.add_argument("--ids", nargs="*")
    s.add_argument("--redo", action="store_true")
    s.add_argument("--max-size", dest="max_size", type=int, default=4)
    s.add_argument("--cap", type=int, default=6, help="programs per task")
    s.add_argument("--heldout-after", dest="heldout_after", default=None, help="CUDA 13 groups newer than this date go to the held-out split")
    s = sub.add_parser("gates")
    s.add_argument("--ids", nargs="*")
    s.add_argument("--redo", action="store_true")
    s = sub.add_parser("author")
    s.add_argument("--ids", nargs="*")
    s = sub.add_parser("stage-gpu")
    s.add_argument("--out")
    s = sub.add_parser("ingest-gpu")
    s.add_argument("--bundle")
    a = ap.parse_args(argv)
    {"mine": cmd_mine, "build": cmd_build, "gates": cmd_gates, "author": cmd_author, "stage-gpu": cmd_stage_gpu, "ingest-gpu": cmd_ingest_gpu}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
