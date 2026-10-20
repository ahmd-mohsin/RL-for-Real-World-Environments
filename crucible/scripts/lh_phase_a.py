"""Phase A (swarm-pipeline/04 "Phase A"): every World built before benchmark v2's leak fix, rebuilt, with evidence. $0.

Per task of every training and held-out store (store3 train/held-out trainable windows, P5a's gated rebreak / stitched /
open tasks, P5b's swarm-built windows, benchmark v2's sealed tasks), one resumable journal ``<out>/tasks/<store>__<id>.json``:

1. ``evidence``: the World the solvers used (the record's image, or its head task version's) is opened and its installed
   copy of the project read (``windows_world.installed_copy``); it is compared with the window's A and B
   (``windows_s3.compare_installed``: per file, and the share of the oracle's added lines it holds) and identified among
   every known start tree of the repository (``identify_copy``: which window's A it is, and whether that A is later);
2. ``refresh``: the leak as a ``world_bug`` through the executed acceptance (``windows_s3.leak_world_fix``): the
   reproduction fails on the oracle state in the old World, passes in the World rebuilt under the current recipe, and the
   rebuilt World passes S3 (60 reruns, every leak probe incl. ``no_installed_project_source``). An accepted fix is a task
   version in ``<out>/versions.jsonl`` (seeded from P5b's, so P4's World fixes are the parents) and the record is saved
   with the new World (``world_previous`` keeps the old one; a backup of the record is kept under ``<out>/backup``);
3. ``gates``: S6, S5, S8, S10 on the refreshed record (S3 is the acceptance's); open tasks run S5, S8, S10.

``report`` writes the before/after table (recorded gate verdicts, the verdict under today's probes, the verdict after).
Containers: ``crucible-pa-*`` (grading) and ``crucible-lh-pa-*`` (Worlds, probes, S3); ``cleanup`` removes only those.

    PYTHONPATH=src python scripts/lh_phase_a.py --out <dir> run [--only id,...] [--parallel 2]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

GRADE_PREFIX = "crucible-pa-"
WORLD_PREFIX = "crucible-lh-pa-"
SCRATCH = Path(__file__).resolve().parents[3]
# the P5b windows Phase A re-targets first (P5b's six "too easy" verdicts)
PRIORITY = ("st-glom--boltons-2ae0bb57", "st-glom--boltons-30eb7ffe", "rb-more-itertools-061d3f39", "sw-python-semver-a2e121df",
            "sw-cachetools-aadb387d", "sw-more-itertools-dedc22c6")


def jload(p: Path, default: Any = None) -> Any:
    return json.loads(p.read_text()) if p.exists() else default


def jsave(p: Path, x: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(x, indent=1, sort_keys=True, default=str))
    tmp.replace(p)


# ---- the stores -------------------------------------------------------------------------------------------------------------
def stores(scratch: Path) -> list[dict[str, Any]]:
    return [
        {"key": "store3-train", "kind": "window", "root": scratch / "lhwork/store3/train", "split": "train"},
        {"key": "store3-heldout", "kind": "window", "root": scratch / "lhwork/store3/heldout", "split": "heldout"},
        {"key": "p5-rebreak", "kind": "rebreak", "root": scratch / "p5/rebreak/store/train", "split": "train", "clones": scratch / "p5/rebreak/clones"},
        {"key": "p5-stitch", "kind": "stitched", "root": scratch / "p5/stitch/store/train", "split": "train", "pairs": scratch / "p5/stitch/pairs.json"},
        {"key": "p5-open", "kind": "open", "root": scratch / "p5/open/store/train", "split": "train", "gen_root": scratch / "p5/open"},
        {"key": "p5b-swarm", "kind": "swarm", "root": scratch / "p5b/run/swarm/store/train", "split": "train", "clones": scratch / "p5b/run/swarm/clones"},
        {"key": "benchv2", "kind": "window", "root": scratch / "benchv2/store/heldout", "split": "heldout", "manifest": scratch / "benchv2/bench/manifest.json"},
    ]


def selected(st: Mapping[str, Any], rec: Mapping[str, Any]) -> bool:
    from crucible.factory.benchmark import trainable

    if st.get("manifest"):
        return rec["id"] in {t["id"] if isinstance(t, Mapping) else t for t in (jload(Path(st["manifest"])) or {}).get("tasks") or []}
    if st["kind"] == "open":
        return rec.get("stage") == "accepted_source"
    return trainable(rec)


def inventory(a: argparse.Namespace) -> list[dict[str, Any]]:
    from crucible.factory.lanes.windows import WindowStore
    from crucible.factory.lanes.windows_s3 import world_of_version
    from crucible.factory.lanes.windows_world import project_name
    from crucible.factory.versioning import VersionStore

    scratch = Path(a.scratch)
    prior = VersionStore(scratch / "p5b/run/versions.jsonl")
    rows = []
    for st in stores(scratch):
        ws = WindowStore(st["root"], split=st["split"])
        for wid in ws.window_ids():
            rec = ws.window(wid)
            if not selected(st, rec):
                continue
            base = ws.dir(wid) / "base"
            head = world_of_version(rec, prior) if st["kind"] == "window" else rec
            if st["kind"] == "stitched":
                names = [x for x in (project_name(base / "host"), project_name(base / "consumer")) if x]
            else:
                names = [(rec.get("world") or {}).get("project") or project_name(base) or rec["repo"]]
            hv = prior.head(wid, "task") if st["kind"] == "window" else None
            fixes = [f for f in ((hv.content.get("world_fixes") or []) if hv else []) if isinstance(f, Mapping)]
            rep = (fixes[-1].get("repair") or {}) if fixes else {}
            used = head["world"].get("image")
            prev = rec.get("world_previous") or {}
            if (rec.get("world_refresh") or {}).get("phase") == "A":  # refreshed already: the World the solvers used is the previous one
                used = prev.get("image_used") or prev.get("image") or used
            rows.append({"store": st["key"], "kind": st["kind"], "id": wid, "repo": rec["repo"], "a_date": rec.get("a_date"), "b_date": rec.get("b_date"),
                         "image_used": used, "record_image": prev.get("image") or (rec.get("world") or {}).get("image"),
                         "head_version": (head["world"].get("version")), "names": names,
                         "carry_repair": {k: rep.get(k) for k in ("dist", "version")} if rep.get("ok") else None})
    rows.sort(key=lambda r: (r["id"] not in PRIORITY, PRIORITY.index(r["id"]) if r["id"] in PRIORITY else 0, r["kind"] == "open", r["store"], r["id"]))
    for spec in [x for x in (getattr(a, "carry", "") or "").split(",") if x]:  # a World expert's version repair: id=dist==version
        wid, _, pin = spec.partition("=")
        dist, _, ver = pin.partition("==")
        for r in rows:
            if r["id"] == wid:
                r["carry_repair"] = {"dist": dist, "version": ver}
    return rows


_TREES: dict[str, list[tuple[str, Path, str | None]]] = {}


def known_trees(scratch: Path, repo: str) -> list[tuple[str, Path, str | None]]:
    """Every window start tree (A) of ``repo`` the stores hold: (label, base dir, A date)."""
    if not _TREES:
        for root in [scratch / "lhwork/store/train", scratch / "lhwork/store/heldout", scratch / "lhwork/store2/train", scratch / "lhwork/store2/heldout",
                     scratch / "lhwork/store3/train", scratch / "lhwork/store3/heldout", scratch / "benchv2/store/heldout", scratch / "p5/stitch/store/train",
                     scratch / "p5/rebreak/store/train", scratch / "p5/open/store/train"]:
            for f in sorted(root.glob("tasks/*/window.json")):
                try:
                    w = json.loads(f.read_text())
                except ValueError:
                    continue
                if (f.parent / "base").is_dir():  # an open task's A is its perf commit's parent: its date stands in (b_date)
                    _TREES.setdefault(str(w.get("repo")), []).append((f"{root.parent.name}/{root.name}:{w['id']}", f.parent / "base",
                                                                      w.get("a_date") or w.get("b_date")))
    return _TREES.get(repo, [])


# ---- lanes and tasks --------------------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, a: argparse.Namespace):
        from crucible.factory.generators import open as open_mod
        from crucible.factory.generators.stitch import StitchImageBook, stitch_sandboxes
        from crucible.factory.lanes.windows import sandbox_factory
        from crucible.factory.lanes.windows_world import ImageBook
        from crucible.factory.versioning import VersionStore

        open_mod.BENCH_PREFIX, open_mod.TRACE_PREFIX = f"{WORLD_PREFIX}bench-", f"{WORLD_PREFIX}trace-"
        self.a, self.out, self.scratch = a, Path(a.out), Path(a.scratch)
        vp = self.out / "versions.jsonl"
        if not vp.exists():  # P5b's chain (P4's World fixes) copied byte for byte: the leak fixes derive from those heads
            vp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.scratch / "p5b/run/versions.jsonl", vp)
        self.versions = VersionStore(vp)
        self.images = ImageBook(parallel=3)
        meta = jload(self.scratch / "p5/stitch/pairs.json", {}) or {}
        self.stitch_images = StitchImageBook(parallel=2)
        for m in meta.values():
            self.stitch_images.host_versions.update(m.get("host_versions") or {})
        self.s3_sbx = sandbox_factory(parallel=a.s3_parallel, name_prefix=f"{WORLD_PREFIX}s3-", timeout_s=600.0)
        self.s3_stitch = stitch_sandboxes(parallel=a.s3_parallel, name_prefix=f"{WORLD_PREFIX}s3-")
        self.grade = sandbox_factory(parallel=a.grade_parallel, memory="1536m", name_prefix=f"{GRADE_PREFIX}grade-")
        self.grade_stitch = stitch_sandboxes(parallel=a.grade_parallel, name_prefix=f"{GRADE_PREFIX}grade-")
        self.stores = {s["key"]: s for s in stores(self.scratch)}
        self._lanes: dict[str, Any] = {}

    def store(self, key: str) -> Any:
        from crucible.factory.lanes.windows import WindowStore

        st = self.stores[key]
        return WindowStore(st["root"], split=st["split"])

    def lane(self, key: str) -> Any:
        if key in self._lanes:
            return self._lanes[key]
        from crucible.factory.generators.base import GenLane
        from crucible.factory.lanes.windows import WindowLane

        st = self.stores[key]
        ws = self.store(key)
        if st["kind"] == "window":
            lane = WindowLane(ws, sandboxes=self.grade)
        elif st["kind"] == "stitched":
            from crucible.factory.generators.stitch import NAME, StitchTask

            meta = jload(Path(st["pairs"]), {}) or {}
            lane = GenLane(ws, generator=NAME, task_cls=StitchTask, repo_dirs={k: str(v["mono"]) for k, v in meta.items()}, sandboxes=self.grade_stitch)
        elif st["kind"] in ("rebreak", "swarm"):
            if st["kind"] == "rebreak":
                from crucible.factory.generators.rebreak import NAME
                from crucible.factory.generators.rebreak import RebreakTask as Cls
            else:
                from crucible.factory.generators.swarm import NAME  # type: ignore[assignment]
                from crucible.factory.generators.swarm import SwarmTask as Cls
            lane = GenLane(ws, generator=NAME, task_cls=Cls, repo_dirs={p.name: str(p) for p in Path(st["clones"]).iterdir() if p.is_dir()},
                           sandboxes=self.grade)
        elif st["kind"] == "open":
            from crucible.factory.generators.base import Funnel
            from crucible.factory.generators.open import OpenGen
            from crucible.factory.sources.lease import EvalLease

            lane = OpenGen(Path(st["gen_root"]), {}, funnel=Funnel(self.out / "open_funnel.jsonl"), lease=EvalLease.default(None), images=self.images,
                           sandboxes=self.grade)
        else:
            raise ValueError(st["kind"])
        if hasattr(lane, "world_prefix"):
            lane.world_prefix = f"{WORLD_PREFIX}world-"
        self._lanes[key] = lane
        return lane


# ---- the steps ----------------------------------------------------------------------------------------------------------------
async def step_evidence(c: Ctx, row: Mapping[str, Any]) -> dict[str, Any]:
    from crucible.factory.lanes.windows_s3 import compare_installed, identify_copy
    from crucible.factory.lanes.windows_world import image_id, installed_copy, workspace_modules

    ws = c.store(row["store"])
    rec = ws.window(row["id"])
    if "world_previous" in rec and (rec.get("world_refresh") or {}).get("phase") == "A":  # resumed after the record was saved
        rec = {**rec, "world": {**rec["world"], **rec["world_previous"]}}
    base = ws.dir(row["id"]) / "base"
    img = row["image_used"]
    if not img or image_id(img) is None:
        return {"ok": False, "why": f"the World the solvers used is gone: {img}"}
    cp = await installed_copy(img, row["names"], modules=workspace_modules(base), name_prefix=f"{WORLD_PREFIX}copy-")
    if not cp["ok"]:
        return {"ok": False, "why": cp.get("error")}
    cmp = compare_installed(cp["files"], base, rec.get("oracle_files") or {})
    trees = [(lab, p, d) for lab, p, d in known_trees(c.scratch, row["repo"]) if p.resolve() != base.resolve()]
    same_a = identify_copy(cp["files"], {"this_A": base}) if cp["files"] else {"equal": 0, "of": 0, "exact": False}
    if same_a.get("exact"):  # the installed copy is this window's own A: nothing later is reachable
        return {"ok": True, "image": img, "dists": cp["dists"], "installed_py_files": len(cp["files"]), "vs_this_A": same_a, "compare": cmp,
                "identified": {"label": "this_A", "exact": True, "later_than_this_A": False, "at_or_after_B": False}}
    ident = identify_copy(cp["files"], {lab: p for lab, p, _ in trees}) if cp["files"] and trees else {"label": None}
    when = next((d for lab, _p, d in trees if lab == ident.get("label")), None)
    later = None
    if ident.get("exact") and when and row.get("a_date"):
        later = str(when) > str(row["a_date"])
    return {"ok": True, "image": img, "dists": cp["dists"], "installed_py_files": len(cp["files"]), "vs_this_A": same_a, "compare": cmp,
            "identified": {**ident, "a_date": when, "later_than_this_A": later,
                           "at_or_after_B": (str(when) >= str(row["b_date"])) if (ident.get("exact") and when and row.get("b_date")) else None}}


async def step_refresh(c: Ctx, row: Mapping[str, Any]) -> dict[str, Any]:
    from crucible.factory.lanes.windows_s3 import leak_world_fix

    ws = c.store(row["store"])
    wid = row["id"]
    rec = ws.window(wid)
    bdir = c.out / "backup" / row["store"]
    bdir.mkdir(parents=True, exist_ok=True)
    if not (bdir / f"{wid}.window.json").exists():
        shutil.copyfile(ws.dir(wid) / "window.json", bdir / f"{wid}.window.json")
    base = ws.dir(wid) / "base"
    kind = row["kind"]
    images = c.stitch_images if kind == "stitched" else c.images
    sbx = c.s3_stitch if kind == "stitched" else c.s3_sbx
    s3_rec = {**rec, "milestones": [], "tests_b": {}} if kind == "open" else None

    res = await leak_world_fix(rec, base, images=images, versions=c.versions, names=row["names"], current_image=row["image_used"], sandboxes=sbx,
                               name_prefix=WORLD_PREFIX, s3_reruns=c.a.s3_reruns, concurrency=4, carry_repair=row.get("carry_repair"), s3_rec=s3_rec)
    if res["status"] == "accepted":
        fix = res["fix"]
        new = dict(rec)
        new["world_previous"] = {k: (rec.get("world") or {}).get(k) for k in ("image", "digest", "recipe", "leak")}
        if row["image_used"] != (rec.get("world") or {}).get("image"):
            new["world_previous"]["image_used"] = row["image_used"]
        card = res["built"].get("card") or {}
        new["world"] = {**(rec.get("world") or {}), **{k: v for k, v in card.items() if k in ("project", "projects", "recipe", "packaging_hash", "needs")},
                        "image": fix["image"], "digest": fix["digest"], "projects": list(row["names"]), "version": res["version"],
                        "leak": (res.get("s3") or {}).get("leak")}
        new["world_refresh"] = {"ok": True, "phase": "A", "version": res["version"], "proposal": res["proposal"], "at": time.time()}
        ws.save_window(new)
    return res


async def step_s3_before(c: Ctx, row: Mapping[str, Any]) -> dict[str, Any]:
    """S3 on the World the solvers used (the old image), light: A's suite, the leak probes (today's set, so
    ``no_installed_project_source`` fails a leaky image) and the verifier check, with ``--before-reruns`` reruns (their
    bound is not a verdict at that n). It tells a pre-existing S3 failure from one the rebuild caused."""
    from crucible.factory.lanes.windows_s3 import world_s3
    from crucible.factory.lanes.windows_world import image_id

    ws = c.store(row["store"])
    rec = ws.window(row["id"])
    if "world_previous" in rec and (rec.get("world_refresh") or {}).get("phase") == "A":
        rec = {**rec, "world": {**rec["world"], **{k: v for k, v in rec["world_previous"].items() if v is not None}}}
    rec = {**rec, "world": {**(rec.get("world") or {}), "projects": list(row["names"])}}
    if row["kind"] == "open":
        rec = {**rec, "milestones": [], "tests_b": {}}
    img = row["image_used"]
    if not img or image_id(img) is None:
        return {"ok": None, "why": f"the old World is gone: {img}"}
    r = await world_s3(rec, ws.dir(row["id"]) / "base", img, reruns=c.a.before_reruns, concurrency=2,
                       sandboxes=c.s3_stitch if row["kind"] == "stitched" else c.s3_sbx, name_prefix=f"{WORLD_PREFIX}s3b-")
    checks = dict(r.get("checks") or {})
    core = {k: v for k, v in checks.items() if k != "reruns_clean"}
    return {"ok_except_reruns": all(core.values()), "checks": checks, "leak": (r.get("leak") or {}).get("probes"),
            "existing": {k: (r.get("existing_tests") or {}).get(k) for k in ("counts", "why", "exit")}, "reruns": r.get("reruns"),
            "grading": r.get("grading")}


async def step_gates(c: Ctx, row: Mapping[str, Any]) -> dict[str, Any]:
    kind = row["kind"]
    lane = c.lane(row["store"])
    ws = c.store(row["store"])
    rec = ws.window(row["id"])
    if kind == "window":
        from crucible.factory.benchmark import GateConfig, run_gates, unauthored_task

        task = unauthored_task(lane, rec)
        return await run_gates(task, rec, cfg=GateConfig(gates=("s6", "s5", "s8", "s10")))
    if kind == "open":
        from crucible.factory.generators.open import open_gates

        task = lane.task(row["id"])
        return await open_gates(task, gates_run=("S5", "S8", "S10"), s3_prefix=f"{WORLD_PREFIX}s3-")
    from crucible.factory.generators.base import GateConfig as GC
    from crucible.factory.generators.base import deterministic_gates

    lane._made.clear()
    task = lane.task(row["id"])
    return await deterministic_gates(task, cfg=GC(gates=("S6", "S5", "S8", "S10"), stop_on_fail=False), s3_prefix=f"{WORLD_PREFIX}s3-")


async def one(c: Ctx, row: Mapping[str, Any], steps: tuple[str, ...], redo: tuple[str, ...]) -> dict[str, Any]:
    jp = c.out / "tasks" / f"{row['store']}__{row['id']}.json"
    j = jload(jp, {}) or {}
    j.update(row=dict(row))
    for name, fn in (("evidence", step_evidence), ("s3_before", step_s3_before), ("refresh", step_refresh), ("gates", step_gates)):
        if name not in steps or (name in j and name not in redo):
            continue
        if name == "gates" and (j.get("refresh") or {}).get("status") not in ("accepted",) and not c.a.gate_unrefreshed:
            continue
        t0 = time.monotonic()
        try:
            res = await fn(c, row)
        except Exception as e:  # noqa: BLE001 - one task's failure is recorded, never the run's
            import traceback

            res = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400], "trace": traceback.format_exc()[-1500:]}
        res["seconds"] = round(time.monotonic() - t0, 1)
        j[name] = res
        jsave(jp, j)
        brief = {"evidence": lambda r: (r.get("installed_py_files"), (r.get("compare") or {}).get("solution_present"), (r.get("identified") or {}).get("label")),
                 "s3_before": lambda r: (r.get("ok_except_reruns"), r.get("leak")),
                 "refresh": lambda r: (r.get("status"), (r.get("s3") or {}).get("ok"), r.get("version")),
                 "gates": lambda r: (r.get("ok"), r.get("passed"))}[name](res)
        print(json.dumps({"task": row["id"], "step": name, "s": res["seconds"], "brief": brief, "error": res.get("error")}, default=str), flush=True)
    return j


async def cmd_run(a: argparse.Namespace) -> None:
    c = Ctx(a)
    rows = inventory(a)
    jsave(c.out / "inventory.json", rows)
    if a.only:
        rows = [r for r in rows if r["id"] in a.only.split(",")]
    if a.stores:
        rows = [r for r in rows if r["store"] in a.stores.split(",")]
    steps = tuple(a.steps.split(","))
    redo = tuple(x for x in a.redo.split(",") if x)
    sem = asyncio.Semaphore(a.parallel)
    opens = [r for r in rows if r["kind"] == "open"]
    rest = [r for r in rows if r["kind"] != "open"]

    async def gated(r: Mapping[str, Any]) -> None:
        async with sem:
            await one(c, r, steps, redo)

    await asyncio.gather(*(gated(r) for r in rest))
    for r in opens:  # timing-measured tasks alone (S10 compares speed: no other task of this run meanwhile)
        await one(c, r, steps, redo)
    print(json.dumps({"done": len(rows)}), flush=True)


# ---- the report ----------------------------------------------------------------------------------------------------------------
def recorded_before(scratch: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    """The gate verdicts recorded before Phase A (P2/P4/P5a/P5b/P6/benchmark v2), per gate where any exists."""
    st, wid = row["store"], row["id"]
    out: dict[str, Any] = {}
    if st in ("p5-rebreak", "p5-stitch", "p5-open", "p5b-swarm"):
        root = {"p5-rebreak": "p5/rebreak/store/train", "p5-stitch": "p5/stitch/store/train", "p5-open": "p5/open/store/train",
                "p5b-swarm": "p5b/run/swarm/store/train"}[st]
        g = jload(scratch / root / "tasks" / wid / "gates.json") or {}
        out = {k: v.get("ok") for k, v in (g.get("gates") or {}).items()}
        out["source"] = "P5a/P5b gates.json"
    elif st == "benchv2":
        g = jload(scratch / "benchv2/bench/gates" / f"{wid}.json") or {}
        out = {k.upper(): v.get("ok") for k, v in g.items() if isinstance(v, Mapping) and "ok" in v}
        out["source"] = "benchmark v2 gates"
    elif st == "store3-heldout":
        g = (jload(scratch / "p6bench/gates" / f"{wid}.json") or {}).get("gates") or {}
        out = {k.upper(): v for k, v in g.items() if k.startswith("s") and isinstance(v, bool)}
        out["source"] = "P6 gates" if g else "none recorded"
    else:
        res = jload(scratch / "lhwork/p2/windows" / wid / "results.json") or {}
        m = (res.get("machine") or {}) if isinstance(res, Mapping) else {}
        if m:  # P2's gate machine: S5 solvability, S8 ground truth (failed rows), S9 (verifier-1 defects)
            out["S5"] = bool(m.get("s5"))
            out["S8"] = not m.get("s8_failed")
            out["S9"] = not (m.get("s9") or {}).get("verifier1")
            out["P2_machine"] = m.get("status")
        w = jload(scratch / "p4/run/world" / f"{wid}.json") or {}
        if w.get("s3_after"):
            out["S3"] = (w["s3_after"] or {}).get("ok")
        out["source"] = "P2 machine" + (", P4 S3" if w else "")
    return out


def cmd_report(a: argparse.Namespace) -> dict[str, Any]:
    out, scratch = Path(a.out), Path(a.scratch)
    rows = []
    for jp in sorted((out / "tasks").glob("*.json")):
        j = json.loads(jp.read_text())
        row, ev, rf, g = j["row"], j.get("evidence") or {}, j.get("refresh") or {}, j.get("gates") or {}
        cmp = ev.get("compare") or {}
        ident = dict(ev.get("identified") or {})
        if (ev.get("vs_this_A") or {}).get("exact"):  # the installed copy is this window's own A (evidence of an older run kept the other label)
            ident = {"label": "this_A", "exact": True, "later_than_this_A": False, "at_or_after_B": False}
        elif ident.get("exact") and ident.get("later_than_this_A") is None:
            when = next((d for lab, _p, d in known_trees(scratch, row["repo"]) if lab == ident.get("label")), None)
            if when and row.get("a_date"):
                ident["later_than_this_A"] = str(when) > str(row["a_date"])
        before = recorded_before(scratch, row)
        rec_ok = [v for k, v in before.items() if k in ("S3", "S5", "S6", "S8", "S10") and v is not None]  # the gates Phase A re-runs
        leaky = (ev.get("installed_py_files") or 0) > 0
        g2 = g.get("gates") if isinstance(g.get("gates"), Mapping) else g  # generator gates nest them; benchmark.run_gates does not
        gates_after = {k.upper(): (v.get("ok") if isinstance(v, Mapping) else None) for k, v in g2.items() if isinstance(v, Mapping) and "ok" in v}
        if rf.get("status") == "accepted":
            gates_after["S3"] = (rf.get("s3") or {}).get("ok")
        elif rf.get("status") == "rejected" and not leaky:
            gates_after["S3"] = "unchanged World (no installed source)"
        after_ok = None
        if rf.get("status") == "accepted" and g:
            after_ok = all(v is True for k, v in gates_after.items() if not (k == "S6" and row["kind"] == "open")) and gates_after.get("S3") is True
        sb = j.get("s3_before") or {}
        s3_old_core = sb.get("ok_except_reruns")
        s3_old_no_leak = None
        if sb.get("checks"):  # A's suite and the verifier on the old World, the leak probe set aside
            s3_old_no_leak = all(v for k, v in sb["checks"].items() if k not in ("reruns_clean", "leak_probes_hold")) and all(
                v in ("pass", "n/a") for k, v in (sb.get("leak") or {}).items() if k != "no_installed_project_source")
        rows.append({
            "s3_before": {"core_ok_with_todays_probes": s3_old_core, "ok_without_the_new_probe": s3_old_no_leak,
                          "new_probe": (sb.get("leak") or {}).get("no_installed_project_source"), "existing": sb.get("existing")},
            "store": row["store"], "id": row["id"], "kind": row["kind"], "image_used": row["image_used"],
            "installed_py_files": ev.get("installed_py_files"), "installed_dists": ev.get("dists"),
            "installed_vs_this_A": {k: (ev.get("vs_this_A") or {}).get(k) for k in ("equal", "of", "exact")},
            "installed_classes": cmp.get("classes"), "oracle_added_lines": cmp.get("added_lines"), "oracle_added_present": cmp.get("added_present"),
            "solution_present": cmp.get("solution_present"), "changed_files": cmp.get("changed"),
            "identified_as": ident.get("label"), "identified_exact": ident.get("exact"),
            "identified_later_than_A": ident.get("later_than_this_A"), "identified_at_or_after_B": ident.get("at_or_after_B"),
            "refresh": rf.get("status"), "refresh_reasons": rf.get("reasons"), "new_image": (rf.get("fix") or {}).get("image"), "version": rf.get("version"),
            "s3_after": {k: (rf.get("s3") or {}).get(k) for k in ("ok", "why")}, "s3_leak_after": ((rf.get("s3") or {}).get("leak") or {}).get("probes"),
            "before_recorded": before, "before_recorded_ok": all(rec_ok) if rec_ok else None,
            "before_current_probes_ok": (all(rec_ok) if rec_ok else None) if not leaky else False,
            "s3_regressed_by_rebuild": (s3_old_no_leak is True and rf.get("status") == "rejected" and leaky) or None,
            "after": gates_after, "after_ok": after_ok, "errors": {k: (j.get(k) or {}).get("error") for k in ("evidence", "refresh", "gates") if (j.get(k) or {}).get("error")},
        })
    changed = [r for r in rows if r["before_recorded_ok"] is not None and r["after_ok"] is not None and r["before_recorded_ok"] != r["after_ok"]]
    vs = None
    try:
        from crucible.factory.versioning import VersionStore

        vs = VersionStore(out / "versions.jsonl").verify()
    except Exception as e:  # noqa: BLE001
        vs = [str(e)]
    rep = {
        "phase": "A", "task": "1: every World rebuilt with the site-packages leak fix", "spend_usd": 0.0, "tasks": len(rows),
        "by_store": {s: sum(1 for r in rows if r["store"] == s) for s in sorted({r["store"] for r in rows})},
        "leaky_before": sum(1 for r in rows if (r["installed_py_files"] or 0) > 0),
        "installed_copy_is_another_A": sum(1 for r in rows if (r["installed_py_files"] or 0) > 0 and not (r["installed_vs_this_A"] or {}).get("exact")),
        "solution_present": [r["id"] for r in rows if r["solution_present"]],
        "identified_later_A": [r["id"] for r in rows if r["identified_later_than_A"]],
        "refreshed": sum(1 for r in rows if r["refresh"] == "accepted"), "refresh_status": {str(s): sum(1 for r in rows if r["refresh"] == s) for s in {r["refresh"] for r in rows}},
        "s3_after_ok": sum(1 for r in rows if (r["s3_after"] or {}).get("ok")),
        "s3_failing_before_and_after": [r["id"] for r in rows if r["refresh"] == "rejected" and (r["installed_py_files"] or 0) > 0
                                        and r["s3_before"]["ok_without_the_new_probe"] is False],
        "s3_regressed_by_rebuild": [r["id"] for r in rows if r["s3_regressed_by_rebuild"]],
        "after_ok": sum(1 for r in rows if r["after_ok"]), "verdict_changed_vs_recorded": [(r["id"], r["before_recorded_ok"], r["after_ok"]) for r in changed],
        "versions_verify": vs, "rows": rows,
    }
    dest = Path(a.report_to) if a.report_to else out / "phase_a_worlds.json"
    jsave(dest, rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, default=str, indent=1))
    return rep


# ---- task 2: P5b's "too easy" windows again, on the refreshed Worlds (paid; one USD ledger, hard cap) -------------------------
P5B_SPECS = {"st-glom--boltons-2ae0bb57": "stitched:st-glom--boltons-2ae0bb57=m12c+m12@locate",
             "st-glom--boltons-30eb7ffe": "stitched:st-glom--boltons-30eb7ffe=m12c+m12@locate",
             "rb-more-itertools-061d3f39": "rebreak:rb-more-itertools-061d3f39=m04@locate",
             "sw-python-semver-a2e121df": "swarm:sw-python-semver-a2e121df=m03@prerequisites",
             "sw-cachetools-aadb387d": "swarm:sw-cachetools-aadb387d=m02@prerequisites",
             "sw-more-itertools-dedc22c6": "swarm:sw-more-itertools-dedc22c6=m03@prerequisites"}


def target_ns(a: argparse.Namespace, windows: str) -> argparse.Namespace:
    """``scripts/lh_swarm.py``'s arguments for this phase's ledger (``<out>/target/spend.jsonl``, ``--cap``) and stores."""
    import lh_swarm

    scratch, out = Path(a.scratch), Path(a.out) / "target"
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "frontier.json").exists():  # P5b's choice, unchanged: the same frontier model
        shutil.copyfile(scratch / "p5b/run/frontier.json", out / "frontier.json")
    if not (out / "swarm").exists():  # the swarm-built windows' store (read; their records carry the refreshed Worlds)
        (out / "swarm").symlink_to(scratch / "p5b/run/swarm")
    lh_swarm.GRADE_PREFIX, lh_swarm.WORLD_PREFIX = GRADE_PREFIX, WORLD_PREFIX
    return lh_swarm.parse_args(["target", "--out", str(out), "--store", str(scratch / "lhwork/store3/train"), "--clones", str(scratch / "lhrepos"),
                                "--p5", str(scratch / "p5"), "--prompts", str(scratch / "lhwork/p2/windows"), "--cap", str(a.cap), "--slack", str(a.slack),
                                "--attempts", str(a.attempts), "--windows", windows, "--concurrency", "4", "--parallel", "6"])


async def cmd_retarget(a: argparse.Namespace) -> None:
    """P5b's procedure (``lh_swarm``: same frontier, tool limit 150, checkpoint sidecar, ``targeting``), staged: the pilot
    episode first (its extrapolation written before anything scales), then per window the scripted probe, the probe
    episode and the full arm; the simplified arm runs only when the full arm is below ceiling (at 4/4 the counterfactual
    cannot reject, so the old verdict stands by construction). Windows in ``--windows`` order (the ones whose
    site-packages held a later version first); a window whose estimate does not fit under the cap is skipped."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import lh_run
    import lh_swarm

    from crucible.factory.longtask import ground_truth
    from crucible.factory.targeting import (
        PROBE_PROMPT,
        TargetEvidence,
        band,
        counterfactual,
        outcomes_of,
        prerequisite_probe,
        probe_episode,
        scripted_probe,
        target_verdict,
    )

    ids = [w for w in a.windows.split(",") if w]
    specs = [P5B_SPECS[w] for w in ids]
    ns = target_ns(a, ",".join(specs))
    out = Path(ns.out)
    old = (jload(Path(a.scratch) / "p5b/run/p5b_report.json") or {}).get("target") or {}
    if not (out / "pilot.json").exists():
        ns.windows = specs[0]
        await lh_swarm.stage_pilot(ns)
        print(json.dumps({"pilot_written": str(out / "pilot.json")}), flush=True)
        if a.pilot_only:
            return
    pilot = jload(out / "pilot.json")
    run = lh_swarm.p5b_run(ns)
    model = lh_swarm.frontier_id(ns)
    cfg = lh_swarm.episode_cfg(ns)
    pol = lh_run.policy_ref(run, model, cfg)
    n = a.attempts
    for wid, raw in zip(ids, specs, strict=True):
        spec = lh_swarm.target_spec(raw)
        jp = out / "target" / f"{wid}.json"
        j = jload(jp, {}) or {}
        task = lh_swarm.build_task(run, ns, spec)
        targets = spec["targets"] or task.target_milestones()
        rec = task.window
        j.update(window=wid, kind=spec["kind"], targets=targets, simplify=spec["simplify"], world=rec["world"].get("image"),
                 world_previous=(rec.get("world_previous") or {}).get("image"), world_refresh=rec.get("world_refresh"))
        if (rec.get("world_refresh") or {}).get("phase") != "A":
            j["skipped"] = "the World is not refreshed yet"
            jsave(jp, j)
            print(json.dumps({"window": wid, "skipped": j["skipped"]}), flush=True)
            continue
        p5b = old.get(wid) or {}
        est = float(p5b.get("usd") or 1.0) / 2 * (n / 4)  # P5b's per-window cost, full arm only
        if "arms_full" not in j and run.spent() + est > a.cap - a.slack:
            j["skipped"] = f"estimate {est:.2f} USD does not fit: spent {run.spent():.3f} of {a.cap} (slack {a.slack})"
            jsave(jp, j)
            print(json.dumps({"window": wid, "skipped": j["skipped"]}), flush=True)
            continue
        if "truths" not in (j.get("probe_scripted") or {}):
            j["probe_scripted"] = await scripted_probe(task, out / "probe" / wid, command=lh_swarm.probe_command(task))
            jsave(jp, j)
        if "probe_episode" not in j:
            pt = lh_swarm.build_task(run, ns, spec, prompt={"prompt": PROBE_PROMPT, "requirements": [], "spans": {}}, tag="~probe")
            pcfg = lh_swarm.episode_cfg(ns, tool_limit=12, max_steps=20)
            await lh_run.run_episodes(run, f"probe-{wid}", [pt], [lh_run.policy_ref(run, model, pcfg)], k=1, concurrency=1, stage="probe", role="probe",
                                      cfg=pcfg, checkpoints=False)
            for e in lh_run.episodes_of(run, f"probe-{wid}"):
                calls, last = __import__("lh_target").episode_trail(Path(e["src_dir"]))
                j["probe_episode"] = {**probe_episode(calls, last, j["probe_scripted"].get("truths")), "usd": e["usd"], "stop": e["stop"], "masked": e["masked"]}
            jsave(jp, j)
        await lh_run.run_episodes(run, f"full-{wid}", [task], [pol], k=n, concurrency=a.solver_concurrency, stage="arms", role="solver", cfg=cfg, checkpoints=True)
        full_eps = [e for e in lh_run.episodes_of(run, f"full-{wid}") if e["task"] == task.task_id]
        if (pilot or {}).get("window", {}).get("id") == wid:
            full_eps += [e for e in lh_run.episodes_of(run, f"pilot-{wid}") if e["task"] == task.task_id]
        fo = outcomes_of(full_eps, task, targets, targets)
        j["arms_full"] = [vars(o) for o in fo]
        j["episodes"] = {"full": [lh_swarm._ep_row(e) for e in full_eps]}
        succ = sum(1 for o in fo if o.success and not o.masked)
        live = sum(1 for o in fo if not o.masked)
        cf = None
        if live and succ < live:  # below ceiling: the simplified arm decides attribution (P5b's procedure)
            simple = lh_swarm.simplify_task(task, spec["simplify"], targets=targets, root=out / "simplified")
            gt = await ground_truth(simple)
            j["s8_simple"] = {"ok": gt["ok"], "failed": gt["failed"]}
            if gt["ok"]:
                await lh_run.run_episodes(run, f"simple-{wid}", [simple], [pol], k=n, concurrency=a.solver_concurrency, stage="arms", role="solver", cfg=cfg, checkpoints=True)
                simple_eps = [e for e in lh_run.episodes_of(run, f"simple-{wid}") if e["task"] == simple.task_id]
                so = outcomes_of(simple_eps, simple, targets, targets)
                j["arms_simple"] = [vars(o) for o in so]
                j["episodes"]["simple"] = [lh_swarm._ep_row(e) for e in simple_eps]
                cf = counterfactual(fo, so)
        j["counterfactual"] = cf or {"verdict": "not_run", "why": f"full arm at ceiling ({succ}/{live}): no simplified arm can show a gain"}
        j["band"] = band([o.reward for o in fo if not o.masked])
        j["probe"] = prerequisite_probe(j["probe_scripted"], lh_swarm.two_repo_probe(j.get("probe_episode"), task))
        j["phi_full"] = lh_swarm._mean([o.phi for o in fo if not o.masked])
        j["usd"] = round(sum(e["usd"] for eps in j["episodes"].values() for e in eps), 5)
        j["verdict"] = target_verdict(TargetEvidence(task=wid, band=j["band"], counterfactual=cf, probe=j["probe"], paraphrase=None))
        j["old"] = {"full": (p5b.get("full")), "simple": p5b.get("simple"), "verdict": p5b.get("verdict"), "counterfactual": (p5b.get("counterfactual") or {}).get("verdict"),
                    "band": (p5b.get("band") or {}).get("in_band"), "phi": p5b.get("phi"), "usd": p5b.get("usd"), "world": (p5b.get("episodes") or {}) and None}
        j["new"] = {"full": f"{succ}/{live}", "simple": (f"{cf.get('success_simple')}/{cf.get('n_simple')}" if cf else "not run"),
                    "too_easy": live > 0 and succ == live and all((o.reward or 0) >= 1.0 for o in fo if not o.masked),
                    "route": j["verdict"].get("route"), "band": j["band"].get("in_band")}
        jsave(jp, j)
        print(json.dumps({"window": wid, "new": j["new"], "usd": j["usd"], "spent": round(run.spent(), 4)}), flush=True)
    rows = {p.stem: jload(p) for p in sorted((out / "target").glob("*.json"))}
    rep = {"phase": "A", "task": "2: P5b's too-easy windows on the refreshed Worlds", "cap_usd": a.cap, "frontier": jload(out / "frontier.json"),
           "pilot": {k: (pilot or {}).get(k) for k in ("model", "window", "usd", "extrapolation")}, "spend_total": round(run.spent(), 5),
           "spend_by_role": lh_swarm.spend_by_role(out),
           "windows": {w: {k: r.get(k) for k in ("kind", "targets", "simplify", "world_previous", "world", "old", "new", "counterfactual", "band", "probe",
                                                   "verdict", "usd", "skipped", "phi_full")} for w, r in rows.items()}}
    jsave(Path(a.report_to) if a.report_to else out / "phase_a_retarget.json", rep)
    print(json.dumps({"spend_total": rep["spend_total"]}), flush=True)


# ---- task 5: the auditor's forced final-verdict turn, measured (AuditorHarness.refine; paid, one ledger, hard cap) ------------
def capped_trails(scratch: Path, n: int) -> list[str]:
    """P5b trails whose first auditor session ended at the 24-turn cap with no verdict, one per window first, shortest first."""
    rows = []
    for f in sorted((scratch / "p5b/run/audit/auditor").glob("*/session.json")):
        r = json.loads(f.read_text())
        if str(r.get("stop") or "").startswith("budget:turns") and int(r.get("turns") or 0) == 24 and not r.get("finished"):
            ctx = f.parent / "context.txt"
            rows.append((r["trace"], ctx.stat().st_size if ctx.exists() else 10**9))
    seen: dict[str, int] = {}
    for tid, size in sorted(dict(rows).items(), key=lambda x: x[1]):
        seen.setdefault(tid, size)
    by_window: dict[str, list[str]] = {}
    for tid in sorted(seen, key=lambda t: seen[t]):
        by_window.setdefault(tid.split("/", 1)[0], []).append(tid)
    out: list[str] = []
    while len(out) < n and any(by_window.values()):
        for w in sorted(by_window):
            if by_window[w] and len(out) < n:
                out.append(by_window[w].pop(0))
    return out


async def cmd_auditor_ab(a: argparse.Namespace) -> None:
    """The forced final-verdict turn through ``AuditorHarness.refine`` (``HarnessState.evaluate``): base = the auditor's
    harness without the switch, candidate = with ``FINAL_VERDICT_EDIT``; per seed one batch of P5b trails whose P3-capped
    (24-turn) session gave no verdict plus one seeded control; interleaved arms, alternating order, a noise re-run of the
    base. Primary ``label_rate`` (validated verdicts on the trails), guard ``seeded_accuracy``. The deepseek auditor and
    P3's configuration (24 turns, 48k window, temperature 0) unchanged; one ledger capped at ``--auditor-cap``."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import lh_audit
    import lh_run

    from crucible.factory.audit import AuditConfig, audit
    from crucible.factory.models import parse_spec
    from crucible.factory.traceaudit import FINAL_VERDICT_EDIT, AuditorHarness, P3Launcher, RLMAuditor, facts, first_wrong_step, load_bundle, switch_on, trace_of

    scratch, out = Path(a.scratch), Path(a.out) / a.ab_dir
    out.mkdir(parents=True, exist_ok=True)
    run = lh_run.Run(argparse.Namespace(work=str(out), out=str(out / "lh_out"), cap=a.auditor_cap, store=str(scratch / "lhwork/store3/train"),
                                        clones=str(scratch / "lhrepos"), parallel=2, windows="", routes=False))
    spec = parse_spec("openrouter:deepseek/deepseek-v4-flash")
    client = run.solver_client(spec.ref.split(":", 1)[-1], tag="auditor")
    seeds = list(range(1 + a.ab_offset, a.ab_seeds + 1 + a.ab_offset))
    trails = capped_trails(scratch, a.ab_per_seed * (len(seeds) + a.ab_offset))[a.ab_per_seed * a.ab_offset:]
    bundles = []
    for tid in trails:
        rname, sample, att = tid.split("/")
        b = load_bundle(scratch / "p5b/run/runs" / rname / "samples" / sample / att, run=rname)
        bundles.append(b)
    traces = [trace_of(b, facts(b), first_wrong_step(b, [])) for b in bundles]
    batches = {sd: traces[i * a.ab_per_seed:(i + 1) * a.ab_per_seed] for i, sd in enumerate(seeds)}
    ctl = lh_audit.controls(Path(__file__).resolve().parents[1] / "docs/research/task-factory/runs/controls.json")
    harness = AuditorHarness(out / "auditor_harness.json")
    rows: list[dict[str, Any]] = []

    async def fn(entries: dict[str, Any], seed: int) -> dict[str, float]:
        arm = "candidate" if switch_on(entries) else "base"
        k = sum(1 for r in rows if r["arm"] == arm and r["seed"] == seed)
        aud = RLMAuditor(client, model=spec.model, workdir=out / "sessions" / f"{arm}-{seed}-{k}", launcher=lambda: P3Launcher(prefix=f"{GRADE_PREFIX}audit-"),
                         config=AuditConfig(concurrency=3),
                         harness=harness, entries=entries)
        async with run.spend("audit_ab", "auditor", f"{arm}-{seed}-{k}"):
            rep = await audit(batches[seed], aud, prompt="(each trace's CONTEXT holds its own request)", seeded=[ctl[seed % len(ctl)]], salt=f"pa-ab-{seed}")
        labs = [rep.labels[t.id] for t in batches[seed]]
        m = {"label_rate": sum(1 for lab in labs if lab.status == "labelled") / len(labs),
             "seeded_accuracy": sum(1 for r in rep.seeded if r["ok"]) / len(rep.seeded)}
        rows.append({"arm": arm, "seed": seed, "run": k, **m, "labels": {t.id: {"status": lab.status, "cause": lab.cause, "why": lab.why[:200]}
                                                                          for t, lab in zip(batches[seed], labs, strict=True)},
                     "seeded": rep.seeded, "sessions": aud.sessions, "spent": round(run.spent(), 5)})
        jsave(out / "rows.json", rows)
        print(json.dumps({"arm": arm, "seed": seed, **m, "spent": round(run.spent(), 4)}), flush=True)
        return m

    ev = await harness.refine([FINAL_VERDICT_EDIT], fn, seeds, primary="label_rate", guards=("seeded_accuracy",),
                              trigger="Phase A: forced final-verdict turn",
                              evidence="P3: 39 of 120 sessions ended at the 24-turn cap with no verdict; P5b: 30 of 41 first sessions")
    sess = [s for r in rows for s in r["sessions"]]
    capped = [s for s in sess if str(s.get("stop") or "").startswith("budget:turns")]
    rep = {"phase": "A", "task": "5: auditor forced final-verdict turn", "cap_usd": a.auditor_cap, "spend_total": round(run.spent(), 5),
           "trails": trails, "seeds": seeds, "event": ev, "accepted": ev["accepted"], "harness_version": harness.version,
           "verdict_rate": {"base": ev["metrics"]["label_rate"]["base"], "candidate": ev["metrics"]["label_rate"]["candidate"]},
           "sessions": {"n": len(sess), "capped": len(capped), "forced_turns": sum(1 for s in capped if s.get("final_turn")),
                        "forced_verdicts": sum(1 for s in capped if (s.get("final_turn") or {}).get("verdict"))}, "rows": rows}
    jsave(Path(a.report_to) if a.report_to else out / "phase_a_auditor.json", rep)
    print(json.dumps({k: rep[k] for k in ("accepted", "verdict_rate", "sessions", "spend_total")}, default=str), flush=True)


# ---- task 3: S4.5 on every stored swarm-built window, in its World ($0) -----------------------------------------------------
async def cmd_oracle_plan(a: argparse.Namespace) -> None:
    """``oracleplan.check`` on each swarm-built window P5b stored (accepted or not, wherever a plan and an oracle exist), in
    the window's World (``world_runner``); the tests that contradict a planned removal are run on the oracle there."""
    from crucible.factory.lanes.windows import WindowStore, run_tests, sandbox_factory
    from crucible.factory.lanes.windows_s3 import WindowWorldPort
    from crucible.factory.oracleplan import check, world_runner

    scratch = Path(a.scratch)
    ws = WindowStore(scratch / "p5b/run/swarm/store/train", split="train")
    sbx = sandbox_factory(parallel=4, name_prefix=f"{GRADE_PREFIX}op-")
    rows: dict[str, Any] = {}
    for wid in ws.window_ids():
        rec = ws.window(wid)
        plan = (rec.get("swarm_window") or {}).get("plan")
        if not plan or not rec.get("oracle_files"):
            continue
        base = ws.dir(wid) / "base"
        image = rec["world"]["image"]
        port = WindowWorldPort(rec, base, image=image, name_prefix=WORLD_PREFIX)
        p2p = list(rec.get("p2p") or [])
        first = await check(plan, base, rec["oracle_files"], runner=world_runner(port), regression_ids=p2p)
        on_oracle: dict[str, str] = {}
        if first.get("contradicting"):
            ids = sorted(first["contradicting"])
            tr = await run_tests(sbx(image), base, dict(rec["oracle_files"]), {}, sorted({t.split("::", 1)[0] for t in ids}))
            on_oracle = {t: tr.results.get(t, "missing") for t in ids}
        res = await check(plan, base, rec["oracle_files"], runner=world_runner(port), regression_ids=p2p, regression_on_oracle=on_oracle)
        rows[wid] = {"stage": rec.get("stage"), "trainable": rec.get("stage") == "mutants_valid" and rec.get("independent") is False, "world": image,
                     "ok": res["ok"], "owner": res.get("owner"), "findings": res["findings"],
                     "probes": [{k: r.get(k) for k in ("milestone", "kind", "name", "verdict", "at_a", "at_b")} for r in res["probes"]],
                     "contradicting": len(res["contradicting"]), "contradicting_pass_on_oracle": sum(1 for v in on_oracle.values() if v == "passed"),
                     "p2p_before": len(p2p), "p2p_after_drop": len(set(p2p) - set(res["drop_from_regression"]))}
        print(json.dumps({"window": wid, "ok": res["ok"], "findings": res["findings"][:2], "contradicting": rows[wid]["contradicting"]}), flush=True)
    rep = {"phase": "A", "task": "3: S4.5 executed oracle-vs-plan check on the stored swarm-built windows", "spend_usd": 0.0, "windows": rows,
           "caught": sorted(w for w, r in rows.items() if not r["ok"])}
    jsave(Path(a.report_to) if a.report_to else Path(a.out) / "phase_a_oracleplan.json", rep)


def cmd_cleanup(_a: argparse.Namespace) -> None:
    r = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True, check=False)
    mine = [n for n in r.stdout.split() if n.startswith((GRADE_PREFIX, WORLD_PREFIX))]
    if mine:
        subprocess.run(["docker", "rm", "-f", *mine], capture_output=True, check=False)
    print(json.dumps({"removed": len(mine)}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["inventory", "run", "report", "cleanup", "retarget", "auditor-ab", "oracle-plan"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=str(SCRATCH))
    ap.add_argument("--only", default="")
    ap.add_argument("--stores", default="")
    ap.add_argument("--steps", default="evidence,refresh,gates")
    ap.add_argument("--redo", default="")
    ap.add_argument("--parallel", type=int, default=2, help="tasks at once")
    ap.add_argument("--s3-parallel", type=int, default=6)
    ap.add_argument("--grade-parallel", type=int, default=6)
    ap.add_argument("--s3-reruns", type=int, default=60)
    ap.add_argument("--gate-unrefreshed", action="store_true", help="gate tasks whose World needed no fix too")
    ap.add_argument("--before-reruns", type=int, default=4, help="s3_before: reruns of A's suite on the old World")
    ap.add_argument("--carry", default="", help="run: version repairs layered on a rebuilt World, id=dist==version,...")
    ap.add_argument("--report-to", default="")
    ap.add_argument("--windows", default=",".join(PRIORITY), help="retarget: window ids, in priority order")
    ap.add_argument("--cap", type=float, default=8.0, help="retarget: hard USD cap of the phase's ledger")
    ap.add_argument("--slack", type=float, default=0.3)
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--pilot-only", action="store_true")
    ap.add_argument("--solver-concurrency", type=int, default=4, help="retarget: episodes at once (OpenRouter refuses past the in-flight credit)")
    ap.add_argument("--auditor-cap", type=float, default=1.0, help="auditor-ab: hard USD cap")
    ap.add_argument("--ab-seeds", type=int, default=3)
    ap.add_argument("--ab-per-seed", type=int, default=2)
    ap.add_argument("--ab-dir", default="auditor_ab")
    ap.add_argument("--ab-offset", type=int, default=0, help="auditor-ab: skip this many seeds' trails (a replication on fresh trails)")
    a = ap.parse_args()
    if a.cmd == "inventory":
        rows = inventory(a)
        jsave(Path(a.out) / "inventory.json", rows)
        print(json.dumps({"tasks": len(rows), "by_store": {s: sum(1 for r in rows if r["store"] == s) for s in sorted({r["store"] for r in rows})}}))
        return
    if a.cmd == "report":
        cmd_report(a)
        return
    if a.cmd == "cleanup":
        cmd_cleanup(a)
        return
    asyncio.run({"retarget": cmd_retarget, "auditor-ab": cmd_auditor_ab, "oracle-plan": cmd_oracle_plan}.get(a.cmd, cmd_run)(a))


if __name__ == "__main__":
    sys.exit(main())
