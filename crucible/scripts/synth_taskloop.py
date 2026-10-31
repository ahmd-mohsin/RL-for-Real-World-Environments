#!/usr/bin/env python3
"""Denser Worlds for the held-out-blocked tasks and the first real run of the per-task loop (DESIGN §7.7 item 23).

    synth_taskloop.py densify  --manifest <recert-1 manifest> --tag pool-2 --workers 6   # Part A: levels measured per task (CPU)
    synth_recert.py   det      --run <pool-2 run> --max-mutants 480 --workers 6          # full re-certification, phase 1 (CPU)
    synth_recert.py   paid     --run <pool-2 run> --cap 2                                # phase 2 (panels, ALT, ATK, LEAK)
    synth_recert.py   report   --run <pool-2 run> --tag pool-2 --publish                 # certificates beside the old ones
    synth_taskloop.py manifest --base <recert-1 manifest> --run <pool-2 run>             # the updated manifest (every row)
    synth_taskloop.py loop     --manifest <updated manifest> --cap 18 --tag taskloop-1   # Part B: TaskLoop on the pool
    synth_taskloop.py report   --run <loop run>

``densify`` re-instantiates every task the re-certification left provisional only because of the held-out share (status
provisional, ``status_h0`` certified) on a denser superset World of its family (``synth.densify``; finance_ap:
``synth_wire.finance_ap_denser``), measuring MUT's shown and held-out splits per level, and picks the smallest level whose
shown split reaches 59 distinct effects. It also gives every task whose id another source shares a unique id (the
loop's archive and the student's seeds are keyed by task id; ``unique_ids``). Both kinds get a full re-certification
in a recert run directory (``items.json`` with ``task_file``), with the archives copied (records of a renamed task
re-keyed). Nothing is edited: instruction, intent card, contract and reference program stay as they are.

Runs and receipts sit under ``<data-root>/recert/`` (densify, re-certification) and ``<data-root>/taskloop/`` (the loop).
The shared Part 2 total ``MILESTONE_CAP_USD`` (150) is checked before every paid run. No Qwen model outside the
student probe; the key comes from the environment only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1]
if str(HERE / "src") not in sys.path:
    sys.path.insert(0, str(HERE / "src"))
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

DENSIFIERS = {"finance_ap": "crucible.factory.synth_wire:finance_ap_denser"}
LEVELS = (15, 30, 45, 60, 90)
MAX_MUTANTS = 480  # above a 1-invoice decision's whole neighbourhood on these Worlds (~400 mutants): measured, see densify
HOLDOUT = 0.2


def data_root(a: argparse.Namespace) -> Path:
    if a.data_root:
        return Path(a.data_root)
    if os.environ.get("CRUCIBLE_DATA_DIR"):
        return Path(os.environ["CRUCIBLE_DATA_DIR"]) / "synth"
    from crucible.core.config import get_config

    return get_config().data_path() / "synth"


def _save(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    tmp.replace(p)


def unique_ids(rows: list[Any]) -> dict[str, str]:
    """manifest key -> a new task id for every task whose id more than one source holds: ``<run>/<id without its arm
    prefix's duplicate>`` (``yield-2/a1`` + ``a1/finance_ap.s0.k0-b0-0`` -> ``yield-2/a1/finance_ap.s0.k0-b0-0``)."""
    c = Counter(r.task_id for r in rows)
    out = {}
    for r in rows:
        if c[r.task_id] > 1:
            run = r.source.split("/", 1)[0]
            out[r.key] = f"{run}/{r.task_id}"
    assert len(set(out.values())) == len(out)
    return out


def copy_records(src: Path, dst: Path, domain: str, old_id: str, new_id: str) -> int:
    """The source archive's counterexamples of ``old_id`` into ``dst`` under ``new_id`` (re-keyed; the source is read
    only)."""
    from crucible.synth.archive import Archive
    from crucible.synth.canon import content_hash

    a = Archive(src)
    if not a.path(domain).exists():
        return 0
    recs = [c.model_copy(update={"task_id": new_id, "id": c.id if new_id == old_id else "cx-" + content_hash({"from": c.id, "task": new_id}, 24)})
            for c in a.records(domain, old_id)]
    return len(Archive(dst).add_all(recs))


# ---------------------------------------------------------------------------------------------------------- Part A

def cmd_densify(a: argparse.Namespace) -> None:
    import multiprocessing as mp

    from crucible.synth.densify import DenseLevel, densify_job
    from crucible.synth.recert import Item, OldCert, load_manifest, slug
    from crucible.synth.task import Task

    root = data_root(a)
    man = Path(a.manifest)
    base_run = man.parent
    rows = load_manifest(man)
    items0 = {i["key"]: Item.model_validate(i) for i in json.loads((base_run / "items.json").read_text())}
    rename = unique_ids(rows)
    blocked = [r for r in rows if r.status == "provisional" and r.status_h0 == "certified"]
    if a.keys:  # a re-measurement of named rows only (e.g. the renamed ones a first pass measured under their old id)
        want = set(a.keys.split(","))
        blocked = [r for r in blocked if r.key in want]
        rename = {k: v for k, v in rename.items() if k in want}
        rows = [r for r in rows if r.key in want]
    d = Path(a.run) if a.run else root / "recert" / (time.strftime("%Y%m%d-%H%M%S") + f"-{a.tag}")
    (d / "densify").mkdir(parents=True, exist_ok=True)
    levels = tuple(int(x) for x in a.levels.split(","))
    todo = [r for r in blocked if r.domain in DENSIFIERS and not (d / "densify" / f"{slug(r.key)}.json").exists()]
    print(f"{len(blocked)} held-out-blocked tasks, {len(todo)} to measure; {len(rename)} tasks get a unique id; run {d}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {}
        for r in todo:
            work = d / "work" / slug(r.source)
            work.mkdir(parents=True, exist_ok=True)
            t0_ = r.load_task()
            if r.key in rename:  # the unique id first: MUT's split and sample are keyed by the task id (taskloop-1 §2, issue 3)
                t0_ = t0_.model_copy(update={"id": rename[r.key]}, deep=True)
            futs[ex.submit(densify_job, t0_.model_dump_json(), str(work), levels, DENSIFIERS[r.domain], HOLDOUT, a.max_mutants)] = r
        for k, f in enumerate(as_completed(futs), 1):
            r = futs[f]
            try:
                doc = json.loads(f.result())
            except Exception as e:  # noqa: BLE001 - recorded
                doc = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
            doc["key"] = r.key
            _save(d / "densify" / f"{slug(r.key)}.json", doc)
            lv = [(x["level"], x["n_shown"], x["check"]["ok"] if x.get("check") else None) for x in doc.get("levels", [])]
            print(f"[{k}/{len(todo)} {time.time() - t0:.0f}s] {r.key}: chosen {doc.get('chosen')} {lv} {doc.get('error', '')[:100]}", flush=True)
    # the items to re-certify: the densified versions, and every renamed task (its MUT split and archive keys change)
    items: list[Item] = []
    summary: dict[str, Any] = {"run": str(d), "blocked": len(blocked), "levels": list(levels), "max_mutants": a.max_mutants, "rows": []}
    (d / "tasks").mkdir(exist_ok=True)
    for r in rows:
        dense_doc = None
        if r in blocked:
            p = d / "densify" / f"{slug(r.key)}.json"
            dense_doc = json.loads(p.read_text()) if p.exists() else {"error": "not measured"}
        dense = Task.model_validate(dense_doc["dense_task"]) if dense_doc and dense_doc.get("dense_task") else None
        new_id = rename.get(r.key)
        if dense is None and new_id is None:
            if dense_doc is not None:
                summary["rows"].append({"key": r.key, "densified": False, "chosen": None, "error": dense_doc.get("error"),
                                        "levels": dense_doc.get("levels")})
            continue
        t = dense if dense is not None else r.load_task()
        if new_id is not None and t.id != new_id:
            t = t.model_copy(update={"id": new_id}, deep=True)
        it0 = items0[r.key]
        tf = d / "tasks" / f"{slug(r.key)}.json"
        _save(tf, {"task": t.model_dump(mode="json"), "from": {"key": r.key, "task_id": r.task_id, "version": r.version, "digest": r.digest,
                                                                 "task_path": r.task_path},
                   "densified": dense is not None, "renamed": new_id is not None, "chosen_level": dense_doc.get("chosen") if dense_doc else None})
        old = OldCert(status=r.status, reasons=r.reasons, classes=_classes(r.certificate), path=r.certificate)
        items.append(Item(key=r.key, source=r.source, kinds=r.kinds, run_dir=it0.run_dir, n=it0.n, task=t, old=old, builders=it0.builders,
                          seed0=False, alt_names_shared=False, archive_root=str(base_run / "archives" / slug(r.source)), task_file=str(tf)))
        if dense_doc is not None:
            summary["rows"].append({"key": r.key, "densified": dense is not None, "chosen": dense_doc.get("chosen"), "renamed": new_id,
                                    "levels": [DenseLevel.model_validate(x).model_dump(mode="json", exclude={"check"}) | {"check": x.get("check")}
                                               for x in dense_doc.get("levels", [])]})
    copied = 0
    for it in items:
        frm = json.loads(Path(it.task_file).read_text())["from"]
        copied += copy_records(Path(it.archive_root), d / "archives" / slug(it.source), it.task.domain, frm["task_id"], it.task.id)
    _save(d / "items.json", [i.model_dump(mode="json") for i in items])
    summary |= {"items": len(items), "densified": sum(1 for x in summary["rows"] if x.get("densified")), "renamed": len(rename),
                "archive_records_copied": copied, "chosen": dict(Counter(str(x.get("chosen")) for x in summary["rows"]))}
    _save(d / "densify.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))


def _classes(cert_path: str) -> dict[str, Any]:
    try:
        doc = json.loads(Path(cert_path).read_text())
    except (OSError, ValueError):
        return {}
    return ((doc.get("certificate") or {}).get("classes")) or {}


def cmd_manifest(a: argparse.Namespace) -> None:
    """recert-1's rows with every re-certified item's row (pool-2 report) in its place: the pool TaskLoop starts from."""
    from crucible.synth.recert import SCHEMA, load_manifest, summarize

    base = load_manifest(a.base)
    run = Path(a.run)
    new = {r.key: r for r in load_manifest(run / f"manifest.{a.tag}.json")}
    rows = [new.get(r.key, r) for r in base]
    summ = summarize(rows)
    dens = json.loads((run / "densify.json").read_text())
    blocked = {x["key"] for x in dens["rows"]}
    summ["blocked"] = {"n": len(blocked), "status": dict(Counter(new[k].status if k in new else "not densified" for k in blocked))}
    doc = {"schema": SCHEMA, "run": str(run), "tag": a.tag, "base": str(a.base), "made": time.strftime("%Y-%m-%d %H:%M:%S"), "summary": summ,
           "rows": [r.model_dump(mode="json") for r in rows]}
    out = run / f"manifest.{a.tag}.pool.json"
    _save(out, doc)
    print(json.dumps(summ, indent=1, default=str))
    print(f"manifest: {out}")



# ---------------------------------------------------------------------------------------------------------- Part B

def _recert_mod() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("synth_recert_cli", Path(__file__).with_name("synth_recert.py"))
    m = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


def _skills(task: Any) -> list[dict[str, str]]:
    from crucible.synth.domain import finance_ap_spec, spec_from_world
    from crucible.synth.meta.reskill import domain_skills

    if task.domain == "finance_ap":
        return domain_skills(finance_ap_spec())
    d = task.world.params.get("dir")
    return domain_skills(spec_from_world(d)) if d else []


class _NoopStudent:
    """``--dry``: a scripted student that does nothing (wiring checks only; costs nothing)."""

    student, family = "dry-run/noop", "noop"

    async def rollouts(self, task: Any, n: int, *, start: int = 0) -> list[Any]:
        from crucible.synth.learn import Rollout

        return [Rollout(i=start + j, passed=False, reasons=["dry"], program={"steps": [], "answer": None}) for j in range(n)]


def cmd_loop(a: argparse.Namespace) -> None:
    import asyncio

    from crucible.factory import models as fm
    from crucible.factory.synth_wire import SYNTH_ROLES, SynthWire, milestone_spent
    from crucible.synth import policy
    from crucible.synth.archive import Archive
    from crucible.synth.falsify import FalsifyContext
    from crucible.synth.falsify.agents import Panel, PanelMember, calibrate
    from crucible.synth.falsify.leak import LeakPanel, LeakProbe, calibrate_leak_panel
    from crucible.synth.falsify.oracle import ORACLE
    from crucible.synth.falsify.twinjudge import calibrate_twin_panel, twin_anchors
    from crucible.synth.meta.lift import LemmaLift, pairs_from_run, repairer_notes, swarm_abstract
    from crucible.synth.meta.loopkit import DomainClassifier, Recertifier, TaskKeyedContext, label_cell
    from crucible.synth.meta.reskill import leak_checker, novelty_checker, p3_checker, swarm_deriver
    from crucible.synth.meta.rsi import ComposeKnobs, LearnClassifier, ProbeReplay, TaskLoop, TaskRules, replay_program
    from crucible.synth.meta.update import entropy
    from crucible.synth.pipeline import make_repairer
    from crucible.synth.recert import Item, load_manifest, resolve_truths, slug, work_world

    rc = _recert_mod()
    root = data_root(a)
    if not os.environ.get("MILESTONE_CAP_USD"):
        raise SystemExit("set MILESTONE_CAP_USD (the shared Part 2 total, 150) in the environment")
    shared = float(os.environ["MILESTONE_CAP_USD"])
    d = Path(a.run) if a.run else root / "taskloop" / (time.strftime("%Y%m%d-%H%M%S") + f"-{a.tag}")
    d.mkdir(parents=True, exist_ok=True)
    ms = milestone_spent(root)
    done_here = milestone_spent(d)
    cap = round(a.cap - done_here, 4)  # a resumed run's cap is what is left of it
    print(f"shared Part 2 spend {ms:.4f} of {shared}; this run spent {done_here:.4f}; cap now {cap:.4f}", flush=True)
    if not a.dry and ms + cap > shared + 1e-9:
        raise SystemExit("the shared Part 2 total would pass its cap: refusing to start")
    if not a.dry and not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not in the environment (set -a; . <repo>/.env; set +a)")
    man = json.loads(Path(a.manifest).read_text())
    rows = load_manifest(a.manifest)
    all_rows = list(rows)  # the panels calibrate on the whole pool's valid tasks, also when --only selects a few (taskloop-1 §2)
    if a.only:
        rows = [r for r in rows if any(x in r.key for x in a.only.split(","))]
    if a.ids:  # exact task ids (taskloop-2: the affected lineages only; a substring of one id can be part of another)
        want = set(a.ids.split(","))
        rows = [r for r in rows if r.load_task().id in want]
        missing = want - {r.load_task().id for r in rows}
        if missing:
            raise SystemExit(f"task ids not in the manifest: {sorted(missing)}")
    if a.limit:
        rows = rows[: a.limit]
    tasks = {}
    for r in rows:
        t = r.load_task()
        if t.id in tasks:
            raise SystemExit(f"task id {t.id} twice in the pool: run `densify` (unique ids) first")
        tasks[t.id] = (r, t)
    # builders and archives: the pool run's items (renamed, densified) else the base re-certification's
    runs = []  # the manifest chain, oldest first: every run's items (builders, archives); a later run's item wins
    m = man
    while True:
        runs.insert(0, Path(m["run"]))
        if not m.get("base"):
            break
        m = json.loads(Path(m["base"]).read_text())
    items: dict[str, Any] = {}
    for run in runs:
        p = run / "items.json"
        if p.exists():
            for x in json.loads(p.read_text()):
                items[x["key"]] = (Item.model_validate(x), run)
    archive = Archive(d / "archive")
    if not (d / "archive.copied").exists():
        n = 0
        for r, t in tasks.values():
            _it, run = items[r.key]
            n += copy_records(run / "archives" / slug(r.source), d / "archive", t.domain, t.id, t.id)
        (d / "archive.copied").write_text(str(n))
    builders = {t.id: sorted(set(items[r.key][0].builders) | {"deepseek"}) for r, t in tasks.values()}
    truths = {"finance_ap": [rc.TRUTHS["finance_ap"]]}
    work = d / "work"
    work.mkdir(exist_ok=True)

    def make_ctx() -> FalsifyContext:
        return FalsifyContext(work_world(work), max_mutants=a.max_mutants)

    kctx = TaskKeyedContext(work_world(work), max_mutants=a.max_mutants)
    student_cap = min(a.student_cap, cap)
    swarm = None if a.dry else SynthWire(d / "swarm", cap_usd=max(0.01, cap - student_cap),
                                          table=dict(SYNTH_ROLES) | {"alt_solver": tuple(getattr(fm, m) for _, m, _ in rc.ALT_SET)})
    stud = None
    if a.dry:
        probe: Any = _NoopStudent()
    else:
        from crucible.factory.synth_student_wire import StudentWire
        from crucible.synth.learn import ContextPool

        stud = StudentWire(d / "student", cap_usd=student_cap)
        probe = stud.probe(make_ctx, concurrency=a.rollout_concurrency)
        probe.contexts = ContextPool(make_ctx, size=a.rollout_pool)
    errors: list[dict[str, Any]] = []
    t_start = time.time()

    async def go() -> dict[str, Any]:
        doms = sorted({t.domain for _, t in tasks.values()})
        valid = {dom: [r.load_task() for r in all_rows if r.domain == dom and r.status in ("certified", "provisional")] for dom in doms}
        panels, twins, opanels, cal = {}, {}, {}, {}
        leak = None
        if swarm is not None:
            roles = swarm.run_roles()
            c0 = make_ctx()
            try:
                for dom in doms:
                    good = [t for t in valid[dom] if "wrong" not in [f(t) for f in ORACLE(resolve_truths(truths.get(dom, []))).truths]] or valid[dom]
                    anc = await rc._panel_anchors(good, c0, k=6)
                    p = Panel(members=[PanelMember(n, c) for n, c in roles.panel], lam=0.7, queue=d / "human_queue.jsonl", tag=f"calibrate:{dom}")
                    rep = rep2 = await calibrate(p, anc, **policy.panel_calibration())  # R-panel (item 24): an uncalibrated panel abstains
                    panels[dom] = opanels[dom] = p
                    tp = Panel(members=p.members, lam=p.lam, queue=d / "human_queue.jsonl", tag=f"calibrate:twin:{dom}")
                    tanc = (await twin_anchors(good, c0))[:9] + anc[:9]
                    trep = await calibrate_twin_panel(tp, tanc, min_decided=6)
                    twins[dom] = tp
                    cal[dom] = {"panel": {"lam": rep.lam, "calibrated": rep.calibrated}, "outcome_panel": {"lam": rep2.lam, "calibrated": rep2.calibrated},
                                "twin": {"lam": trep.lam, "calibrated": trep.calibrated, "anchors": len(tanc)}}
            finally:
                await c0.close()
            srcs = [t for dom in doms for t in valid[dom] if t.intent.derive]
            lp = LeakPanel(members=[PanelMember(f"leak_panel-{c.family}", c) for c in swarm.clients("leak_panel")], lam=0.7, queue=d / "leak_queue.jsonl")
            anchors = rc._f2().leak_anchors(srcs, list(rc._f2().DEV_TEMPLATES))[:40]
            lc = await calibrate_leak_panel(lp, anchors) if anchors else {"calibrated": False}
            cal["leak"] = {k: v for k, v in lc.items() if k != "votes"}
            leak = LeakProbe(swarm.client("leak_probe"), panel=lp, skip_structural=True, max_calls=a.leak_calls)
            _save(d / "calibration.json", cal)

        async def paid(task: Any) -> tuple[Any, Any, Any, Any, Any]:
            base = panels[task.domain]
            key = f"loop:{task.id}@v{task.version}"
            pan = Panel(members=base.members, lam=base.lam, queue=base.queue, tag=f"adjudicate:{key}")
            tw = twins.get(task.domain)
            tpan = Panel(members=tw.members, lam=tw.lam, queue=tw.queue, tag=f"adjudicate:twin:{key}") if tw is not None else None
            return pan, tpan, rc._alt_agents(swarm, key), rc._atk_agents(swarm, key), [leak] if leak is not None else []

        import multiprocessing as mp
        from functools import partial

        from crucible.synth.recert import det_job

        pool = ProcessPoolExecutor(max_workers=a.det_workers, mp_context=mp.get_context("spawn"))

        async def det_fn(t: Any) -> dict[str, Any]:  # phase 1 of a full re-certification in a worker: CPU off the event loop
            f = partial(det_job, t.model_dump_json(), str(work), a.max_mutants, HOLDOUT, truths.get(t.domain, []), str(d / "archive"),
                        mut_all=False, probe_replay=True)
            return json.loads(await asyncio.get_running_loop().run_in_executor(pool, f))

        rec = Recertifier(make_ctx=make_ctx, archive=archive, holdout=HOLDOUT, truths=lambda t: truths.get(t.domain, []),
                          paid=None if swarm is None else paid, builders_of=lambda t: builders.get(t.id, ["deepseek"]), log=d / "recert.jsonl",
                          extra_replay=[ProbeReplay()], det_fn=det_fn)
        alt_map: dict[str, list[int]] = {}
        for r, t in tasks.values():
            ps = sum(int(v.get("passes", 0)) for v in r.alt.values())
            n = sum(int(v.get("n", 0)) for v in r.alt.values())
            rec.seed(t, r.status, r.reasons, (ps, n))
            if n:
                alt_map[t.id] = [ps, n]
        clf = DomainClassifier({dom: LearnClassifier(kctx, panel=opanels.get(dom), alt=alt_map) for dom in doms})

        async def tools_of(t: Any) -> list[Any]:
            return list(await (await kctx.world(t)).tools())

        repair = make_repairer(swarm.client("repairer"), kctx, tools_of=tools_of, log_path=d / "repairs.jsonl") if swarm is not None else (lambda t, c: _none())
        deriver = swarm_deriver(swarm.client("abstractor"), _skills, tag="reskill") if swarm is not None and not a.no_reskill else None
        # R-lemma-lift (DESIGN item 25): accepted repairs are recorded as pairs; a recurring class becomes a candidate the
        # abstractor (non-builder) drafts and the gate decides; with no gate budget it is deferred, never admitted
        lift = LemmaLift(d / "lift", abstract=swarm_abstract(swarm.client("abstractor")) if swarm is not None else None,
                         gate=None, gate_budget_usd=lambda: float(a.lemma_gate_usd), notes_of=repairer_notes(d / "repairs.jsonl"))
        if a.lift_seed:
            seeded = pairs_from_run(Path(a.lift_seed), lift)
            print(f"lift: {len(seeded)} accepted repair(s) of {a.lift_seed} seeded", flush=True)
            await lift.seed(seeded)

        async def p3(t: Any) -> tuple[bool | None, list[str]]:  # R-p3: EVID through the SQL evidence channel (item 25)
            pool = [x for x in loop.tasks.values() if x.domain == t.domain and x.id != t.id]
            return await p3_checker(kctx, pool=pool, evidence=True)(t)

        def cell_of(t: Any) -> str:
            return label_cell(t.domain, t.intent.skill, _skills(t))

        cells = Counter(cell_of(t) for _, t in tasks.values())
        loop = TaskLoop(d / "loop", probe=probe, classify=clf, repair=repair, replay=lambda t, prog: replay_program(t, prog, kctx), recheck=rec.recheck,
                        knobs=ComposeKnobs(kctx), ctx=kctx, oracle=ORACLE(resolve_truths(truths["finance_ap"])), p3_check=p3, alt_check=rec.alt_solves,
                        cell_of=cell_of, archive=archive,
                        rules=TaskRules(student_family="qwen", h_min=round(0.9 * entropy(dict(cells)), 4), max_rounds=a.rounds),
                        derive_intent=deriver, skills_of=lambda t: [x["id"] for x in _skills(t)], leak_check=leak_checker(kctx),
                        novel_check=novelty_checker(kctx), validity=rec.validity, lift_lemmas=lift)
        lift.lineage_of = lambda tid: loop.of.get(tid, tid)
        loop.stored = stored_rollouts(root, probe)
        loop.add_tasks(t for _, t in tasks.values())
        loop.save()

        def on_error(tid: str, e: BaseException) -> None:
            errors.append({"task": tid, "error": f"{type(e).__name__}: {str(e)[:400]}"})
            print(f"  ! {tid}: {type(e).__name__}: {str(e)[:200]}", flush=True)

        rounds = []
        start = 1 + max((x.round for x in loop.decisions), default=0)
        for rnd in range(start, a.rounds + 1):
            t0 = time.time()
            decs = await loop.run_typed_round(rnd, concurrency=a.concurrency, on_error=on_error)
            al = loop.monitor.alarm()
            rounds.append({"round": rnd, "decisions": len(decs), "moves": dict(Counter(x.move for x in decs)), "seconds": round(time.time() - t0, 1),
                           "drift": al})
            print(json.dumps(rounds[-1], default=str), flush=True)
            if al.get("alarm"):
                loop.rules.max_repairs = 0  # prereg §8.1: the drift alarm stops every repair for review
            if not decs or any("Budget" in x["error"] for x in errors):
                break
        pool.shutdown(wait=False, cancel_futures=True)
        return {"rounds": rounds, "calibration": cal, "recert": [x.model_dump(mode="json") for x in rec.rows.values() if x.source != "manifest"],
                "summary": loop.typed_summary(), "knob_calibration": loop.knob_calibration, "world_fixes": loop.world_fixes,
                "outcomes": [r.model_dump(mode="json") for r in clf.records], "lemmas": lift.summary(), "policy_version": policy.tag()}

    async def runner() -> dict[str, Any]:
        try:
            return await go()
        finally:
            await kctx.close()

    try:
        out = asyncio.run(runner())
    finally:
        cost = {"swarm": swarm.cost() if swarm is not None else None, "student": stud.cost() if stud is not None else None,
                "run_spent": milestone_spent(d), "wall_s": round(time.time() - t_start, 1)}
        _save(d / f"cost-{time.strftime('%H%M%S')}.json", cost)
    out |= {"errors": errors, "cost": cost}
    _save(d / "loop_report.json", out)
    print(json.dumps({"summary": out["summary"], "errors": len(errors), "cost": cost}, indent=1, default=str))



def cmd_report(a: argparse.Namespace) -> None:
    """The loop run's tables (runs/taskloop-1.md): moves, the band before and after, the drift alarm, examples, the
    lessons and the power to accept them, cost per perfect task and wall-clock. P3 and novelty of the kept tasks are
    measured here (CPU)."""
    import asyncio

    from crucible.factory.synth_wire import milestone_spent
    from crucible.synth.archive import Archive
    from crucible.synth.meta.abstract import natural_clusters
    from crucible.synth.meta.loopkit import TaskKeyedContext
    from crucible.synth.meta.reskill import novelty_checker, p3_checker
    from crucible.synth.meta.rsi import Lineage, TaskDecision, TuningMonitor
    from crucible.synth.meta.stats import mde
    from crucible.synth.recert import work_world
    from crucible.synth.task import Task

    d = Path(a.run)
    st = json.loads((d / "loop" / "task_state.json").read_text())
    ps = json.loads((d / "loop" / "state.json").read_text())
    decs = [TaskDecision.model_validate(x) for x in st["decisions"]]
    lins = {k: Lineage.model_validate(v) for k, v in st["lineages"].items()}
    tasks = {k: Task.model_validate(v) for k, v in ps["tasks"].items()}
    rep = json.loads((d / "loop_report.json").read_text()) if (d / "loop_report.json").exists() else {}
    out: dict[str, Any] = {"run": str(d)}
    acc = Counter(x.move for x in decs if x.accepted)
    out["moves"] = {"decisions": dict(Counter(x.move for x in decs)), "accepted": dict(acc),
                    "refused_by": dict(Counter(r.split(":", 1)[0] for x in decs for r in x.refused_by)),
                    "lineages": dict(Counter(v.status for v in lins.values())),
                    "per_round": {r: dict(Counter(x.move for x in decs if x.round == r)) for r in sorted({x.round for x in decs})}}
    first: dict[str, Any] = {}
    last: dict[str, Any] = {}
    for x in decs:
        if x.band and x.band.get("decision"):
            first.setdefault(x.task_id, x.band["decision"])
            last[x.task_id] = x.band["decision"]
        if x.accepted and x.acceptance_band and x.acceptance_band.get("decision"):
            last[x.variant_id or x.task_id] = x.acceptance_band["decision"]
    out["band"] = {"before": dict(Counter(first.values())), "after": dict(Counter(last.values())), "measured_tasks": len(first)}
    # item 25: the conditions the policy resolved, by (condition, rule, action), and the lemma lift
    out["conditions"] = dict(Counter(f"{r['condition']}:{r['rule']}:{r['action']}" for x in decs for r in x.remedies))
    out["lemmas"] = rep.get("lemmas")
    mon = TuningMonitor.model_validate(ps["monitor"]) if ps.get("monitor") else TuningMonitor()
    out["drift"] = {"alarm": mon.alarm(), "rows": mon.rows}
    ex: dict[str, list[dict[str, Any]]] = {}
    for x in decs:
        k = f"{x.move}{'' if x.accepted or x.move in ('HOLD', 'RESERVE') else ' (refused)'}"
        if len(ex.setdefault(k, [])) < 4:
            ex[k].append({"task": x.task_id, "round": x.round, "why": x.why[-3:], "refused_by": x.refused_by[:3], "band": (x.band or {}).get("decision"),
                          "acceptance": (x.acceptance_band or {}).get("decision"), "reskill": x.reskill and {"from": x.reskill["from"]["skill"],
                                                                                                             "to": x.reskill["to"]["skill"], "cell": x.reskill["cell"]},
                          "oracle": x.oracle.get("status"), "certificate": x.oracle.get("certificate")})
    out["examples"] = ex
    arch = Archive(d / "archive")
    probe = [c for dom in ("finance_ap", "it_helpdesk") if arch.path(dom).exists() for c in arch.records(dom) if c.cls == "PROBE"]
    cl = natural_clusters(probe)
    out["lessons"] = {"probe_counterexamples": len(probe), "clusters": [{"defect": c.defect, "view": c.view, "members": len(c.members)} for c in cl],
                      "world_fixes": ps.get("world_fixes", []),
                      "repairs_accepted": [{"task": x.task_id, "to": x.new_version, "why": x.why[:2]} for x in decs if x.move == "REPAIR" and x.accepted]}
    # the R1-R10 gate on lifted operators: a cluster's candidate is tested on fresh decision units of OTHER tasks; the
    # most units any cluster could supply is its members on other tasks: the minimum detectable gain at that n
    out["power"] = {str(n): mde(n, alpha=0.005) for n in sorted({max(1, len(c.members)) for c in cl} | {30, 60, 120, 240})}
    # perfect: kept lineages whose final version is certified, on intent (P3) and novel (P4) in its cell
    recs = {}
    if (d / "recert.jsonl").exists():
        for line in (d / "recert.jsonl").read_text().splitlines():
            r = json.loads(line)
            recs[(r["task_id"], r["version"])] = r["status"]
    man = {}
    if a.manifest:
        from crucible.synth.recert import load_manifest

        for r in load_manifest(a.manifest):
            t = r.load_task()
            man[r.task_id if r.task_id in tasks else t.id] = r.status
            man[t.id] = r.status

    async def perfect() -> list[dict[str, Any]]:
        ctx = TaskKeyedContext(work_world(d / "work-report"), max_mutants=480)
        rows = []
        try:
            kept = [tasks[m] for k, v in lins.items() if v.status == "kept" for m in v.members if m in tasks]
            nov = novelty_checker(ctx)
            for t in kept:
                cert = recs.get((t.id, t.version)) or man.get(t.id)
                pool = [x for x in tasks.values() if x.id != t.id and x.domain == t.domain]
                on, why = await p3_checker(ctx, pool=pool)(t)
                n_ok, n_why = await nov(t, t.intent.skill, [x for x in pool if x.intent.skill == t.intent.skill])
                rows.append({"task": t.id, "version": t.version, "certificate": cert, "on_intent": on, "novel": n_ok,
                             "perfect": cert == "certified" and on is True and n_ok, "why": (why + n_why)[:3]})
        finally:
            await ctx.close()
        return rows

    pf = asyncio.run(perfect())
    spent = milestone_spent(d)
    npf = sum(1 for r in pf if r["perfect"])
    out["perfect"] = {"kept": len(pf), "perfect": npf, "rows": pf, "spent_usd": spent, "usd_per_perfect": round(spent / npf, 3) if npf else None,
                      "kept_provisional_ok": sum(1 for r in pf if r["certificate"] in ("certified", "provisional") and r["on_intent"] is True and r["novel"])}
    out["cost"] = rep.get("cost")
    out["rounds"] = rep.get("rounds")
    out["errors"] = rep.get("errors")
    _save(d / "report.json", out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("examples", "lessons")}, indent=1, default=str)[:6000])

def cmd_lift(a: argparse.Namespace) -> None:
    """R-lemma-lift on an earlier loop run (DESIGN item 25): its accepted repairs as (counterexample class, repair) pairs, the
    recurring classes as candidates, drafted by the abstractor (a non-builder family) and decided by the gate's policy
    remedy. No student, no task edit. The shared Part 2 total is checked before the abstractor call."""
    import asyncio

    from crucible.factory.synth_wire import SynthWire, milestone_spent
    from crucible.synth.meta.lift import LemmaLift, pairs_from_run, swarm_abstract

    root = data_root(a)
    d = Path(a.out) if a.out else root / "taskloop" / (time.strftime("%Y%m%d-%H%M%S") + "-lift")
    d.mkdir(parents=True, exist_ok=True)
    wire = None
    if not a.dry:
        if not os.environ.get("MILESTONE_CAP_USD"):
            raise SystemExit("set MILESTONE_CAP_USD (the shared Part 2 total, 150) in the environment")
        ms = milestone_spent(root)
        if ms + a.cap > float(os.environ["MILESTONE_CAP_USD"]) + 1e-9:
            raise SystemExit("the shared Part 2 total would pass its cap: refusing to start")
        wire = SynthWire(d / "swarm", cap_usd=a.cap)
    lift = LemmaLift(d, abstract=swarm_abstract(wire.client("abstractor"), tag="lift") if wire is not None else None,
                     gate_budget_usd=lambda: float(a.lemma_gate_usd))
    pairs = pairs_from_run(Path(a.run), lift)
    asyncio.run(lift.seed(pairs))
    out = {"run": str(a.run), "pairs": [p.model_dump(mode="json") for p in pairs], "summary": lift.summary(),
           "candidates": {k: v.model_dump(mode="json") for k, v in lift.candidates.items()}, "cost": wire.cost() if wire is not None else None}
    _save(d / "lift_report.json", out)
    print(json.dumps({"pairs": len(pairs), "summary": out["summary"], "cost": out["cost"]}, indent=1, default=str))


def stored_rollouts(root: Path, probe: Any) -> Any:
    """prereg §8.1: the learnability store's measured rollouts (``perfect/difficulty.jsonl``) of the exact task version
    (id, version, digest), the same student and simulator, are the round-1 diagnosis; None when the probe is not the
    store's student or the version was never measured (a densified or renamed version is a new version)."""
    from crucible.synth.learn import DifficultyStore

    p = root / "perfect" / "difficulty.jsonl"
    if not p.exists():
        return None
    by: dict[tuple[str, int, str], list[Any]] = {}
    for r in DifficultyStore(p).all():
        if r.source == "measured" and r.key.student == getattr(probe, "student", None) and r.key.simulator == getattr(probe, "simulator", None):
            by[(r.key.task_id, r.key.task_version, r.key.task_digest)] = list(r.rollouts)

    def get(task: Any) -> list[Any] | None:
        return by.get((task.id, task.version, task.digest()))

    return get


async def _none() -> None:
    return None

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    dz = sub.add_parser("densify")
    dz.add_argument("--manifest", required=True)
    dz.add_argument("--tag", default="pool-2")
    dz.add_argument("--run", default="")
    dz.add_argument("--levels", default=",".join(str(x) for x in LEVELS))
    dz.add_argument("--max-mutants", type=int, default=MAX_MUTANTS)
    dz.add_argument("--workers", type=int, default=6)
    dz.add_argument("--keys", default="", help="only these manifest keys (comma-separated)")
    mf = sub.add_parser("manifest")
    mf.add_argument("--base", required=True)
    mf.add_argument("--run", required=True)
    mf.add_argument("--tag", default="pool-2")
    lp = sub.add_parser("loop")
    lp.add_argument("--manifest", required=True)
    lp.add_argument("--tag", default="taskloop-1")
    lp.add_argument("--run", default="")
    lp.add_argument("--cap", type=float, default=18.0, help="USD for this loop run (student + swarm)")
    lp.add_argument("--student-cap", type=float, default=12.0)
    lp.add_argument("--rounds", type=int, default=3)
    lp.add_argument("--concurrency", type=int, default=6, help="tasks decided concurrently")
    lp.add_argument("--rollout-concurrency", type=int, default=4)
    lp.add_argument("--rollout-pool", type=int, default=16)
    lp.add_argument("--max-mutants", type=int, default=MAX_MUTANTS)
    lp.add_argument("--leak-calls", type=int, default=120)
    lp.add_argument("--det-workers", type=int, default=4)
    lp.add_argument("--only", default="")
    lp.add_argument("--ids", default="", help="exact task ids (comma-separated): only these lineages")
    lp.add_argument("--lift-seed", default="", help="an earlier loop run whose accepted repairs seed the lemma lift (R-lemma-lift)")
    lp.add_argument("--lemma-gate-usd", type=float, default=0.0, help="what the lemma gate may spend (0: an underpowered gate defers)")
    lp.add_argument("--limit", type=int, default=0)
    lp.add_argument("--no-reskill", action="store_true")
    lp.add_argument("--dry", action="store_true", help="a no-op scripted student and no paid classes (wiring only, no spend)")
    rp = sub.add_parser("report")
    rp.add_argument("--run", required=True)
    rp.add_argument("--manifest", default="")
    lf = sub.add_parser("lift", help="the lemma lift of an earlier loop run's accepted repairs (R-lemma-lift; no student)")
    lf.add_argument("--run", required=True, help="the loop run whose accepted repairs are lifted")
    lf.add_argument("--out", default="", help="the lift directory (default: <data-root>/taskloop/<now>-lift)")
    lf.add_argument("--cap", type=float, default=0.05, help="USD for the abstractor")
    lf.add_argument("--lemma-gate-usd", type=float, default=0.0)
    lf.add_argument("--dry", action="store_true", help="no abstractor call")
    a = ap.parse_args()
    {"densify": cmd_densify, "manifest": cmd_manifest, "loop": cmd_loop, "report": cmd_report, "lift": cmd_lift}[a.cmd](a)


if __name__ == "__main__":
    main()
