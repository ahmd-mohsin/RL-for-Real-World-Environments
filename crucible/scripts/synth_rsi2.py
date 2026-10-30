#!/usr/bin/env python3
"""RSI-2 statistics on real data (runs/rsi-stats.md, runs/rsi-2-prereg.md). CPU work runs on worker processes
(``meta/par.ShardPool``); the swarm calls of ``evolve`` and ``e10`` go through the Tinker wire with a hard cap.

    synth_rsi2.py aa       --run-id rsi-2 --replicates 5            # A/A: δ_AA on the dev pool vs RSI-1's between-view δ
    synth_rsi2.py retro    --run-id rsi-2                           # RSI-1's R2 decisions re-tested by the paired machinery (report only)
    synth_rsi2.py power    --run-id rsi-2                           # power per class on the locked set; sizing for the 10 x 10 pool
    synth_rsi2.py headroom --run-id rsi-2                           # dev headroom of the base population
    synth_rsi2.py locked   --run-id rsi-2 --name rsi-2b --max-per-op 6   # a rebuilt locked set (data/sealed/synth-locked), its power
    synth_rsi2.py evolve   --run-id rsi-2 --label pilot --rounds 2 --cap 8   # the paired loop on the swarm (pilot)
    synth_rsi2.py e10      --run-id rsi-2 --label e10-pilot --ablations none,R7,NL --seeds 0,1,2 --cap 4

Outputs: ``<data-root>/rsi/<run-id>/stats/`` and the run directories ``<data-root>/rsi/<run-id>/rsi2-pilot-*``. The pilot's
spend (every receipt under a path carrying "rsi2-pilot") is capped at ``PILOT_CAP_USD``. Keys come from the environment
only (``set -a; . <repo>/.env; set +a`` in the shell that starts this script). The student is never called.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE / "src") not in sys.path:
    sys.path.insert(0, str(HERE / "src"))
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

PILOT_CAP_USD = 30.0
RSI1_DELTA = 0.04073400617738517  # RSI-1's between-view δ (rsi-2 evolve-e2 ledger)


def data_root(a: argparse.Namespace) -> Path:
    if a.data_root:
        return Path(a.data_root)
    if os.environ.get("CRUCIBLE_DATA_DIR"):
        return Path(os.environ["CRUCIBLE_DATA_DIR"]) / "synth"
    from crucible.core.config import get_config

    return get_config().data_path() / "synth"


def rsi_dir(a: argparse.Namespace) -> Path:
    return data_root(a) / "rsi" / a.run_id


def stats_dir(a: argparse.Namespace) -> Path:
    d = rsi_dir(a) / "stats"
    d.mkdir(parents=True, exist_ok=True)
    return d


def locked_root(a: argparse.Namespace) -> Path:
    return Path(a.locked_root) if a.locked_root else data_root(a).parent / "sealed" / "synth-locked"


def _load(a: argparse.Namespace):
    from crucible.synth.meta.recall import SetStore
    from crucible.synth.task import Task

    out = rsi_dir(a)
    store = SetStore(out / "sets", locked_root(a))
    dev = store.read(f"{a.run_id}-dev", "dev")
    locked = store.read(a.locked or f"{a.run_id}-locked", "locked", purpose="report")
    src = json.loads((out / "sources.json").read_text())
    tasks = {k: Task.model_validate(v) for k, v in src["tasks"].items()}
    return out, dev, locked, [tasks[i] for i in src["dev"]], [tasks[i] for i in src["locked"]]


def _pool(dev, dev_src):
    from crucible.synth.meta.rsi import Pool

    pool = Pool(dev_src)
    pool.add("P@1", dev.plants)
    return pool


def _dump(path: Path, doc: object) -> None:
    path.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"wrote {path}", flush=True)


def pilot_cap(a: argparse.Namespace, root: Path) -> float:
    from crucible.factory.synth_wire import work_spent

    spent = work_spent(root, "rsi2-pilot")
    cap = min(float(a.cap), round(PILOT_CAP_USD - spent, 4))
    print(f"RSI-2 pilot spend so far {spent:.4f} of {PILOT_CAP_USD}; this run's cap {cap:.4f}", flush=True)
    if cap < 0.05:
        raise SystemExit("the pilot budget is spent: refusing to start")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not in the environment (set -a; . <repo>/.env; set +a)")
    return cap


# ---------------------------------------------------------------------------------------------------------- aa


async def _units_at(ev, theta_f, ids, anchors, seeds):
    ms = await asyncio.gather(*(ev.measure(theta_f, ids, anchors, seed=s) for s in seeds))
    return dict(zip(seeds, ms, strict=True))


def cmd_aa(a: argparse.Namespace) -> None:
    from crucible.synth.canon import content_hash
    from crucible.synth.falsify import FalsifyContext
    from crucible.synth.meta.par import ShardPool
    from crucible.synth.meta.rsi import FEvaluator, Sampler
    from crucible.synth.meta.stats import aa_floor

    _out, dev, _locked, dev_src, _ = _load(a)
    pool = _pool(dev, dev_src)
    k = a.replicates
    seeds = [0, *range(1, 2 * k + 1)]

    async def go() -> dict:
        shards = ShardPool(a.workers)
        try:
            ev = FEvaluator(pool, FalsifyContext(), shards=shards)
            ids = list(pool.plants)
            t = time.time()
            ms = await _units_at(ev, {}, ids, [x.id for x in pool.anchors], seeds)
            secs = time.time() - t
        finally:
            shards.close()
        sampler = Sampler(pool, frac=0.6, seed=int(content_hash({"arm": "loop", "seed": 0}, 8), 16) % 10_000)
        view = sampler.draw("0/calibrate/1")  # RSI-1's first calibration view (75 units)
        pos = {u: i for i, u in enumerate(ids)}
        pairs = [(2 * r + 1, 2 * r + 2) for r in range(k)]
        full = aa_floor([(ms[x].j_units, ms[y].j_units) for x, y in pairs], seeds=pairs, view="dev-pool")
        sub = aa_floor([([ms[x].j_units[pos[u]] for u in view], [ms[y].j_units[pos[u]] for u in view]) for x, y in pairs], seeds=pairs, view="rsi1-view-75")
        flips = {}
        for s in seeds[1:]:
            d = [ids[i] for i, (p, q) in enumerate(zip(ms[0].j_units, ms[s].j_units, strict=True)) if p != q]
            flips[s] = [{"plant": x, "op": pool.plants[x].op.key, "domain": pool.plants[x].domain} for x in d]
        return {"replicates": k, "seed_pairs": pairs, "n_units": len(ids), "j_by_seed": {s: round(m.j, 4) for s, m in ms.items()},
                "fa_by_seed": {s: m.fa for s, m in ms.items()}, "delta_aa_pool": full.model_dump(mode="json"), "delta_aa_view75": sub.model_dump(mode="json"),
                "rsi1_delta_between_views": RSI1_DELTA, "flips_vs_seed0": flips, "seconds": round(secs, 1), "workers": a.workers}

    doc = asyncio.run(go())
    _dump(stats_dir(a) / "aa.json", doc)
    print(json.dumps({k2: doc[k2] for k2 in ("j_by_seed", "rsi1_delta_between_views", "seconds")}, indent=1))
    print("delta_AA pool", doc["delta_aa_pool"]["delta_aa"], "view75", doc["delta_aa_view75"]["delta_aa"])


# ---------------------------------------------------------------------------------------------------------- retro


def cmd_retro(a: argparse.Namespace) -> None:
    """RSI-1's evaluated F decisions, re-tested (report only; nothing is accepted): the view is reconstructed from the
    sampler's seeds and matched to the ledger's hash; Θ and Θ′ are re-measured at seed 0 (RSI-1's own setting, checked
    against the ledger's J) and at independent seeds (the paired mode's setting)."""
    from crucible.synth.canon import content_hash
    from crucible.synth.falsify import FalsifyContext
    from crucible.synth.meta.library import Library, OperatorSpec
    from crucible.synth.meta.par import ShardPool
    from crucible.synth.meta.rsi import FEvaluator, Sampler
    from crucible.synth.meta.stats import betting_lcb, betting_test, bootstrap_ci, effect, paired

    out, dev, _locked, dev_src, _ = _load(a)
    pool = _pool(dev, dev_src)
    aa = json.loads((stats_dir(a) / "aa.json").read_text())
    d_aa = aa["delta_aa_view75"]["delta_aa"]
    anchors = sorted(x.id for x in pool.anchors)
    labels = [x for x in a.evolve.split(",") if x]

    def find_view(arm: str, seed: int, h: str, rnd: int) -> list[str] | None:
        s = Sampler(pool, frac=0.6, seed=int(content_hash({"arm": arm, "seed": seed}, 8), 16) % 10_000)
        for n in range(1, 120):
            for purpose in ("decide", "calibrate", "prune", "audit"):
                ids = s.draw(f"{rnd}/{purpose}/{n}")
                if content_hash({"plants": sorted(ids), "anchors": anchors}, 32) == h:
                    return ids
        return None

    async def go() -> list[dict]:
        shards = ShardPool(a.workers)
        rows_out = []
        try:
            ev = FEvaluator(pool, FalsifyContext(), shards=shards)
            for lab in labels:
                d = out / lab
                cfg = json.loads((d / "config.json").read_text())["config"]
                rows = json.loads((d / "rows.json").read_text())
                specs = {c["op"]: c["spec"] for r in rows for c in r["candidates"] if c.get("spec")}
                led = [json.loads(x)["rec"] for x in (d / "loop" / "ledger.jsonl").read_text().splitlines()]
                decs = [r for r in led if r["kind"] == "update" and r["evaluated"] and r["component"] == "F"]
                T = len(decs)
                for i, r in enumerate(decs, 1):
                    op = r["update_id"].split("/")[-1]
                    ids = find_view("loop", cfg.get("seed", 0), r["view"], r["round"])
                    if ids is None or op not in specs:
                        rows_out.append({"evolve": lab, "update": r["update_id"], "error": "view or spec not reconstructed"})
                        continue
                    lib = Library().add(OperatorSpec.model_validate(specs[op]), r["round"]).model_dump(mode="json")
                    base0, cand0 = await asyncio.gather(ev.measure({}, ids, anchors, seed=0), ev.measure(lib, ids, anchors, seed=0))
                    d0 = paired(cand0.j_units, base0.j_units)
                    indep = []
                    for sa, sb in ((1, 2), (3, 4), (5, 6)):
                        b, c = await asyncio.gather(ev.measure({}, ids, anchors, seed=sa), ev.measure(lib, ids, anchors, seed=sb))
                        dd = paired(c.j_units, b.j_units)
                        indep.append({"seeds": [sa, sb], "mean": round(sum(dd) / len(dd), 4), "e_value_vs_dAA_at_0.05/T": round(betting_test(dd, d_aa, 0.05 / T).max_wealth, 3),
                                      "reject_at_0.05/T": betting_test(dd, d_aa, 0.05 / T).reject, "reject_unadjusted_0.05": betting_test(dd, d_aa, 0.05).reject})
                    eff = effect(d0, 0.05 / T, clusters=[pool.plants[u].source_id for u in ids])
                    rows_out.append({"evolve": lab, "round": r["round"], "update": op, "n": len(ids), "j_star_ledger": r["metrics"]["j_star"], "j_star_now": round(base0.j, 4),
                                     "j_new_ledger": r["metrics"]["j_new"], "j_new_now": round(cand0.j, 4), "rsi1_lcb": r["metrics"]["lcb"], "rsi1_delta": r["metrics"]["delta"],
                                     "rsi1_rules": [f["rule"] for f in r["failed"]], "gains": eff.gains, "losses": eff.losses, "mean_gain": round(eff.mean, 4),
                                     "boot95": [round(x, 4) for x in eff.ci95], "boot95_cluster": [round(x, 4) for x in (eff.ci95_cluster or (0, 0))],
                                     "delta_aa": d_aa, "tests_in_run": T, "alpha_t": 0.05 / T,
                                     "crn_e_value": round(betting_test(d0, d_aa, 0.05 / T).max_wealth, 3), "crn_reject_at_0.05/T": betting_test(d0, d_aa, 0.05 / T).reject,
                                     "crn_lcb_anytime_0.05/T": round(betting_lcb(d0, 0.05 / T), 4), "crn_lcb_anytime_0.05": round(betting_lcb(d0, 0.05), 4),
                                     "boot_lcb_one_sided_95": round(bootstrap_ci(d0, level=0.90)[0], 4), "independent_seeds": indep})
                    print(json.dumps(rows_out[-1])[:400], flush=True)
        finally:
            shards.close()
        return rows_out

    _dump(stats_dir(a) / "retro.json", asyncio.run(go()))


# ---------------------------------------------------------------------------------------------------------- power


def _counts(plants) -> dict[str, int]:
    out: dict[str, int] = {f"D{i}": 0 for i in range(1, 12)}
    for p in plants:
        out[p.defect] = out.get(p.defect, 0) + 1
    return out


def cmd_power(a: argparse.Namespace) -> None:
    from crucible.synth.meta.stats import mde, plants_needed, power_table

    _out, dev, locked, _dev_src, locked_src = _load(a)
    aa_p = stats_dir(a) / "aa.json"
    p01 = 0.0
    if aa_p.exists():
        p01 = json.loads(aa_p.read_text())["delta_aa_pool"]["discord_rate"] / 2  # a discordant A/A unit flips one way or the other
    counts = _counts(locked.plants)
    tab = power_table(counts, gains=(0.1, 0.2, 0.3), p01=p01, classes_tested=11)
    need = {g: {"p01=0": plants_needed(g), f"p01={p01:.4f}": plants_needed(g, p01=p01), "bonferroni/11": plants_needed(g, alpha=0.05 / 11, p01=p01)}
            for g in (0.1, 0.15, 0.2, 0.3)}
    n_src = len(locked_src)
    per_task = {d: c / max(1, n_src) for d, c in counts.items()}
    proj = {}
    for tasks in (40, 30, 20):  # the 10 x 10 target: 4, 3 or 2 held-out environments of 10 tasks each
        pc = {d: round(r * tasks) for d, r in per_task.items()}
        proj[f"{tasks}_locked_tasks"] = {"plants": pc, "mde": {d: mde(n, p01=p01) if n else None for d, n in pc.items()}}
    doc = {"locked_set": locked.name, "plants": len(locked.plants), "sources": n_src, "counts": counts, "p01": p01, "table": tab, "plants_needed": need,
           "per_task_yield": {d: round(x, 3) for d, x in per_task.items()}, "projection_10x10": proj, "dev_counts": _counts(dev.plants)}
    _dump(stats_dir(a) / f"power-{locked.name}.json", doc)
    print("| class | n | MDE (α .05) | MDE (Bonferroni /11) | power @0.2 |")
    for r in tab:
        print(f"| {r['class']} | {r['n']} | {r['mde']} | {r['mde_bonferroni']} | {r['power@0.2']} |")
    print(json.dumps(need, indent=1))
    print(json.dumps(proj["40_locked_tasks"], indent=1))


# ---------------------------------------------------------------------------------------------------------- headroom


def cmd_headroom(a: argparse.Namespace) -> None:
    from crucible.synth.falsify import FalsifyContext, cheap
    from crucible.synth.meta.headroom import catchable
    from crucible.synth.meta.rsi import verdicts

    _out, dev, _locked, dev_src, _ = _load(a)

    async def go() -> dict:
        ctx = FalsifyContext()
        try:
            t = time.time()
            vs, _ = await verdicts(dev.plants, cheap(), ctx)
            missed = [p.id for p in dev.plants if not vs[p.id].hit]
            h = await catchable(dev.plants, missed, dev_src, ctx)
            secs = time.time() - t
        finally:
            await ctx.close()
        by_op: dict[str, list[int]] = {}
        for p in dev.plants:
            if p.id in set(missed):
                r = by_op.setdefault(p.op.key, [0, 0])
                r[0] += 1
                r[1] += p.id in set(h.catchable)
        return {**h.model_dump(mode="json"), "missed_by_op": by_op, "seconds": round(secs, 1)}

    doc = asyncio.run(go())
    _dump(stats_dir(a) / "headroom.json", doc)
    print(json.dumps({k: doc[k] for k in ("plants", "missed", "headroom_abs", "headroom_rel", "admissible", "operators", "by_class", "missed_by_op", "runs", "seconds")},
                     indent=1))


# ---------------------------------------------------------------------------------------------------------- locked


def cmd_locked(a: argparse.Namespace) -> None:
    """A rebuilt locked set: the held-out planter's variants with a higher per-task cap, on the locked sources (or, with
    ``--domains``, on every clean source of those held-out domains); its per-class counts and minimum detectable effects."""
    from crucible.synth.falsify import FalsifyContext
    from crucible.synth.meta.plant import held_out_planter, plant_all
    from crucible.synth.meta.recall import PlantSet, SetStore
    from crucible.synth.meta.stats import mde
    from crucible.synth.task import Task

    out, _dev, _locked, dev_src, locked_src = _load(a)
    src = json.loads((out / "sources.json").read_text())
    tasks = {k: Task.model_validate(v) for k, v in src["tasks"].items()}
    if a.held_out_domains:
        doms = set(a.held_out_domains.split(","))
        sources = [t for t in tasks.values() if t.domain in doms]
        if {t.domain for t in dev_src} & doms:
            print(f"note: held-out domains {sorted(doms)} also have dev sources; their dev plants would have to be dropped for a domain hold-out")
    else:
        sources = list(locked_src)
    planter = held_out_planter().model_copy(update={"max_per_op": a.max_per_op})

    async def go():
        ctx = FalsifyContext()
        try:
            return await plant_all(sources, planter, ctx)
        finally:
            await ctx.close()

    t = time.time()
    batch = asyncio.run(go())
    ps = PlantSet(name=f"{a.name}-locked", kind="locked", plants=batch.kept).sealed()
    SetStore(out / "sets", locked_root(a)).write(ps)
    counts = _counts(batch.kept)
    clusters = {d: len({p.source_id for p in batch.kept if p.defect == d}) for d in counts}
    doc = {"name": ps.name, "sources": len(sources), "domains": sorted({t.domain for t in sources}), "max_per_op": a.max_per_op, "plants": len(batch.kept),
           "discarded": len(batch.discarded), "counts": counts, "source_tasks_per_class": clusters,
           "mde": {d: mde(n) if n else None for d, n in counts.items()}, "seconds": round(time.time() - t, 1)}
    _dump(stats_dir(a) / f"locked-{a.name}.json", doc)
    print(json.dumps(doc, indent=1))


# ---------------------------------------------------------------------------------------------------------- evolve / e10


def _paired_config(a: argparse.Namespace):
    from crucible.synth.meta.rsi import EvolveConfig, StopRules
    from crucible.synth.meta.update import StatsConfig, UpdateConfig

    st = StatsConfig(paired=True, alpha=0.05, schedule="bonferroni", t_max=a.t_max, aa_replicates=5, seeds=list(range(1, 13)), bet_c=0.75, chunk=a.chunk,
                     n_max=a.n_max, n_min=a.n_min, n_futility=a.n_futility)
    return EvolveConfig(rounds=a.rounds, candidates=a.candidates, view_frac=0.6, seed=a.seed, decide_frac=a.decide_frac, planter_loop=not a.no_planter,
                        meta_loop=not a.no_meta, headroom=not a.no_headroom, planter_reward="headroom" if not a.no_headroom else "miss",
                        locked_gate=a.locked_gate, update=UpdateConfig(rounds=a.rounds, min_band=0.02, stats=st),
                        stop=StopRules(max_rounds=a.rounds, futility_rounds=a.futility_rounds))


def _natural(root: Path, domains: list[str]):
    from crucible.synth.archive import Archive
    from crucible.synth.meta.rsi import load_tasks

    arch = Archive(root)
    cxs = [c for d in domains if arch.path(d).exists() for c in arch.records(d)]
    nat_tasks = {t.id: t for d in domains for t, _ in load_tasks(root / d, statuses=("rejected",))}
    return arch, cxs, nat_tasks


def _make_loop(a: argparse.Namespace, cfg, d: Path, spend, shards_factory, seeded=None):
    from crucible.synth.falsify import FalsifyContext
    from crucible.synth.meta.rsi import Loop, load_real_defects, real_defect_features

    _out, dev, locked, dev_src, locked_src = _load(a)
    root = data_root(a)
    domains = [x for x in a.domains.split(",") if x]
    arch, cxs, nat_tasks = _natural(root, domains)
    ref = real_defect_features(load_real_defects(a.real_defects)) if a.real_defects else None

    def refresh():
        return [c for dm in domains if arch.path(dm).exists() for c in arch.records(dm)]

    return Loop(d, _pool(dev, dev_src), locked, locked_src, cxs, ctx=FalsifyContext(), report_ctx=FalsifyContext(), config=cfg, dev_sources=dev_src,
                builders=["deepseek", "glm", "kimi"], natural_sources=nat_tasks, spend=spend, reference=ref, natural_refresh=refresh, shards=shards_factory,
                seeded=seeded)


def cmd_evolve(a: argparse.Namespace) -> None:
    from crucible.factory.synth_wire import SynthWire
    from crucible.synth.meta.par import ShardPool
    from crucible.synth.meta.rsi import refusals, rows_markdown, swarm_proposers

    root = data_root(a)
    cap = pilot_cap(a, root)
    d = rsi_dir(a) / f"rsi2-pilot-{a.label}"
    d.mkdir(parents=True, exist_ok=a.resume)
    wire = SynthWire(d, cap_usd=cap)
    cfg = _paired_config(a)
    pools: list[ShardPool] = []

    def shards():
        sp = ShardPool(a.workers)
        pools.append(sp)
        return sp

    loop = _make_loop(a, cfg, d, lambda: wire.meter.spent, shards)
    props = swarm_proposers(wire.client("abstractor"), planter_client=wire.client("abstractor"))

    def save() -> None:
        rows = [r for arm in loop.arms for r in arm.rows]
        (d / "rows.json").write_text(json.dumps([r.model_dump(mode="json") for r in rows], indent=1, default=str))
        (d / "rows.md").write_text(rows_markdown(rows) + "\n" + json.dumps(refusals(rows), indent=1) + "\n")
        (d / "cost.json").write_text(json.dumps(wire.cost(), indent=1))

    async def go() -> None:
        try:
            await loop.run(props, control=not a.no_control, save=save, resume=a.resume)
        finally:
            save()
            for p in pools:
                p.close()
        (d / "config.json").write_text(json.dumps({"config": cfg.model_dump(mode="json"), "mu": loop.mu, "args": vars(a)}, indent=1, default=str))

    t = time.time()
    asyncio.run(go())
    print((d / "rows.md").read_text())
    print(json.dumps({"dir": str(d), "cost": wire.cost(), "seconds": round(time.time() - t, 1)}, indent=1, default=str))


def cmd_e10(a: argparse.Namespace) -> None:
    from crucible.factory.synth_wire import SynthWire
    from crucible.synth.meta.ablate import e10_summary, e10_table, run_e10
    from crucible.synth.meta.par import ShardPool
    from crucible.synth.meta.rsi import Proposers, random_proposer, swarm_proposers

    root = data_root(a)
    d = rsi_dir(a) / f"rsi2-pilot-{a.label}"
    d.mkdir(parents=True, exist_ok=True)
    wire = None
    if not a.random:
        wire = SynthWire(d, cap_usd=pilot_cap(a, root))
    cfg = _paired_config(a)
    sp = ShardPool(a.workers)  # one pool for every arm: the shards' class caches serve every ablation (same plants, same seeds)
    seeded: dict = {}

    def props(seed: int):
        if wire is None:
            return Proposers(ops=random_proposer(seed))
        return swarm_proposers(wire.client("abstractor"), planter_client=wire.client("abstractor"))

    rows_out: list = []

    def on_row(r) -> None:
        rows_out.append(r)
        _dump(d / "e10_rows.json", [x.model_dump(mode="json") for x in rows_out])

    async def go():
        return await run_e10(lambda c, dd: _make_loop(a, c, dd, (lambda: wire.meter.spent) if wire else (lambda: 0.0), lambda: sp, seeded), props, cfg, d,
                             ablations=[x for x in a.ablations.split(",") if x], seeds=[int(x) for x in a.seeds.split(",")],
                             spend=(lambda: wire.meter.spent) if wire else None, on_row=on_row)

    t = time.time()
    try:
        rows = asyncio.run(go())
    finally:
        sp.close()
    (d / "e10.md").write_text(e10_table(rows))
    _dump(d / "e10_summary.json", e10_summary(rows))
    print(e10_table(rows))
    print(json.dumps({"dir": str(d), "cost": wire.cost() if wire else 0.0, "seconds": round(time.time() - t, 1)}, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run-id", required=True)
        p.add_argument("--locked-root", default=None)
        p.add_argument("--locked", default=None, help="the locked set's name (default <run-id>-locked)")
        p.add_argument("--workers", type=int, default=10)
        p.add_argument("--domains", default="finance_ap,it_helpdesk")

    p = sub.add_parser("aa")
    common(p)
    p.add_argument("--replicates", type=int, default=5)
    p.set_defaults(fn=cmd_aa)
    p = sub.add_parser("retro")
    common(p)
    p.add_argument("--evolve", default="evolve-e2,evolve-e3-it,evolve-e3-it-specs")
    p.set_defaults(fn=cmd_retro)
    p = sub.add_parser("power")
    common(p)
    p.set_defaults(fn=cmd_power)
    p = sub.add_parser("headroom")
    common(p)
    p.set_defaults(fn=cmd_headroom)
    p = sub.add_parser("locked")
    common(p)
    p.add_argument("--name", required=True)
    p.add_argument("--max-per-op", type=int, default=6)
    p.add_argument("--held-out-domains", default="", help="plant every clean source of these domains (a domain hold-out); default: the locked sources")
    p.set_defaults(fn=cmd_locked)
    for name, fn in (("evolve", cmd_evolve), ("e10", cmd_e10)):
        p = sub.add_parser(name)
        common(p)
        p.add_argument("--label", required=True)
        p.add_argument("--rounds", type=int, default=2)
        p.add_argument("--candidates", type=int, default=4)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--decide-frac", type=float, default=0.5)
        p.add_argument("--t-max", type=int, default=20)
        p.add_argument("--chunk", type=int, default=10)
        p.add_argument("--n-max", type=int, default=40)
        p.add_argument("--n-min", type=int, default=10)
        p.add_argument("--n-futility", type=int, default=20)
        p.add_argument("--futility-rounds", type=int, default=None)
        p.add_argument("--no-planter", action="store_true")
        p.add_argument("--no-meta", action="store_true")
        p.add_argument("--no-headroom", action="store_true")
        p.add_argument("--no-control", action="store_true")
        p.add_argument("--locked-gate", action="store_true")
        p.add_argument("--resume", action="store_true")
        p.add_argument("--real-defects", default=None)
        p.add_argument("--cap", type=float, default=5.0)
        p.set_defaults(fn=fn)
        if name == "e10":
            p.add_argument("--ablations", default="none,R7,NL")
            p.add_argument("--seeds", default="0,1,2")
            p.add_argument("--random", action="store_true", help="the random-DSL proposer (model-free)")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
