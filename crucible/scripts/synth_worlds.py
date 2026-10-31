"""``synth.py worlds ...``: World families and instances on the Tinker swarm (part2-paper-core §11, §12).

    synth.py worlds catalogue --n 40 --per-area 3 --cap 1                 # the swarm lists domains, deduped (catalogue.json)
    synth.py worlds build --pick 5 --parallel 5 --cap 15                  # families from the catalogue, concurrently
    synth.py worlds build --keywords "IT service desk ticket handling" --cap 3
    synth.py worlds instances --family all --n 3                          # no model call
    synth.py worlds calibrate --cap 0.5                                   # the realism panel's lambda on reviewed anchors
    synth.py worlds gate --family all [--no-panel] [--only density,surface] --cap 1
    synth.py worlds report                                                # report.json + report.md

Everything lives under ``<data-root>/worlds/`` (default data root: the main checkout's data dir + ``/synth``). **Spend:**
every receipt is under a path carrying "worlds"; each run is capped by ``--cap``, by what ``WORLDS_CAP_USD`` (30) has left
over those receipts, and by what the milestone budget has left. Concurrent family builds split the run's cap between them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any


def data_root(a: argparse.Namespace) -> Path:
    if a.data_root:
        return Path(a.data_root)
    if os.environ.get("CRUCIBLE_DATA_DIR"):
        return Path(os.environ["CRUCIBLE_DATA_DIR"]) / "synth"
    from crucible.core.config import get_config

    return get_config().data_path() / "synth"


def worlds_root(a: argparse.Namespace) -> Path:
    return Path(a.worlds_root) if a.worlds_root else data_root(a) / "worlds"


def worlds_cap(a: argparse.Namespace) -> float:
    from crucible.factory.synth_wire import MILESTONE_CAP_USD, WORLDS_CAP_USD, milestone_spent, work_spent

    root = data_root(a)
    ms = milestone_spent(root)
    ws = work_spent(root, "worlds")
    # the shared backstop: the Part 2 total (lead, 2026-10-29: $150, set as MILESTONE_CAP_USD in the environment; the
    # module's 60 is the M4-era cap and stays the default)
    shared = float(os.environ.get("MILESTONE_CAP_USD") or MILESTONE_CAP_USD)
    cap = min(float(a.cap), round(WORLDS_CAP_USD - ws, 4), round(shared - ms, 4))
    print(f"shared spend {ms:.4f} of {shared}; worlds spend {ws:.4f} of {WORLDS_CAP_USD}; this run's cap {cap:.4f}", flush=True)
    if cap < 0.02:
        raise SystemExit("the worlds budget is spent: refusing to start")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not in the environment (set -a; . <repo>/.env; set +a)")
    return cap


def _wire(run_dir: Path, cap_total: float) -> Any:
    from crucible.factory.synth_wire import SYNTH_ROLES, WORLD_ROLE_TABLE, SynthWire

    return SynthWire(run_dir, cap_usd=cap_total, table={**SYNTH_ROLES, **WORLD_ROLE_TABLE}, seed_by_tag=True)


def _prior(run_dir: Path) -> float:
    p = run_dir / "spend.jsonl"
    if not p.exists():
        return 0.0
    return sum(float(json.loads(x).get("usd") or 0) for x in p.read_text().splitlines() if x.strip())


# ---------------------------------------------------------------------------------------------------------- catalogue

def cmd_catalogue(a: argparse.Namespace) -> None:
    from crucible.synth.catalogue import build_catalogue, load

    root = worlds_root(a)
    cap = worlds_cap(a)
    d = root / "_catalogue"
    wire = _wire(d, _prior(d) + cap)
    t0 = time.monotonic()
    path = root / "catalogue.json"
    cat = asyncio.run(build_catalogue(wire.client("domain_cataloguer"), per_area=a.per_area, cat=load(path), path=path, target=a.n))
    out = {"entries": len(cat.entries), "duplicates": len(cat.duplicates), "refused": len(cat.refused), "calls": len(cat.calls),
           "elapsed_s": round(time.monotonic() - t0, 1), "cost": wire.cost()}
    print(json.dumps(out, indent=1))
    for e in cat.entries:
        print(f"  [{e.function.split(':')[0]}] {e.keyword}")


# ---------------------------------------------------------------------------------------------------------- build

def cmd_build(a: argparse.Namespace) -> None:
    from crucible.synth.catalogue import load, pick
    from crucible.synth.families import FamilyConfig, build_family, family_id, load_state

    root = worlds_root(a)
    cap = worlds_cap(a)
    if a.keywords:
        fn_of = {e.keyword.lower(): e.function for e in load(root / "catalogue.json").entries}
        todo = [(k.strip(), fn_of.get(k.strip().lower(), "")) for k in a.keywords.split(";") if k.strip()]
    else:
        cat = load(root / "catalogue.json")
        done = [p.name for p in root.iterdir() if (p / "state.json").exists()] if root.exists() else []
        chosen = pick(cat, a.pick, exclude=[e.keyword for e in cat.entries if family_id(e.keyword) in done and not a.include_built])
        todo = [(e.keyword, e.function) for e in chosen]
    if not todo:
        raise SystemExit("nothing to build")
    each = round(min(a.per_family_cap, cap / len(todo)), 4)
    cfg = FamilyConfig(**({"k_min": a.k_min} if a.k_min else {}), max_debug_rounds=a.rounds, actors=not a.no_actors, guidance=a.guidance, density=not a.no_density,
                       data_attempts=a.data_attempts)
    print(f"building {len(todo)} families, {a.parallel} at a time, at most {each} USD each", flush=True)

    async def one(kw: str, fn: str, sem: asyncio.Semaphore) -> dict[str, Any]:
        fdir = root / family_id(kw)
        async with sem:
            prior = _prior(fdir)
            wire = _wire(fdir, prior + each)
            t0 = time.monotonic()
            try:
                st = await build_family(kw, fdir, wire.family_clients(), function=fn, cfg=cfg, spent=lambda: wire.meter.spent,
                                        restart_from=a.restart_from or None, redo=[x for x in a.redo.split(",") if x])
                status = st.status
            except Exception as e:  # noqa: BLE001 - a budget stop or a crash is recorded per family; the others go on
                status = f"stopped: {type(e).__name__}: {str(e)[:300]}"
                st = load_state(fdir)
            (fdir / "build_cost.json").write_text(json.dumps(wire.cost(), indent=1) + "\n")
            return {"family": fdir.name, "keyword": kw, "status": status, "wall_s": round(time.monotonic() - t0, 1), "usd_this_run": round(wire.meter.spent - prior, 4),
                    "usd_total": round(wire.meter.spent, 4), "stages": {k: [v.status, v.elapsed_s, v.usd] for k, v in (st.stages.items() if st else [])},
                    "reasons": (st.reasons[-2:] if st else [])}

    async def go() -> list[dict[str, Any]]:
        sem = asyncio.Semaphore(a.parallel)
        return list(await asyncio.gather(*(one(k, f, sem) for k, f in todo)))

    t0 = time.monotonic()
    res = asyncio.run(go())
    out = {"wall_s": round(time.monotonic() - t0, 1), "families": res}
    root.mkdir(parents=True, exist_ok=True)
    (root / f"_build-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps(out, indent=1, default=str))


# ---------------------------------------------------------------------------------------------------------- instances

def _families(root: Path, which: str) -> list[Path]:
    if which == "all":
        return sorted(p for p in root.iterdir() if (p / "app" / "world.json").exists()) if root.exists() else []
    return [root / f for f in which.split(",")]


def cmd_instances(a: argparse.Namespace) -> None:
    from crucible.synth.families import instance_config, make_instance

    root = worlds_root(a)
    out = []
    for fdir in _families(root, a.family):
        for i in range(a.start, a.start + a.n):
            t0 = time.monotonic()
            try:
                cfg = instance_config(fdir.name, i, base_seed=a.base_seed)
                r = make_instance(fdir, cfg, k_min=a.k_min)
                if a.pick_actor:  # the acting user chosen by execution (worldgates.choose_actor), kept in the config
                    from crucible.synth.domain import DomainSpec
                    from crucible.synth.worldgates import choose_actor

                    ch = asyncio.run(choose_actor(Path(r.dir)))
                    acting = DomainSpec.model_validate_json((Path(r.dir) / "domain.json").read_text()).permissions.acting_param
                    if ch.get("user") is not None and ch["user"] != r.params.get(acting):
                        r = make_instance(fdir, cfg.model_copy(update={"params": {**cfg.params, acting: ch["user"]}}), k_min=a.k_min)
                    (Path(r.dir) / "actor.json").write_text(json.dumps(ch, indent=1, default=str) + "\n")
                out.append({"family": fdir.name, "instance": r.id, "rows": r.rows, "noise": r.noise, "params": r.params, "s": round(time.monotonic() - t0, 3)})
            except Exception as e:  # noqa: BLE001 - recorded per instance
                out.append({"family": fdir.name, "index": i, "error": f"{type(e).__name__}: {str(e)[:300]}"})
    print(json.dumps(out, indent=1, default=str))


# ---------------------------------------------------------------------------------------------------------- panel

async def _finance_ap_summaries(work: Path, seeds: tuple[int, ...] = (0, 1)) -> list[dict[str, Any]]:
    from crucible.factory.synth_wire import finance_ap_world
    from crucible.synth.domain import finance_ap_spec
    from crucible.synth.worldgates import world_summary

    spec = finance_ap_spec()
    out = []
    for s in seeds:
        w = await finance_ap_world(s, str(work))
        try:
            await w.reset(s)
            docs = {}
            d = await w.call("kb.read_doc", {"path": "ap_policy.md"})
            if d.ok:
                docs["AP policy (policy)"] = d.value if isinstance(d.value, str) else json.dumps(d.value)
            out.append(await world_summary(w, domain=f"{spec.keyword}: {spec.description}", skills=[f"{x.id}: {x.description}" for x in spec.skills], docs=docs, seed=s))
        finally:
            await w.close()
    return out


def _panel(wire: Any, root: Path) -> Any:
    from crucible.synth.worldgates import RealismPanel

    cl = wire.clients("realism_rater")
    return RealismPanel(members=[(f"realism-{c.family}", c) for c in cl], queue=root / "human_queue.jsonl")


def cmd_calibrate(a: argparse.Namespace) -> None:
    from crucible.synth.worldgates import anchors_from, calibrate_panel, rejected_anchors

    root = worlds_root(a)
    cap = worlds_cap(a)
    d = root / "_panel"
    wire = _wire(d, _prior(d) + cap)
    panel = _panel(wire, root)

    async def go() -> dict[str, Any]:
        pos = await _finance_ap_summaries(d / "work")
        anchors = anchors_from(pos)
        if a.rejected_anchors:  # worlds-2: instances a realism gate rejected are negative anchors with known truth
            anchors += await rejected_anchors(root, per_family=a.rejected_anchors)
        return await calibrate_panel(panel, anchors)

    cal = asyncio.run(go())
    cal["cost"] = wire.cost()
    (root / "realism_calibration.json").write_text(json.dumps(cal, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in cal.items() if k != "votes"}, indent=1, default=str))


# ---------------------------------------------------------------------------------------------------------- gates

def cmd_gate(a: argparse.Namespace) -> None:
    import math

    from crucible.synth import policy
    from crucible.synth.families import InstanceConfig, list_instances, make_instance
    from crucible.synth.worldgates import gate_instance

    root = worlds_root(a)
    panel = None
    wire = None
    if not a.no_panel:
        cal_p = root / "realism_calibration.json"
        if not cal_p.exists():
            raise SystemExit("calibrate the realism panel first (worlds calibrate), or pass --no-panel")
        cap = worlds_cap(a)
        d = root / "_panel"
        wire = _wire(d, _prior(d) + cap)
        panel = _panel(wire, root)
        panel.lam = float(json.loads(cal_p.read_text())["lam"])
    only = [x for x in a.only.split(",") if x] or None
    rows = []
    max_reseed = int(policy.param("R-density", "max_reseed"))

    async def act_on_remedies(fdir: Path, inst: Path, g: Any) -> Any:
        """Carry out policy.resolve's remedies (DESIGN item 24, R-density): reseed_denser regenerates the instance with
        1.5x the rows per target collection (same seed, profile, policy variant and acting user; the earlier gates
        record is kept beside it) and gates it again, up to max_reseed times; rebuild_stage is recorded in the family's
        rebuild_requests.jsonl (a paid build the operator runs under its own cap); not_used keeps the reasons."""
        tries = 0
        while any(r["action"] == "reseed_denser" for r in g.remedies) and tries < max_reseed:
            tries += 1
            info = json.loads((inst / "instance.json").read_text())
            cfg = InstanceConfig.model_validate(info["config"])
            man = json.loads((fdir / "app" / "world.json").read_text())
            k = int(cfg.k_min or a.k_min or man.get("k_min") or policy.param("R-density", "k_min"))
            (inst / "gates.json").replace(inst / f"gates.reseed{tries - 1}.json")
            make_instance(fdir, cfg.model_copy(update={"k_min": math.ceil(k * 1.5)}))
            # a new start state: every gate runs again, whatever --only asked
            g = await gate_instance(fdir, inst, panel=panel, reseed_tries=tries, can_reseed=tries < max_reseed, restartable=a.allow_rebuild)
        for r in g.remedies:
            if r["action"] == "rebuild_stage":
                with (fdir / "rebuild_requests.jsonl").open("a") as fh:
                    fh.write(json.dumps({"t": time.time(), "instance": inst.name, **r}) + "\n")
        return g

    async def go() -> None:
        sem = asyncio.Semaphore(a.parallel)

        async def one(fdir: Path, inst: Path) -> None:
            async with sem:
                t0 = time.monotonic()
                g = await gate_instance(fdir, inst, panel=panel, k_min=a.k_min, only=only, restartable=a.allow_rebuild)
                g = await act_on_remedies(fdir, inst, g)
                rows.append({"family": fdir.name, "instance": inst.name, "usable": g.usable, "why": g.why, "s": round(time.monotonic() - t0, 1),
                             "remedies": [(r["gate"], r["action"]) for r in g.remedies],
                             "gates": {k: v.passed for k, v in g.gates.items()},
                             "density": g.gates["density"].details.get("per_tool") and [(p["tool"], p.get("n_distinct"), p.get("n_heldout"))
                                                                                       for p in g.gates["density"].details["per_tool"]]
                             if "density" in g.gates else None,
                             "density_by_collection": {"shown": g.gates["density"].details.get("best_by_collection"),
                                                       "heldout": g.gates["density"].details.get("heldout_by_collection")} if "density" in g.gates else None})

        jobs = []
        for fdir in _families(root, a.family):
            insts = list_instances(fdir)
            if a.instances != "all":
                insts = [p for p in insts if p.name in a.instances.split(",")]
            jobs += [one(fdir, i) for i in insts]
        await asyncio.gather(*jobs)

    t0 = time.monotonic()
    asyncio.run(go())
    out = {"wall_s": round(time.monotonic() - t0, 1), "instances": sorted(rows, key=lambda r: (r["family"], r["instance"])),
           "cost": wire.cost() if wire else None}
    print(json.dumps(out, indent=1, default=str))


# ---------------------------------------------------------------------------------------------------------- report

def cmd_report(a: argparse.Namespace) -> None:
    from crucible.synth.families import list_instances, load_state

    root = worlds_root(a)
    fams = []
    for fdir in sorted(p for p in root.iterdir() if (p / "state.json").exists()) if root.exists() else []:
        st = load_state(fdir)
        assert st is not None
        insts = []
        for inst in list_instances(fdir):
            gp = inst / "gates.json"
            g = json.loads(gp.read_text()) if gp.exists() else None
            info = json.loads((inst / "instance.json").read_text())
            dens = (g or {}).get("gates", {}).get("density", {}).get("details", {})
            insts.append({"id": inst.name, "gen_s": info.get("elapsed_s"), "rows": sum(info.get("rows", {}).values()), "noise": info.get("noise"),
                          "usable": bool(g and g.get("usable")), "gates": {k: v["passed"] for k, v in (g or {}).get("gates", {}).items()},
                          "gate_s": round(sum(v.get("elapsed_s", 0) for v in (g or {}).get("gates", {}).values()), 1),
                          "why": (g or {}).get("why"), "density_best": dens.get("best"), "density_median": dens.get("median"),
                          "density_split": dens.get("split", "all"), "density_shown": dens.get("best_by_collection"), "density_heldout": dens.get("heldout_by_collection"),
                          "density_reached": f"{dens.get('reached')}/{dens.get('references')}" if dens else None})
        fams.append({"family": fdir.name, "keyword": st.keyword, "status": st.status, "reasons": st.reasons[-2:], "wall_s": st.elapsed_s, "usd": st.usd,
                     "tools": st.tools, "tables": st.tables, "tests": st.tests_written, "dropped": len(st.tests_dropped), "debug_rounds": st.debug_rounds,
                     "pass_history": st.pass_history, "stages": {k: {"status": v.status, "s": v.elapsed_s, "usd": v.usd, "attempts": v.attempts} for k, v in st.stages.items()},
                     "instances": insts})
    frozen = [f for f in fams if f["status"] == "frozen"]
    allinst = [i for f in fams for i in f["instances"]]
    gate_names = sorted({k for i in allinst for k in i["gates"]})
    summary = {"families": len(fams), "frozen": len(frozen), "instances": len(allinst), "usable": sum(i["usable"] for i in allinst),
               "gate_pass": {g: f"{sum(1 for i in allinst if i['gates'].get(g))}/{sum(1 for i in allinst if g in i['gates'])}" for g in gate_names},
               "usd_per_frozen_family": round(statistics.mean([f["usd"] for f in frozen]), 4) if frozen else None,
               "wall_s_per_frozen_family": round(statistics.mean([f["wall_s"] for f in frozen]), 1) if frozen else None,
               "usd_total_families": round(sum(f["usd"] for f in fams), 4)}
    rep = {"summary": summary, "families": fams}
    (root / "report.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    lines = ["# World families", "", "| family | status | wall s | $ | tools | tables | tests (dropped) | debug | instances usable |", "|---|---|---|---|---|---|---|---|---|"]
    for f in fams:
        lines.append(f"| {f['family']} | {f['status']} | {f['wall_s']} | {f['usd']:.3f} | {f['tools']} | {f['tables']} | {f['tests']} ({f['dropped']}) | "
                     f"{f['debug_rounds']} | {sum(i['usable'] for i in f['instances'])}/{len(f['instances'])} |")
    lines += ["", "| family | instance | gen s | rows | noise | gates passed | density best / median / reached | usable | why |", "|---|---|---|---|---|---|---|---|---|"]
    for f in fams:
        for i in f["instances"]:
            ok = ",".join(k for k, v in i["gates"].items() if v)
            lines.append(f"| {f['family']} | {i['id']} | {i['gen_s']} | {i['rows']} | {i['noise']} | {ok} | {i['density_best']} / {i['density_median']} / "
                         f"{i['density_reached']} | {i['usable']} | {'; '.join(i['why'] or [])[:200]} |")
    lines += ["", "```", json.dumps(summary, indent=1), "```"]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def add_parser(sub: Any) -> None:
    w = sub.add_parser("worlds", help="World families and instances (catalogue, build, instances, calibrate, gate, report)")
    w.add_argument("action", choices=["catalogue", "build", "instances", "calibrate", "gate", "report"])
    w.add_argument("--worlds-root", default=None)
    w.add_argument("--cap", type=float, default=1.0)
    w.add_argument("--n", type=int, default=40, help="catalogue: entries wanted; instances: per family")
    w.add_argument("--per-area", type=int, default=3)
    w.add_argument("--keywords", default="", help="build: ';'-separated keywords (else --pick from the catalogue)")
    w.add_argument("--pick", type=int, default=5)
    w.add_argument("--include-built", action="store_true", help="build: also resume families already started")
    w.add_argument("--parallel", type=int, default=5)
    w.add_argument("--per-family-cap", type=float, default=4.0)
    w.add_argument("--k-min", type=int, default=None, help="rows per target collection (default: R-density's k_min)")
    w.add_argument("--rounds", type=int, default=3)
    w.add_argument("--no-actors", action="store_true")
    w.add_argument("--restart-from", default="", help="build: rebuild from this stage on (old files copied aside)")
    w.add_argument("--redo", default="", help="build: rebuild exactly these stages, e.g. tests,debug,base,freeze")
    w.add_argument("--guidance", default="")
    w.add_argument("--family", default="all")
    w.add_argument("--instances", default="all")
    w.add_argument("--start", type=int, default=0)
    w.add_argument("--base-seed", type=int, default=0)
    w.add_argument("--no-panel", action="store_true")
    w.add_argument("--pick-actor", action="store_true", help="instances: choose the acting user by execution (worldgates.choose_actor)")
    w.add_argument("--only", default="")
    w.add_argument("--allow-rebuild", action="store_true", help="gate: failing gates may resolve to rebuild_stage (recorded in rebuild_requests.jsonl)")
    w.add_argument("--no-density", action="store_true", help="build: skip the data stage's density measurement and feedback")
    w.add_argument("--data-attempts", type=int, default=3, help="build: data-generator conversations (each may be refused or sent back for density)")
    w.add_argument("--rejected-anchors", type=int, default=0, help="calibrate: up to this many realism-gate-rejected instances per family as negative anchors")
    w.set_defaults(fn=lambda a: {"catalogue": cmd_catalogue, "build": cmd_build, "instances": cmd_instances, "calibrate": cmd_calibrate,
                                 "gate": cmd_gate, "report": cmd_report}[a.action](a))
