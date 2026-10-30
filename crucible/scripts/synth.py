#!/usr/bin/env python3
"""crucible.synth on the Tinker swarm (part2-paper-core M4): World synthesis, task synthesis, certification, reports.

    synth.py worldgen --keyword "IT helpdesk ticket triage" --name it_helpdesk --cap 8
    synth.py propose  --domain finance_ap --n 20 --cap 2              # proposals only (no oracles, no falsifiers)
    synth.py certify  --domain finance_ap --n 20 --cap 25             # propose -> oracle -> contract -> falsify/repair -> certify
    synth.py certify  --domain it_helpdesk --world <run dir of worldgen>/world --n 20 --cap 25
    synth.py report   --domain finance_ap [--run-id R]                # a run's report.md (default: the latest run)
    synth.py budget                                                   # the milestone's Tinker spend so far
    synth.py smoke --cap 0.5                                          # one JSON call per model of the assignment
  the regularised RSI loop (part2-paper-core §9; every receipt under a path with "rsi", capped by RSI_CAP_USD = 40):
    synth.py reseed  --world <world dir> --name it_helpdesk --target tickets=40,agents=20   # denser seeded World
    synth.py certify --domain finance_ap --invoices 30 --budget rsi --tag rsi-dense ...     # tasks on a denser World
    synth.py plant   --run-id R --fixtures            # sources -> dev set + locked set (data/sealed/synth-locked)
    synth.py recall  --run-id R [--adjudicate 40]     # E1: FR/FA on the locked set, residuals, capture-recapture, PPI
    synth.py evolve  --run-id R --label e2 --rounds 4 # E2: loop arm vs equal-budget random-DSL control
    synth.py evolve  --run-id R --label e3-it --evidence it_helpdesk --no-control           # E3: one domain's evidence
    synth.py lemma-test --run-id R --domain finance_ap --lemmas <json>                      # S: lemma acceptance

Artifacts: ``<data-root>/<domain>/<run-id>/`` (default data root: ``$CRUCIBLE_DATA_DIR/synth``, else the config's data
dir + ``/synth``), the counterexample archive ``<data-root>/<domain>/archive.jsonl``. **Spend:** each run has one hard
cap (``--cap``, booked in its ``spend.jsonl``), and the cap is clipped to what the milestone budget
(``MILESTONE_CAP_USD`` = 60) has left over every ``spend.jsonl`` under the data root; a run with nothing left is refused.
The key comes from the environment only: ``set -a; . <repo>/.env; set +a`` in the shell that starts this script.
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
if str(HERE) not in sys.path:  # the fixture domains' World factories live in tests.unit.synth_fixtures
    sys.path.append(str(HERE))


def data_root(a: argparse.Namespace) -> Path:
    if a.data_root:
        return Path(a.data_root)
    if os.environ.get("CRUCIBLE_DATA_DIR"):
        return Path(os.environ["CRUCIBLE_DATA_DIR"]) / "synth"
    from crucible.core.config import get_config

    return get_config().data_path() / "synth"


def run_cap(a: argparse.Namespace, root: Path) -> float:
    """The run's hard cap: ``--cap``, clipped to what the M4 milestone budget has left and, for the RSI work
    (``--budget rsi``, every run path carrying "rsi"), to what ``RSI_CAP_USD`` has left over that work's receipts."""
    from crucible.factory.synth_wire import MILESTONE_CAP_USD, RSI_CAP_USD, milestone_spent, work_spent

    spent = milestone_spent(root)
    left = round(MILESTONE_CAP_USD - spent, 4)
    cap = min(float(a.cap), left)
    print(f"milestone spend so far {spent:.4f} of {MILESTONE_CAP_USD}", flush=True)
    if getattr(a, "budget", "milestone") == "rsi":
        if "rsi" not in (getattr(a, "tag", "") or "") and getattr(a, "cmd", "") not in ("recall", "evolve", "lemma-test", "plant"):
            raise SystemExit("an RSI run needs a tag containing 'rsi' (its receipts are attributed by path)")
        ws = work_spent(root, "rsi")
        cap = min(cap, round(RSI_CAP_USD - ws, 4))
        print(f"RSI work spend so far {ws:.4f} of {RSI_CAP_USD}", flush=True)
    print(f"this run's cap {cap:.4f}", flush=True)
    if cap < 0.05:
        raise SystemExit("the Tinker budget is spent: refusing to start")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not in the environment (set -a; . <repo>/.env; set +a)")
    return cap


def new_run(root: Path, domain: str, tag: str) -> tuple[str, Path]:
    rid = time.strftime("%Y%m%d-%H%M%S") + (f"-{tag}" if tag else "")
    d = root / domain / rid
    d.mkdir(parents=True, exist_ok=False)
    return rid, d


def cmd_worldgen(a: argparse.Namespace) -> None:
    from crucible.factory.synth_wire import SynthWire
    from crucible.synth.worldgen import worldgen

    root = data_root(a)
    cap = run_cap(a, root)
    _rid, d = new_run(root, a.name, a.tag)
    wire = SynthWire(d, cap_usd=cap)
    rep = asyncio.run(worldgen(a.keyword, d / "world", wire.worldgen_clients(), guidance=a.guidance, max_debug_rounds=a.rounds))
    out = {"run": str(d), "report": rep.model_dump(mode="json"), "cost": wire.cost()}
    (d / "worldgen_cost.json").write_text(json.dumps(out["cost"], indent=1) + "\n")
    print(json.dumps(out, indent=1, default=str))


def _env(a: argparse.Namespace, d: Path):
    from crucible.factory.synth_wire import finance_ap_env, generated_env

    if a.domain == "finance_ap":
        return finance_ap_env(d / "work", seeds=tuple(int(x) for x in a.seeds.split(",")), invoices=a.invoices)
    if not a.world:
        raise SystemExit("a generated domain needs --world <worldgen output dir>")
    w = Path(a.world)
    rep = json.loads((w / "worldgen_report.json").read_text())
    if rep.get("status") != "accepted":
        raise SystemExit(f"the World at {w} was not accepted: {rep.get('status')}")
    design = json.loads((w / "design.json").read_text())
    return generated_env(w, d / "work", name=a.domain, description=f"{rep['keyword']}: {design.get('description', '')}")


def _runner(a: argparse.Namespace, *, stage_only: bool = False):
    from crucible.factory.synth_wire import SynthWire
    from crucible.synth.falsify.agents import AgentBudget
    from crucible.synth.pipeline import RunConfig, Runner

    root = data_root(a)
    cap = run_cap(a, root)
    rid, d = new_run(root, a.domain, a.tag)
    wire = SynthWire(d, cap_usd=cap)
    cfg = RunConfig(n_tasks=a.n, batch=a.batch, max_rounds=a.rounds, concurrency=a.concurrency, calibrate=not a.no_calibrate,
                    alt_budget=AgentBudget(max_calls=a.alt_calls, max_turns=a.alt_calls + 4),
                    atk_budget=AgentBudget(max_calls=a.atk_calls, max_turns=a.atk_calls + 2), show_reasons=not a.blind_panel)
    env = _env(a, d)
    r = Runner(env, wire.run_roles(), cfg, d, root, run_id=rid, cost=wire.cost, families=wire.families())
    return r, d, wire


def cmd_propose(a: argparse.Namespace) -> None:
    r, d, wire = _runner(a)

    async def go() -> None:
        await r.survey()
        recs = await r.propose()
        r.report.tasks = recs
        r.save()

    asyncio.run(go())
    print(json.dumps({"run": str(d), "proposed": len(r.report.tasks), "cost": wire.cost()}, indent=1))


def cmd_certify(a: argparse.Namespace) -> None:
    r, d, wire = _runner(a)
    rep = asyncio.run(r.run())
    print((d / "report.md").read_text())
    print(json.dumps({"run": str(d), "funnel": rep.funnel, "cost": wire.cost(), "stopped": rep.stopped}, indent=1, default=str))


def cmd_report(a: argparse.Namespace) -> None:
    root = data_root(a) / a.domain
    runs = sorted(p for p in root.iterdir() if (p / "report.md").exists()) if root.exists() else []
    if not runs:
        raise SystemExit(f"no runs under {root}")
    d = root / a.run_id if a.run_id else runs[-1]
    print((d / "report.md").read_text())


def cmd_budget(a: argparse.Namespace) -> None:
    from crucible.factory.synth_wire import MILESTONE_CAP_USD, RSI_CAP_USD, milestone_spent, work_spent

    root = data_root(a)
    rows = {}
    for p in sorted(root.rglob("spend.jsonl")):
        tot = sum(float(json.loads(x).get("usd") or 0) for x in p.read_text().splitlines() if x.strip())
        rows[str(p.parent.relative_to(root))] = round(tot, 4)
    print(json.dumps({"runs": rows, "spent": milestone_spent(root), "cap": MILESTONE_CAP_USD,
                      "work": {"tag": a.work, "spent": work_spent(root, a.work), "cap": RSI_CAP_USD if a.work == "rsi" else None}}, indent=1))


def cmd_smoke(a: argparse.Namespace) -> None:
    """One small JSON call per distinct model of the assignment: the key, the tokenizers, the meter and the parser."""
    from crucible.factory.synth_wire import SynthWire

    root = data_root(a)
    cap = run_cap(a, root)
    _rid, d = new_run(root, "_smoke", a.tag)
    wire = SynthWire(d, cap_usd=cap)
    seen: dict[str, tuple[str, int]] = {}
    for role, ms in wire.table.items():
        for i, m in enumerate(ms):
            seen.setdefault(m, (role, i))

    async def go() -> dict[str, object]:
        out: dict[str, object] = {}
        for m, (role, i) in seen.items():
            c = wire.client(role, i)
            t0 = time.time()
            try:
                doc = await c.complete_json([{"role": "user", "content": 'Reply with the JSON object {"ok": true, "sum": <the value of 2+3>}.'}],
                                            schema=["ok"], max_tokens=1024, tag="smoke:models")
                out[m] = {"ok": doc.get("ok"), "sum": doc.get("sum"), "s": round(time.time() - t0, 1)}
            except Exception as e:  # noqa: BLE001 - the smoke reports every model
                out[m] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        return out

    res = asyncio.run(go())
    print(json.dumps({"run": str(d), "models": res, "cost": wire.cost()}, indent=1, default=str))


# ---------------------------------------------------------------------------------------------------------- RSI (§9, E1-E3)

def cmd_reseed(a: argparse.Namespace) -> None:
    """A denser start state for an accepted World (the seeding knob): the swarm seeder adds sibling rows per table."""
    from crucible.factory.synth_wire import SynthWire
    from crucible.synth.worldgen import reseed

    root = data_root(a)
    cap = run_cap(a, root)
    _rid, d = new_run(root, a.name, a.tag)
    wire = SynthWire(d, cap_usd=cap)
    target = {k: int(v) for k, v in (x.split("=") for x in a.target.split(","))}
    rep = asyncio.run(reseed(Path(a.world), d / "world", wire.client("seeder"), target=target, guidance=a.guidance))
    out = {"run": str(d), "report": rep.model_dump(mode="json"), "cost": wire.cost()}
    (d / "reseed_cost.json").write_text(json.dumps(out["cost"], indent=1) + "\n")
    print(json.dumps(out, indent=1, default=str))


def rsi_dir(a: argparse.Namespace) -> Path:
    return data_root(a) / "rsi" / a.run_id


def locked_root(a: argparse.Namespace) -> Path:
    return Path(a.locked_root) if a.locked_root else data_root(a).parent / "sealed" / "synth-locked"


def _fixture_tasks() -> list:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from tests.unit import synth_fixtures as fx  # hand-written fixtures (the M2/M3 test domains ap-test, desk-test)

    return [fx.anchors()[i] for i in range(5)] + [fx.in_domain(fx.t5_batch())]


def _domains(a: argparse.Namespace) -> list[str]:
    return [x for x in a.domains.split(",") if x]


def cmd_plant(a: argparse.Namespace) -> None:
    """Sources (admitted tasks of the domains + the fixtures), re-checked clean, split into dev and locked sources; the
    dev set (base planter) and the locked set (held-out variants, under the locked root; never committed)."""
    from crucible.synth.falsify import FalsifyContext
    from crucible.synth.meta.recall import SetStore
    from crucible.synth.meta.rsi import SourceInfo, build_sets, load_tasks, recheck, split_sources

    root = data_root(a)
    out = rsi_dir(a)
    out.mkdir(parents=True, exist_ok=False)

    async def go() -> dict:
        cands = []
        for dom in _domains(a):
            cands += load_tasks(root / dom)
        if a.fixtures:
            cands += [(t, SourceInfo(task_id=t.id, domain=t.domain, run="fixtures", status="certified")) for t in _fixture_tasks()]
        ctx = FalsifyContext()
        try:
            clean, infos = await recheck(cands, ctx)
            dev_src, locked_src = split_sources(clean, seed=a.seed)
            dev, locked, info = await build_sets(dev_src, locked_src, ctx, name=a.run_id)
        finally:
            await ctx.close()
        store = SetStore(out / "sets", locked_root(a))
        store.write(dev)
        lp = store.write(locked)
        (out / "sources.json").write_text(json.dumps({"dev": [t.id for t in dev_src], "locked": [t.id for t in locked_src],
                                                      "infos": [i.model_dump(mode="json") for i in infos],
                                                      "tasks": {t.id: t.model_dump(mode="json") for t in clean}}, indent=1, default=str))
        rep = {"run": str(out), "locked_set": str(lp), "dev_plants": len(dev.plants), "locked_plants": len(locked.plants), "sources": len(cands),
               "clean": len(clean), "dev_sources": len(dev_src), "locked_sources": len(locked_src), "plant": info,
               "alarms": {i.task_id: i.alarms for i in infos if i.alarms}, "mut_n": {i.task_id: i.mut_n for i in infos}}
        (out / "plant_report.json").write_text(json.dumps(rep, indent=1, default=str))
        return rep

    print(json.dumps(asyncio.run(go()), indent=1, default=str))


def _load(a: argparse.Namespace):
    from crucible.synth.meta.recall import SetStore
    from crucible.synth.task import Task

    out = rsi_dir(a)
    store = SetStore(out / "sets", locked_root(a))
    dev = store.read(f"{a.run_id}-dev", "dev")
    locked = store.read(f"{a.run_id}-locked", "locked", purpose="report")  # E1-E3 report on it; the loop never reads it
    src = json.loads((out / "sources.json").read_text())
    tasks = {k: Task.model_validate(v) for k, v in src["tasks"].items()}
    return out, dev, locked, [tasks[i] for i in src["dev"]], [tasks[i] for i in src["locked"]]


def _natural(a: argparse.Namespace, domains: list[str]):
    from crucible.synth.archive import Archive
    from crucible.synth.meta.rsi import load_tasks

    root = data_root(a)
    arch = Archive(root)
    cxs = [c for d in domains if arch.path(d).exists() for c in arch.records(d)]
    nat_tasks = {t.id: t for d in domains for t, _ in load_tasks(root / d, statuses=("rejected",))}
    return cxs, nat_tasks


def cmd_recall(a: argparse.Namespace) -> None:
    """E1: the FR/FA matrix on the locked set (Wilson intervals), residual estimates and capture–recapture; with
    ``--adjudicate N`` also MUT's suspects decided by the panel (at most N judgements) and their PPI interval."""
    from crucible.synth.falsify import FalsifyContext, cheap
    from crucible.synth.meta.recall import measure, report_markdown
    from crucible.synth.meta.rsi import e1_estimates, summary

    out, _dev, locked, _dev_src, locked_src = _load(a)
    cxs, _ = _natural(a, _domains(a))
    res_dir = out / "e1"
    res_dir.mkdir(exist_ok=True)

    async def go() -> dict:
        ctx = FalsifyContext()
        try:
            m = await measure(locked.plants, locked_src, cheap(), ctx, set_name=locked.name, set_kind="locked", purpose="report")
            dev_m = await measure(_dev.plants, _dev_src, cheap(), ctx, set_name=_dev.name, set_kind="dev")
            res, cap = e1_estimates(m, cxs)
            doc = {"matrix": m.model_dump(mode="json"), "summary": summary(m), "dev_summary": summary(dev_m), "residuals": res, "capture_recapture": cap,
                   "natural": {"counterexamples": len(cxs), "domains": _domains(a)}}
            adj = None
            if a.adjudicate:
                adj = await _adjudicated(a, out, locked, locked_src, ctx)
                doc["adjudicated"] = adj
        finally:
            await ctx.close()
        (res_dir / "e1.json").write_text(json.dumps(doc, indent=1, default=str))
        md = [report_markdown(m), "", report_markdown(dev_m)]
        (res_dir / "e1.md").write_text("\n".join(md))
        return {"summary": doc["summary"], "dev_summary": doc["dev_summary"], "residuals": res, "capture_recapture": cap,
                "adjudicated": (adj or {}).get("summary")}

    print(json.dumps(asyncio.run(go()), indent=1, default=str))


async def _adjudicated(a: argparse.Namespace, out: Path, locked, locked_src, ctx) -> dict:
    from crucible.factory.synth_wire import SynthWire
    from crucible.synth.falsify import cheap
    from crucible.synth.falsify.agents import Adjudicated, Anchor, Attempt, Panel, PanelMember, PanelVerdict, calibrate, trace_view
    from crucible.synth.meta.ppi import adjudicated_rates
    from crucible.synth.meta.recall import measure
    from crucible.synth.meta.rsi import summary

    root = data_root(a)
    a.tag = a.tag or "rsi-e1"
    cap = run_cap(a, root)
    d = out / f"e1-adjudicate-{time.strftime('%H%M%S')}"
    d.mkdir()
    wire = SynthWire(d, cap_usd=cap)
    limit = int(a.adjudicate)

    class Capped(Panel):
        n: int = 0

        async def decide(self, task, probe, outcome, ctx=None):  # type: ignore[override]
            if self.n >= limit:
                return PanelVerdict(decision="abstain", votes=[], lam=self.lam)
            self.n += 1
            return await super().decide(task, probe, outcome, ctx)

    panel = Capped(members=[PanelMember(n, c) for n, c in wire.run_roles().panel], lam=0.7, queue=d / "human_queue.jsonl", tag="adjudicate:e1")
    anchors = []
    for t in locked_src:
        r = await ctx.oracle(t)
        anchors.append(Anchor(task=t, attempt=Attempt(trace=trace_view(r.calls), answer=r.answer), valid=True, kind="oracle"))
        anchors.append(Anchor(task=t, attempt=Attempt(), valid=False, kind="noop"))
        writes = [i for i, c in enumerate(r.calls) if not c.read_only]
        if writes:
            cut = [c for i, c in enumerate(r.calls) if i != writes[-1]]
            anchors.append(Anchor(task=t, attempt=Attempt(trace=trace_view(cut), answer=r.answer), valid=False, kind="minus_last_write"))
    cal = await calibrate(panel, anchors, floor=0.7)
    fs = [Adjudicated(f, panel) if f.name == "MUT" else f for f in cheap()]
    from crucible.synth.falsify import FalsifyContext

    actx = FalsifyContext()
    try:
        m = await measure(locked.plants, locked_src, fs, actx, set_name=locked.name, set_kind="locked", purpose="report")
    finally:
        await actx.close()
    rates = adjudicated_rates(panel.log, cal.votes)
    (d / "panel.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in panel.log))
    doc = {"calibration": cal.model_dump(mode="json"), "matrix": m.model_dump(mode="json"), "summary": summary(m), "ppi": rates, "judged": panel.n,
           "limit": limit, "cost": wire.cost()}
    (d / "adjudicated.json").write_text(json.dumps(doc, indent=1, default=str))
    return {"summary": {"matrix": summary(m), "ppi": rates, "judged": panel.n, "lam": cal.lam, "calibrated": cal.calibrated, "cost": wire.cost()},
            "dir": str(d)}


def cmd_evolve(a: argparse.Namespace) -> None:
    """E2 (both arms) or E3 (``--evidence <domain>``, loop arm only by default): the regularised loop (meta/rsi.py)."""
    from crucible.factory.synth_wire import SynthWire
    from crucible.synth.falsify import FalsifyContext
    from crucible.synth.meta.rsi import EvolveConfig, Loop, Pool, refusals, rows_markdown, swarm_proposers
    from crucible.synth.meta.update import UpdateConfig

    out, dev, locked, dev_src, locked_src = _load(a)
    cxs, nat_tasks = _natural(a, _domains(a))
    root = data_root(a)
    a.tag = a.tag or "rsi-evolve"
    cap = run_cap(a, root)
    name = f"evolve-{a.label}"
    d = out / name
    d.mkdir(exist_ok=False)
    wire = SynthWire(d, cap_usd=cap)
    ev = [x for x in a.evidence.split(",") if x] or None
    cfg = EvolveConfig(rounds=a.rounds, candidates=a.candidates, view_frac=a.view_frac, seed=a.seed, evidence_domains=ev, planter_loop=not a.no_planter,
                       meta_loop=not a.no_meta, update=UpdateConfig(rounds=a.rounds, min_band=0.02))
    pool = Pool(dev_src)
    pool.add("P@1", dev.plants)
    builders = sorted({"deepseek", "glm", "kimi"})  # M4's builder families (proposer, oracle, repairer; contract writer; F1 World)
    rows_out: list = []
    refinfo: dict = {}

    async def go() -> None:
        ctx, rctx = FalsifyContext(), FalsifyContext()
        ref = None
        if a.real_defects:
            from crucible.synth.meta.rsi import load_real_defects, real_defect_features

            ref = real_defect_features(load_real_defects(a.real_defects))
            if not ref:
                raise SystemExit(f"{a.real_defects}: no independently found P1 defect with a class")
        loop = Loop(d, pool, locked, locked_src, cxs, ctx=ctx, report_ctx=rctx, config=cfg, dev_sources=dev_src, builders=builders,
                    natural_sources=nat_tasks, spend=lambda: wire.meter.spent, reference=ref)
        refinfo.update({"kind": loop.reference_kind, "n": len(loop.reference), "mu": loop.mu})
        props = swarm_proposers(wire.client("abstractor"), planter_client=wire.client("abstractor"))

        def save() -> None:
            rows = [r for arm in loop.arms for r in arm.rows]
            rows_out[:] = rows
            (d / "rows.json").write_text(json.dumps([r.model_dump(mode="json") for r in rows], indent=1, default=str))
            (d / "rows.md").write_text(rows_markdown(rows) + "\n" + json.dumps(refusals(rows), indent=1) + "\n")
            (d / "cost.json").write_text(json.dumps(wire.cost(), indent=1))

        try:
            await loop.run(props, control=not a.no_control, save=save)
        finally:
            save()
            await ctx.close()
            await rctx.close()
        (d / "config.json").write_text(json.dumps({"config": cfg.model_dump(mode="json"), "mu": loop.mu, "reference": refinfo, "evidence": ev, "dev_plants": len(dev.plants),
                                                   "locked_plants": len(locked.plants), "natural": len(cxs)}, indent=1))

    asyncio.run(go())
    print((d / "rows.md").read_text())
    print(json.dumps({"dir": str(d), "cost": wire.cost()}, indent=1))


def cmd_lemma_test(a: argparse.Namespace) -> None:
    """The synthesizer loop's acceptance test (S, §9.5): the same cells built with and without a lemma batch, at equal
    spend, measured by first-round confirmed counterexamples (no repair, no ALT/ATK); exact McNemar on the pairs."""
    from crucible.factory.synth_wire import SynthWire
    from crucible.synth.meta.lemmas import Rate, lemma_acceptance
    from crucible.synth.pipeline import RunConfig, Runner

    out = rsi_dir(a)
    lem = json.loads(Path(a.lemmas).read_text())
    texts = [x["rule"] if isinstance(x, dict) else str(x) for x in lem]
    texts = [t for t in texts if t][: a.max_lemmas]
    root = data_root(a)
    a.tag = a.tag or "rsi-lemma"
    cap = run_cap(a, root)
    d = out / f"lemma-test-{a.domain}-{time.strftime('%H%M%S')}"
    d.mkdir(parents=True)
    wire = SynthWire(d, cap_usd=cap)
    res: dict = {}

    async def arm(name: str, lemmas: tuple[str, ...]) -> list:
        ad = d / name
        ad.mkdir()
        cfg = RunConfig(n_tasks=a.n, batch=a.batch, max_rounds=1, concurrency=a.concurrency, calibrate=False, lemmas=lemmas, paid=False, repair=False)
        r = Runner(_env(a, ad), wire.run_roles(), cfg, ad, ad / "archive", run_id=f"{a.run_id}-{name}", cost=wire.cost, families=wire.families())
        r.roles.panel = ()  # MUT suspects stay suspects: the rate counts confirmed counterexamples only
        before = wire.meter.spent
        rep = await r.run()
        res[name] = {"spend": round(wire.meter.spent - before, 5), "funnel": rep.funnel}
        return rep.tasks

    async def go() -> None:
        ctl = await arm("control", ())
        trt = await arm("treated", tuple(texts))
        had = lambda t: bool(t.first_round_confirmed)
        built = lambda ts: [t for t in ts if t.stage == "refined"]
        bc, bt = built(ctl), built(trt)
        pairs = [(had(x), had(y)) for x, y in zip(ctl, trt, strict=False) if x.stage == "refined" and y.stage == "refined"]
        tr = Rate(k=sum(had(t) for t in bt), n=len(bt), spend=res["treated"]["spend"])
        cr = Rate(k=sum(had(t) for t in bc), n=len(bc), spend=res["control"]["spend"])
        jt = sum(1 for t in bt if not had(t)) / max(tr.spend, 1e-9)
        jc = sum(1 for t in bc if not had(t)) / max(cr.spend, 1e-9)
        dec = lemma_acceptance(tr, cr, j_rises=jt > jc, pairs=pairs or None, budget_tol=a.budget_tol)
        res["decision"] = dec.model_dump(mode="json")
        res["rates"] = {"treated": tr.model_dump(mode="json"), "control": cr.model_dump(mode="json"), "j_proxy": {"treated": jt, "control": jc},
                        "pairs": len(pairs)}
        res["lemmas"] = texts
        res["cost"] = wire.cost()

    try:
        asyncio.run(go())
    finally:
        (d / "lemma_test.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("worldgen")
    w.add_argument("--keyword", required=True)
    w.add_argument("--name", required=True)
    w.add_argument("--guidance", default="")
    w.add_argument("--rounds", type=int, default=3)
    w.add_argument("--cap", type=float, default=8.0)
    w.add_argument("--tag", default="")
    w.set_defaults(fn=cmd_worldgen)

    for name, fn in (("propose", cmd_propose), ("certify", cmd_certify)):
        p = sub.add_parser(name)
        p.add_argument("--domain", required=True)
        p.add_argument("--world", default=None, help="a worldgen output dir (generated domains)")
        p.add_argument("--seeds", default="0,1,2,3", help="finance_ap start-state seeds")
        p.add_argument("--n", type=int, default=20)
        p.add_argument("--batch", type=int, default=5)
        p.add_argument("--rounds", type=int, default=3)
        p.add_argument("--concurrency", type=int, default=6)
        p.add_argument("--alt-calls", type=int, default=24)
        p.add_argument("--atk-calls", type=int, default=14)
        p.add_argument("--no-calibrate", action="store_true")
        p.add_argument("--blind-panel", action="store_true", help="the panel never sees the grader's reason codes")
        p.add_argument("--invoices", type=int, default=None, help="finance_ap density knob: pending invoices per World (default 7-14)")
        p.add_argument("--budget", default="milestone", choices=["milestone", "rsi"])
        p.add_argument("--cap", type=float, default=25.0)
        p.add_argument("--tag", default="")
        p.set_defaults(fn=fn)

    r = sub.add_parser("report")
    r.add_argument("--domain", required=True)
    r.add_argument("--run-id", default=None)
    r.set_defaults(fn=cmd_report)

    s = sub.add_parser("smoke")
    s.add_argument("--cap", type=float, default=0.5)
    s.add_argument("--tag", default="")
    s.set_defaults(fn=cmd_smoke)

    b = sub.add_parser("budget")
    b.add_argument("--work", default="rsi", help="also report one piece of work's spend (run paths carrying this tag)")
    b.set_defaults(fn=cmd_budget)

    rs = sub.add_parser("reseed", help="a denser start state for an accepted World (more sibling entities)")
    rs.add_argument("--world", required=True)
    rs.add_argument("--name", required=True)
    rs.add_argument("--target", required=True, help="table=rows,... (rows in total per table)")
    rs.add_argument("--guidance", default="")
    rs.add_argument("--cap", type=float, default=2.0)
    rs.add_argument("--budget", default="rsi", choices=["milestone", "rsi"])
    rs.add_argument("--tag", default="rsi-dense")
    rs.set_defaults(fn=cmd_reseed)

    def rsi_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run-id", required=True)
        p.add_argument("--domains", default="finance_ap,it_helpdesk")
        p.add_argument("--locked-root", default=None)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--budget", default="rsi", choices=["milestone", "rsi"])
        p.add_argument("--tag", default="")

    pl = sub.add_parser("plant", help="dev + locked planted sets (part2-paper-core §8)")
    rsi_common(pl)
    pl.add_argument("--fixtures", action="store_true", help="add the hand-written fixture domains ap-test and desk-test")
    pl.set_defaults(fn=cmd_plant)

    rc = sub.add_parser("recall", help="E1: the FR/FA matrix on the locked set")
    rsi_common(rc)
    rc.add_argument("--adjudicate", type=int, default=0, help="judge up to N MUT suspects with the panel (PPI interval)")
    rc.add_argument("--cap", type=float, default=2.0)
    rc.set_defaults(fn=cmd_recall)

    ev = sub.add_parser("evolve", help="E2 / E3: the regularised loop")
    rsi_common(ev)
    ev.add_argument("--label", required=True)
    ev.add_argument("--rounds", type=int, default=4)
    ev.add_argument("--candidates", type=int, default=4)
    ev.add_argument("--view-frac", type=float, default=0.6)
    ev.add_argument("--evidence", default="", help="grow only from these domains' evidence (E3)")
    ev.add_argument("--no-control", action="store_true")
    ev.add_argument("--no-planter", action="store_true")
    ev.add_argument("--no-meta", action="store_true")
    ev.add_argument("--real-defects", default=None, help="the M5 real-defect set (realdefects.json): its independently found defects are the "
                                                         "natural-likeness reference (else the archive)")
    ev.add_argument("--cap", type=float, default=5.0)
    ev.set_defaults(fn=cmd_evolve)

    lt = sub.add_parser("lemma-test", help="the synthesizer loop's equal-budget acceptance test")
    rsi_common(lt)
    lt.add_argument("--lemmas", required=True, help="a JSON list of lemmas ({rule} objects or strings)")
    lt.add_argument("--max-lemmas", type=int, default=6)
    lt.add_argument("--domain", required=True)
    lt.add_argument("--world", default=None)
    lt.add_argument("--seeds", default="0,1,2,3")
    lt.add_argument("--invoices", type=int, default=None)
    lt.add_argument("--n", type=int, default=8)
    lt.add_argument("--batch", type=int, default=4)
    lt.add_argument("--concurrency", type=int, default=4)
    lt.add_argument("--budget-tol", type=float, default=0.35)
    lt.add_argument("--cap", type=float, default=3.0)
    lt.set_defaults(fn=cmd_lemma_test)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
