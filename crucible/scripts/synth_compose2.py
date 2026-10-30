#!/usr/bin/env python3
"""compose-2 (DESIGN §7.7 item 18): certified parts, per-part repair and the ALT solver pool on long composites.

    synth_compose2.py parts      --domain it_helpdesk --world <dir> --from <compose-1 run> --chains 0,1   # certify stored links in order
    synth_compose2.py chain      --domain finance_ap --invoices 30 --links 8 --cands 2                    # new chains, certify-before-join
    synth_compose2.py composites --domain it_helpdesk --world <dir> --from <parts or chain run> --min-calls 20 [--leak]
    synth_compose2.py solvers    --long <composites run>,... --single <parts run>,... --configs ultra:v2,ultra:v3,inkling:v3 --budget-usd 0.3
    synth_compose2.py budget

Runs go under ``<data-root>/<domain>/<ts>-compose2-<tag>`` (solvers: ``<data-root>/compose2/<ts>-<tag>``); the spend is
capped by ``COMPOSE2_CAP_USD`` (8) over every receipt whose run path carries "compose2", and by the shared Part 2 total
``MILESTONE_CAP_USD`` from the environment (150) over every receipt under the data root. Swarm only: builders DeepSeek
(proposer, oracle, repairer, instruction writer) and GLM (contract); ALT from the families that built nothing
(``--alt``); no Qwen model in any role and the student never called. The key comes from the environment only.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1]
if str(HERE / "src") not in sys.path:
    sys.path.insert(0, str(HERE / "src"))
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

COMPOSE2_CAP_USD = __import__("crucible.synth.policy", fromlist=["stage_cap"]).stage_cap("compose2")  # R-budget in the policy ledger (DESIGN §7.7 item 24)
WORK = "compose2"
MODELS = {"ultra": "NEMOTRON_ULTRA", "super": "NEMOTRON", "gptoss": "GPT_OSS", "inkling": "INKLING", "inkling-small": "INKLING_SMALL"}


def _synth():
    spec = importlib.util.spec_from_file_location("synth_cli", Path(__file__).with_name("synth.py"))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


S = _synth()


def data_root(a: argparse.Namespace) -> Path:
    return S.data_root(a)


def spent(root: Path) -> tuple[float, float]:
    from crucible.factory.synth_wire import milestone_spent, work_spent

    return milestone_spent(root), work_spent(root, WORK)


def cap_for(a: argparse.Namespace, root: Path) -> float:
    if not os.environ.get("MILESTONE_CAP_USD"):
        raise SystemExit("set MILESTONE_CAP_USD (the shared Part 2 total, 150) in the environment")
    shared = float(os.environ["MILESTONE_CAP_USD"])
    ms, ws = spent(root)
    cap = min(float(a.cap), round(shared - ms, 4), round(COMPOSE2_CAP_USD - ws, 4))
    print(f"shared Part 2 spend {ms:.4f} of {shared}; compose-2 work {ws:.4f} of {COMPOSE2_CAP_USD}; this run's cap {cap:.4f}", flush=True)
    if cap < 0.02:
        raise SystemExit("the compose-2 budget is spent: refusing to start")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not in the environment (set -a; . <repo>/.env; set +a)")
    return cap


def new_run(root: Path, domain: str, tag: str) -> tuple[str, Path]:
    rid = time.strftime("%Y%m%d-%H%M%S") + f"-{WORK}-{tag}"
    d = root / domain / rid
    d.mkdir(parents=True, exist_ok=False)
    return rid, d


def table_for(alt: str | None) -> dict[str, tuple[str, ...]]:
    from crucible.factory import models as fm
    from crucible.factory.synth_wire import SYNTH_ROLES

    t = dict(SYNTH_ROLES)
    if alt:
        t["alt_solver"] = tuple(getattr(fm, MODELS[x]) for x in alt.split(",") if x)
    return t


def wire_for(d: Path, cap: float, alt: str | None = None):
    from crucible.factory.synth_wire import SynthWire

    return SynthWire(d, cap_usd=cap, table=table_for(alt), seed_by_tag=True)


def run_cfg(a: argparse.Namespace):
    from crucible.synth.pipeline import RunConfig

    v3 = a.harness == "v3"
    return RunConfig(max_rounds=a.rounds, concurrency=a.concurrency, max_mutants=a.max_mutants, variant="v3" if v3 else "v2", oracle_sees_derive=False,
                     alt_finish_nudge=True, alt_progress=v3, alt_window=16 if v3 else None)


def population(a: argparse.Namespace):
    """The deterministic population plus ORACLE where the World registers policy truth (finance_ap: the pack's own
    policy decision per invoice, ``synth_wire.finance_ap_truth``; a task on a prefix variant is not covered: no trial)."""
    if a.domain != "finance_ap":
        return None
    from crucible.factory.synth_wire import finance_ap_truth
    from crucible.synth.falsify import cheap
    from crucible.synth.falsify.oracle import ORACLE

    def truth(task: Any) -> Any:
        v = finance_ap_truth(task.model_dump(mode="json"))
        return v if v in ("right", "wrong") else None

    return lambda: [*cheap(), ORACLE([truth])]


def env_for(a: argparse.Namespace, d: Path):
    a.n = getattr(a, "n", 12)
    return S._env(a, d)


def _save(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=str) + "\n")


def _reid(links: list, domain: str, rid: str, ci: int) -> list:
    """compose-1's it_helpdesk link ids repeat across chains (``link02/t01`` is two different tasks): a fresh id per link,
    so the archive never replays one link's counterexamples against another."""
    return [t.model_copy(update={"id": f"{domain}/{rid}/c{ci}/l{j + 1:02d}", "version": 1}) for j, t in enumerate(links)]


async def _calibrate(runner: Any, tasks: list, twin: bool = True) -> None:
    """The panel (and the TWIN panel) once per run, on the parts' oracle traces (executed truth)."""
    from crucible.synth.falsify import FalsifyContext
    from crucible.synth.pipeline import TaskRecord

    ctx = FalsifyContext()
    try:
        ready = [(TaskRecord(n=900 + k, id=t.id, seed=t.seed), t, await ctx.oracle(t)) for k, t in enumerate(tasks[: runner.cfg.calib_tasks])]
    finally:
        await ctx.close()
    await runner.calibrate_panel(ready=ready)
    if twin and not runner.stop:
        await runner.calibrate_twin(tasks)


# ---------------------------------------------------------------------------------------------------------- parts

def cmd_parts(a: argparse.Namespace) -> None:
    """Certify the links of stored chains in order on their own World states (``parts.certify_links``)."""
    from crucible.synth.falsify import FalsifyContext
    from crucible.synth.parts import CertBook, certify_links
    from crucible.synth.pipeline import Runner
    from crucible.synth.task import Task

    root = data_root(a)
    cap = cap_for(a, root)
    rid, d = new_run(root, a.domain, a.tag)
    wire = wire_for(d, cap, a.alt)
    env = env_for(a, d)
    book = CertBook(d / "certs.json")
    out: dict[str, Any] = {"run": str(d), "source": a.source, "chains": []}

    async def go() -> None:
        cal = None
        for ci in [int(x) for x in a.chains.split(",")]:
            links = [Task.model_validate(x) for x in json.loads((Path(a.source) / f"chain{ci}" / "links.json").read_text())][: a.links]
            orig = [t.id for t in links]
            links = _reid(links, a.domain, rid, ci)
            runner = Runner(env, wire.run_roles(), run_cfg(a), d / f"chain{ci}", root, run_id=f"{rid}-c{ci}", cost=wire.cost, families=wire.families())
            if population(a) is not None:
                runner.cheap_factory = population(a)
            await runner.survey()
            if cal is None:
                await _calibrate(runner, links, twin=not a.no_twin)
                cal = (runner.panel, runner.twin_panel)
            else:
                runner.panel, runner.twin_panel = cal
            log: list[dict[str, Any]] = []
            ctx = FalsifyContext()
            try:
                got = await certify_links(links, runner, book, ctx, log=log)
            finally:
                await ctx.close()
            _save(d / f"chain{ci}" / "links.json", [t.model_dump(mode="json") for t in links])
            _save(d / f"chain{ci}" / "links-certified.json", [t.model_dump(mode="json") for t in got])
            row = {"chain": ci, "links": len(links), "admitted": len(got), "log": log, "source_ids": orig}
            out["chains"].append(row)
            _save(d / "parts_summary.json", out | {"cost": wire.cost()})
            if runner.stop:
                out["stopped"] = runner.stop
                break

    try:
        asyncio.run(go())
    finally:
        out["cost"] = wire.cost()
        _save(d / "parts_summary.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "chains"} | {"chains": [{k: v for k, v in c.items() if k != "log"} | {
        "statuses": [x.get("status") for x in c["log"]]} for c in out["chains"]]}, indent=1, default=str))


# ---------------------------------------------------------------------------------------------------------- new chains

def cmd_chain(a: argparse.Namespace) -> None:
    """New chains with the swarm, every link certified before it joins (``compose.build_chain(certify=True)``)."""
    from crucible.synth.compose import build_chain
    from crucible.synth.falsify import FalsifyContext
    from crucible.synth.parts import CertBook

    root = data_root(a)
    cap = cap_for(a, root)
    _rid, d = new_run(root, a.domain, a.tag)
    wire = wire_for(d, cap, a.alt)
    env = env_for(a, d)
    book = CertBook(d / "certs.json")
    seeds = [int(x) for x in a.seeds.split(",")]
    out: dict[str, Any] = {"run": str(d), "chains": [], "start_from": a.start_from}
    start: list[Any] = []
    if a.start_from:  # resume: the certified links of an earlier chain run, with their certificates
        from crucible.synth.task import Task

        src = Path(a.start_from)
        for row in CertBook(src / "certs.json").rows.values():
            book.put(row)
        start = [Task.model_validate(x) for x in json.loads((src / "chain0" / "links-certified.json").read_text())]

    async def chain(ci: int) -> None:
        ctx = FalsifyContext(max_mutants=a.max_mutants)
        cdir = d / f"chain{ci}"

        def save(j: int, links: list, choice: Any) -> None:
            _save(cdir / "links-certified.json", [t.model_dump(mode="json") for t in links])
            with (cdir / "choices.jsonl").open("a") as fh:
                fh.write(json.dumps(choice.model_dump(mode="json"), default=str) + "\n")

        try:
            links, log = await build_chain(env, wire.run_roles(), length=a.links, run_dir=cdir, archive_root=root, seed=seeds[ci % len(seeds)],
                                           candidates=a.cands, cfg=run_cfg(a), cost=wire.cost, families=wire.families(), ctx=ctx, on_link=save,
                                           retries=a.link_retries, max_horizon=a.max_horizon, certify=True, book=book, certify_tries=a.tries,
                                           twin=not a.no_twin, start=start if ci == 0 else [], cheap_factory=population(a))
        finally:
            await ctx.close()
        _save(cdir / "links.json", [t.model_dump(mode="json") for t in links])
        out["chains"].append({"chain": ci, "links": len(links), "choices": [{"link": c.link, "chosen": c.chosen, "dep": c.dep, "horizon": c.horizon,
                                                                               "note": c.note, "certs": [r.get("cert") for r in c.candidates if "cert" in r]}
                                                                              for c in log]})

    async def go() -> None:
        for ci in range(a.chains):
            await chain(ci)

    try:
        asyncio.run(go())
    finally:
        out["cost"] = wire.cost()
        _save(d / "chain_summary.json", out)
    print(json.dumps(out, indent=1, default=str))


# ---------------------------------------------------------------------------------------------------------- composites

def _load_parts(src: Path) -> tuple[list[tuple[int, list]], Any]:
    from crucible.synth.parts import CertBook
    from crucible.synth.task import Task

    book = CertBook(src / "certs.json")
    out = []
    for p in sorted(src.glob("chain*/links-certified.json")):
        ci = int(p.parent.name.removeprefix("chain"))
        out.append((ci, [Task.model_validate(x) for x in json.loads(p.read_text())]))
    return out, book


def leak_factory(wire: Any, d: Path, parts: list) -> Any:
    """The L2 LEAK class (DESIGN §7.7 item 20) for composites: the leak panel calibrated on anchors from the certified
    parts (a template sentence with a derive value: a leak; the part's own sentences: not), Inkling-Small probing."""
    from crucible.synth.falsify.agents import PanelMember
    from crucible.synth.falsify.leak import LeakPanel, LeakProbe, calibrate_leak_panel

    spec = importlib.util.spec_from_file_location("synth_f2", Path(__file__).with_name("synth_falsifiers2.py"))
    f2 = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(f2)  # type: ignore[union-attr]
    panel = LeakPanel(members=[PanelMember(f"leak_panel-{c.family}", c) for c in wire.clients("leak_panel")], lam=0.7, queue=d / "leak_queue.jsonl")
    anchors = f2.leak_anchors([t for t in parts if t.intent.derive], list(f2.DEV_TEMPLATES))[:40]

    async def calibrate() -> dict[str, Any]:
        return await calibrate_leak_panel(panel, anchors) if anchors else {"calibrated": False, "anchors": 0}

    probe = LeakProbe(wire.client("leak_probe"), panel=panel, skip_structural=True, max_calls=20)
    return calibrate, (lambda: [probe])


def cmd_composites(a: argparse.Namespace) -> None:
    """Compose each chain's admitted links (at least ``--min-calls`` calls) and certify the composites: per-part repair,
    TWIN suspects adjudicated, the LEAK class (``--leak``), ALT from ``--alt`` under the unchanged rule."""
    from crucible.synth.compose import CompositionError, certify_composed, compose
    from crucible.synth.falsify import FalsifyContext
    from crucible.synth.pipeline import Runner

    root = data_root(a)
    cap = cap_for(a, root)
    rid, d = new_run(root, a.domain, a.tag)
    wire = wire_for(d, cap, a.alt)
    env = env_for(a, d)
    out: dict[str, Any] = {"run": str(d), "sources": a.source, "composites": []}
    writer = wire.client("instruction_writer")

    async def go() -> None:
        cands, parts_of, books = [], {}, []
        ctx = FalsifyContext(max_mutants=a.max_mutants)
        from crucible.synth.parts import CertBook

        book = CertBook(d / "certs.json")
        for src in [Path(x) for x in a.source.split(",") if x]:
            chains, b = _load_parts(src)
            books.append(b)
            for row in b.rows.values():
                book.put(row)
            for ci, links in chains:
                if a.only and ci not in {int(x) for x in a.only.split(",")}:
                    continue
                k = len(links)
                while k >= 2:
                    try:
                        c = await compose(links[:k], ctx=ctx, writer=writer, task_id=f"{a.domain}/{rid}/{src.name[-24:]}-c{ci}-k{k}", tag=f"compose:c{ci}",
                                          certs=book)
                    except CompositionError as e:
                        out["composites"].append({"source": src.name, "chain": ci, "parts": k, "error": str(e)[:400]})
                        k -= 1
                        continue
                    row = S._composed_row(c) | {"source": src.name, "chain": ci, "closed": c.comp.closed, "statuses": [s.status for s in c.comp.segments]}
                    out["composites"].append(row)
                    if c.comp.horizon >= a.min_calls and c.comp.dep >= a.min_dep:
                        cands.append(c)
                        parts_of[c.task.id] = list(links[:k])
                        _save(d / "composed" / f"{src.name[-24:]}-c{ci}-k{k}.json", c.dump() | {"parts": [t.model_dump(mode="json") for t in links[:k]]})
                    break
        await ctx.close()
        _save(d / "compose_summary.json", out | {"cost": wire.cost()})
        if not a.certify or not cands:
            return
        extra = None
        if a.leak:
            calibrate, extra = leak_factory(wire, d, [t for ps in parts_of.values() for t in ps])
            out["leak_calibration"] = await calibrate()
        part_runner = Runner(env, wire.run_roles(), run_cfg(a), d / "parts", root, run_id=f"{rid}-parts", cost=wire.cost, families=wire.families())
        await part_runner.survey()
        rep = await certify_composed(cands, env, wire.run_roles(), run_dir=d / "certify", archive_root=root, run_id=rid, cfg=run_cfg(a), cost=wire.cost,
                                     families=wire.families(), parts=parts_of, book=book, writer=writer, part_runner=part_runner, twin=not a.no_twin,
                                     extra_paid=extra, cheap_factory=population(a), alt_per_call=a.per_call)
        out["certify"] = S._cert_rows(rep)

    try:
        asyncio.run(go())
    finally:
        out["cost"] = wire.cost()
        _save(d / "compose_summary.json", out)
    print(json.dumps({k: v for k, v in out.items()}, indent=1, default=str)[:20000])


# ---------------------------------------------------------------------------------------------------------- solvers

def _long_tasks(src: Path) -> list[tuple[str, Any, int]]:
    from crucible.synth.compose import load_composed

    out = []
    finals = sorted(src.glob("certify/tasks/*/composed-final.json"))
    paths = finals or sorted(src.glob("composed/*.json"))
    for p in paths:
        t, c = load_composed(json.loads(p.read_text()))
        out.append(("long", t, c.horizon))
    return out


def _single_tasks(src: Path, k: int) -> list[tuple[str, Any, int]]:
    from crucible.synth.parts import ADMIT, CertBook
    from crucible.synth.task import Task

    book = CertBook(src / "certs.json")
    out = []
    for p in sorted(src.glob("chain*/links-certified.json")):
        for x in json.loads(p.read_text()):
            t = Task.model_validate(x)
            if book.admitted(t, ADMIT):
                out.append(("single", t, len(t.oracle.steps)))
    return out[:k]


def _yield_tasks(glob: str, root: Path, k: int) -> list[tuple[str, Any, int]]:
    from crucible.synth.task import Task

    out = []
    for p in sorted(root.glob(glob)):
        d = json.loads(p.read_text())
        # finance_ap: only oracles the pack's own policy confirms (yield-1 §5: certified tasks with a wrong oracle exist)
        if d.get("status") == "certified" and d.get("task") and (d.get("truth") == "right" or d.get("domain") != "finance_ap"):
            t = Task.model_validate(d["task"])
            out.append(("single", t, len(t.oracle.steps)))
    return out[:k]


def cmd_solvers(a: argparse.Namespace) -> None:
    """The ALT solver pool at equal spend (``synth.solvers``) on certified single tasks and long composites."""
    from crucible.factory import models as fm
    from crucible.factory.synth_wire import SynthWire
    from crucible.synth.solvers import SolverConfig, evaluate, report_md, summarize

    root = data_root(a)
    cap = cap_for(a, root)
    d = root / WORK / (time.strftime("%Y%m%d-%H%M%S") + f"-solvers-{a.tag}")
    d.mkdir(parents=True)
    names = [c.split(":")[0] for c in a.configs.split(",")]
    uniq = list(dict.fromkeys(names))
    alt = [getattr(fm, MODELS[n]) for n in uniq]
    if len({m.split(":")[1].split("/")[0] for m in alt}) < 2:  # the ALT role needs two families in the table; the extra one is never asked
        alt.append(fm.GPT_OSS if "gptoss" not in uniq else fm.NEMOTRON)
    wire = SynthWire(d, cap_usd=cap, table=table_for(None) | {"alt_solver": tuple(alt)}, seed_by_tag=True)
    clients = {n: wire.client("alt_solver", i) for i, n in enumerate(uniq)}
    cfgs = [SolverConfig(name=c.replace(":", "-"), client=clients[c.split(":")[0]], harness=c.split(":")[1] if ":" in c else "v2", per_call=a.per_call) for c in a.configs.split(",")]
    tasks: list[tuple[str, Any, int]] = []
    for s in [x for x in (a.single or "").split(",") if x]:
        tasks += _single_tasks(Path(s), a.max_single)
    if a.yield_glob:
        tasks += _yield_tasks(a.yield_glob, root, a.max_single)
    for s in [x for x in (a.long or "").split(",") if x]:
        tasks += _long_tasks(Path(s))
    if a.max_long is not None:
        longs = [t for t in tasks if t[0] == "long"][: a.max_long]
        tasks = [t for t in tasks if t[0] != "long"] + longs
    if a.match:
        tasks = [t for t in tasks if a.match in t[1].id]
    builders = {"deepseek", "glm"} | set(a.exclude_families.split(",") if a.exclude_families else [])
    bad = [c.name for c in cfgs if c.family in builders]
    if bad:
        raise SystemExit(f"family separation: {bad} share a family with the builders {sorted(builders)}")

    def cost_of(prefix: str) -> float:
        return sum(r["usd"] for r in wire.usage.rows if str(r.get("tag", "")).startswith(prefix))

    _save(d / "tasks.json", [{"set": s, "id": t.id, "horizon": h} for s, t, h in tasks])
    print(f"{len(tasks)} tasks x {len(cfgs)} configs, budget {a.budget_usd} per task and config", flush=True)
    rows = asyncio.run(evaluate(tasks, cfgs, budget_usd=a.budget_usd, max_attempts=a.max_attempts, cost_of=cost_of, concurrency=a.concurrency, out=d,
                                stop=lambda: wire.meter.spent >= cap * 0.97))
    summ = summarize(rows, a.budget_usd)
    _save(d / "summary.json", {"summary": summ, "cost": wire.cost()})
    (d / "report.md").write_text(report_md(summ, f"solver pool {d.name}"))
    print((d / "report.md").read_text())
    print(json.dumps(wire.cost(), indent=1))


def cmd_budget(a: argparse.Namespace) -> None:
    root = data_root(a)
    ms, ws = spent(root)
    print(json.dumps({"shared_spent": ms, "shared_cap": os.environ.get("MILESTONE_CAP_USD"), "compose2_spent": ws, "compose2_cap": COMPOSE2_CAP_USD}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--domain", required=True)
        p.add_argument("--world", default=None)
        p.add_argument("--seeds", default="0")
        p.add_argument("--invoices", type=int, default=None)
        p.add_argument("--rounds", type=int, default=3)
        p.add_argument("--concurrency", type=int, default=3)
        p.add_argument("--max-mutants", type=int, default=240)
        p.add_argument("--alt", default=None, help="comma list of ALT models: " + ",".join(MODELS))
        p.add_argument("--harness", default="v2", choices=["v2", "v3"])
        p.add_argument("--per-call", type=float, default=1.5, help="ALT call budget per oracle call when certifying composites")
        p.add_argument("--no-twin", action="store_true")
        p.add_argument("--cap", type=float, default=2.0)
        p.add_argument("--tag", default="run")

    p = sub.add_parser("parts")
    common(p)
    p.add_argument("--from", dest="source", required=True)
    p.add_argument("--chains", default="0")
    p.add_argument("--links", type=int, default=99)
    p.set_defaults(fn=cmd_parts)

    p = sub.add_parser("chain")
    common(p)
    p.add_argument("--links", type=int, default=8)
    p.add_argument("--chains", type=int, default=1)
    p.add_argument("--cands", type=int, default=2)
    p.add_argument("--tries", type=int, default=2)
    p.add_argument("--link-retries", type=int, default=1)
    p.add_argument("--max-horizon", type=int, default=48)
    p.add_argument("--start-from", default=None, help="an earlier chain run: chain 0 resumes after its certified links")
    p.set_defaults(fn=cmd_chain)

    p = sub.add_parser("composites")
    common(p)
    p.add_argument("--from", dest="source", required=True, help="comma list of parts/chain run dirs")
    p.add_argument("--only", default=None)
    p.add_argument("--min-calls", type=int, default=20)
    p.add_argument("--min-dep", type=int, default=1)
    p.add_argument("--certify", action="store_true")
    p.add_argument("--leak", action="store_true")
    p.set_defaults(fn=cmd_composites)

    p = sub.add_parser("solvers")
    p.add_argument("--single", default=None, help="parts/chain run dirs whose admitted links are the single tasks")
    p.add_argument("--yield-glob", default=None, help="certified yield candidates (relative to the data root)")
    p.add_argument("--long", default=None, help="composites run dirs")
    p.add_argument("--max-single", type=int, default=8)
    p.add_argument("--max-long", type=int, default=None)
    p.add_argument("--match", default=None, help="only tasks whose id contains this text")
    p.add_argument("--configs", required=True, help="model:harness, e.g. ultra:v2,ultra:v3,inkling:v3")
    p.add_argument("--exclude-families", default="", help="World-builder families to keep out (kimi for worldgen Worlds)")
    p.add_argument("--budget-usd", type=float, default=0.3)
    p.add_argument("--per-call", type=float, default=1.5, help="ALT call budget per oracle call (compose.agent_budget_for)")
    p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--cap", type=float, default=2.0)
    p.add_argument("--tag", default="pool")
    p.set_defaults(fn=cmd_solvers)

    p = sub.add_parser("budget")
    p.set_defaults(fn=cmd_budget)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
