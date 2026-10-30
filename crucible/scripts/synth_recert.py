#!/usr/bin/env python3
"""Re-certify the stored training-side pool under the fixed framework (DESIGN §7.7 item 22; ``synth/recert.py``).

    synth_recert.py collect --tag recert-1                       # the learnability pool (89), M4, the frontier round, every yield arm
    synth_recert.py det     --run <run dir> --workers 6          # phase 1: deterministic classes, CPU only, resumable
    synth_recert.py paid    --run <run dir> --cap 6              # phase 2: panels, ALT, ATK, LEAK on the phase-1 survivors
    synth_recert.py report  --run <run dir> [--publish]          # statuses, causes, manifest; --publish writes the certificates beside the old ones
    synth_recert.py budget

Runs go under ``<data-root>/recert/<ts>-<tag>`` (every receipt there carries "recert" in its path). Spend: ``RECERT_CAP_USD``
(8) over those receipts and the shared Part 2 total ``MILESTONE_CAP_USD`` from the environment (150) over every receipt
under the data root; both are checked before every paid run. Swarm only: ALT Nemotron-3-Super, gpt-oss-120b and
Inkling-Small, all on the v3 harness (compose-2: it helps these at equal spend, not Ultra, and Ultra cost 53% of the
pilot's spend); ATK Super + gpt-oss; panel Ultra + gpt-oss; leak probe Inkling-Small, leak panel Ultra + gpt-oss. No Qwen model in any role, the
student is never called, and no stored task is edited. The key comes from the environment only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1]
if str(HERE / "src") not in sys.path:
    sys.path.insert(0, str(HERE / "src"))
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

RECERT_CAP_USD = __import__("crucible.synth.policy", fromlist=["stage_cap"]).stage_cap("recert")  # R-budget (policy ledger)
WORK = "recert"
HOLDOUT = __import__("crucible.synth.policy", fromlist=["holdout"]).holdout()  # R-holdout: h = 0.2, fixed (policy ledger)
TRUTHS = {"finance_ap": "crucible.factory.synth_wire:finance_ap_truth"}  # executed policy truth (item 21's ORACLE)
POOL_CHECK = "perfect/20261029-232613-perfect-check3/perfect.json"  # the learnability agent's 89-task pool (its P1-P4 verdicts)
M4_RUNS = {"finance_ap": ["20261029-143755-smoke", "20261029-144119-smoke2", "20261029-152544-full", "20261029-160300-full2"],
           "it_helpdesk": ["20261029-150517-smoke2", "20261029-151753-smoke3", "20261029-160257-full2"]}
FRONTIER_RUNS = ["20261029-232517-perfect-frontier-control", "20261029-232517-perfect-frontier-targeted"]
SEEDED_YIELD = {"yield-3", "speed-ser2"}  # yield runs after the batch wire seeded by tag (yield-1 §2 bug 8)
# compose-2's v3 set: about 10x cheaper per attempt than Ultra x2 + gpt-oss v2 and passing at least as often at equal spend.
# The pilot (4 tasks with Ultra v2 + Super v3 + gpt-oss v3) cost USD 0.062 per task, Ultra 53% of it: the whole pool would
# not fit the cap, so Ultra stays out of ALT (it still sits on the calibrated panel).
ALT_SET = (("super", "NEMOTRON", "v3"), ("gptoss", "GPT_OSS", "v3"), ("inkling-small", "INKLING_SMALL", "v3"))


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


def spent(root: Path) -> tuple[float, float]:
    from crucible.factory.synth_wire import milestone_spent, work_spent

    return milestone_spent(root), work_spent(root, WORK)


def cap_for(a: argparse.Namespace, root: Path) -> float:
    if not os.environ.get("MILESTONE_CAP_USD"):
        raise SystemExit("set MILESTONE_CAP_USD (the shared Part 2 total, 150) in the environment")
    shared = float(os.environ["MILESTONE_CAP_USD"])
    ms, ws = spent(root)
    cap = min(float(a.cap), round(shared - ms, 4), round(RECERT_CAP_USD - ws, 4))
    print(f"shared Part 2 spend {ms:.4f} of {shared}; recert work {ws:.4f} of {RECERT_CAP_USD}; planned {a.cap}; this run's cap {cap:.4f}",
          flush=True)
    if ms + cap > shared + 1e-9:
        raise SystemExit("the shared Part 2 total would pass its cap: refusing to start")
    if cap < 0.02:
        raise SystemExit("the recert budget is spent: refusing to start")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not in the environment (set -a; . <repo>/.env; set +a)")
    return cap


# ---------------------------------------------------------------------------------------------------------- collect

def sources(root: Path) -> list[Any]:
    from crucible.synth.recert import Source

    check = json.loads((root / POOL_CHECK).read_text())
    ids = [v["task_id"] for v in check["verdicts"]]
    out = []
    for dom in ("finance_ap", "it_helpdesk"):  # the 89: M4 and RSI-1 runs, every status (P1 kept the rejected ones out)
        out.append(Source(name=f"pool-{dom}", kind="pool", dir=str(root / dom), archive_root=str(root), statuses=["certified", "provisional", "rejected"],
                          ids=[i for i in ids if i.startswith(dom + "/")], seed0=True, alt_names_shared=True))
    for dom, runs in M4_RUNS.items():
        out.append(Source(name=f"m4-{dom}", kind="m4", dir=str(root / dom), archive_root=str(root), runs=runs, seed0=True, alt_names_shared=True))
    out.append(Source(name="frontier", kind="frontier", dir=str(root / "finance_ap"), archive_root=str(root), runs=FRONTIER_RUNS, seed0=True))
    for arm in sorted(p for p in (root / "yield").glob("*/*") if p.is_dir() and any((p / d / "report.json").exists() for d in ("finance_ap", "it_helpdesk"))):
        run = arm.parent.name
        out.append(Source(name=f"{run}/{arm.name}", kind="yield", dir=str(arm), archive_root=str(arm / "archive"), seed0=run not in SEEDED_YIELD))
    return out


def cmd_collect(a: argparse.Namespace) -> None:
    from crucible.synth.recert import collect, copy_archive

    root = data_root(a)
    rid = time.strftime("%Y%m%d-%H%M%S") + f"-{a.tag if WORK in a.tag else WORK + '-' + a.tag}"
    d = root / WORK / rid
    d.mkdir(parents=True, exist_ok=False)
    srcs = sources(root)
    items = collect(srcs)
    copied = copy_archive(items, d / "archives")
    _save(d / "sources.json", [s.model_dump(mode="json") for s in srcs])
    _save(d / "items.json", [i.model_dump(mode="json") for i in items])
    summ = {"run": str(d), "items": len(items), "by_source": dict(Counter(i.source for i in items)), "kinds": dict(Counter(k for i in items for k in i.kinds)),
            "old_status": dict(Counter(i.old.status for i in items)), "archive_records_copied": copied}
    _save(d / "collect.json", summ)
    print(json.dumps(summ, indent=1))


def _items(d: Path) -> list[Any]:
    from crucible.synth.recert import Item

    items = [Item.model_validate(x) for x in json.loads((d / "items.json").read_text())]
    return items


# ---------------------------------------------------------------------------------------------------------- phase 1

def cmd_det(a: argparse.Namespace) -> None:
    import multiprocessing as mp

    from crucible.synth.recert import det_job, det_verdict, load_det, save_det, slug

    d = Path(a.run)
    items = _items(d)
    if a.only:
        items = [i for i in items if a.only in i.key]
    todo = [i for i in items if not (d / "det" / f"{slug(i.key)}.json.gz").exists()]
    print(f"{len(items)} items, {len(todo)} to run, {a.workers} workers, holdout {a.holdout}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {}
        for it in todo:
            work = d / "work" / slug(it.source)
            work.mkdir(parents=True, exist_ok=True)
            truths = [TRUTHS[it.task.domain]] if it.task.domain in TRUTHS else []
            futs[ex.submit(det_job, it.task.model_dump_json(), str(work), a.max_mutants, a.holdout, truths, str(d / "archives" / slug(it.source)))] = it
        from concurrent.futures import as_completed

        for done, f in enumerate(as_completed(futs), 1):
            it = futs[f]
            try:
                doc = json.loads(f.result())
            except Exception as e:  # noqa: BLE001 - a task whose World cannot be rebuilt is recorded, not fatal
                doc = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
            doc |= {"key": it.key, "holdout": a.holdout, "max_mutants": a.max_mutants}
            save_det(d / "det" / f"{slug(it.key)}.json.gz", doc)
            print(f"[{done}/{len(todo)} {time.time() - t0:.0f}s] {it.key} {doc.get('cpu_s')}s {doc.get('error', '')[:120]}", flush=True)
    rows = []
    for it in items:
        p = d / "det" / f"{slug(it.key)}.json.gz"
        if not p.exists():
            continue
        doc = load_det(p)
        if "error" in doc:
            rows.append({"key": it.key, "old": it.old.status, "det": "error", "reasons": [doc["error"]]})
            continue
        st, why, susp = det_verdict(it.task, doc)
        rows.append({"key": it.key, "old": it.old.status, "det": st, "reasons": why, "suspects": susp, "cpu_s": doc.get("cpu_s")})
    _save(d / "det_summary.json", {"rows": rows, "det": dict(Counter(r["det"] for r in rows)),
                                   "by_old": dict(Counter(f"{r['old']}->{r['det']}" for r in rows))})
    print(json.dumps({"det": dict(Counter(r["det"] for r in rows)), "by_old": dict(Counter(f"{r['old']}->{r['det']}" for r in rows))}, indent=1))


# ---------------------------------------------------------------------------------------------------------- phase 2

def _wire(d: Path, cap: float) -> Any:
    from crucible.factory import models as fm
    from crucible.factory.synth_wire import SYNTH_ROLES, SynthWire

    table = dict(SYNTH_ROLES) | {"alt_solver": tuple(getattr(fm, m) for _, m, _ in ALT_SET)}
    return SynthWire(d, cap_usd=cap, table=table)  # seeds by tag, salted by this run's directory name (item 22)


def _alt_agents(wire: Any, key: str) -> list[Any]:
    from crucible.synth.falsify.agents import AgentBudget, llm_alt
    from crucible.synth.roles import variant

    out = []
    for i, (short, _m, h) in enumerate(ALT_SET):
        name = f"alt_solver-{short}-{h}"
        out.append(llm_alt(name, wire.client("alt_solver", i), role=variant("alt_solver", h), budget=AgentBudget(), tag=f"ALT:{key}#{name}",
                           finish_nudge=True, progress_check=h == "v3", window=16 if h == "v3" else None))
    return out


def _atk_agents(wire: Any, key: str) -> list[Any]:
    from crucible.synth.falsify.agents import AgentBudget, llm_atk

    return [llm_atk(n, c, budget=AgentBudget(max_calls=14, max_turns=18), tag=f"ATK:{key}#{n}") for n, c in wire.run_roles().atk]


def _panel_anchors(tasks: list[Any], ctx: Any, k: int = 4) -> list[Any]:
    async def go() -> list[Any]:
        from crucible.synth.falsify.agents import Anchor, Attempt, trace_view

        out = []
        for t in tasks[:k]:
            r = await ctx.oracle(t)
            out.append(Anchor(task=t, attempt=Attempt(trace=trace_view(r.calls), answer=r.answer), valid=True, kind="oracle"))
            out.append(Anchor(task=t, attempt=Attempt(), valid=False, kind="noop"))
            w = [i for i, c in enumerate(r.calls) if not c.read_only]
            if w:
                cut = [c for i, c in enumerate(r.calls) if i != w[-1]]
                out.append(Anchor(task=t, attempt=Attempt(trace=trace_view(cut), answer=r.answer), valid=False, kind="minus_last_write"))
        return out

    return go()  # type: ignore[return-value]


def _f2() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("synth_f2", Path(__file__).with_name("synth_falsifiers2.py"))
    m = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


def cmd_paid(a: argparse.Namespace) -> None:
    from crucible.actors.llm import BudgetExceeded
    from crucible.synth.archive import Archive
    from crucible.synth.falsify import FalsifyContext
    from crucible.synth.falsify.agents import Panel, PanelMember, calibrate
    from crucible.synth.falsify.leak import LeakPanel, LeakProbe, calibrate_leak_panel
    from crucible.synth.falsify.oracle import ORACLE
    from crucible.synth.falsify.twinjudge import calibrate_twin_panel, twin_anchors
    from crucible.synth.recert import certify_paid, det_results, det_verdict, load_det, paid_rows, resolve_truths, slug, status_of, work_world

    root = data_root(a)
    d = Path(a.run)
    cap = cap_for(a, root)
    pd = d / f"paid-{time.strftime('%Y%m%d-%H%M%S')}"
    pd.mkdir(parents=True)
    wire = _wire(pd, cap)
    items = _items(d)
    cands = []
    for it in items:
        p = d / "det" / f"{slug(it.key)}.json.gz"
        if not p.exists() or (d / "paid" / f"{slug(it.key)}.json").exists():
            continue
        doc = load_det(p)
        if "error" in doc or det_verdict(it.task, doc)[0] != "paid":
            continue
        cands.append((it, doc))
    if a.only:
        cands = [(i, x) for i, x in cands if a.only in i.key]
    if a.limit:
        cands = cands[: a.limit]
    print(f"{len(cands)} phase-2 candidates", flush=True)

    async def go() -> dict[str, Any]:
        roles = wire.run_roles()
        cal: dict[str, Any] = {}
        work = d / "work" / "paid"
        work.mkdir(parents=True, exist_ok=True)
        ctx = FalsifyContext(work_world(work))
        panels: dict[str, Panel] = {}
        twins: dict[str, Panel] = {}
        try:
            for dom in sorted({i.task.domain for i, _ in cands}):
                ts = [i.task for i, _ in cands if i.task.domain == dom]
                truths = resolve_truths([TRUTHS[dom]] if dom in TRUTHS else [])
                good = []  # anchors only from tasks whose reference decision is not wrong (yield-2: wrong oracles poisoned λ)
                for t in ts:
                    o = ORACLE(truths)
                    v = [tr(t) for tr in o.truths]
                    if "wrong" not in v:
                        good.append(t)
                p = Panel(members=[PanelMember(n, c) for n, c in roles.panel], lam=0.7, queue=pd / "human_queue.jsonl", tag=f"calibrate:{dom}")
                rep = await calibrate(p, await _panel_anchors(good, ctx), floor=0.7)
                panels[dom] = p
                cal[dom] = {"panel": {"lam": rep.lam, "calibrated": rep.calibrated, "rows": rep.rows}}
                need_twin = any((det_results(x).get("TWIN") and det_results(x)["TWIN"].suspects) for i, x in cands if i.task.domain == dom)
                if need_twin:
                    tp = Panel(members=p.members, lam=p.lam, queue=pd / "human_queue.jsonl", tag=f"calibrate:twin:{dom}")
                    anc = (await twin_anchors(good, ctx))[:9] + (await _panel_anchors(good, ctx, k=3))
                    trep = await calibrate_twin_panel(tp, anc, min_decided=6)
                    twins[dom] = tp
                    cal[dom]["twin"] = {"lam": trep.lam, "calibrated": trep.calibrated, "anchors": len(anc), "rows": trep.rows}
            leak_srcs = [i.task for i, _ in cands if i.task.intent.derive and i.old.status in ("certified", "provisional")]
            lp = LeakPanel(members=[PanelMember(f"leak_panel-{c.family}", c) for c in wire.clients("leak_panel")], lam=0.7, queue=pd / "leak_queue.jsonl")
            anchors = _f2().leak_anchors(leak_srcs, list(_f2().DEV_TEMPLATES))[:40]
            cal["leak"] = await calibrate_leak_panel(lp, anchors) if anchors else {"calibrated": False, "anchors": 0}
            cal["leak"] = {k: v for k, v in cal["leak"].items() if k != "votes"}
        finally:
            await ctx.close()
        _save(pd / "calibration.json", cal)
        probe = LeakProbe(wire.client("leak_probe"), panel=lp, skip_structural=True, max_calls=max(20, 2 * len(cands)))
        sem = asyncio.Semaphore(a.concurrency)
        stop: list[str] = []

        async def one(it: Any, doc: dict[str, Any]) -> None:
            if stop:
                return
            async with sem:
                if stop:
                    return
                base = panels[it.task.domain]
                panel = Panel(members=base.members, lam=base.lam, queue=base.queue, tag=f"adjudicate:{it.key}")
                tw = twins.get(it.task.domain)
                tpan = Panel(members=tw.members, lam=tw.lam, queue=tw.queue, tag=f"adjudicate:twin:{it.key}") if tw is not None else None
                c2 = FalsifyContext(work_world(work))
                t0 = time.time()
                try:
                    rep, sink = await certify_paid(it, doc, ctx=c2, archive=Archive(d / "archives" / slug(it.source)), holdout=float(doc.get("holdout") or 0),
                                                   truths=[TRUTHS[it.task.domain]] if it.task.domain in TRUTHS else [], panel=panel, twin_panel=tpan,
                                                   alt=_alt_agents(wire, it.key), atk=_atk_agents(wire, it.key), extra=[probe])
                    st, why = status_of(rep)
                    row = {"key": it.key, "status": st, "reasons": why, "report": rep.model_dump(mode="json"), "attempts": paid_rows(rep, sink),
                           "panel_log": panel.log + (tpan.log if tpan is not None else []), "seconds": round(time.time() - t0, 1)}
                except BudgetExceeded as e:
                    stop.append(str(e))
                    row = {"key": it.key, "status": "budget", "reasons": [str(e)[:300]]}
                except Exception as e:  # noqa: BLE001
                    row = {"key": it.key, "status": "error", "reasons": [f"{type(e).__name__}: {str(e)[:500]}"]}
                finally:
                    await c2.close()
                if row["status"] != "budget":
                    _save(d / "paid" / f"{slug(it.key)}.json", row | {"paid_run": str(pd)})
                print(f"{it.key}: {row['status']} {row['reasons'][:4]} ${wire.meter.spent:.3f}", flush=True)

        await asyncio.gather(*(one(i, x) for i, x in cands))
        return {"calibration": cal, "stopped": stop[:1]}

    try:
        out = asyncio.run(go())
    finally:
        _save(pd / "cost.json", wire.cost())
    print(json.dumps({"cost": wire.cost()["total_usd"], **{k: v for k, v in out.items() if k != "calibration"}}, indent=1, default=str))


# ---------------------------------------------------------------------------------------------------------- report

def cmd_report(a: argparse.Namespace) -> None:
    from crucible.synth.recert import SCHEMA, ManifestRow, attribute, beside, det_verdict, load_det, slug, status_h0, summarize, write_new

    d = Path(a.run)
    items = _items(d)
    rows: list[ManifestRow] = []
    details = []
    for it in items:
        dp, pp = d / "det" / f"{slug(it.key)}.json.gz", d / "paid" / f"{slug(it.key)}.json"
        doc = load_det(dp) if dp.exists() else None
        paid = json.loads(pp.read_text()) if pp.exists() else None
        if doc is None:
            st, why, att = "error", ["phase 1 did not run"], []
        elif "error" in doc:
            st, why, att = "error", [doc["error"]], []
        else:
            ds, dwhy, _ = det_verdict(it.task, doc)
            if ds == "rejected":
                st, why, att = "rejected", dwhy, []
            elif paid is None:
                st, why, att = "error", ["phase 2 did not run"], []
            else:
                st, why, att = paid["status"], paid["reasons"], paid.get("attempts") or []
        attr = attribute(it, st, why, doc if doc and "error" not in doc else None, att)
        st0 = status_h0(st, why, (doc or {}).get("mut_all"))
        alt = {}
        for r in att:
            if r["cls"] == "ALT":
                x = alt.setdefault(str(r["agent"]), [0, 0])
                x[0] += int(r["passed"])
                x[1] += 1
        cert_doc = {"schema": "crucible.synth.recert.certificate/v1", "recert_run": str(d), "tag": a.tag, "key": it.key, "task_id": it.task.id,
                    "task_version": it.task.version, "task_digest": it.task.digest(), "source": it.source, "kinds": it.kinds,
                    "old": it.old.model_dump(mode="json"), "status": st, "reasons": why, "status_h0": st0, "attribution": attr,
                    "config": {"holdout": (doc or {}).get("holdout"), "max_mutants": (doc or {}).get("max_mutants"), "alt": [f"{s}:{h}" for s, _, h in ALT_SET],
                               "seeds": "by tag, salted per run and client (item 22)", "policy": "pipeline.admission_policy()"},
                    "certificate": ((paid or {}).get("report") or {}).get("certificate"), "final": ((paid or {}).get("report") or {}).get("final"),
                    "attempts": att}
        local = d / "certs" / f"{slug(it.key)}.json"
        if not local.exists():
            write_new(local, cert_doc)
        path = str(local)
        if a.publish and st not in ("error", "budget"):
            b = beside(it, a.tag)
            if not b.exists():
                write_new(b, cert_doc)
            path = str(b)
        rows.append(ManifestRow(key=it.key, task_id=it.task.id, version=it.task.version, digest=it.task.digest(), domain=it.task.domain, source=it.source,
                                kinds=it.kinds, task_path=it.old.path, old_status=it.old.status, old_reasons=it.old.reasons, old_certificate=it.old.path,
                                status=st, reasons=why, certificate=path, heldout=float((doc or {}).get("holdout") or 0.0),
                                status_h0=st0, outcome=attr["outcome"],
                                causes=attr["causes"], alt={k: {"passes": v[0], "n": v[1]} for k, v in alt.items()}))
        details.append({"key": it.key, **attr})
    summ = summarize(rows)
    man = {"schema": SCHEMA, "run": str(d), "tag": a.tag, "made": time.strftime("%Y-%m-%d %H:%M:%S"), "summary": summ, "rows": [r.model_dump(mode="json") for r in rows]}
    out = d / f"manifest.{a.tag}.json"
    _save(out, man)
    print(json.dumps(summ, indent=1))
    print(f"manifest: {out}")


def cmd_budget(a: argparse.Namespace) -> None:
    root = data_root(a)
    ms, ws = spent(root)
    print(json.dumps({"shared_part2_usd": ms, "recert_usd": ws, "recert_cap": RECERT_CAP_USD}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--tag", default="recert-1")
    dt = sub.add_parser("det")
    dt.add_argument("--run", required=True)
    dt.add_argument("--workers", type=int, default=6)
    dt.add_argument("--holdout", type=float, default=HOLDOUT)
    dt.add_argument("--max-mutants", type=int, default=240)
    dt.add_argument("--only", default="")
    pd = sub.add_parser("paid")
    pd.add_argument("--run", required=True)
    pd.add_argument("--cap", type=float, default=6.0)
    pd.add_argument("--concurrency", type=int, default=6)
    pd.add_argument("--only", default="")
    pd.add_argument("--limit", type=int, default=0)
    rp = sub.add_parser("report")
    rp.add_argument("--run", required=True)
    rp.add_argument("--tag", default="recert-1")
    rp.add_argument("--publish", action="store_true")
    sub.add_parser("budget")
    a = ap.parse_args()
    {"collect": cmd_collect, "det": cmd_det, "paid": cmd_paid, "report": cmd_report, "budget": cmd_budget}[a.cmd](a)


if __name__ == "__main__":
    main()
