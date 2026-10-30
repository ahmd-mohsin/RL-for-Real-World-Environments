"""The perfect-task subcommands of ``scripts/synth.py`` (P2 learnability, P4 novelty, the P1-P4 check, the K frontier).

    synth.py learnability --domains finance_ap,it_helpdesk --limit 30 --cap 8 --tag perfect-pilot   # student band (Tinker)
    synth.py learnability ... --dry-run                       # the same pipeline with a scripted no-op student (no model)
    synth.py novelty      --domains finance_ap,it_helpdesk [--paraphrase 20 --cap 0.3]               # signatures, θ, coverage
    synth.py perfect      --domains finance_ap,it_helpdesk --store <records.jsonl> --calibration <novelty.json>
    synth.py frontier     --domains finance_ap,it_helpdesk --store <records.jsonl>                    # attribution -> requests

Artifacts: ``<data-root>/perfect/<run-id>/`` (every run id carries "perfect", so ``work_spent(root, "perfect")`` books
the work's receipts); the difficulty store defaults to ``<data-root>/perfect/difficulty.jsonl``. **Spend:** the student
and proxy calls of every run share the work's cap ``P2_CAP_USD`` (the lead: $15) over every receipt under a path with
"perfect"; each run's ``--cap`` is clipped to what is left. Only training-side synthesis runs are read (``load_pool``
refuses sealed roots). The Tinker key comes from the environment only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------------------------------------- shared


def _root(a: argparse.Namespace, data_root: Callable[[argparse.Namespace], Path]) -> Path:
    return data_root(a)


def _cap(a: argparse.Namespace, root: Path, *, needs_key: bool = True) -> float:
    from crucible.factory.synth_student_wire import P2_CAP_USD, WORK_TAG
    from crucible.factory.synth_wire import work_spent

    spent = work_spent(root, WORK_TAG)
    cap = round(min(float(a.cap), P2_CAP_USD - spent), 4)
    print(f"perfect-task work spend so far {spent:.4f} of {P2_CAP_USD}; this run's cap {cap:.4f}", flush=True)
    if needs_key:
        if cap < 0.05:
            raise SystemExit("the work's Tinker budget is spent: refusing to start")
        if not os.environ.get("TINKER_API_KEY"):
            raise SystemExit("TINKER_API_KEY is not in the environment (set -a; . <repo>/.env; set +a)")
    return max(cap, 0.0)


def _run_dir(root: Path, tag: str) -> tuple[str, Path]:
    t = tag if "perfect" in tag else f"perfect-{tag}" if tag else "perfect"
    rid = time.strftime("%Y%m%d-%H%M%S") + f"-{t}"
    d = root / "perfect" / rid
    d.mkdir(parents=True, exist_ok=False)
    return rid, d


def _pool(a: argparse.Namespace, root: Path) -> list[Any]:
    from crucible.synth.perfect import load_pool

    items = []
    for dom in [x for x in a.domains.split(",") if x]:
        items += load_pool(root / dom, statuses=tuple(a.statuses.split(",")))
    if getattr(a, "tasks", None):
        want = [x for x in a.tasks.split(",") if x]
        by = {i.task.id: i for i in items}
        items = [by[t] for t in want if t in by]
    return items


def _store(a: argparse.Namespace, root: Path) -> Any:
    from crucible.synth.learn import DifficultyStore

    return DifficultyStore(Path(a.store) if a.store else root / "perfect" / "difficulty.jsonl")


def _jw(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, indent=1, default=str) + "\n")


ORDER = {"certified": 0, "provisional": 1, "rejected": 2}


def _select(items: list[Any], limit: int) -> list[Any]:
    """Certified and provisional tasks first; then rejected ones alternating by ALT history (a pass share > 0, then 0)
    so the predictor sees both ends; ``limit`` in all."""
    head = sorted([i for i in items if i.status != "rejected"], key=lambda i: (ORDER[i.status], i.task.id))
    rej = [i for i in items if i.status == "rejected"]
    pos = sorted([i for i in rej if i.alt and i.alt[0] > 0], key=lambda i: i.task.id)
    zer = sorted([i for i in rej if not (i.alt and i.alt[0] > 0)], key=lambda i: i.task.id)
    tail = []
    while pos or zer:
        if pos:
            tail.append(pos.pop(0))
        if zer:
            tail.append(zer.pop(0))
    return (head + tail)[:limit] if limit else head + tail


# ---------------------------------------------------------------------------------------------------------- learnability


def cmd_learnability(a: argparse.Namespace, data_root: Callable[[argparse.Namespace], Path]) -> None:
    from crucible.synth.falsify import FalsifyContext
    from crucible.synth.falsify.agents import AgentBudget
    from crucible.synth.learn import (
        AgentStudentProbe,
        BandSpec,
        ContextPool,
        calibrated,
        cost_report,
        features,
        key_of,
        loo,
        measure,
        operating_characteristic,
        predictor_accuracy,
        proxy_estimate,
        records_json,
        training_rows,
    )

    root = _root(a, data_root)
    cap = _cap(a, root, needs_key=not a.dry_run)
    rid, d = _run_dir(root, a.tag)
    store = _store(a, root)
    spec = BandSpec(looks=tuple(int(x) for x in a.looks.split(",")), alpha=a.alpha)
    items = _select(_pool(a, root), a.limit)
    budget = AgentBudget(max_calls=a.max_calls, max_turns=a.max_calls + 4, obs_chars=a.obs_chars)
    wire = None
    if a.dry_run:
        async def noop(view: Any, world: Any) -> Any:
            return None

        probe: Any = AgentStudentProbe(runner=lambda tag: noop, contexts=ContextPool(FalsifyContext, size=a.concurrency), student="dry-run/noop",
                                       concurrency=a.concurrency)
    else:
        from crucible.factory.synth_student_wire import StudentWire

        wire = StudentWire(d, cap_usd=cap)
        probe = wire.probe(FalsifyContext, budget=budget, temperature=a.temperature, concurrency=a.concurrency)
    report: dict[str, Any] = {"run": str(d), "store": str(store.path), "student": probe.student, "spec": spec.__dict__, "tasks": [], "issues": []}

    async def go() -> None:
        ctx = FalsifyContext()
        try:
            prior_rows = training_rows([r for r in store.all() if r.key.student == probe.student])
            predictor = calibrated(prior_rows, trust_min=a.trust_min, trust_accuracy=a.trust_accuracy) if len({r[0] for r in prior_rows}) >= 6 else None
            report["predictor_before"] = predictor.to_json() if predictor else None
            feats_of: dict[str, dict[str, float]] = {}
            for it in items:  # static features and the free ALT history first: they order the work
                feats_of[it.task.id] = features(it.task, await ctx.oracle(it.task), alt=tuple(it.alt) if it.alt else None)
            order = items
            if predictor is not None:
                order = sorted(items, key=lambda i: abs(predictor.p(feats_of[i.task.id]) - 0.5))
            pp = wire.proxy_probe(FalsifyContext, budget=budget, temperature=a.temperature, concurrency=a.concurrency) if wire and a.proxy_rollouts else None
            sem = asyncio.Semaphore(max(1, a.parallel_tasks))  # tasks in flight; every rollout still gets its own context
            probe.contexts.size = a.concurrency * max(1, a.parallel_tasks)
            if pp is not None:
                pp.contexts.size = a.concurrency * max(1, a.parallel_tasks)

            async def one(it: Any) -> None:
                async with sem:
                    left = (wire.meter.cap_usd - wire.meter.spent) if wire is not None else 1.0
                    if wire is not None and left < a.reserve:
                        report["issues"].append(f"skipped {it.task.id}: ${left:.3f} left < reserve {a.reserve}")
                        return
                    await measure_one(it)

            async def measure_one(it: Any) -> None:
                t0 = time.time()
                px_est = px_roll = None
                prev = store.get(key_of(it.task, probe))
                done = prev is not None and prev.source == "measured" and prev.decision is not None and (prev.confident or prev.n >= spec.looks[-1])
                if done:  # resumed: the stored record answers; no proxy is bought again
                    pass
                elif a.proxy_estimate and wire is not None:  # the paid non-student proxies, just before the task's band test
                    o = await ctx.oracle(it.task)
                    tools = await (await ctx.world(it.task)).tools()
                    px_est = await proxy_estimate(wire.proxy_client(), it.task, tools, o, student=probe.student, tag=f"proxy-est:{it.task.id}")
                if pp is not None and not done:
                    rs = await pp.rollouts(it.task, a.proxy_rollouts)
                    m = [r.passed for r in rs if r.passed is not None]
                    px_roll = (sum(m) / len(m)) if m else None
                if px_est is not None or px_roll is not None:
                    f = dict(feats_of[it.task.id])
                    f.update({"proxy": px_roll if px_roll is not None else px_est, "has_proxy": 1.0, "proxy_est": -1.0 if px_est is None else px_est,
                              "proxy_roll": -1.0 if px_roll is None else px_roll})
                    feats_of[it.task.id] = f
                rec = await measure(it.task, probe, spec=spec, predictor=predictor, feats=feats_of[it.task.id], store=store,
                                    note=f"{it.status}; cell={it.cell}; run {rid}")
                row = {"task": it.task.id, "status": it.status, "cell": it.cell, "alt": it.alt, "decision": rec.decision, "confident": rec.confident,
                       "error": rec.error, "passes": rec.passes, "n": rec.n, "masked": rec.masked, "looks": rec.looks, "source": rec.source,
                       "predictor": rec.predictor.get("band_call"), "cost_usd": rec.cost_usd, "stops": dict(Counter(r.stop for r in rec.rollouts)),
                       "reasons": dict(Counter(x.split(":", 1)[0] for r in rec.rollouts if not r.passed for x in r.reasons)),
                       "mean_calls": round(sum(r.calls for r in rec.rollouts) / len(rec.rollouts), 2) if rec.rollouts else None,
                       "elapsed_s": round(time.time() - t0, 1)}
                report["tasks"].append(row)
                print(json.dumps(row, default=str), flush=True)
                _jw(d / "learnability.json", report)

            await asyncio.gather(*(one(it) for it in order))
        finally:
            await ctx.close()
            await probe.contexts.close()

    try:
        asyncio.run(go())
    finally:
        recs = [r for r in store.all() if r.key.student == probe.student]
        rows = training_rows(recs)
        static = ("bias", "steps", "writes", "reads", "depth", "required", "derive", "obs_literals", "answer", "world_log", "ins_words_log")
        variants = {"all": None, "static": static, "alt": ("bias", "alt_share", "has_alt"), "proxy_est": ("bias", "proxy_est"),
                    "proxy_roll": ("bias", "proxy_roll"), "static+alt": (*static, "alt_share", "has_alt"),
                    "static+alt+proxy": (*static, "alt_share", "has_alt", "proxy_est", "proxy_roll")}
        if len({r[0] for r in rows}) >= 3:
            report["loo"] = {}
            for name, keys in variants.items():
                kw = {} if keys is None else {"keys": keys}
                if keys and any(k.startswith("proxy_") for k in keys):  # only rows that have the proxy
                    sub = [r for r in rows if all(r[1].get(k, -1.0) >= 0 for k in keys if k.startswith("proxy_"))]
                else:
                    sub = rows
                if len({r[0] for r in sub}) >= 3:
                    rep = loo(sub, **kw)
                    report["loo"][name] = {k: v for k, v in rep.items() if k != "predictions"}
            report["loo_predictions"] = loo(rows).get("predictions")
        report["predictor_online"] = predictor_accuracy(recs)
        report["cost"] = cost_report(recs)
        report["oc"] = operating_characteristic(spec)
        report["decisions"] = dict(Counter(r.decision for r in recs if r.source == "measured"))
        report["wire_cost"] = wire.cost() if wire is not None else None
        report["records"] = records_json(recs)
        _jw(d / "learnability.json", report)
        print(json.dumps({k: report.get(k) for k in ("run", "decisions", "cost", "loo", "predictor_online", "wire_cost")}, indent=1, default=str))


# ---------------------------------------------------------------------------------------------------------- novelty


def cmd_novelty(a: argparse.Namespace, data_root: Callable[[argparse.Namespace], Path]) -> None:
    from crucible.synth.falsify import FalsifyContext
    from crucible.synth.novelty import assign_cell, calibrate, coverage, diversity, known_pairs, near_duplicates, signature
    from crucible.synth.pipeline import CELLS

    root = _root(a, data_root)
    cap = _cap(a, root, needs_key=a.paraphrase > 0)
    _rid, d = _run_dir(root, a.tag or "novelty")
    items = _pool(a, root)
    grid = {f"c{i + 1}": desc for i, desc in enumerate(CELLS)}
    para = None
    wire = None
    if a.paraphrase > 0:
        from crucible.factory.synth_student_wire import StudentWire
        from crucible.synth.roles import Role, _Out

        class ParaOut(_Out):
            instruction: str

        role = Role(name="paraphraser", stage="P4", output=ParaOut, sees=("instruction",), temperature=0.7, max_tokens=800,
                    system=("Rewrite the user's request in different words, as another employee would write it. Keep every fact, name, id, number "
                            "and requirement; change the wording and sentence order. Reply with one JSON object {\"instruction\": <the rewrite>}."))
        wire = StudentWire(d, cap_usd=cap)
        done = {"n": 0}

        async def para(text: str) -> str | None:
            if done["n"] >= a.paraphrase:
                return None
            done["n"] += 1
            try:
                doc = await wire.proxy_client().complete_json(role.render(instruction=text), schema=ParaOut, max_tokens=800, temperature=0.7,
                                                              tag="paraphrase")
                return ParaOut.model_validate(doc).instruction
            except Exception as e:
                if type(e).__name__ in ("TinkerCapReached", "BudgetExceeded"):
                    raise
                return None

    out: dict[str, Any] = {"run": str(d), "tasks": len(items), "grid": grid, "domains": {}}

    async def go() -> None:
        ctx = FalsifyContext()
        try:
            cell_of = {i.task.id: assign_cell(i.task, grid, recorded=i.cell) for i in items}
            tasks = [i.task for i in items]
            pairs = await known_pairs(tasks, ctx, cells={i.task.id: i.cell or "" for i in items}, paraphrase=para,
                                      cell_of=lambda t: cell_of.get(t.id) or t.intent.skill)
            cal = calibrate(pairs)
            out["calibration"] = cal.model_dump(mode="json")
            out["pairs"] = dict(Counter(f"{k}:{'dup' if lab else 'distinct'}" for *_, lab, k in pairs))
            sigs = {}
            for t in tasks:
                sigs[t.id] = signature(t, await ctx.oracle(t), cell=cell_of.get(t.id) or "", tools=await (await ctx.world(t)).tools())
            for dom in sorted({t.domain for t in tasks}):
                ss = [sigs[t.id] for t in tasks if t.domain == dom]
                adm = [sigs[i.task.id] for i in items if i.task.domain == dom and i.status in ("certified", "provisional")]
                out["domains"][dom] = {"all": diversity(ss, w=cal.w, theta=cal.theta, gate=cal.gate), "admitted": diversity(adm, w=cal.w, theta=cal.theta, gate=cal.gate) if adm else None,
                                       "coverage_all": coverage({t.id: cell_of[t.id] for t in tasks if t.domain == dom}, grid),
                                       "coverage_admitted": coverage({i.task.id: cell_of[i.task.id] for i in items if i.task.domain == dom
                                                                      and i.status in ("certified", "provisional")}, grid),
                                       "near_duplicates": near_duplicates(ss, w=cal.w, theta=cal.theta, gate=cal.gate)}
            out["signatures"] = {k: v.model_dump(mode="json") for k, v in sigs.items()}
            out["cells"] = cell_of
        finally:
            await ctx.close()

    try:
        asyncio.run(go())
    finally:
        out["cost"] = wire.cost() if wire is not None else None
        _jw(d / "novelty.json", out)
    print(json.dumps({k: out.get(k) for k in ("run", "calibration", "pairs", "cost")}, indent=1, default=str))
    for dom, v in out["domains"].items():
        print(dom, json.dumps({"all": {k: v["all"][k] for k in ("tasks", "distinct_signatures", "near_duplicate_pairs", "effective_tasks")},
                               "coverage_all": v["coverage_all"]["histogram"]}, default=str))


# ---------------------------------------------------------------------------------------------------------- perfect


def cmd_perfect(a: argparse.Namespace, data_root: Callable[[argparse.Namespace], Path]) -> None:
    from crucible.synth.falsify import FalsifyContext
    from crucible.synth.falsify.taint import TAINT
    from crucible.synth.falsify.twin import TWIN
    from crucible.synth.intent import EVID, SWAP, p3, pool_report
    from crucible.synth.learn import band_examples, band_verdicts, calibrated, training_rows
    from crucible.synth.novelty import DEFAULT_THETA, DEFAULT_W, assign_cell, signature
    from crucible.synth.perfect import admit_pool, export_schemas
    from crucible.synth.pipeline import CELLS

    root = _root(a, data_root)
    _rid, d = _run_dir(root, a.tag or "check")
    items = sorted(_pool(a, root), key=lambda i: (ORDER.get(i.status, 9), i.task.id))
    store = _store(a, root)
    recs = store.all()
    w, theta, gate = DEFAULT_W, DEFAULT_THETA, 0.5
    if a.calibration:
        cal = json.loads(Path(a.calibration).read_text()).get("calibration") or {}
        w, theta, gate = float(cal.get("w", w)), float(cal.get("theta", theta)), float(cal.get("gate", gate))
    grid = {f"c{i + 1}": desc for i, desc in enumerate(CELLS)}
    out: dict[str, Any] = {"run": str(d), "w": w, "theta": theta, "gate": gate, "p3": {}, "verdicts": [], "by_domain": {}}

    async def go() -> None:
        ctx = FalsifyContext()
        try:
            entries = []
            p3s = []
            for it in items:
                peers = [j.task for j in items if j.task.domain == it.task.domain]
                r = await p3(it.task, ctx, classes=[TAINT(), TWIN(), EVID(), SWAP(pool=peers)])
                p3s.append((it, r))
                out["p3"][it.task.id] = r.model_dump(mode="json")
                sig = signature(it.task, await ctx.oracle(it.task), cell=assign_cell(it.task, grid, recorded=it.cell) or "",
                                tools=await (await ctx.world(it.task)).tools())
                entries.append((it.task, it.status, r, sig))
            for dom in sorted({it.task.domain for it in items}):
                sub = [(t, s, r, g) for t, s, r, g in entries if t.domain == dom]
                verdicts, summ = admit_pool(sub, recs, student=a.student or None, accept_provisional=a.accept_provisional, w=w, theta=theta, gate=gate)
                out["verdicts"] += [v.model_dump(mode="json") for v in verdicts]
                out["by_domain"][dom] = {"summary": summ, "on_intent": pool_report([r for it, r in p3s if it.task.domain == dom]),
                                         "on_intent_admitted_only": pool_report([r for it, r in p3s if it.task.domain == dom and it.status != "rejected"])}
        finally:
            await ctx.close()

    try:
        asyncio.run(go())
    finally:
        rows = training_rows(recs)
        pred = calibrated(rows) if len({r[0] for r in rows}) >= 3 else None
        out["exports"] = {"band_verdicts": [v.model_dump(mode="json") for v in band_verdicts(recs)],
                          "band_examples": [e.model_dump(mode="json") for e in band_examples(recs, pred)]}
        _jw(d / "perfect.json", out)
        _jw(d / "schemas.json", export_schemas())
    print(json.dumps({"run": str(d), "by_domain": {k: {"summary": v["summary"], "on_intent_share": v["on_intent"]["share"],
                                                       "on_intent_by_class": v["on_intent"]["by_class"]} for k, v in out["by_domain"].items()}},
                     indent=1, default=str))


# ---------------------------------------------------------------------------------------------------------- frontier


def cmd_frontier(a: argparse.Namespace, data_root: Callable[[argparse.Namespace], Path]) -> None:
    from crucible.synth.falsify import FalsifyContext
    from crucible.synth.frontier import FrontierPolicy, PoolEntry, admit, attribute, attribution_summary, export_attributions, requests_from
    from crucible.synth.novelty import assign_cell
    from crucible.synth.pipeline import CELLS

    root = _root(a, data_root)
    _rid, d = _run_dir(root, a.tag or "frontier")
    items = {i.task.id: i for i in _pool(a, root)}
    recs = [r for r in _store(a, root).all() if r.key.task_id in items and (not a.student or r.key.student == a.student) and r.source == "measured"]
    grid = {f"c{i + 1}": desc for i, desc in enumerate(CELLS)}
    out: dict[str, Any] = {"run": str(d), "records": len(recs), "attributions": []}

    async def go() -> list[Any]:
        ctx = FalsifyContext()
        atts = []
        try:
            for r in recs:
                it = items[r.key.task_id]
                if it.task.version != r.key.task_version:
                    continue
                cell = assign_cell(it.task, grid, recorded=it.cell) or ""
                for ro in r.rollouts:
                    if ro.passed is None:
                        continue
                    atts.append(await attribute(it.task, ro, ctx, cell=cell, student=r.key.student, checkpoint=r.key.checkpoint,
                                                budget_calls=a.max_calls))
        finally:
            await ctx.close()
        return atts

    atts = asyncio.run(go())
    reqs = requests_from(atts, recs, min_traces=a.min_traces)
    pool = [PoolEntry(task_id=i.task.id, cell=assign_cell(i.task, grid, recorded=i.cell) or "") for i in items.values() if i.status != "rejected"]
    cands = [PoolEntry(task_id=f"{r.id}#{j}", cell=r.cell, targeted=True, request=r.id) for r in reqs for j in range(a.per_request)]
    admitted, refused, r4 = admit(pool, cands, FrontierPolicy(kappa=a.kappa))
    out.update({"summary": attribution_summary(atts), "requests": [r.model_dump(mode="json") for r in reqs],
                "attributions": [x.model_dump(mode="json") for x in atts], "export": export_attributions(atts),
                "admission_plan": {"pool": len(pool), "candidates": len(cands), "admitted": len(admitted), "refused": [w for _, w in refused], "r4": r4}})
    _jw(d / "frontier.json", out)
    with (d / "attributions.jsonl").open("w") as fh:
        for x in export_attributions(atts):
            fh.write(json.dumps(x) + "\n")
    print(json.dumps({k: out[k] for k in ("run", "records", "summary", "admission_plan")}, indent=1, default=str))
    for r in reqs:
        print(json.dumps({"cell": r.cell, "mode": r.mode, "direction": r.direction, "knobs": r.knobs, "traces": r.traces, "tasks": len(r.source_tasks)}))


def cmd_frontier_round(a: argparse.Namespace, data_root: Callable[[argparse.Namespace], Path]) -> None:
    """One real K round on one domain: the targeted batch (the proposer's cells are the frontier's requests in words)
    and the untargeted control (the default cells), each ``--n`` tasks under the same cap, both certified by the swarm
    pipeline, then measured on the student (P2), probed (P3), signed (P4) and checked; compared at equal budget and
    admitted under the 20% cap and R4."""
    from crucible.factory.synth_student_wire import StudentWire
    from crucible.factory.synth_wire import SynthWire, finance_ap_env, generated_env
    from crucible.synth.falsify import FalsifyContext
    from crucible.synth.falsify.agents import AgentBudget
    from crucible.synth.falsify.taint import TAINT
    from crucible.synth.falsify.twin import TWIN
    from crucible.synth.frontier import BatchResult, FrontierPolicy, PoolEntry, TargetRequest, TaskOutcome, admit, attribute, compare, request_text
    from crucible.synth.intent import EVID, SWAP, p3
    from crucible.synth.learn import BandSpec, features, measure
    from crucible.synth.novelty import assign_cell, signature
    from crucible.synth.perfect import load_pool, perfect
    from crucible.synth.pipeline import CELLS, RunConfig, Runner

    root = _root(a, data_root)
    cap = _cap(a, root)
    fr = json.loads(Path(a.attributions).read_text())
    reqs = [TargetRequest.model_validate(r) for r in fr["requests"]]
    reqs = sorted([r for r in reqs if any(t.startswith(a.domain + "/") for t in r.source_tasks)], key=lambda r: -r.traces)[: a.requests]
    if not reqs:
        raise SystemExit("no target request for this domain")
    grid = {f"c{i + 1}": desc for i, desc in enumerate(CELLS)}
    base = {r.id: next((d for d in CELLS if d == r.cell or grid.get(r.cell) == d), grid.get(r.cell, r.cell)) for r in reqs}
    t_cells = tuple(request_text(r, base[r.id]) for r in reqs)
    rid = time.strftime("%Y%m%d-%H%M%S")
    out: dict[str, Any] = {"requests": [r.model_dump(mode="json") for r in reqs], "targeted_cells": list(t_cells), "arms": {}}
    pool_items = load_pool(root / a.domain, statuses=("certified", "provisional"))
    store = _store(a, root)

    async def arm(name: str, cells: tuple[str, ...]) -> BatchResult:
        d = root / a.domain / f"{rid}-perfect-frontier-{name}"
        d.mkdir(parents=True)
        wire = SynthWire(d, cap_usd=a.cap_arm)
        cfg = RunConfig(n_tasks=a.n, batch=a.batch, max_rounds=a.rounds, concurrency=4, cells=cells,
                        alt_budget=AgentBudget(max_calls=20, max_turns=24), atk_budget=AgentBudget(max_calls=12, max_turns=14))
        if a.domain == "finance_ap":
            env = finance_ap_env(d / "work", seeds=tuple(int(x) for x in a.seeds.split(",")), invoices=a.invoices)
        else:
            if not a.world:
                raise SystemExit("a generated domain needs --world <worldgen output dir>")
            wrep = json.loads((Path(a.world) / "worldgen_report.json").read_text())
            design = json.loads((Path(a.world) / "design.json").read_text())
            env = generated_env(a.world, d / "work", name=a.domain, description=f"{wrep['keyword']}: {design.get('description', '')}")
        r = Runner(env, wire.run_roles(), cfg, d, root, run_id=d.name, cost=wire.cost, families=wire.families())
        await r.run()
        synth_usd = wire.cost()["total_usd"]
        sw = StudentWire(d / "student", cap_usd=min(cap, 2.0))
        probe = sw.probe(FalsifyContext, concurrency=4)
        items = load_pool(root / a.domain, statuses=("certified", "provisional", "rejected"), runs=[d.name])
        ctx = FalsifyContext()
        outcomes = []
        pool_sigs = [signature(i.task, await ctx.oracle(i.task), cell=assign_cell(i.task, grid, recorded=i.cell) or "",
                               tools=await (await ctx.world(i.task)).tools()) for i in pool_items]
        try:
            for it in items:
                o = await ctx.oracle(it.task)
                req = next((r.id for r, c in zip(reqs, t_cells, strict=True) if c == it.cell), None) if name == "targeted" else None
                cell = assign_cell(it.task, grid, recorded=base.get(req or "", it.cell)) or ""
                modes: list[str] = []
                rec = None
                if it.status in ("certified", "provisional") and ctx.grade(o, it.task.contract).passed:
                    rec = await measure(it.task, probe, spec=BandSpec(), feats=features(it.task, o, alt=tuple(it.alt) if it.alt else None),
                                        store=store, note=f"frontier round {rid} {name}")
                    for ro in rec.rollouts:
                        if ro.passed is False:
                            att = await attribute(it.task, ro, ctx, cell=cell, student=probe.student)
                            if att.mode and not att.routed:
                                modes.append(att.mode)
                pr = await p3(it.task, ctx, classes=[TAINT(), TWIN(), EVID(), SWAP(pool=[i.task for i in items])])
                sig = signature(it.task, o, cell=cell, tools=await (await ctx.world(it.task)).tools())
                v = perfect(it.task, pool_sigs, [rec] if rec else [], status=it.status, p3=pr, signature=sig, accept_provisional=a.accept_provisional)
                outcomes.append(TaskOutcome(task_id=it.task.id, cell=cell, perfect=v.admit, decision=rec.decision if rec else None,
                                            cost_usd=rec.cost_usd if rec else 0.0, request=req, failed_modes=sorted(set(modes))))
                out["arms"].setdefault(name, {}).setdefault("verdicts", []).append(v.model_dump(mode="json"))
        finally:
            await ctx.close()
            await probe.contexts.close()
        spent = synth_usd + sw.cost()["total_usd"]
        out["arms"][name].update({"run": str(d), "synth_usd": synth_usd, "student_usd": sw.cost()["total_usd"], "tasks": len(items),
                                  "status": dict(Counter(i.status for i in items))})
        return BatchResult(arm=name, budget_usd=a.cap_arm, spent_usd=round(spent, 5), tasks=outcomes)  # type: ignore[arg-type]

    async def go() -> None:
        t_res = await arm("targeted", t_cells)
        c_res = await arm("control", ())
        cmp = compare(t_res, c_res, reqs, tol=0.35)
        pool = [PoolEntry(task_id=i.task.id, cell=assign_cell(i.task, grid, recorded=i.cell) or "") for i in pool_items]
        cands = [PoolEntry(task_id=o.task_id, cell=o.cell, targeted=True, request=o.request) for o in t_res.tasks if o.perfect]
        cands += [PoolEntry(task_id=o.task_id, cell=o.cell) for o in c_res.tasks if o.perfect]
        adm, ref, r4 = admit(pool, cands, FrontierPolicy(kappa=a.kappa)) if cmp["equal_budget"] else ([], [(c, "unequal budget") for c in cands], {})
        out.update({"targeted": t_res.model_dump(mode="json"), "control": c_res.model_dump(mode="json"), "comparison": cmp,
                    "admitted": [x.model_dump(mode="json") for x in adm], "refused": [{"entry": c.model_dump(mode="json"), "why": w} for c, w in ref],
                    "r4": r4})

    d0 = root / "perfect" / f"{rid}-perfect-frontier-round"
    d0.mkdir(parents=True)
    try:
        asyncio.run(go())
    finally:
        _jw(d0 / "frontier_round.json", out)
    print(json.dumps({"run": str(d0), "comparison": out.get("comparison"), "admitted": len(out.get("admitted") or []),
                      "arms": {k: {x: v.get(x) for x in ("synth_usd", "student_usd", "tasks", "status")} for k, v in out["arms"].items()}},
                     indent=1, default=str))


# ---------------------------------------------------------------------------------------------------------- outcome


def cmd_outcome(a: argparse.Namespace, data_root: Callable[[argparse.Namespace], Path]) -> None:
    """The student-probe outcome classifier: ``--calibrate`` on the known-label fixture cases; else every stored rollout
    of the pool's tasks, with the calibrated swarm panel deciding the residue when ``--panel``."""
    from crucible.synth.falsify import FalsifyContext
    from crucible.synth.outcome import classify, confusion, export_outcomes, summary, task_evidence

    root = _root(a, data_root)
    _rid, d = _run_dir(root, a.tag or "outcome")
    out: dict[str, Any] = {"run": str(d)}
    if a.calibrate:
        import sys

        here = Path(__file__).resolve().parents[1]
        if str(here) not in sys.path:
            sys.path.insert(0, str(here))
        from tests.unit.synth_outcome_cases import CASES, rollout_for

        async def cal() -> list[dict[str, Any]]:
            rows = []
            for name, mk, plan, want in CASES:
                t = mk()
                ro = await rollout_for(t, plan)
                ctx = FalsifyContext()
                try:
                    tev = await task_evidence(t, ctx, mut=True)
                    r = await classify(t, ro, ctx, tev, student="scripted")
                finally:
                    await ctx.close()
                rows.append({"case": name, "expected": want, "got": r.label, "cause": r.cause, "passed": ro.passed})
            return rows

        rows = asyncio.run(cal())
        out["cases"] = rows
        out["confusion"] = confusion([(r["expected"], r["got"]) for r in rows])
        _jw(d / "outcome_calibration.json", out)
        print(json.dumps(out["confusion"], indent=1))
        return
    items = {i.task.id: i for i in _pool(a, root)}
    recs = [r for r in _store(a, root).all() if r.key.task_id in items and r.source == "measured" and (not a.student or r.key.student == a.student)]
    panel = None
    wire = None
    if a.panel:
        from crucible.factory.synth_wire import SynthWire
        from crucible.synth.falsify.agents import Anchor, Attempt, Panel, PanelMember, calibrate, trace_view

        cap = _cap(a, root)
        wire = SynthWire(d, cap_usd=cap)
        panel = Panel(members=[PanelMember(f"panel-{c.family}", c) for c in wire.clients("panel")], lam=0.7, queue=d / "human_queue.jsonl")
    records = []

    async def go() -> None:
        ctx = FalsifyContext()
        try:
            if panel is not None:  # λ from executed-truth anchors: π*'s own trace is valid, doing nothing is not
                from crucible.synth.falsify.base import empty_run

                anchors = []
                for it in list(items.values())[: a.anchors]:
                    o = await ctx.oracle(it.task)
                    anchors.append(Anchor(task=it.task, attempt=Attempt(trace=trace_view(o.calls), answer=o.answer), valid=True, kind="oracle"))
                    v = ctx.grade(empty_run(o.start, it.task.instruction), it.task.contract)
                    anchors.append(Anchor(task=it.task, attempt=Attempt(trace=[], answer=None, reasons=v.reasons), valid=False, kind="noop"))
                cal = await calibrate(panel, anchors, floor=0.7)
                out["panel_calibration"] = cal.model_dump(mode="json")
            tevs: dict[str, Any] = {}
            for r in recs:
                it = items[r.key.task_id]
                if it.task.version != r.key.task_version:
                    continue
                if it.task.id not in tevs:
                    tevs[it.task.id] = await task_evidence(it.task, ctx, alt=it.alt, mut=a.mut)
                for ro in r.rollouts:
                    oc = await classify(it.task, ro, ctx, tevs[it.task.id], student=r.key.student, checkpoint=r.key.checkpoint, cell=it.cell or "",
                                        panel=panel, budget_calls=a.max_calls)
                    records.append(oc)
            out["task_evidence"] = {k: v.model_dump(mode="json") for k, v in tevs.items()}
        finally:
            await ctx.close()

    try:
        asyncio.run(go())
    finally:
        out["summary"] = summary(records)
        out["records"] = export_outcomes(records)
        out["cost"] = wire.cost() if wire is not None else None
        _jw(d / "outcome.json", out)
        with (d / "outcomes.jsonl").open("w") as fh:
            for x in export_outcomes(records):
                fh.write(json.dumps(x, default=str) + "\n")
    print(json.dumps({"run": str(d), "summary": out["summary"], "cost": out["cost"]}, indent=1, default=str))


# ---------------------------------------------------------------------------------------------------------- the parser


def register(sub: Any, data_root: Callable[[argparse.Namespace], Path]) -> None:
    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--domains", default="finance_ap,it_helpdesk")
        p.add_argument("--statuses", default="certified,provisional,rejected")
        p.add_argument("--tasks", default="", help="comma-separated task ids (default: the selection rule)")
        p.add_argument("--store", default=None, help="the difficulty store (default <data-root>/perfect/difficulty.jsonl)")
        p.add_argument("--tag", default="")
        p.add_argument("--student", default="", help="only this student's records (default: all)")

    lp = sub.add_parser("learnability", help="P2: predictor-first sequential band measurement on the student")
    common(lp)
    lp.add_argument("--limit", type=int, default=30)
    lp.add_argument("--looks", default="4,8")
    lp.add_argument("--alpha", type=float, default=0.1)
    lp.add_argument("--max-calls", type=int, default=20)
    lp.add_argument("--obs-chars", type=int, default=2000)
    lp.add_argument("--temperature", type=float, default=1.0)
    lp.add_argument("--concurrency", type=int, default=4)
    lp.add_argument("--parallel-tasks", type=int, default=3, help="tasks measured at once")
    lp.add_argument("--proxy-estimate", action="store_true", help="one non-student proxy-model estimate per task (a feature)")
    lp.add_argument("--proxy-rollouts", type=int, default=0, help="k proxy-model rollouts per task (a feature)")
    lp.add_argument("--trust-min", type=int, default=12)
    lp.add_argument("--trust-accuracy", type=float, default=0.7)
    lp.add_argument("--reserve", type=float, default=0.4, help="stop when less than this is left of the run's cap")
    lp.add_argument("--dry-run", action="store_true", help="a scripted no-op student: no model call")
    lp.add_argument("--cap", type=float, default=5.0)
    lp.set_defaults(fn=lambda a: cmd_learnability(a, data_root))

    nv = sub.add_parser("novelty", help="P4: signatures, near-duplicate threshold calibration, coverage and diversity")
    common(nv)
    nv.add_argument("--paraphrase", type=int, default=0, help="paraphrase positives from the proxy model (calls)")
    nv.add_argument("--cap", type=float, default=0.3)
    nv.set_defaults(fn=lambda a: cmd_novelty(a, data_root))

    pf = sub.add_parser("perfect", help="P1-P4 per task, the on-intent share per pool, admission")
    common(pf)
    pf.add_argument("--calibration", default=None, help="a novelty.json with the calibrated (w, theta)")
    pf.add_argument("--accept-provisional", action="store_true")
    pf.set_defaults(fn=lambda a: cmd_perfect(a, data_root))

    fr = sub.add_parser("frontier", help="K: snapshot-replay attribution of the student's failures, target requests, the cap/R4 plan")
    common(fr)
    fr.add_argument("--max-calls", type=int, default=20)
    fr.add_argument("--min-traces", type=int, default=2)
    fr.add_argument("--per-request", type=int, default=2)
    fr.add_argument("--kappa", type=float, default=0.1)
    fr.set_defaults(fn=lambda a: cmd_frontier(a, data_root))

    rd = sub.add_parser("frontier-round", help="K: one real targeted vs equal-budget untargeted batch (swarm + student on Tinker)")
    common(rd)
    rd.add_argument("--attributions", required=True, help="a frontier.json (its requests)")
    rd.add_argument("--domain", default="finance_ap")
    rd.add_argument("--world", default=None, help="a generated domain's worldgen output dir")
    rd.add_argument("--requests", type=int, default=2)
    rd.add_argument("--n", type=int, default=6, help="tasks per arm")
    rd.add_argument("--batch", type=int, default=3)
    rd.add_argument("--rounds", type=int, default=2)
    rd.add_argument("--seeds", default="0,1")
    rd.add_argument("--invoices", type=int, default=30)
    rd.add_argument("--cap-arm", type=float, default=1.5, help="the swarm's cap per arm (equal budget)")
    rd.add_argument("--accept-provisional", action="store_true")
    rd.add_argument("--kappa", type=float, default=0.1)
    rd.add_argument("--cap", type=float, default=4.0)
    rd.set_defaults(fn=lambda a: cmd_frontier_round(a, data_root))

    oc = sub.add_parser("outcome", help="the student-probe outcome classifier (intended / unintended pass or fail) for the RSI loop")
    common(oc)
    oc.add_argument("--calibrate", action="store_true", help="the known-label fixture cases: the confusion matrix (no model)")
    oc.add_argument("--panel", action="store_true", help="the calibrated swarm panel decides the residue (Tinker)")
    oc.add_argument("--anchors", type=int, default=8, help="tasks whose oracle and no-op traces calibrate the panel's λ")
    oc.add_argument("--mut", action="store_true", help="run MUT in the task evidence (slow on large Worlds)")
    oc.add_argument("--max-calls", type=int, default=20)
    oc.add_argument("--cap", type=float, default=1.0)
    oc.set_defaults(fn=lambda a: cmd_outcome(a, data_root))
