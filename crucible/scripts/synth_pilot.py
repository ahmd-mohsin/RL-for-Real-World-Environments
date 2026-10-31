#!/usr/bin/env python3
"""The Part 2 10 x 10 pilot (DESIGN §7.7 item 26; runs/pilot-10x10.md): 10 World instances x 10 tasks each, end to end,
with quality and generation speed measured. Every stage is the framework's own code; this script only orders them and
reads their records.

    synth_pilot.py budget   --plan 24 [--apply]                       # R-budget: fund the pilot's stage through policy.resolve(over_budget)
    synth.py       batch    --work pilot-10x10 --run <R> --arms a1v3 --cells 10 --quota 1 --max-per-cell 1 --world n=<inst> ...
    synth_pilot.py ladder   --run <R> --cap 2                         # R-ALT-whole: the solver ladder on ALT:no_pass rejections
    synth_pilot.py manifest --run <R>                                 # the batch's tasks as the loop's pool (stored certificates)
    synth_taskloop.py loop  --work pilot-10x10 --manifest <R>/manifest.pilot.pool.json --run <R>/loop ...
    synth_taskloop.py report --run <R>/loop --manifest ...
    synth_pilot.py report   --run <R>                                 # the funnel, speed, USD per stage / certified / perfect task

Runs sit under ``<data-root>/pilot-10x10/<R>`` (every receipt there carries "pilot-10x10" in its path; the stage cap is
R-budget's ``stage_caps_usd["pilot-10x10"]``). The shared Part 2 total ``MILESTONE_CAP_USD`` (150) is checked before every
paid stage. Swarm only: no Qwen model outside the student probe; the key comes from the environment only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1]
if str(HERE / "src") not in sys.path:
    sys.path.insert(0, str(HERE / "src"))
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

WORK = "pilot-10x10"


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


def shared_cap() -> float:
    if not os.environ.get("MILESTONE_CAP_USD"):
        raise SystemExit("set MILESTONE_CAP_USD (the shared Part 2 total, 150) in the environment")
    return float(os.environ["MILESTONE_CAP_USD"])


def stage_left(root: Path, work: str = WORK) -> tuple[float, float, float]:
    """(the stage cap, spent under ``<root>/<work>``, the shared Part 2 total's remainder)."""
    from crucible.factory.synth_wire import milestone_spent
    from crucible.synth import policy

    cap = float((policy.param("R-budget", "stage_caps_usd") or {}).get(work, 0.0))
    spent = milestone_spent(root / work) if (root / work).exists() else 0.0
    return cap, spent, round(shared_cap() - milestone_spent(root), 4)


# ---------------------------------------------------------------------------------------------------------- budget

def cmd_budget(a: argparse.Namespace) -> None:
    """R-budget through the policy: the pilot's stage is funded by ``resolve(over_budget)`` (reallocate within the Part 2
    total, an inferred ledger entry) or trimmed, or escalated beyond the total; ``--apply`` appends the entry."""
    from crucible.synth import policy
    from crucible.synth.policy import Condition, PolicyLedger

    root = data_root(a)
    if a.run_stage:
        return _run_stage_budget(a, root)
    cap, spent, remaining = stage_left(root, a.stage)
    facts = {"stage": a.stage, "planned_usd": float(a.plan), "stage_remaining_usd": round(cap - spent, 4), "min_useful_usd": float(a.min_useful),
             "part2_remaining_usd": remaining, "arms": 1}
    c = Condition(kind="over_budget", facts=facts)
    r = policy.resolve(c)
    out = {"condition": c.model_dump(mode="json"), "remedy": r.model_dump(mode="json"), "policy_version": policy.tag()}
    print(json.dumps({"facts": facts, "action": r.action, "rule": r.rule, "changes": [x.model_dump(mode="json") for x in r.changes],
                      "policy_version": out["policy_version"]}, indent=1, default=str))
    if a.apply and r.action == "reallocate_within_total" and r.changes:
        ch = r.changes[0]
        e = PolicyLedger().append({"kind": "amend_rule", "rule_id": "R-budget", "set": {"stage_caps_usd": ch.new},
                                   "rationale": f"{WORK}: policy.resolve(over_budget) -> reallocate_within_total under {out['policy_version']}; the "
                                                f"{a.stage} stage has {facts['stage_remaining_usd']} USD left of {cap}, the lead's pilot brief plans "
                                                f"{a.plan} USD (10 World instances x 10 tasks, end to end)",
                                   "source": {"kind": "inferred", "ref": "policy.resolve(over_budget); the lead's 10 x 10 pilot brief (USD 24 cap)",
                                              "date": time.strftime("%Y-%m-%d")},
                                   "data": {"condition": c.model_dump(mode="json"), "part2_remaining_usd": remaining}})
        out["ledger"] = {"seq": e["seq"], "sha": e["sha"], "tag": policy.tag()}
        print(f"ledger entry seq {e['seq']}: {policy.tag()}")
    if a.out:
        _save(Path(a.out), out)


def _run_stage_budget(a: argparse.Namespace, root: Path) -> None:
    """A batch's own per-stage cap (``--stage-cap``) bound: the same procedure one level down. ``resolve(over_budget)``
    with the stage's remainder and the ENCLOSING total being what the arm's hard cap has left; reallocating raises only
    that run's stage cap (a run setting, not a ledger parameter), appended to ``<run>/budget-run-stages.jsonl``. The arm's cap,
    the pilot's ledger stage cap and the Part 2 total are unchanged."""
    from crucible.synth import policy
    from crucible.synth.policy import Condition

    arm = root / WORK / a.run / a.arm
    rows = _spend_rows(arm)
    stage_spent = sum(float(r.get("usd") or 0.0) for r in rows if r.get("stage") == a.run_stage)
    arm_spent = sum(float(r.get("usd") or 0.0) for r in rows)
    facts = {"stage": a.run_stage, "planned_usd": float(a.plan), "stage_remaining_usd": round(float(a.stage_cap_now) - stage_spent, 4),
             "min_useful_usd": float(a.min_useful), "part2_remaining_usd": round(float(a.arm_cap) - arm_spent, 4), "arms": 1}
    c = Condition(kind="over_budget", subject=f"{a.run}/{a.arm}:{a.run_stage}", facts=facts, raised_by="scripts/synth_pilot.py budget --run-stage")
    r = policy.resolve(c)
    new = round(stage_spent + float(a.plan), 2) if r.action == "reallocate_within_total" else (
        float(a.stage_cap_now) if r.action == "trim_plan" else None)
    out = {"condition": c.model_dump(mode="json"), "action": r.action, "rule": r.rule, "policy_version": policy.tag(), "stage_cap_before": float(a.stage_cap_now),
           "stage_cap_after": new, "note": "a run-local stage cap (synth.py batch --stage-cap); the enclosing total is the arm's hard cap",
           "considered": [x for x in (r.model_dump(mode="json").get("considered") or [])]}
    with (root / WORK / a.run / "budget-run-stages.jsonl").open("a") as fh:  # every decision kept, in order
        fh.write(json.dumps(out | {"t": round(time.time(), 3)}, default=str) + "\n")
    print(json.dumps({k: out[k] for k in ("action", "rule", "policy_version", "stage_cap_before", "stage_cap_after")} | {"facts": facts}, indent=1))


# ---------------------------------------------------------------------------------------------------------- the batch's tasks

def run_dir(a: argparse.Namespace) -> Path:
    return data_root(a) / WORK / a.run


def arm_dir(a: argparse.Namespace) -> Path:
    return run_dir(a) / a.arm


def cands(arm: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((arm / "cands").glob("*.json"))]


def task_dir(arm: Path, c: dict[str, Any]) -> Path:
    return arm / c["domain"].replace("/", "_") / "tasks" / f"t{int(c['n']):02d}"


LADDER_CERT = "certificate.ladder.json"


async def _ladder(a: argparse.Namespace, todo: list[dict[str, Any]], cap: float) -> list[dict[str, Any]]:
    from crucible.factory.synth_wire import SYNTH_ROLES, SynthWire, ladder_configs, ladder_models
    from crucible.synth import policy
    from crucible.synth.certificate import Certificate, with_ladder
    from crucible.synth.falsify import FalsifyContext
    from crucible.synth.falsify.agents import Attempt, Panel, PanelMember, trace_view
    from crucible.synth.policy import Condition
    from crucible.synth.recert import world_builders, write_new
    from crucible.synth.solvers import escalate_alt
    from crucible.synth.task import Task

    arm = arm_dir(a)
    d = run_dir(a) / "ladder"
    wire = SynthWire(d, cap_usd=cap, table=dict(SYNTH_ROLES) | {"alt_solver": ladder_models()})
    configs = ladder_configs(wire)
    rp = policy.current().rule("R-ALT-whole").params
    per_task = float(rp["usd_per_config_task"]) * sum(len(r["configs"]) for r in rp["ladder_rungs"][1:])

    def cost_of(prefix: str) -> float:
        return sum(float(r.get("usd") or 0.0) for r in wire.usage.rows if str(r.get("tag", "")).startswith(prefix))

    out: list[dict[str, Any]] = []
    for c in todo:
        tdir = task_dir(arm, c)
        raw = json.loads((tdir / "cegis.json").read_text())
        task = Task.model_validate(raw["task"])
        cert = Certificate.model_validate(raw["certificate"])
        left = round(cap - wire.meter.spent, 4)
        facts = {"truth_available": False, "oracle_checked": False, "rungs_tried": 1, "budget_ok": left >= per_task, "planned_usd": per_task,
                 "left_usd": left}
        cond = Condition(kind="no_solver_pass", subject=task.id, facts=facts, raised_by="scripts/synth_pilot.py ladder")
        rem = policy.resolve(cond)
        row: dict[str, Any] = {"task": task.id, "cand": c["id"], "domain": c["domain"], "facts": facts, "action": rem.action, "rule": rem.rule,
                               "policy_version": rem.policy_version}
        if rem.action == "escalate_solver_ladder":
            cal = json.loads((arm / c["domain"].replace("/", "_") / "calibration.json").read_text()) if (
                arm / c["domain"].replace("/", "_") / "calibration.json").exists() else {}
            panel = Panel(members=[PanelMember(f"panel-{x.family}-{i}", x) for i, x in enumerate(wire.clients("panel"))], lam=float(cal.get("lam", 1.01)),
                          queue=d / "human_queue.jsonl", tag=f"adjudicate:ladder:{task.id}")
            gctx = FalsifyContext()

            async def judge(t: Task, run: Any, panel: Any = panel, gctx: Any = gctx) -> bool | None:
                v = gctx.grade(run, t.contract)
                pv = await panel.judge(t, Attempt(trace=trace_view(run.calls), answer=run.answer, reasons=list(v.reasons)))
                return {"valid": True, "invalid": False}.get(pv.decision)

            builders = sorted({"deepseek", "glm"} | set(world_builders(task)))
            try:
                lo = await escalate_alt(task, max(1, len(task.oracle.steps)), configs, start=1, builders=builders, judge=judge, cost_of=cost_of,
                                        out=d, stop=lambda: wire.meter.spent >= cap * 0.97)
            finally:
                await gctx.close()
            new = with_ladder(cert, lo.rows, lo.verdicts)
            row |= {"passed": lo.passed, "rung": lo.rung, "rungs_tried": lo.rungs_tried, "attempts": len(lo.rows), "skipped": lo.skipped,
                    "verdicts": {str(k): v for k, v in lo.verdicts.items()}, "status": new.status, "reasons": new.reasons}
            p = tdir / LADDER_CERT
            if not p.exists():
                write_new(p, {"certificate": new.model_dump(mode="json"), "ladder": lo.model_dump(mode="json"), "remedy": rem.model_dump(mode="json")})
        out.append(row)
        with (d / "ladder.jsonl").open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        print(json.dumps({k: row.get(k) for k in ("task", "action", "passed", "rung", "status", "reasons")}, default=str), flush=True)
    _save(d / "cost.json", wire.cost())
    return out


def cmd_ladder(a: argparse.Namespace) -> None:
    """R-ALT-whole on the batch's rejections whose only reason is ``ALT:no_pass``: ``resolve(no_solver_pass)`` decides; the
    ladder climbs from L1 (the batch's ALT was L0, the certifying set) at the rule's equal per-configuration budget, failed
    attempts go to the calibrated panel, and the certificate is re-decided by ``with_ladder`` (written beside the batch's,
    never over it). Resumable: tasks with a ladder certificate are skipped."""
    import asyncio

    root = data_root(a)
    stage, spent, remaining = stage_left(root)
    cap = min(float(a.cap), round(stage - spent, 4), remaining)
    print(f"{WORK} {spent:.4f} of {stage}; Part 2 remainder {remaining:.4f}; ladder cap {cap:.4f}", flush=True)
    if cap < 0.05:
        raise SystemExit("no budget left for the ladder")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not in the environment (set -a; . <repo>/.env; set +a)")
    arm = arm_dir(a)
    def alt_only(why: list[str]) -> bool:  # the only failed rule is ALT:no_pass (open suspects and MUT's bound are not rules)
        return [r for r in why if ":suspects=" not in r and not r.startswith("MUT:upper")] == ["ALT:no_pass"]

    todo = [c for c in cands(arm) if c["status"] == "rejected" and alt_only(c["reasons"])
            and (task_dir(arm, c) / "cegis.json").exists() and not (task_dir(arm, c) / LADDER_CERT).exists()]
    print(f"{len(todo)} rejections whose only failed rule is ALT:no_pass", flush=True)
    asyncio.run(_ladder(a, todo, cap))


def cmd_manifest(a: argparse.Namespace) -> None:
    """The batch's built tasks as the per-task loop's pool: ``recert.collect`` over the arm (every status), the archive
    copied per source, and one row per task from its STORED certificate (the ladder's when the ladder re-decided it)."""
    from collections import Counter

    from crucible.synth.recert import SCHEMA, Source, collect, copy_archive, stored_rows, world_builders

    arm = arm_dir(a)
    d = run_dir(a)
    src = Source(name=f"{a.run}/{a.arm}", kind="pilot", dir=str(arm), archive_root=str(arm / "archive"), statuses=["certified", "provisional", "rejected"],
                 seed0=False)
    items = collect([src])
    for it in items:
        it.builders = sorted(set(it.builders) | set(world_builders(it.task)))
        lad = it.tdir / LADDER_CERT
        if lad.exists():  # R-ALT-whole re-decided it: that certificate is the task's
            doc = json.loads(lad.read_text())
            it.old.status, it.old.reasons, it.old.path = doc["certificate"]["status"], list(doc["certificate"]["reasons"]), str(lad)
    copied = copy_archive(items, d / "archives")
    _save(d / "items.json", [i.model_dump(mode="json") for i in items])
    rows = []
    for r in stored_rows(items):
        lad = Path(r.certificate)
        if lad.name == LADDER_CERT:  # the row reads the task from the batch's cegis.json, the certificate from the ladder's
            r = r.model_copy(update={"task_path": str(lad.parent / "cegis.json")})
        rows.append(r)
    doc = {"schema": SCHEMA, "run": str(d), "tag": "pilot", "base": None, "made": time.strftime("%Y-%m-%d %H:%M:%S"),
           "summary": {"tasks": len(rows), "status": dict(Counter(r.status for r in rows)), "by_domain": dict(Counter(r.domain for r in rows)),
                       "archive_records_copied": copied},
           "rows": [r.model_dump(mode="json") for r in rows]}
    out = d / "manifest.pilot.pool.json"
    _save(out, doc)
    print(json.dumps(doc["summary"], indent=1))
    print(f"manifest: {out}")


# ---------------------------------------------------------------------------------------------------------- report

def _spend_rows(d: Path) -> list[dict[str, Any]]:
    out = []
    for p in d.rglob("spend.jsonl") if d.exists() else []:
        for line in p.read_text().splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line) | {"_file": str(p.relative_to(d))})
                except ValueError:
                    continue
    return out


def funnel(cs: list[dict[str, Any]], ladder: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """proposed -> valid (built: an executed oracle and a contract) -> certificate status (after the ladder)."""
    from collections import Counter

    st = Counter()
    for c in cs:
        s, why = c["status"], c["reasons"]
        if c["id"] in ladder and ladder[c["id"]].get("status"):
            s, why = ladder[c["id"]]["status"], ladder[c["id"]].get("reasons") or []
        if s == "rejected" and not [r for r in why if ":suspects=" not in r and not r.startswith("MUT:upper")]:
            s = "on_hold"  # open suspects only (R-panel: held, the recert statuses; ``recert.stored_rows``)
        st[s] += 1
    built = sum(1 for c in cs if c.get("task"))
    return {"proposed": len(cs), "duplicate": st["duplicate"], "filtered": st["filtered"], "dropped": st["dropped"], "valid": built,
            "certified": st["certified"], "provisional": st["provisional"], "on_hold": st["on_hold"], "rejected": st["rejected"],
            "error": st["error"], "budget": st["budget"]}


def speed(cs: list[dict[str, Any]], t_end: float | None = None, gap_s: float = 180.0) -> dict[str, Any]:
    """Wall-clock from the first proposer call: to the first certified task, to the 100th attempted (proposal landed),
    the whole batch; tasks per hour overall and per stage; per stage the summed busy time and its median."""
    import statistics

    if not cs:
        return {}
    t0 = min(c["times"]["propose"][0] for c in cs if c["times"].get("propose"))
    ends = [t[1] for c in cs for t in c["times"].values() if t and t[1]]
    t1 = t_end or max(ends)
    landed = sorted(c["times"]["propose"][1] for c in cs if c["times"].get("propose"))
    cert = sorted(c["times"]["refine"][1] for c in cs if c["status"] == "certified" and c["times"].get("refine"))
    adm = sorted(c["times"]["refine"][1] for c in cs if c["status"] in ("certified", "provisional") and c["times"].get("refine"))
    # a stopped and resumed batch: a gap of more than ``gap_s`` with no candidate event is downtime, not generation time
    stamps = sorted(x for c in cs for t in c["times"].values() if t for x in t if x)
    down = sum(b - a for a, b in pairwise(stamps) if b - a > gap_s)

    def active(t: float) -> float:  # seconds from t0 to t without the downtime before t
        return (t - t0) - sum(b - a for a, b in pairwise(stamps) if b - a > gap_s and b <= t)

    wall_h = max(1e-9, (t1 - t0 - down) / 3600)
    stages = {}
    for s in ("propose", "build", "refine"):
        xs = [c["times"][s][1] - c["times"][s][0] for c in cs if c["times"].get(s) and c["times"][s][1] and c["times"][s][0]]
        span = [(c["times"][s][0], c["times"][s][1]) for c in cs if c["times"].get(s) and c["times"][s][1]]
        sw = (max(b for _, b in span) - min(a for a, _ in span)) / 3600 if span else 0.0
        stages[s] = {"n": len(xs), "busy_s": round(sum(xs), 1), "median_s": round(statistics.median(xs), 1) if xs else None,
                     "span_h": round(sw, 3), "per_hour_over_span": round(len(xs) / sw, 1) if sw else None}
    return {"wall_s": round(t1 - t0 - down, 1), "downtime_s": round(down, 1), "to_first_certified_s": round(active(cert[0]), 1) if cert else None,
            "to_first_admitted_s": round(active(adm[0]), 1) if adm else None,
            "to_100_attempted_s": round(active(landed[99]), 1) if len(landed) >= 100 else None, "attempted": len(landed),
            "attempted_per_hour": round(len(landed) / wall_h, 1), "valid_per_hour": round(sum(1 for c in cs if c.get("task")) / wall_h, 1),
            "certified_per_hour": round(len(cert) / wall_h, 2), "admitted_per_hour": round(len(adm) / wall_h, 2), "stages": stages,
            "serial_equivalent_s": round(sum(v["busy_s"] for v in stages.values()), 1)}


def cmd_report(a: argparse.Namespace) -> None:
    """The pilot's numbers (runs/pilot-10x10.md): the funnel per World, speed, USD per stage, per certified task and per
    perfect task, the probe outcome classes and what the loop did, from the batch's candidates, the ladder's log, the
    loop's state and receipts. No model call."""
    from collections import Counter, defaultdict

    d = run_dir(a)
    arm = arm_dir(a)
    cs = cands(arm)
    ladder = {}
    lp = d / "ladder" / "ladder.jsonl"
    if lp.exists():
        for line in lp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                ladder[r["cand"]] = r
    out: dict[str, Any] = {"run": str(d), "policy_version": None}
    out["funnel"] = {"all": funnel(cs, ladder), "by_world": {w: funnel([c for c in cs if c["domain"] == w], ladder) for w in sorted({c["domain"] for c in cs})}}
    out["reasons"] = dict(Counter(r.split(":")[0] + ":" + (r.split(":")[1].split("=")[0] if ":" in r else "")
                                  for c in cs for r in c["reasons"] if c["status"] in ("rejected", "dropped", "filtered")).most_common(30))
    out["speed"] = speed(cs)
    # USD per stage: the batch's receipts carry the stage; the ladder and the loop are stages of their own
    rows = _spend_rows(arm)
    by_stage: dict[str, float] = defaultdict(float)
    for r in rows:
        by_stage[r.get("stage") or ("unbooked_estimate" if r.get("kind") == "unbooked_estimate" else "batch:other")] += float(r.get("usd") or 0.0)
    # the loop runs, oldest first (a re-run on the lineages a fix affected comes after): every receipt counts, and a lineage's
    # latest run decides it
    loops = [Path(x) for x in a.loop.split(",")] if a.loop else sorted(x for x in d.iterdir() if x.is_dir() and x.name.startswith("loop") and
                                                                     not x.name.endswith("-dry") and (x / "loop" / "task_state.json").exists())
    by_stage["ladder"] += sum(float(r.get("usd") or 0.0) for r in _spend_rows(d / "ladder"))
    for lp in loops:
        by_stage["loop:student"] += sum(float(r.get("usd") or 0.0) for r in _spend_rows(lp / "student"))
        by_stage["loop:swarm"] += sum(float(r.get("usd") or 0.0) for r in _spend_rows(lp / "swarm"))
    total_here = sum(float(r.get("usd") or 0.0) for r in _spend_rows(d))
    earlier = {p.name: round(sum(float(r.get("usd") or 0.0) for r in _spend_rows(p)), 4) for p in (d.parent).iterdir() if p.is_dir() and p != d}
    fun = out["funnel"]["all"]
    out["usd"] = {"by_stage": {k: round(v, 4) for k, v in sorted(by_stage.items())}, "this_run": round(total_here, 4), "earlier_runs": earlier,
                  "per_attempted": round(total_here / max(1, fun["proposed"]), 4),
                  "per_certified": round(total_here / fun["certified"], 3) if fun["certified"] else None,
                  "per_admitted": round(total_here / max(1, fun["certified"] + fun["provisional"]), 3) if fun["certified"] + fun["provisional"] else None}
    dom_of = {r["task_id"]: r["domain"] for r in json.loads((d / "manifest.pilot.pool.json").read_text())["rows"]} if (
        d / "manifest.pilot.pool.json").exists() else {}

    def world(tid: str) -> str:
        return dom_of.get(tid) or next((v for k, v in sorted(dom_of.items(), key=lambda kv: -len(kv[0])) if tid.startswith(k)), "?")

    decs: list[dict[str, Any]] = []
    lins: dict[str, dict[str, Any]] = {}
    calib: dict[str, Any] = {}
    for lp in loops:
        st = json.loads((lp / "loop" / "task_state.json").read_text())
        decs += [x | {"run": lp.name} for x in st["decisions"]]
        lins.update(st["lineages"])  # a later run's lineage replaces the earlier one
        if (lp / "calibration.json").exists():
            calib[lp.name] = json.loads((lp / "calibration.json").read_text())
    if decs:
        out["policy_version"] = sorted({x.get("policy_version") for x in decs if x.get("policy_version")} |
                                       {r.get("policy_version") for x in decs for r in x.get("remedies") or [] if r.get("policy_version")})
        first: dict[str, str] = {}
        last: dict[str, str] = {}
        for x in decs:
            b = (x.get("band") or {}).get("decision")
            if b:
                first.setdefault(x["task_id"], b)
                last[x["task_id"]] = b
        counted = Counter()
        for x in decs:
            counted.update(x.get("counted") or {})
        out["loop"] = {"runs": [lp.name for lp in loops], "decisions": dict(Counter(x["move"] for x in decs)),
                       "accepted": dict(Counter(x["move"] for x in decs if x.get("accepted"))),
                       "lineages": dict(Counter(v["status"] for v in lins.values())),
                       "band_first": dict(Counter(first.values())), "band_last": dict(Counter(last.values())),
                       "by_world": {w: {"band": dict(Counter(b for t, b in first.items() if world(t) == w)),
                                        "lineages": dict(Counter(v["status"] for k, v in lins.items() if world(k) == w))}
                                    for w in sorted(set(dom_of.values()))},
                       "conditions": dict(Counter(f"{r['condition']}:{r['rule']}:{r['action']}" for x in decs for r in x.get("remedies") or [])),
                       "refused_by": dict(Counter(r.split(":", 1)[0] for x in decs for r in x.get("refused_by") or [])),
                       "outcome_classes": dict(counted), "rollouts": sum(len(v) for x in decs for v in (x.get("rollouts") or {}).values()),
                       "calibration": calib}
    reps = [json.loads((lp / "loop_report.json").read_text()) for lp in loops if (lp / "loop_report.json").exists()]
    if reps:
        oc = [r for rep in reps for r in rep.get("outcomes") or []]
        out["outcomes"] = {"rollouts": len(oc), "labels": dict(Counter(r.get("label") for r in oc)),
                           "unintended": dict(Counter(f"{r.get('label')}:{r.get('cause') or r.get('mode') or '?'}" for r in oc
                                                      if str(r.get("label", "")).startswith("unintended"))),
                           "world_fixes": [w for rep in reps for w in rep.get("world_fixes") or []], "lemmas": [rep.get("lemmas") for rep in reps],
                           "errors": sum(len(rep.get("errors") or []) for rep in reps), "cost": [rep.get("cost") for rep in reps]}
    lrep = next((lp / "report.json" for lp in reversed(loops) if (lp / "report.json").exists()), None)
    if lrep is not None:
        lr = json.loads(lrep.read_text())
        pf = lr.get("perfect") or {}
        n = int(pf.get("perfect") or 0)
        out["perfect"] = {"from": str(lrep), "kept": pf.get("kept"), "perfect": n, "rows": pf.get("rows"),
                          "usd_per_perfect": round(total_here / n, 3) if n else None, "kept_valid_on_intent_novel": pf.get("kept_provisional_ok")}
    p = Path(a.out) if a.out else d / "pilot_report.json"
    _save(p, out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("perfect",)}, indent=1, default=str)[:12000])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("budget")
    b.add_argument("--stage", default=WORK)
    b.add_argument("--plan", type=float, required=True, help="USD planned (with --run-stage: more spend the stage needs)")
    b.add_argument("--run", default="")
    b.add_argument("--arm", default="a1v3")
    b.add_argument("--run-stage", default="", help="a batch's own stage cap that bound (e.g. oracle)")
    b.add_argument("--stage-cap-now", type=float, default=0.0)
    b.add_argument("--arm-cap", type=float, default=0.0)
    b.add_argument("--min-useful", type=float, default=0.5)
    b.add_argument("--apply", action="store_true")
    b.add_argument("--out", default=None)
    b.set_defaults(fn=cmd_budget)
    for name, fn, help_ in (("ladder", cmd_ladder, "R-ALT-whole's solver ladder on ALT:no_pass rejections"),
                            ("manifest", cmd_manifest, "the batch's tasks as the loop's pool"),
                            ("report", cmd_report, "the pilot's tables")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--run", required=True)
        p.add_argument("--arm", default="a1v3")
        if name == "ladder":
            p.add_argument("--cap", type=float, default=2.0)
        if name == "report":
            p.add_argument("--loop", default="", help="the loop run directory (default <run>/loop)")
            p.add_argument("--out", default="", help="write the report JSON here (default <run>/pilot_report.json)")
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
