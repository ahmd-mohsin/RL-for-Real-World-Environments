#!/usr/bin/env python3
"""DESIGN §7.7 item 20 (Part 2 blind spots and backlog): measurements and the backlog tools.

    synth_falsifiers2.py matrix   --run-id rsi-2                 # FR/FA before (cheap_m3) and after (cheap), dev + locked, TAINT per layer
    synth_falsifiers2.py leak-l2  --run-id rsi-2 --cap 3         # the L2 leak probe + calibrated leak panel on dev, locked and anchors
    synth_falsifiers2.py repo-panel --cap 3                      # the open MUT suspects of the repository pool through the panel
    synth_falsifiers2.py visible  [--execute]                    # the visible-test-data leak detector on the pool and the real-defect set
    synth_falsifiers2.py triage   --queue <human_queue.jsonl>    # the human-review sheet (grouped, ranked; nothing decided)

Every artifact goes under ``<data>/synth/falsifiers-2/`` (``--data-root``: the synth data root, default
``$CRUCIBLE_DATA_DIR/synth`` or the config's). The locked sets are read only through ``recall.SetStore`` with
``purpose="report"`` (E1's rule: never for tuning). Spend: one cap for the whole package, ``CAP_USD`` = 15 over every
``spend.jsonl`` under ``falsifiers-2/``; each paid command also takes ``--cap``. The key comes from the environment only
(``set -a; . <repo>/.env; set +a`` in the shell that starts this script).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1]
if str(HERE / "src") not in sys.path:
    sys.path.insert(0, str(HERE / "src"))
if str(HERE) not in sys.path:  # the fixture domains' World factories live in tests.unit.synth_fixtures
    sys.path.append(str(HERE))

CAP_USD = 15.0
WORK = "falsifiers-2"


def data_root(a: argparse.Namespace) -> Path:
    if a.data_root:
        return Path(a.data_root)
    if os.environ.get("CRUCIBLE_DATA_DIR"):
        return Path(os.environ["CRUCIBLE_DATA_DIR"]) / "synth"
    from crucible.core.config import get_config

    return get_config().data_path() / "synth"


def work_dir(a: argparse.Namespace) -> Path:
    return data_root(a) / WORK


def cap_for(a: argparse.Namespace) -> float:
    from crucible.factory.synth_wire import work_spent

    spent = work_spent(data_root(a), WORK)
    cap = min(float(a.cap), round(CAP_USD - spent, 4))
    print(f"{WORK} spend so far {spent:.4f} of {CAP_USD}; this run's cap {cap:.4f}", flush=True)
    if cap < 0.02:
        raise SystemExit("the package budget is spent")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not in the environment")
    return cap


def write(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=str) + "\n")


def load_sets(a: argparse.Namespace, run_id: str):
    from crucible.synth.meta.recall import SetStore
    from crucible.synth.task import Task

    out = data_root(a) / "rsi" / run_id
    store = SetStore(out / "sets", data_root(a).parent / "sealed" / "synth-locked")
    dev = store.read(f"{run_id}-dev", "dev")
    locked = store.read(f"{run_id}-locked", "locked", purpose="report")  # reporting only
    src = json.loads((out / "sources.json").read_text())
    tasks = {k: Task.model_validate(v) for k, v in src["tasks"].items()}
    return dev, locked, [tasks[i] for i in src["dev"]], [tasks[i] for i in src["locked"]]


def by_op(m: Any) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for o in m.plants:
        c = out.setdefault(f"{o.op.split('@')[0]}:{o.op.split(':')[-1]}", {"n": 0, "hit": 0, "defect": o.defect})  # type: ignore[dict-item]
        c["n"] += 1
        c["hit"] += bool(o.hit)
    return dict(sorted(out.items()))


def cell(m: Any, d: str) -> dict[str, Any]:
    c = m.population.get(d)
    return {"hit": c.hits, "n": c.n, "lo": round(c.lo, 3), "hi": round(c.hi, 3)} if c else {}


# ---------------------------------------------------------------------------------------------------------- matrix

def cmd_matrix(a: argparse.Namespace) -> None:
    from crucible.synth.falsify import MUT, NULL, PERT, RED, REP, TAINT, TWIN, FalsifyContext, cheap, cheap_m3
    from crucible.synth.falsify.leak import LAYERS, SynonymMap
    from crucible.synth.meta.recall import measure, report_markdown
    from crucible.synth.meta.rsi import summary

    syn = SynonymMap.load(a.synonyms) if a.synonyms else SynonymMap()
    out = work_dir(a) / "matrix" / a.run_id
    out.mkdir(parents=True, exist_ok=True)

    async def go() -> dict[str, Any]:
        dev, locked, dev_src, locked_src = load_sets(a, a.run_id)
        ctx = FalsifyContext()
        doc: dict[str, Any] = {"run_id": a.run_id, "synonyms": syn.digest(), "sets": {}}
        md: list[str] = []
        try:
            for kind, plants, anchors, purpose in (("dev", dev.plants, dev_src, "loop"), ("locked", locked.plants, locked_src, "report")):
                row: dict[str, Any] = {}
                for name, pop in (("before", cheap_m3()), ("after", cheap(synonyms=syn))):
                    m = await measure(plants, anchors, pop, ctx, set_name=f"{a.run_id}-{kind}", set_kind=kind, purpose=purpose)  # type: ignore[arg-type]
                    row[name] = {"summary": summary(m), "by_op": by_op(m), "D6": cell(m, "D6"), "D7": cell(m, "D7"),
                                 "fa_by_class": {k: c.hits for k, c in m.fa.items()}, "anchors": m.anchors}
                    md += [f"## {a.run_id} {kind} {name}", "", report_markdown(m)]
                layers: dict[str, Any] = {}
                for i in range(1, len(LAYERS) + 1):  # TAINT per cumulative layer, the rest of the population as after
                    ls = LAYERS[:i]
                    pop = [REP(), RED(), TAINT(layers=ls, synonyms=syn), NULL(), MUT(), TWIN(), PERT()]
                    m = await measure(plants, anchors, pop, ctx, set_name=f"{a.run_id}-{kind}", set_kind=kind, purpose=purpose)  # type: ignore[arg-type]
                    layers["+".join(ls)] = {"D7_taint": {"hit": m.fr["TAINT"]["D7"].hits, "n": m.fr["TAINT"]["D7"].n} if "D7" in m.fr["TAINT"] else {},
                                            "D7_any": cell(m, "D7"), "taint_fa": m.fa["TAINT"].hits, "fa_n": m.fa["TAINT"].n,
                                            "taint_hits_all": sum(c.hits for c in m.fr["TAINT"].values())}
                row["taint_layers"] = layers
                doc["sets"][kind] = row
        finally:
            await ctx.close()
        write(out / "matrix.json", doc)
        (out / "matrix.md").write_text("\n".join(md))
        return {k: {n: {"fr_all": v[n]["summary"]["fr_all"], "hits": v[n]["summary"]["hits"], "n": v[n]["summary"]["n"], "fa": v[n]["summary"]["fa"],
                        "D6": v[n]["D6"], "D7": v[n]["D7"]} for n in ("before", "after")} | {"layers": v["taint_layers"]} for k, v in doc["sets"].items()}

    print(json.dumps(asyncio.run(go()), indent=1, default=str))


# ---------------------------------------------------------------------------------------------------------- repo suspects

def _patch_diff(env: Any, files: dict[str, str | None], limit: int = 24000) -> str:
    import difflib

    out = []
    for p, t in sorted(files.items()):
        old = env.spec.start_files.get(p) if p in env.spec.start_files else env.base_text(p)
        if (old or "") == (t or ""):
            continue
        out += difflib.unified_diff((old or "").splitlines(keepends=True), (t or "").splitlines(keepends=True), f"a/{p}", f"b/{p}" if t is not None else "/dev/null")
    s = "".join(out)
    return s if len(s) <= limit else s[:limit] + f"\n...<{len(s) - limit} more characters of the diff>"


def _attempt(env: Any, files: dict[str, str | None], reasons: list[str] | None = None) -> Any:
    from crucible.synth.falsify.agents import Attempt

    d = _patch_diff(env, files)
    return Attempt(trace=[{"tool": "repo.apply_patch", "args": {"patch": d}, "result": "applied"}] if d else [], answer=None, reasons=list(reasons or []))


def cmd_repo_panel(a: argparse.Namespace) -> None:
    """The open MUT suspects of the repository pool (DropHunk survivors, ``runs/m5/certificates_summary.json``) through
    the calibrated cross-family panel. Calibration anchors carry executed truth: the oracle diff (valid), the lane's
    commit-revert mutants with their executed milestone vectors (each fails its own milestone: invalid), the empty
    patch (invalid). The suspects' grades are the stored ones (every suspect passed the hidden tests); nothing is
    re-graded and no task is changed. Verdicts: decided valid -> clean (the dropped code is not needed); invalid ->
    confirmed D1 (the contract passes an incomplete fix); abstain -> still a suspect (human queue)."""
    from crucible.factory.synth_patch_wire import _NoRunner, open_pool, patch_env
    from crucible.factory.synth_wire import SynthWire
    from crucible.synth.falsify.agents import Anchor, Panel, PanelMember, calibrate
    from crucible.synth.falsify.patch import PatchMUT

    summ = json.loads(Path(a.summary).read_text())["tasks"]
    todo = {w: [s for s in r["findings"]["suspects"] if s["cls"] == "MUT"] for w, r in summ.items()}
    todo = {w: ss for w, ss in todo.items() if ss}
    cap = cap_for(a)
    d = work_dir(a) / f"repo-panel-{time.strftime('%Y%m%d-%H%M%S')}"
    d.mkdir(parents=True)
    from crucible.factory import models as fm
    from crucible.factory.synth_wire import SYNTH_ROLES

    table = {**SYNTH_ROLES, "panel": tuple(getattr(fm, m) for m in a.members.split(","))} if a.members else None
    wire = SynthWire(d, cap_usd=cap, table=table)
    from crucible.synth.roles import PANEL, PANEL_PATCH

    panel = Panel(members=[PanelMember(n, c) for n, c in wire.run_roles().panel], lam=0.7, queue=d / "human_queue.jsonl", tag="adjudicate:repo",
                  role=PANEL if a.tool_prompt else PANEL_PATCH)
    pool = open_pool(Path(a.pool))

    async def go() -> dict[str, Any]:
        anchors, suspects = [], []
        for wid in sorted(summ):
            env = patch_env(pool.task(wid))
            task = env.task()
            anchors.append(Anchor(task=task, attempt=_attempt(env, dict(env.spec.oracle_files)), valid=True, kind="oracle"))
            anchors.append(Anchor(task=task, attempt=_attempt(env, {}), valid=False, kind="noop"))
            for m in env.spec.mutants[: a.mutants]:
                if m.observed and not all(m.observed.values()):  # executed: the mutant fails a milestone
                    anchors.append(Anchor(task=task, attempt=_attempt(env, {**env.spec.oracle_files, **m.files}), valid=False, kind="revert_fix"))
            if wid in todo:
                ctx = env.context(_NoRunner())
                probes = {p.site: p for p in await PatchMUT(env, max_hunks=a.max_hunks).propose(task, ctx) if p.op in ("DropHunk", "DropFile")}
                for s in todo[wid]:
                    p = probes.get(s["site"])
                    if p is None:
                        suspects.append({"task": wid, "site": s["site"], "error": "probe not re-proposed"})
                        continue
                    files = {st.args["path"].v: (st.args["text"].v if st.tool == "repo.write_file" else None) for st in p.program.steps}  # type: ignore[union-attr]
                    suspects.append({"task": wid, "site": s["site"], "op": p.op, "env": env, "probe": p, "files": files})
        if a.dry:
            sizes = [len(json.dumps(x.attempt.trace)) for x in anchors] + [len(_patch_diff(s["env"], s["files"])) for s in suspects if "env" in s]
            return {"anchors": len(anchors), "kinds": [x.kind for x in anchors], "suspects": [(s["task"], s["site"], "error" in s) for s in suspects],
                    "chars_total": sum(sizes), "chars_max": max(sizes)}
        cal = await calibrate(panel, anchors, floor=0.7, min_decided=10, require_both=True, refuse_uncalibrated=True)
        sem = asyncio.Semaphore(6)

        async def judge(s: dict[str, Any]) -> dict[str, Any]:
            if "error" in s:
                return s
            async with sem:
                env = s["env"]
                v = await panel.judge(env.task(), _attempt(env, s["files"]))
            from crucible.synth.falsify.agents import resolve

            row = {"task": s["task"], "site": s["site"], "op": s["op"], "decision": v.decision, "votes": [x.model_dump(mode="json") for x in v.votes], "lam": v.lam}
            if v.decision == "abstain":
                row["label"], row["defect"] = "suspect", "D1"
                with (d / "human_queue.jsonl").open("a") as fh:  # the triage tool's row format (panel log + attempt)
                    fh.write(json.dumps({**row, "cls": "MUT", "version": 1, "grader_passed": True, "instruction": env.task().instruction,
                                         "attempt": _attempt(env, s["files"]).model_dump(mode="json")}, default=str) + "\n")
            else:
                row["label"], row["defect"] = resolve(True, v.decision, "D1")  # every suspect passed the hidden tests
            return row

        rows = list(await asyncio.gather(*(judge(s) for s in suspects)))
        status = {}
        for wid, r in sorted(summ.items()):
            mine = [x for x in rows if x.get("task") == wid]
            reasons = [x for x in r["reasons"] if not x.startswith("MUT:suspects")]
            open_ = sum(1 for x in mine if x.get("label") == "suspect")
            conf = sum(1 for x in mine if x.get("label") == "confirmed")
            if open_:
                reasons.append(f"MUT:suspects={open_}")
            if conf:
                reasons.append(f"MUT:confirmed={conf}")
            st = "rejected" if reasons else ("provisional" if r.get("provisional_for") or r["status"] != "certified" else "certified")
            status[wid] = {"before": r["status"], "after": st, "reasons_after": reasons, "suspects": len(mine)}
        doc = {"calibration": cal.model_dump(mode="json"), "anchors": len(anchors), "rows": rows, "status": status, "cost": wire.cost()}
        write(d / "repo_panel.json", doc)
        return {"lam": cal.lam, "calibrated": cal.calibrated, "cal_rows": cal.rows, "anchors": len(anchors),
                "verdicts": [{k: x.get(k) for k in ("task", "site", "decision", "label")} for x in rows], "status": status, "cost": wire.cost()["total_usd"], "dir": str(d)}

    print(json.dumps(asyncio.run(go()), indent=1, default=str))


# ---------------------------------------------------------------------------------------------------------- L2 leaks

DEV_TEMPLATES = ("The reason code is {v}.", "This is because of {v}.")  # the dev planter's own variants (base_planter)


def _planter_templates(a: argparse.Namespace, run_id: str) -> list[str]:
    """The planter loop's accepted dev templates (rsi-2 E2's P versions): dev variants with the decision in words."""
    out: list[str] = []
    for p in sorted((data_root(a) / "rsi" / run_id).glob("evolve-*/loop/theta/P/v*.json")):
        for op in (json.loads(p.read_text()).get("data") or {}).get("ops") or []:
            t = (op.get("params") or {}).get("template")
            if t and t not in out:
                out.append(t)
    return out


def _sentences(text: str) -> list[str]:
    import re

    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 8]


def leak_anchors(dev_src: list[Any], templates: list[str]) -> list[Any]:
    """Calibration anchors for the leak panel, from dev sources only. Positives (a leak by construction): a dev template
    filled with a derive value, appended to the instruction; the quote is the inserted sentence. Negatives: the same
    value with each sentence of the certified source's own instruction (TAINT-clean, ALT-passed): saying what to check
    is not a give-away. The negative truth rests on the source being certified (a paraphrase leak in a certified source
    would be mislabelled): reported with the numbers."""
    from crucible.synth.falsify.leak import LeakAnchor

    out = []
    by_domain: dict[str, list[Any]] = {}
    for t in dev_src:
        by_domain.setdefault(t.domain, []).extend(v for v in t.intent.derive if isinstance(v, str))
    for t in dev_src:
        for v in t.intent.derive:
            h = str(v).replace("_", " ")
            for k, tmpl in enumerate(templates):
                s = tmpl.format(v=v, h=h)
                out.append(LeakAnchor(instruction=f"{t.instruction} {s}", value=v, quote=s, truth=True, kind=f"planted:{k}"))
            for s in _sentences(t.instruction)[:2]:
                out.append(LeakAnchor(instruction=t.instruction, value=v, quote=s, truth=False, kind="certified"))
            sib = next((x for x in by_domain.get(t.domain, []) if x != v and not any(c.isdigit() for c in x)), None)
            if sib is not None and not any(c.isdigit() for c in str(v)):  # hard negatives: an enumeration and a restated rule
                for kind, s in (("enumeration", f"Depending on the records, the outcome may be {h} or {sib.replace('_', ' ')}."),
                                ("rule", f"The policy says when {h} applies; check whether its condition holds before you act.")):
                    out.append(LeakAnchor(instruction=f"{t.instruction} {s}", value=v, quote=s, truth=False, kind=kind))
    return out


def cmd_leak_l2(a: argparse.Namespace) -> None:
    """The L2 leak probe (Inkling-Small) with the calibrated two-family leak panel, measured as its own layer
    (``skip_structural=False``): recall on D7 plants, false alarms on every other plant and anchor with derive values.
    Dev first; the locked set only with ``--locked`` (report only, after the dev run fixed every setting)."""
    from crucible.factory.synth_wire import SynthWire
    from crucible.synth.falsify import FalsifyContext, falsify
    from crucible.synth.falsify.agents import PanelMember
    from crucible.synth.falsify.leak import LeakPanel, LeakProbe, calibrate_leak_panel

    cap = cap_for(a)
    d = work_dir(a) / f"leak-l2-{a.run_id}-{time.strftime('%Y%m%d-%H%M%S')}"
    d.mkdir(parents=True)
    wire = SynthWire(d, cap_usd=cap)
    dev, locked, dev_src, locked_src = load_sets(a, a.run_id)
    templates = [*DEV_TEMPLATES, *_planter_templates(a, a.run_id)]

    async def go() -> dict[str, Any]:
        panel = LeakPanel(members=[PanelMember(f"leak_panel-{c.family}", c) for c in wire.clients("leak_panel")], lam=0.7, queue=d / "human_queue.jsonl")
        anchors = leak_anchors(dev_src, templates)
        cal = await calibrate_leak_panel(panel, anchors)
        probe = LeakProbe(wire.client("leak_probe"), panel=panel, skip_structural=False, max_calls=a.max_calls)
        ctx = FalsifyContext()
        sem = asyncio.Semaphore(8)
        sets = [("dev", dev.plants, dev_src)] + ([("locked", locked.plants, locked_src)] if a.locked else [])
        out: dict[str, Any] = {}
        for kind, plants, srcs in sets:
            items = [("plant", p.defect, p.op.key, p.task) for p in plants if p.task.intent.derive] + [("anchor", "-", "-", t) for t in srcs if t.intent.derive]

            async def one(it: tuple[str, str, str, Any]) -> dict[str, Any]:
                async with sem:
                    r = await falsify(it[3], probe, ctx)
                f = next((x for x in r.findings), None)
                return {"what": it[0], "defect": it[1], "op": it[2], "task": it[3].id, "n": r.n, "label": f.label if f and f.outcome.valid else None,
                        "finds": (f.outcome.data.get("finds") if f else None), "error": (f.outcome.error if f else None)}

            rows = list(await asyncio.gather(*(one(it) for it in items)))
            d7 = [r for r in rows if r["what"] == "plant" and r["defect"] == "D7"]
            other = [r for r in rows if not (r["what"] == "plant" and r["defect"] == "D7")]
            out[kind] = {"d7_plants": len(d7), "d7_confirmed": sum(r["label"] == "confirmed" for r in d7), "d7_suspect": sum(r["label"] == "suspect" for r in d7),
                         "d7_raised": sum(bool(r["finds"]) for r in d7), "others": len(other), "fa_confirmed": sum(r["label"] == "confirmed" for r in other),
                         "fa_suspect": sum(r["label"] == "suspect" for r in other), "fa_raised": sum(bool(r["finds"]) for r in other),
                         "invalid": sum(r["label"] is None for r in rows),
                         "by_op": {op: {"n": sum(1 for r in d7 if r["op"] == op), "confirmed": sum(1 for r in d7 if r["op"] == op and r["label"] == "confirmed")}
                                   for op in sorted({r["op"] for r in d7})}}
            write(d / f"rows-{kind}.json", rows)
        doc = {"calibration": cal, "anchors": len(anchors), "templates": templates, "results": out, "panel_log": panel.log, "probe_calls": probe.calls,
               "cost": wire.cost()}
        write(d / "leak_l2.json", doc)
        return {"lam": cal["lam"], "calibrated": cal["calibrated"], "cal_rows": cal["rows"], "anchors": len(anchors), "results": out,
                "probe_calls": probe.calls, "cost": wire.cost()["total_usd"], "dir": str(d)}

    print(json.dumps(asyncio.run(go()), indent=1, default=str))


def cmd_natural(a: argparse.Namespace) -> None:
    """The structural leak layers on every swarm-written task (all outcomes, final versions) of the synth runs: how many
    derive values each layer finds beyond the exact matcher, and (``--cap`` > 0) the calibrated leak panel's verdict on
    each such new hit: the precision of the paraphrase layers on natural text. Instruction-only: no labels, no options
    guard (both need the World), so this over-counts the layers' alarms, never under-counts them."""
    from crucible.factory.synth_wire import SynthWire
    from crucible.synth.falsify.agents import PanelMember
    from crucible.synth.falsify.leak import LeakMatcher, LeakPanel, calibrate_leak_panel
    from crucible.synth.meta.rsi import load_tasks

    tasks = {}
    for dom in ("finance_ap", "it_helpdesk"):
        for t, _i in load_tasks(data_root(a) / dom, statuses=("certified", "provisional", "rejected", "dropped")):
            tasks[t.id] = t
    new: list[dict[str, Any]] = []
    counts = {"tasks": len(tasks), "values": 0, "exact": 0}
    for t in tasks.values():
        for v in t.intent.derive:
            counts["values"] += 1
            if LeakMatcher(("exact",)).find(v, t.instruction):
                counts["exact"] += 1
                continue
            hits = LeakMatcher(("sep", "number", "lemma")).find(v, t.instruction)
            if hits:
                counts[hits[0].layer] = counts.get(hits[0].layer, 0) + 1
                new.append({"task": t.id, "value": v, "layer": hits[0].layer, "quote": hits[0].quoted, "instruction": t.instruction})
    doc: dict[str, Any] = {"counts": counts, "new": new}
    if a.cap > 0 and new:
        cap = cap_for(a)
        d = work_dir(a) / f"natural-{time.strftime('%Y%m%d-%H%M%S')}"
        d.mkdir(parents=True)
        wire = SynthWire(d, cap_usd=cap)

        async def go() -> None:
            panel = LeakPanel(members=[PanelMember(f"leak_panel-{c.family}", c) for c in wire.clients("leak_panel")], lam=0.7)
            _dev, _locked, dev_src, _ls = load_sets(a, "rsi-2")
            doc["calibration"] = {k: v for k, v in (await calibrate_leak_panel(panel, leak_anchors(dev_src, [*DEV_TEMPLATES, *_planter_templates(a, "rsi-2")]))).items()
                                  if k != "votes"}
            for x in new:  # the sentence around the hit, so the panel reads the claim in context
                v = await panel.judge(x["instruction"], x["value"], x["quote"])
                x["decision"], x["votes"] = v["decision"], v["votes"]

        asyncio.run(go())
        doc["cost"] = wire.cost()
    write(work_dir(a) / "natural" / "natural.json", doc)
    print(json.dumps({"counts": counts, "decisions": [(x["task"].rsplit("/", 2)[-2:], x["value"], x["layer"], x["quote"], x.get("decision")) for x in new]},
                     indent=1, default=str))


# ---------------------------------------------------------------------------------------------------------- visible data

# The independently found leak_other records (runs/m5/debug_lane_defects.json) and the version each is measured on.
LEAK_RECORDS = {
    "echo-perf-bench-visible-data": ("db-go-echo-61eca05b-e2", None),  # the pool's strip build still carries the lane's flag
    "mvdan-sh-dc951aeb-e1-table-entries": ("db-go-mvdan-sh-dc951aeb-e1", "pre_table_strip"),  # rebuilt clean: re-created from the record
    "mvdan-sh-5e983709-e1-table-entries": (None, None),  # lost: no store holds the version
    "mvdan-sh-5e983709-e2-table-entries": (None, None),
}


def cmd_visible(a: argparse.Namespace) -> None:
    """TAINT ``visible_data`` (static) and ``visible_check`` (executed, ``--execute``: Docker, at most 4 containers) on
    every pool task and on the leak_other real defects (DESIGN §7.7 item 20). Recall counts only the independently found
    records; every other pool task is a false-alarm trial unless a person documented a leak in it."""
    from crucible.factory.synth_patch_wire import PatchEnv, lane_roles, lane_runner, open_pool, patch_env, pre_table_strip, remove_containers
    from crucible.synth.falsify.patch import PatchTAINT

    pool = open_pool(Path(a.pool), parallel=a.parallel)
    out = work_dir(a) / "visible"
    out.mkdir(parents=True, exist_ok=True)
    versions: list[tuple[str, str, str | None]] = [(w, w, None) for w in pool.ids()]
    for rec, (wid, how) in LEAK_RECORDS.items():
        if wid is not None and how is not None:
            versions.append((f"{rec}@{how}", wid, how))

    async def one(name: str, wid: str, how: str | None) -> dict[str, Any]:
        task = pool.task(wid)
        env = patch_env(task)
        if how == "pre_table_strip":
            env = PatchEnv(env.spec.model_copy(update={"start_files": {**env.spec.start_files, **pre_table_strip(task)}}), roles=lane_roles(task))
        runner = lane_runner(task, parallel=a.parallel) if a.execute else None
        try:
            f = PatchTAINT(env, runner, visible=runner is not None)
            st = env.task()
            from crucible.factory.synth_patch_wire import _NoRunner

            ctx = env.context(runner if runner is not None else _NoRunner())
            row: dict[str, Any] = {"version": name}
            for p in await f.propose(st, ctx):
                if p.op not in ("visible_data", "visible_check"):
                    continue
                o = await f.execute(p, st, ctx)
                lab, dfc = f.label(p, o) if o.valid else ("invalid", None)
                row[p.op] = {"label": lab, "defect": dfc, "error": o.error,
                             **{k: v for k, v in o.data.items() if k in ("hidden_cases", "fix_answers", "fix_literals", "fix_definitions", "fail_to_pass", "n_f2p",
                                                                        "milestone_tests")}}
            return row
        finally:
            if runner is not None:
                runner.close()

    async def go() -> list[dict[str, Any]]:
        rows = []
        try:
            for name, wid, how in versions:
                t0 = time.monotonic()
                r = await one(name, wid, how)
                r["seconds"] = round(time.monotonic() - t0, 1)
                print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ("label", "n_f2p", "milestone_tests", "error")})
                                  for k, v in r.items()}, default=str)[:600], flush=True)
                rows.append(r)
        finally:
            if a.execute:
                left = remove_containers()
                if left:
                    print(f"removed leftover containers: {left}")
        return rows

    rows = asyncio.run(go())
    write(out / ("visible_executed.json" if a.execute else "visible_static.json"), {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": rows,
                                                                                   "records": {k: v[0] for k, v in LEAK_RECORDS.items()}})


# ---------------------------------------------------------------------------------------------------------- triage

def cmd_triage(a: argparse.Namespace) -> None:
    """The review sheet for a person (``synth.triage``): grouped, ranked by expected information, nothing decided."""
    from crucible.synth.triage import export, groups, load_queue, triage

    q = Path(a.queue)
    rep = q.parent / "report.json"
    items = triage(load_queue(q), report=json.loads(rep.read_text()) if rep.is_file() else None)
    out = work_dir(a) / "triage" / f"{q.parent.parent.name}-{q.parent.name}"
    paths = export(items, out, title=f"Human review: {q.parent.parent.name} {q.parent.name}")
    write(out / "items.json", [i.model_dump(mode="json") for i in items])
    print(json.dumps({"items": len(items), "groups": groups(items), "top": [(i.rank, i.id, i.task.rsplit('/', 1)[-1], i.op, i.ei) for i in items[:10]],
                      **paths}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("matrix")
    m.add_argument("--run-id", required=True)
    m.add_argument("--synonyms", default="")
    m.set_defaults(fn=cmd_matrix)
    rp = sub.add_parser("repo-panel")
    rp.add_argument("--summary", default=str(HERE / "docs" / "research" / "env-synthesis" / "runs" / "m5" / "certificates_summary.json"))
    rp.add_argument("--pool", default=os.environ.get("CRUCIBLE_DEBUG_POOL") or str(HERE / "data" / "debug-pool"))
    rp.add_argument("--max-hunks", type=int, default=16)
    rp.add_argument("--mutants", type=int, default=2)
    rp.add_argument("--cap", type=float, default=3.0)
    rp.add_argument("--dry", action="store_true")
    rp.add_argument("--tool-prompt", action="store_true", help="the tool-world PANEL prompt (the first run)")
    rp.add_argument("--members", default="", help="panel models by factory.models name, e.g. INKLING,GPT_OSS (default: the calibrated panel)")
    rp.set_defaults(fn=cmd_repo_panel)
    lk = sub.add_parser("leak-l2")
    lk.add_argument("--run-id", required=True)
    lk.add_argument("--cap", type=float, default=2.0)
    lk.add_argument("--max-calls", type=int, default=400)
    lk.add_argument("--locked", action="store_true")
    lk.set_defaults(fn=cmd_leak_l2)
    na = sub.add_parser("natural")
    na.add_argument("--cap", type=float, default=0.0)
    na.set_defaults(fn=cmd_natural)
    vi = sub.add_parser("visible")
    vi.add_argument("--pool", default=os.environ.get("CRUCIBLE_DEBUG_POOL") or str(HERE / "data" / "debug-pool"))
    vi.add_argument("--execute", action="store_true")
    vi.add_argument("--parallel", type=int, default=4)
    vi.set_defaults(fn=cmd_visible)
    tr = sub.add_parser("triage")
    tr.add_argument("--queue", required=True)
    tr.set_defaults(fn=cmd_triage)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
