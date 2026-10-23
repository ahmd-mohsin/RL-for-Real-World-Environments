#!/usr/bin/env python
"""The swarm's difficulty loop on the debug lane (swarm-pipeline/12 §3, 13 §5): measured training tasks in, gated and
re-measured variants out. No Claude in any role: the controller (``factory.difficulty.control`` through
``difficulty_bridge.ControlController``) picks a knob, the swarm's ``prompt_author`` writes the variant's report
(``factory.revise_debug.DebugReviser``), the lane's executed gates decide (``DebugGates``), and the target student and
a fairness solver re-measure it through ``lh_debug_target.py`` (``--preset swarm``).

    set -a; . ../.env; set +a
    python scripts/lh_difficulty_pilot.py propose --tasks a,b          # the controller only (no model call): what it would do
    python scripts/lh_difficulty_pilot.py revise  --tasks a,b --cap 8  # variants: report by the swarm, gates, candidates.json
    python scripts/lh_difficulty_pilot.py measure --ids v1,v2 --cap 90 # student k=8 + frontier k=2 + audit + report (lh_debug_target)
    python scripts/lh_difficulty_pilot.py next    --ids v1,v2          # the controller on the variants' new measurement

Everything goes under ``data/targeting/swarm-pilot`` (``actions.jsonl``: every controller decision; ``revisions.jsonl``:
every revision and its gate verdict; the targeting phase's own ``spend.jsonl``). Held-out tasks are refused (V9).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lh_debug

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "data" / "targeting" / "swarm-pilot"
T300 = HERE / "docs" / "research" / "swarm-pipeline" / "runs" / "s15_debug_targeting_t300.json"
COND = HERE / "docs" / "research" / "swarm-pipeline" / "runs" / "difficulty_conditional.json"
REPATH = HERE / "docs" / "research" / "swarm-pipeline" / "runs" / "repath_install.json"
FRONTIER = "deepseek-ai/DeepSeek-V4.1-Flash:peft:262144"  # no artifact of a mined window's variant comes from its family (provenance)


def jl(p: Path, row: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def repathed_frontier() -> dict[str, dict[str, bool]]:
    """task -> {arm: True} for frontier attempts whose re-pathed verdict (``factory.repath``) is a full pass."""
    out: dict[str, dict[str, bool]] = {}
    if REPATH.exists():
        for f in json.loads(REPATH.read_text()).get("flips") or []:
            if "/arms/student/" not in f["attempt"] and (f.get("new") or {}).get("full_pass"):
                out.setdefault(f["task"], {})[f["arm"]] = True
    return out


def measured(ids: list[str], report: Path | None = None) -> dict[str, dict[str, Any]]:
    """The newest targeting row per task: the pilot's own report (variants) over the 300-call run (parents)."""
    def task_rows(d: dict[str, Any]) -> list[dict[str, Any]]:  # runs/s15_*.json list them, a phase's report.json keys them by id
        t = d.get("tasks") or []
        return [{"task": k, **v} for k, v in t.items()] if isinstance(t, dict) else list(t)

    rows = {t["task"]: t for t in task_rows(json.loads(T300.read_text()))} if T300.exists() else {}
    for rp in ([report] if report else [OUT / "report.json", *sorted(OUT.glob("r*/report.json"))]):  # later rounds override earlier
        if rp.exists():
            rows.update({t["task"]: t for t in task_rows(json.loads(rp.read_text()))})
    return {i: rows[i] for i in ids if i in rows}


def conditional_of(ids: list[str]) -> dict[str, Any]:

    out = json.loads(COND.read_text()) if COND.exists() else {}
    for pc in (OUT / "conditional.json", *sorted(OUT.glob("r*/conditional.json"))):
        if pc.exists():
            out.update(json.loads(pc.read_text()))
    return {i: out.get(i) for i in ids}


class Lane:
    """The debug lane's pieces the reviser and the gates take (``lh_debug``'s own functions, one training store)."""

    def __init__(self) -> None:
        from crucible.factory.lanes.windows import WindowStore

        self.root = lh_debug.ROOT
        self.sd = self.root / "gen" / "store" / "train"
        self.store = WindowStore(self.sd, split="train")
        self._gens: dict[str, Any] = {}

    def gen(self, repo: str) -> Any:
        if repo not in self._gens:
            from crucible.factory.generators.debug import debug_sandboxes

            g = lh_debug.generator([repo], "train")
            g.sandboxes = debug_sandboxes(parallel=2)
            self._gens[repo] = g
        return self._gens[repo]

    def repo_git(self, rec: dict[str, Any]) -> Any:
        from crucible.factory.lanes.windows_git import RepoGit

        clone = self.root / "gen" / "clones" / rec["repo"]
        return RepoGit(clone if clone.exists() else lh_debug.clones()[rec["repo"]][1])

    def task_of(self, wid: str) -> Any:
        rec = self.store.window(wid)
        af = self.root / "authored" / f"{wid}.json"
        return self.gen(rec["repo"]).task(wid, authored=json.loads(af.read_text()) if af.exists() else None)

    def mined_of(self, rec: dict[str, Any], base: Path) -> Any:
        from crucible.factory.generators.debug import scope
        from crucible.factory.lanes.windows import mined_of

        with scope(rec["lang"]):
            return mined_of(rec, base)

    async def write_report(self, vid: str, rec: dict[str, Any], revision: dict[str, Any]) -> dict[str, Any]:
        from crucible.factory.generators.debug import author_debug, faulty_names, hidden_names, lint_debug
        from crucible.factory.models import ModelBook
        from crucible.factory.swarm_journal import SwarmJournal
        from crucible.factory.swarm_prompts import PromptJob, write_prompt

        book = ModelBook(preset="swarm")
        repo = self.repo_git(rec)
        brief = lh_debug.swarm_brief(vid, "train", rec, repo)
        rev = dict(revision)
        if rev.get("format"):
            brief["format"] = rev.pop("format")
        brief["revision"] = rev
        faulty, hidden = faulty_names(rec, repo), hidden_names(rec)
        builder = book.spec("prompt_author").ref
        base = self.sd / "tasks" / vid / "base"

        def lint(draft: dict) -> list[str]:
            return author_debug(rec, {**draft, "builder": builder}, repo, base_dir=base)["findings"]

        job = PromptJob(vid, [m["id"] for m in rec["milestones"]], brief, base, lint, lambda t: lint_debug(t, faulty=faulty, hidden=hidden),
                        self.root / "prompts" / f"{vid}.json")
        sroot = self.root / "swarm" / "train"
        return await write_prompt(job, book=book, root=sroot, journal=SwarmJournal(sroot / "journal"), budget_usd=1.5, max_turns=60)

    def author(self, vid: str) -> dict[str, Any]:
        from crucible.factory.generators.debug import author_debug

        rec = self.store.window(vid)
        res = author_debug(rec, json.loads((self.root / "prompts" / f"{vid}.json").read_text()), self.repo_git(rec), base_dir=self.sd / "tasks" / vid / "base")
        (self.root / "authored").mkdir(exist_ok=True)
        (self.root / "authored" / f"{vid}.json").write_text(json.dumps(res, indent=1))
        return res

    async def world_gates(self, vid: str) -> dict[str, Any]:
        from crucible.factory.generators.base import GateConfig

        rec = self.store.window(vid)
        res = await self.gen(rec["repo"]).gates(vid, cfg=GateConfig(s3_reruns=60, flake_reruns=30))
        from crucible.factory.revise_debug import hidden_hash, tree_hash

        res.update(tree_sha256=tree_hash(self.sd / "tasks" / vid / "base"), hidden_sha256=hidden_hash(self.store.window(vid)))
        (self.root / "gates").mkdir(exist_ok=True)
        (self.root / "gates" / f"{vid}.json").write_text(json.dumps(res, indent=1, default=str))
        return res

    async def start_status(self, vid: str, pre: list[str]) -> dict[str, bool | None]:
        """The ORIGINAL milestones at the variant's start (the empty patch), each graded as its own check over its own
        tests (``WindowTask.grade(checks=...)``): pre-applied ones must pass, the rest must fail. Masked: every value None."""
        from crucible.factory.generators.debug import scope

        rec = self.store.window(vid)
        parent = self.store.window(rec["revision"]["parent"])
        t = self.gen(rec["repo"]).task(vid)
        with scope(rec["lang"]):
            g, _ = await t.grade("", checks={m["id"]: list(m["tests"]) for m in parent["milestones"]})
        if (g.infra or {}).get("status", "ok") != "ok":
            return {m["id"]: None for m in parent["milestones"]}
        return {m["id"]: g.gates.get(m["id"]) == "pass" for m in parent["milestones"]}

    async def repro(self, vid: str) -> dict[str, Any]:
        ns = argparse.Namespace(ids=vid)
        await lh_debug.repro(ns)
        rf = self.root / "repro" / f"{vid}.json"
        return json.loads(rf.read_text()) if rf.exists() else {"ok": False, "status": "unmeasured"}


def controller_stats(lane: Lane, ids: list[str], *, report: Path | None = None) -> dict[str, dict[str, Any]]:
    from crucible.factory.difficulty_bridge import stats_from_targeting

    rows, cond, fo = measured(ids, report), conditional_of(ids), repathed_frontier()
    out = {}
    for i in ids:
        if i not in rows:
            continue
        rec = lane.store.window(i)
        out[i] = stats_from_targeting(rows[i], cond.get(i), frontier_override=fo.get(i), milestones=rec["milestones"],
                                      applied=(rec.get("revision") or {}).get("applied") or ())
    return out


def cmd_propose(a: argparse.Namespace, *, lane: Lane | None = None) -> dict[str, Any]:
    from crucible.factory.difficulty import Journal
    from crucible.factory.difficulty_bridge import ControlController
    from crucible.factory.revise_debug import SUPPORTED

    lane = lane or Lane()
    ids = [x for x in a.tasks.split(",") if x]
    cc = ControlController(supported=SUPPORTED, journal=Journal(OUT / "actions.jsonl"))
    acts = {}
    for i, st in controller_stats(lane, ids).items():
        k = cc.propose(i, st)
        acts[i] = k
        print(json.dumps({"task": i, "action": cc.last["control"]["action"], "knobs": cc.last["control"]["knobs"], "chosen": k.knob if k else None,
                          "milestone": k.milestone if k else None, "checkpoint": cc.last["checkpoint"], "why": (cc.last["why_none"] or (k.reason if k else ""))[:200]}),
              flush=True)
    return acts


async def cmd_revise(a: argparse.Namespace) -> None:
    from crucible.factory.difficulty_hooks import revise_and_gate
    from crucible.factory.revise_debug import DebugGates, DebugReviser
    from crucible.factory.swarm_journal import SwarmJournal
    from crucible.train.bridges.tinker.sampler import ensure_cap

    ensure_cap(a.cap)
    lane = Lane()
    acts = cmd_propose(a, lane=lane)
    reviser = DebugReviser(lane.store, lane.root / "prompts", lane.task_of, lane.mined_of, lane.write_report, scratch=OUT / "scratch")
    gates = DebugGates(lane.store, lane.root / "gates", lane.root / "prompts", lane.author, lane.world_gates, lane.start_status, lane.repro)
    journal = SwarmJournal(lane.root / "swarm" / "train" / "journal")
    ok_ids = []
    for tid, act in acts.items():
        if act is None:
            continue
        if reviser.spent >= a.cap:
            print(f"cap ${a.cap} reached (${reviser.spent:.3f}); stopping before {tid}", flush=True)
            break
        try:
            res = await revise_and_gate(tid, act, reviser=reviser, gates=gates, journal=journal, run=f"difficulty-{tid}")
        except Exception as e:  # noqa: BLE001 -- one task's failure is recorded, never the run's
            res = {"ok": False, "error": f"{type(e).__name__}: {e}"[:600], "action": act.to_json()}
        row = {"task": tid, **{k: v for k, v in res.items() if k != "revision"}, "variant": res["revision"].task_id if res.get("revision") else
               (reviser.log[-1]["variant"] if reviser.log and reviser.log[-1]["parent"] == tid else None), "report": reviser.log[-1] if reviser.log else None}
        jl(OUT / "revisions.jsonl", row)
        print(json.dumps(row, default=str)[:1500], flush=True)
        if res.get("ok"):
            ok_ids.append(res["revision"].task_id)
            lh_debug.cmd_intent(argparse.Namespace())  # candidates.json at once: a gated variant can be measured while the next one gates
    print(json.dumps({"gated_variants": ok_ids, "prompt_usd": round(reviser.spent, 4)}), flush=True)


def cmd_measure(a: argparse.Namespace) -> None:
    py = sys.executable
    tgt = str(Path(__file__).resolve().parent / "lh_debug_target.py")
    cmd = [py, tgt, *a.stages.split(","), "--tasks", a.ids, "--out", str(OUT), "--preset", "swarm", "--frontier", a.frontier, "--k", str(a.k),
           "--frontier-k", str(a.frontier_k), "--cap", str(a.cap), "--seed-offset", str(a.seed_offset), "--no-observe"]
    print(" ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd, env=os.environ.copy()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose")
    p.add_argument("--tasks", required=True)
    r = sub.add_parser("revise")
    r.add_argument("--tasks", required=True)
    r.add_argument("--cap", type=float, default=8.0, help="USD for the swarm's report writing over this run")
    m = sub.add_parser("measure")
    m.add_argument("--ids", required=True)
    m.add_argument("--stages", default="student,frontier,audit,report")
    m.add_argument("--frontier", default=FRONTIER)
    m.add_argument("--k", type=int, default=8)
    m.add_argument("--frontier-k", type=int, default=2)
    m.add_argument("--cap", type=float, default=90.0, help="the targeting phase's one meter (spend.jsonl under the pilot dir)")
    m.add_argument("--seed-offset", type=int, default=700)
    n = sub.add_parser("next")
    n.add_argument("--tasks", required=True)
    a = ap.parse_args()
    if a.cmd in ("propose", "next"):
        cmd_propose(a)
    elif a.cmd == "revise":
        asyncio.run(cmd_revise(a))
    else:
        cmd_measure(a)


if __name__ == "__main__":
    main()
