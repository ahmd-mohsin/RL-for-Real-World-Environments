"""P2 (swarm-pipeline/00 §15): verifier rigor on the dependent training windows.

    python scripts/lh_p2.py --work <lhwork> --store store3 --clones <repos> --runs <lhrun>,<lhrun2> --out <dir> run [--usd 5] [--only a,b]
    python scripts/lh_p2.py ... seed       (the seeded verifier gaps on real windows; after run)
    python scripts/lh_p2.py ... report     (gold sets, verifier cards, spot-check queue, spend by role)

Per dependent training window (held-out windows are never read: mining and grading only, never a model prompt):
1. the release request (author family, ``author_window``), cached;
2. the second verifier's shadow suite (reviewer family, one call + one syntax repair; ``verifier2.write_shadow``), from
   the prompt, the requirements and the repository at A only, the payload checked by ``leaks`` first;
3. two oracle rewrites (author family, imperative; third family, functional; ``alternatives.rewrite_oracle``);
4. both verifiers on every end state: oracle, start, floors, commit-revert mutants, rewrite candidates, and the
   recorded solver and hacker end states of runs 1 and 2 (``artifacts/model.patch``); the strict full suite as the
   executed label of real items;
5. qualification, alternative acceptance, the fairness lint, the ten-competent-people test, the differential with
   owners, and the S5/S8/S9 machine run (``verifier2.gate_handlers``).

Every model call is booked in one USD ledger (``models.booked``) with a hard cap (``--usd``); the report states the
spend by role. Grading containers are named ``crucible-p2-*`` and removed by name at the end.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from crucible.factory.alternatives import Alternative, accept, coverage, rewrite_oracle, salvage
from crucible.factory.fairness import admit_check, lint_checks, ten_competent_people
from crucible.factory.goldset import GoldConfig, GoldItem, executed_label, pair_fracs, score, spot_check_queue, verifier_card
from crucible.factory.labels import item_sha256
from crucible.factory.lanes.executable import NOT_SOURCE
from crucible.factory.lanes.windows import WindowLane, WindowStore, author_window, sandbox_factory, window_checkset
from crucible.factory.lanes.windows_git import RepoGit
from crucible.factory.machine import MachineRun
from crucible.factory.verifier2 import (
    EndStateRec,
    P2Gates,
    ShadowSuite,
    WindowRunner,
    gate_handlers,
    gate_machine,
    qualify,
    run_differential,
    truth_for,
    unsound_defects,
    write_shadow,
)

GRADE_PREFIX = "crucible-p2-run-"  # never "crucible-p2-": the P2 re-mine names its containers crucible-p2-mine-*
ROLES = {"author": "qwen/qwen3.7-flash", "second_verifier": "deepseek/deepseek-v4-flash", "rewriter_a": "qwen/qwen3.7-flash",
         "rewriter_b": "mistralai/mistral-small-3.2-24b-instruct", "rewriter_c": "deepseek/deepseek-v4-flash"}
REWRITERS = (("rewriter_a", "imperative", "qwen"), ("rewriter_b", "functional", "mistral"), ("rewriter_c", "functional", "deepseek"))
AUTHOR_FAMILY = "qwen"  # "another family" for alternatives = not the family that wrote the task (its prompt)
SPEND: dict[str, float] = {}


def log(ev: dict) -> None:
    print(json.dumps(ev, default=str)[:400], flush=True)


def jload(p: Path) -> Any:
    return json.loads(p.read_text()) if p.exists() else None


def jsave(p: Path, x: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(x, indent=1, default=str))
    tmp.replace(p)


# ---- model calls, booked --------------------------------------------------------------------------------------------------
class Spend:
    def __init__(self, cap: float, out: Path):
        from crucible.factory.models import ModelBook

        self.book = ModelBook(roles=ROLES, budget_usd_per_client=cap)  # a client holds max_tokens x price per call in flight; the ledger caps the run
        self.path = out / "spend.jsonl"
        # one cap across every invocation of this run directory: what earlier processes spent is taken off first
        self.prior = sum(float(json.loads(ln).get("usd") or 0.0) for ln in self.path.read_text().splitlines()) if self.path.exists() else 0.0
        self.ledger = self.book.ledger(max(0.0, cap - self.prior))
        self.cap = cap

    async def call(self, role: str, who: str, coro_fn, *, estimate: float = 0.03) -> Any:
        from crucible.factory.models import booked

        async with booked(self.ledger, f"{role}:{who}", estimate_usd=estimate) as scope:
            try:
                return await coro_fn(self.book.client(role))
            finally:
                SPEND[role] = SPEND.get(role, 0.0) + scope.usd
                with self.path.open("a") as fh:
                    fh.write(json.dumps({"t": round(time.time(), 1), "role": role, "model": ROLES[role], "who": who, "usd": scope.usd}) + "\n")


# ---- end states from runs 1 and 2 -----------------------------------------------------------------------------------------
def recorded_states(runs: list[Path]) -> dict[str, list[dict[str, Any]]]:
    """window id -> recorded end states: ``{id, kind (solver | hacker), policy, patch_path, run}``. Isolated-milestone
    (``-iso-``) episodes start from another tree and are skipped."""
    out: dict[str, list[dict[str, Any]]] = {}
    for root in runs:
        for rec in sorted(root.glob("runs/*/samples/*/a*/record.json")):
            r = json.loads(rec.read_text())
            task = str(r.get("task") or "")
            patch = rec.parent / "artifacts" / "model.patch"
            if "-iso-" in task or "~pre" in task or not patch.exists():
                continue
            kind = "hacker" if "#hacker" in task or str(r.get("policy", "")).startswith("hacker") else "solver"
            wid = re.sub(r"-p\d+$", "", task.split("~", 1)[0].split("#", 1)[0])
            run = rec.relative_to(root).parts[1]
            out.setdefault(wid, []).append({"id": f"{root.name}-{run}-{rec.parent.parent.name}-{rec.parent.name}", "kind": kind,
                                            "policy": r.get("policy"), "patch_path": str(patch), "run": f"{root.name}/{run}"})
    return out


# ---- one window -------------------------------------------------------------------------------------------------------------
async def window(a: argparse.Namespace, lane: WindowLane, spend: Spend | None, wid: str, recorded: list[dict[str, Any]]) -> dict[str, Any]:
    out = Path(a.out) / "windows" / wid
    res_path = out / "results.json"
    errored = [x for x in jload(out / "rewrites.json") or [] if any(str(r).startswith("error:") for r in x.get("reasons") or [])]
    if res_path.exists() and not a.redo and not (a.retry_rewrites and errored):
        return json.loads(res_path.read_text())
    store = lane.store
    rec = store.window(wid)
    repo = RepoGit(Path(a.clones) / rec["repo"])
    base = store.dir(wid) / "base"
    t0 = time.monotonic()
    # 1. the prompt
    pj = jload(out / "prompt.json")
    if pj is None:
        if spend is None:
            return {"window": wid, "skipped": "no prompt cached and model calls are off"}
        findings: list[str] = []
        res = None
        for k in range(1, 4):
            try:
                res = await spend.call("author", wid, lambda c, k=k, f=list(findings): author_window(
                    c, rec, repo, base, findings=f, sample_key=f"p2-author-{wid}-{k}" if k > 1 else None))
            except Exception as e:  # noqa: BLE001 - recorded, retried
                findings = [f"error: {e}"[:300]]
                continue
            findings = res["findings"]
            if not findings:
                break
        if res is None:
            return {"window": wid, "error": f"no prompt: {findings}"}
        pj = {"prompt": res["prompt"], "requirements": res["requirements"], "spans": res["spans"], "checkset": res["checkset"], "findings": findings}
        jsave(out / "prompt.json", pj)
    task = lane.materialize({"world": {"task": wid}, "prompt": pj["prompt"], "spec": {"requirements": pj["requirements"]}, "spans": pj["spans"],
                             "checkset": pj["checkset"], "task_id": wid})
    runner = WindowRunner(task)
    reqs = [r for r in task.requirements if r.get("milestone")]
    rids = [r["id"] for r in reqs]
    req_ms = {r["id"]: r["milestone"] for r in reqs}
    # 2. the shadow suite
    sj = jload(out / "shadow.json")
    if sj is None:
        if spend is None:
            return {"window": wid, "skipped": "no shadow suite cached"}
        for k, chunk in ((1, 6), (2, 3), (3, 2)):  # a reply cut off or malformed is asked again in smaller groups (same inputs: nothing new learned)
            try:
                suite = await spend.call("second_verifier", wid, lambda c, n=chunk: write_shadow(
                    c, prompt=task.prompt, requirements=task.requirements, base_dir=task.base_dir, model=ROLES["second_verifier"], family="deepseek",
                    guard={"oracle_files": rec["oracle_files"], "hidden_tests": rec["tests_b"]}, chunk=n), estimate=0.05)
                break
            except Exception as e:
                if k == 3 or "leaks" in str(e):
                    raise
                log({"event": "shadow_retry", "window": wid, "error": str(e)[:200]})
        jsave(out / "shadow.json", suite.to_json())
    else:
        suite = ShadowSuite.from_json(sj)
    # 3. rewrite candidates
    rj = jload(out / "rewrites.json")
    if rj is None:
        rj = []
        if spend is not None:
            for role, style, fam in REWRITERS:
                try:
                    alt = await spend.call(role, wid, lambda c, s=style, f=fam, r=role: rewrite_oracle(c, task, style=s, model=ROLES[r], family=f), estimate=0.05)
                except Exception as e:  # noqa: BLE001 - one rewriter's failure is recorded; the others still run
                    log({"event": "rewrite_error", "window": wid, "role": role, "error": str(e)[:200]})
                    alt = Alternative(f"rewrite-{style}-{fam}", "rewrite", "", ROLES[role], fam, style, reasons=[f"error: {e}"[:300]])
                rj.append(alt.to_json())
            jsave(out / "rewrites.json", rj)
    elif a.retry_rewrites and spend is not None:  # a rewriter that errored (a rate limit, a reply cut off) gets one more call
        for i, x in enumerate(rj):
            if not any(str(r).startswith("error:") for r in x.get("reasons") or []):
                continue
            role = next(r for r, st, f in REWRITERS if f == x["family"] and st == x["style"])
            try:
                alt = await spend.call(role, wid, lambda c, s=x["style"], f=x["family"], r=role: rewrite_oracle(c, task, style=s, model=ROLES[r], family=f),
                                       estimate=0.05)
                rj[i] = alt.to_json()
            except Exception as e:  # noqa: BLE001
                rj[i]["reasons"] = [*rj[i]["reasons"], f"retry error: {e}"[:300]]
        jsave(out / "rewrites.json", rj)
    cands = [Alternative.from_json(x) for x in rj]
    # 4. both verifiers on every end state
    floors = task.floors()
    muts = task.mutants()
    mut_recs = {m["id"]: m for m in rec.get("mutants") or []}
    rec_states = []
    for s in recorded:
        try:
            rec_states.append({**s, "patch": Path(s["patch_path"]).read_text(errors="replace")})
        except OSError:
            continue
    sols: dict[str, Any] = {"oracle": task.oracle_patch(), "start": "", **{f"floor:{f.id}": f.solution for f in floors},
                            **{f"mutant:{m.id}": m.solution for m in muts}, **{f"rewrite:{c.id}": c.patch for c in cands if c.patch},
                            **{f"{s['kind']}:{s['id']}": s["patch"] for s in rec_states}}

    async def both(name: str, sol: Any) -> tuple[str, Any, Any]:
        return name, await runner.hidden(sol), await runner.shadow(suite, sol)

    got = {n: (v1, v2) for n, v1, v2 in await asyncio.gather(*[both(n, s) for n, s in sols.items()])}
    # 5. qualification: phase 1 (oracle-sound) accepts alternatives; phase 2 qualifies on oracle + accepted alternatives
    mutant_runs = {str(m.id): (str(m.milestone), got[f"mutant:{m.id}"][1]) for m in muts}
    q1 = qualify(suite, oracle=got["oracle"][1], alternatives={}, start=got["start"][1], mutants=mutant_runs, req_milestone=req_ms)
    for c in cands:
        if c.patch:
            v1, v2 = got[f"rewrite:{c.id}"]
            await accept(c, v1=v1, v2=q1.verdict(v2))
    # a candidate the real tests fail: salvage the largest chunk-subset of its rewritten definitions that passes (no model call)
    for c in list(cands):
        if c.patch and not c.accepted and (got[f"rewrite:{c.id}"][0] or {}).get("reward") != 1.0:
            async def passes(patch: str) -> bool:
                return ((await runner.hidden(patch)) or {}).get("reward") == 1.0

            sub = await salvage(task, c, passes)
            if sub is not None:
                got[f"rewrite:{sub.id}"] = (await runner.hidden(sub.patch), await runner.shadow(suite, sub.patch))
                await accept(sub, v1=got[f"rewrite:{sub.id}"][0], v2=q1.verdict(got[f"rewrite:{sub.id}"][1]))
                cands.append(sub)
    accepted = {c.id: c.patch for c in cands if c.accepted}
    qual = qualify(suite, oracle=got["oracle"][1], alternatives={n: got[f"rewrite:{n}"][1] for n in accepted}, start=got["start"][1],
                   mutants=mutant_runs, req_milestone=req_ms)
    # 6. fairness: the lint and the ten-competent-people test
    lint = lint_checks(task.checkset["checks"], rec.get("tests_b") or {}, task.prompt)
    pos = {"oracle": (got["oracle"][0] or {}).get("statuses"), **{n: (got[f"rewrite:{n}"][0] or {}).get("statuses") for n in accepted}}
    ff = ten_competent_people(task.check_tests, pos, gating=task.gating)
    spec_flags: dict[str, list[str]] = {}
    for f in lint.findings:
        if f.kind == "unstated_format":
            for r in f.requirement_ids:
                spec_flags.setdefault(r, []).append(f.test)
    # 7. the differential over every end state
    states = [EndStateRec("oracle", "oracle", sols["oracle"], truth_for("oracle", rids, milestone_of=req_ms)),
              EndStateRec("start", "start", "", truth_for("start", rids, milestone_of=req_ms))]
    states += [EndStateRec(f"floor:{f.id}", "floor", f.solution, truth_for("floor", rids, milestone_of=req_ms)) for f in floors if f.id != "test_edit"]
    states += [EndStateRec(f"mutant:{m.id}", "mutant", m.solution, truth_for("mutant", rids, milestone_of=req_ms, mutant=mut_recs.get(str(m.id))))
               for m in muts]
    for c in cands:
        if c.patch:
            kind = "alternative" if c.accepted else "rewrite_candidate"
            states.append(EndStateRec(f"rewrite:{c.id}", kind, c.patch, truth_for(kind, rids, milestone_of=req_ms)))
    states += [EndStateRec(f"{s['kind']}:{s['id']}", s["kind"], s["patch"], truth_for(s["kind"], rids, milestone_of=req_ms)) for s in rec_states]
    diff = await run_differential(runner, suite, qual, states, requirements=task.requirements, spec_flags=spec_flags)
    diff.defects += unsound_defects(wid, qual, suite, req_ms)
    # 8. executed labels for the real items (strict full suite), agreement with the second verifier
    real = {}
    for s in rec_states:
        name = f"{s['kind']}:{s['id']}"
        strict = await runner.strict(s["patch"])
        v2all = qual.verdict(got[name][1])
        vals = [v for v in v2all.values() if v is not None]
        v2_pass = None if not vals else all(vals)
        label = None if strict is None else strict["pass"]
        real[name] = {"strict": strict, "v2_pass": v2_pass, "label": label if (v2_pass is None or v2_pass == label) else None,
                      "disputed": label is not None and v2_pass is not None and v2_pass != label, "v1_reward": (got[name][0] or {}).get("reward"),
                      "policy": s["policy"], "run": s["run"], "kind": s["kind"]}
    # 9. the S5/S8/S9 machine run (the gates as factory/machine.py runs them)
    gates = P2Gates(task=task, runner=runner, alternatives=accepted, suite=suite, states=states, lint=lint)
    shutil.rmtree(out / "machine", ignore_errors=True)  # a recomputation starts a fresh journal
    mrun = MachineRun(gate_machine(), run_dir=out / "machine", handlers=gate_handlers(), services=gates, run_id=f"p2-{wid}"[:60])
    mres = await mrun.run()
    rows = {n: {"v1": None if v1 is None else {k: v1[k] for k in ("reward", "milestones", "phi", "gates")},
                "v2": None if v2 is None else qual.verdict(v2)} for n, (v1, v2) in got.items()}
    result = {"window": wid, "repo": rec["repo"], "milestones": [m["id"] for m in task.milestones], "requirements": rids,
              "prompt_findings": pj.get("findings"), "suite": {"requirements": sorted(suite.requirements), "problems": suite.problems, "tests": len(qual.tests)},
              "qualification": qual.to_json(), "rewrites": [c.to_json() | {"patch": f"<{len(c.patch)} bytes>"} for c in cands],
              "alternatives": coverage(cands, writer_family=AUTHOR_FAMILY), "lint": lint.to_json(), "false_fail": [f.to_json() for f in ff],
              "differential": diff.to_json(), "rows": rows, "real": real, "mutants": {str(m.id): str(m.milestone) for m in muts},
              "mutant_status": {k: v.get("status") for k, v in mut_recs.items()}, "mutant_expected": {k: v.get("expected") for k, v in mut_recs.items()},
              "machine": {"status": mres.status, "s5": (gates.results.get("s5") or {}).get("clean_world", {}).get("ok"),
                          "s8_failed": ((gates.results.get("s8") or {}).get("ground_truth") or {}).get("failed"),
                          "s9": (gates.results.get("s9") or {}).get("differential", {}).get("confirmed_by_owner"),
                          "decision": getattr(mres, "decision_request", None)},
              "verifier_version": task.verifier_version, "seconds": round(time.monotonic() - t0, 1)}
    jsave(res_path, result)
    log({"event": "window", "window": wid, "gating": qual.gating(), "accepted_alts": len(accepted), "defects": diff.by_owner(), "machine": mres.status,
         "seconds": result["seconds"]})
    return result


# ---- the gold set -----------------------------------------------------------------------------------------------------------
def gold_items(results: list[dict[str, Any]]) -> tuple[list[GoldItem], dict[str, Any], list[Any], list[tuple[bool | None, bool | None]]]:
    """Gold items (end-state level), their verifier-1 verdicts, executed labels, and milestone-level (truth, verdict) pairs."""
    items: list[GoldItem] = []
    verdicts: dict[str, Any] = {}
    labels: list[Any] = []
    pairs: list[tuple[bool | None, bool | None]] = []
    for r in results:
        fam = "window"
        wid = r["window"]
        mut_ms = r["mutants"]
        for name, row in r["rows"].items():
            v1 = row["v1"]
            kind = name.split(":", 1)[0]
            if v1 is None:
                continue
            if kind == "oracle":
                src, truth_ms = "oracle", dict.fromkeys(r["milestones"], True)
            elif kind == "floor":
                if name == "floor:test_edit":
                    continue  # restored before grading: the same end state as the start
                src, truth_ms = "floor", dict.fromkeys(r["milestones"], False)
            elif kind == "start":
                continue  # the no_op floor is the same end state
            elif kind == "mutant":
                mid = name.split(":", 1)[1]
                src = "mutant"
                exp = (r.get("mutant_expected") or {}).get(mid) or {mut_ms.get(mid): False}
                if (r.get("mutant_status") or {}).get(mid) == "broad":
                    exp = {mut_ms.get(mid): False}  # a broad revert is known only on its own milestone
                truth_ms = {m: exp.get(m) for m in r["milestones"]}
            elif kind == "rewrite":
                rid = name.split(":", 1)[1]
                rw = next((x for x in r["rewrites"] if x["id"] == rid), None)
                if not rw or not rw["accepted"]:
                    continue  # a rejected candidate has no constructed truth (it is in the differential instead)
                src, truth_ms = "alternative", dict.fromkeys(r["milestones"], True)
            elif kind in ("solver", "hacker"):
                lab = r["real"].get(name) or {}
                src = "real_pass" if lab.get("label") else "real_fail"
                truth_ms = {}
            else:
                continue
            content = {"window": wid, "end_state": name}
            it = GoldItem(f"{wid}:{name}"[:200].replace(" ", "_"), src, content, item_sha256(content), proven_by=f"{kind}:{name}", family=fam,  # type: ignore[arg-type]
                          meta={"window": wid, "repo": r["repo"], "kind": kind})
            items.append(it)
            verdicts[it.id] = {"reward": v1["reward"]}
            if kind in ("solver", "hacker"):
                lab = r["real"].get(name) or {}
                if lab.get("label") is not None:
                    labels.append(executed_label(it, bool(lab["label"]), reason=f"strict full suite {'passes' if lab['label'] else 'fails'}; v2 agrees"))
            for m, t in truth_ms.items():
                pairs.append((t, (v1.get("milestones") or {}).get(m)))
    return items, verdicts, labels, pairs


def funnels(a: argparse.Namespace) -> dict[str, Any]:
    """The re-mine's funnel per split (mining and grading only: the held-out store is counted, never prompted), and the
    window-for-window comparison with the store it re-validated (``store2``) when that store is there."""
    from crucible.factory.lanes.windows import funnel

    out: dict[str, Any] = {}
    for split in ("train", "heldout"):
        root = Path(a.work) / a.store / split
        if not (root / "split.json").exists():
            continue
        s3 = WindowStore(root, split=split)
        row: dict[str, Any] = {"total": funnel(s3)["total"]}
        src = Path(a.work) / "store2" / split
        if (src / "split.json").exists():
            s2 = WindowStore(src, split=split)

            def dep(s: WindowStore, w: str) -> bool:
                return s.window(w).get("stage") in ("dependent", "mutants_valid")

            both = set(s2.window_ids()) & set(s3.window_ids())
            row["store2_total"] = funnel(s2)["total"]
            row["same_windows"] = len(both)
            row["dependent_kept"] = sum(1 for w in both if dep(s2, w) and dep(s3, w))
            row["dependent_gained"] = sorted(w for w in both if not dep(s2, w) and dep(s3, w))
            row["dependent_lost"] = sorted(w for w in both if dep(s2, w) and not dep(s3, w))
        out[split] = row
    return out


def report(a: argparse.Namespace) -> dict[str, Any]:
    out = Path(a.out)
    results = [json.loads(p.read_text()) for p in sorted((out / "windows").glob("*/results.json"))]
    items, verdicts, labels, pairs = gold_items(results)
    cfg = GoldConfig(min_items=40, required_sources=("real_fail", "mutant", "alternative"), fp_upper_max=0.10, fn_upper_max=0.10)
    families: dict[str, Any] = {}

    def card_for(name: str, its: list[GoldItem], prs: list) -> dict[str, Any]:
        ids = {it.id for it in its}
        rep = score(its, {k: v for k, v in verdicts.items() if k in ids}, [lab for lab in labels if lab.item in ids], config=cfg)
        rs = [r for r in results if name == "window" or r["repo"] == name]
        mk = sum(1 for r in rs for n, row in r["rows"].items() if n.startswith("mutant:") and row["v1"] and not row["v1"]["milestones"].get(r["mutants"][n.split(":", 1)[1]]))
        mn = sum(1 for r in rs for n, row in r["rows"].items() if n.startswith("mutant:") and row["v1"])
        alts = sum(r["alternatives"]["accepted"] for r in rs)
        dd = sum(len(r["differential"]["defects"]) for r in rs)
        dc = sum(r["differential"]["compared"] for r in rs)
        fl = {}
        for r in rs:
            for n, row in r["rows"].items():
                if n.startswith("floor:") and row["v1"]:
                    fl[n] = max(fl.get(n, 0.0), float(row["v1"]["phi"] or 0.0))
        card = verifier_card(rep, verifier_version="window-grader/p0", family=name, mutant_kill=(mk, mn), alternatives_pass=(alts, alts),
                             differential=(dd, max(dc, dd)), hacks_confirmed=0, floors=fl, units={"milestone": pair_fracs(prs)})
        card["reasons"] = rep.reasons()
        card["windows"] = len(rs)
        return card

    families["window"] = card_for("window", items, pairs)
    for repo in sorted({r["repo"] for r in results}):
        its = [it for it in items if it.meta["repo"] == repo]
        families[repo] = card_for(repo, its, [])
    truth = {}
    from crucible.factory.goldset import truths

    truth, _ = truths(items, labels)
    queue = spot_check_queue(items, truth, share=0.10, seed="p2")
    with (out / "spot_check_queue.jsonl").open("w") as fh:
        for row in queue:
            fh.write(json.dumps(row) + "\n")
    with (out / "defects.jsonl").open("w") as fh:
        for r in results:
            for d in r["differential"]["defects"]:
                fh.write(json.dumps(d) + "\n")
    spend: dict[str, float] = {}
    sp = out / "spend.jsonl"
    if sp.exists():
        for ln in sp.read_text().splitlines():
            x = json.loads(ln)
            spend[x["role"]] = round(spend.get(x["role"], 0.0) + float(x["usd"] or 0.0), 6)
    defects: dict[str, int] = {}
    confirmed: dict[str, int] = {}
    for r in results:
        for d in r["differential"]["defects"]:
            key = f"{d['owner']}:{d['defect']}"
            defects[key] = defects.get(key, 0) + 1
            if d["confirmed"]:
                confirmed[key] = confirmed.get(key, 0) + 1
    summary = {
        "windows": len(results), "gold_items": len(items), "executed_labels": len(labels),
        "real_disputed": sum(1 for r in results for v in r["real"].values() if v["disputed"]),
        "alternatives": {"accepted": sum(r["alternatives"]["accepted"] for r in results), "windows_with_2": sum(1 for r in results if r["alternatives"]["ok"]),
                         "candidates": sum(1 for r in results for x in r["rewrites"] if x["patch"] != "<0 bytes>")},
        "shadow": {"tests": sum(r["suite"]["tests"] for r in results), "requirements": sum(len(r["requirements"]) for r in results),
                   "gating": sum(len(r["qualification"]["gating"]) for r in results),
                   "fails_oracle": sum(r["qualification"]["counts"]["fails_oracle"] for r in results),
                   "fails_alternative": sum(r["qualification"]["counts"]["fails_alternative"] for r in results)},
        "lint": {k: sum(r["lint"]["by_kind"].get(k, 0) for r in results) for k in ("source_text", "source_structure", "private_name", "unstated_format",
                                                                                     "no_requirement")},
        "lint_refused_checks": sum(len(r["lint"]["refused_checks"]) for r in results),
        "false_fail_checks": sum(len(r["false_fail"]) for r in results),
        "defects": defects, "confirmed": confirmed,
        "machine": {s: sum(1 for r in results if r["machine"]["status"] == s) for s in {r["machine"]["status"] for r in results}},
        "spend_usd": spend, "spend_total": round(sum(spend.values()), 4), "spot_check_queue": len(queue),
        "spot_check_decide": sum(1 for q in queue if q.get("task") == "decide"),
        # independent false-fail evidence: rewrite candidates the real tests fail and the second verifier passes
        "false_fail_candidates": [d["id"] for r in results for d in r["differential"]["defects"] if d["kind"] == "rewrite_candidate" and not d["v1"] and d["v2"]],
        "verifier_gap_candidates": [d["id"] for r in results for d in r["differential"]["defects"] if d["owner"] == "verifier1" and not d["confirmed"]],
        "gold_errors": families["window"]["errors"], "funnel": funnels(a)}
    rep = {"summary": summary, "families": families}
    jsave(out / "p2_report.json", rep)
    print(json.dumps(summary, indent=1))
    return rep


# ---- seeded gaps on real windows ----------------------------------------------------------------------------------------------
def _module_of(path: str) -> str:
    p = path.removeprefix("src/")
    return p.removesuffix(".py").removesuffix("/__init__").replace("/", ".")


def _locals(src: str) -> set[str]:
    import ast

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    fn = tree.body[0] if tree.body else None
    return {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)} if fn is not None else set()


async def seed(a: argparse.Namespace, lane: WindowLane) -> dict[str, Any]:
    """On real windows with an accepted rewrite: (i) a hidden test that checks a local variable name through the source
    text, (ii) a milestone check weakened to a regression test (it no longer tests its requirement), (iii) a hidden test
    that pins a local variable name of the oracle through the code object (a valid alternative renamed it)."""
    out = Path(a.out)
    rows: list[dict[str, Any]] = []
    for p in sorted((out / "windows").glob("*/results.json")):
        r = json.loads(p.read_text())
        if len([x for x in rows if "skipped" not in x]) >= a.seed_windows or not [x for x in r["rewrites"] if x["accepted"]]:
            continue
        row = await seed_window(lane, out, r)
        rows.append(row)
        log({"event": "seed", **{k: row.get(k) for k in ("window", "function", "local", "skipped")},
             "i": (row.get("i_variable_name") or {}).get("lint_caught"), "ii": (row.get("ii_weak_check") or {}).get("caught"),
             "iii": (row.get("iii_false_fail") or {}).get("caught")})
    jsave(out / "seeded.json", rows)
    return {"rows": rows}


def subset_patch(task: Any, entry: Mapping[str, Any]) -> str:
    """A salvaged subset's patch, rebuilt from the definitions it kept (``evidence.rewritten``)."""
    from crucible.factory.alternatives import pick_defs, rebuild

    rewritten = dict((entry.get("evidence") or {}).get("rewritten") or {})
    files = dict(task.window["oracle_files"])
    spans = [d for d in pick_defs(files, task.base_dir) if f"{d.path}::{d.qualname}" in rewritten]
    return task.patch_of(rebuild(files, spans, rewritten)[0])


def _pick_local(task: Any, alts: Mapping[str, str]) -> tuple[str, str, str, str] | None:
    """(alternative, path, function, local name) for a rewritten top-level function whose B source binds a local name the
    alternative no longer has."""
    from crucible.factory.alternatives import changed_defs

    rec = task.window
    for aid, patch in alts.items():
        files, _, _ = task._files(patch)
        for path, b_text in rec["oracle_files"].items():
            if not (b_text and path.endswith(".py") and path in files and files[path] != b_text) or NOT_SOURCE.search(path):
                continue  # library code only (a docs/conf.py helper is not behaviour the tests import)
            a_file = task.base_dir / path
            alt_defs = {x.qualname: x for x in changed_defs(path, b_text, files[path] or "")}
            for d in changed_defs(path, a_file.read_text(errors="replace") if a_file.is_file() else None, b_text):
                if "." in d.qualname or d.source.lstrip().startswith("@") or d.qualname not in alt_defs:
                    continue
                gone = sorted(_locals(d.source) - _locals(alt_defs[d.qualname].source))
                if gone:
                    return aid, path, d.qualname, gone[0]
    return None


async def seed_window(lane: WindowLane, out: Path, r: Mapping[str, Any]) -> dict[str, Any]:
    import textwrap

    from crucible.factory.lanes.windows import WindowTask
    from crucible.factory.verifier2 import Qualification

    wid = r["window"]
    pj = jload(out / "windows" / wid / "prompt.json")
    rw = {x["id"]: Alternative.from_json(x) for x in jload(out / "windows" / wid / "rewrites.json")}
    task = lane.materialize({"world": {"task": wid}, "prompt": pj["prompt"], "spec": {"requirements": pj["requirements"]}, "spans": pj["spans"],
                             "checkset": pj["checkset"], "task_id": wid})
    rec = task.window
    alts = {x["id"]: rw[x["id"]].patch if x["id"] in rw else subset_patch(task, x) for x in r["rewrites"] if x["accepted"]}
    pick = _pick_local(task, alts)
    if not pick:
        return {"window": wid, "skipped": "no rewritten top-level function lost a local name"}
    aid, path, fn, var = pick
    mod = _module_of(path)
    gating = r["qualification"]["gating"]
    req_ms = {x["id"]: x["milestone"] for x in pj["requirements"] if x.get("milestone")}
    target_m = next((req_ms[g] for g in gating), task.milestones[-1]["id"])
    tfile = min((p for p in rec["tests_b"] if p.rsplit("/", 1)[-1].startswith("test_")), default=min(rec["tests_b"]))

    def seeded(name: str, body: str, *, replace_with: list[str] | None = None) -> WindowTask:
        tests_b = dict(rec["tests_b"])
        if body:
            tests_b[tfile] = tests_b[tfile] + "\n\n" + textwrap.dedent(body)
        tid = f"{tfile}::{name}"
        ms = [{**m, "tests": replace_with if replace_with is not None else [*m["tests"], tid]} if m["id"] == target_m else m for m in rec["milestones"]]
        rec2 = {**rec, "tests_b": tests_b, "milestones": ms}
        return WindowTask(lane, task.mined, rec2, task_id=f"{wid}~seed-{name}", prompt=task.prompt, requirements=task.requirements, spans=task.spans,
                          checkset=window_checkset(rec2, task.requirements), import_tolerant=task.import_tolerant)

    async def admit(t: WindowTask, cid: str) -> dict[str, Any]:
        rn = WindowRunner(t)

        async def run(sol: Any, checks: Mapping[str, Any]) -> Any:
            v = await rn.hidden(sol, checks=checks)
            return None if v is None else v["statuses"]

        return await admit_check(cid, t.check_tests[cid], run, oracle=t.oracle_patch(), alternatives=alts)

    cid = f"c_{target_m}"
    row: dict[str, Any] = {"window": wid, "function": f"{mod}.{fn}", "local": var, "alternative": aid, "milestone": target_m}
    # (i) a variable-name check through the source text
    t1 = seeded("test_crucible_seed_source", f"""
        def test_crucible_seed_source():
            import importlib, inspect
            assert {var!r} in inspect.getsource(importlib.import_module({mod!r}).{fn})
        """)
    lint1 = lint_checks(t1.checkset["checks"], t1.window["tests_b"], t1.prompt)
    adm1 = await admit(t1, cid)
    row["i_variable_name"] = {"lint": [f.kind for f in lint1.findings if f.test.endswith("test_crucible_seed_source")],
                              "lint_caught": cid in lint1.refused_checks(), "admitted": adm1["admitted"], "rows": adm1["rows"],
                              "execution_caught": not adm1["admitted"] and adm1["rows"].get("oracle") is True}
    # (ii) a milestone check weakened to one regression test: it no longer tests its requirement's behaviour
    rid = next((g for g in gating if req_ms[g] == target_m), None)
    reg = (task.check_tests.get("c_regress") or [None])[0]
    if rid and reg:
        t2 = seeded("weak", "", replace_with=[reg])
        suite = ShadowSuite.from_json(jload(out / "windows" / wid / "shadow.json"))
        rids = list(req_ms)
        mut_recs = {m["id"]: m for m in rec.get("mutants") or []}
        states = [EndStateRec(f"mutant:{m.id}", "mutant", m.solution, truth_for("mutant", rids, milestone_of=req_ms, mutant=mut_recs.get(str(m.id))))
                  for m in t2.mutants() if m.milestone == target_m]
        states.append(EndStateRec("start", "start", "", truth_for("start", rids, milestone_of=req_ms)))
        q = r["qualification"]
        diff = await run_differential(WindowRunner(t2), suite, Qualification(q["tests"], q["requirements"]), states, requirements=t2.requirements)
        hits = [d for d in diff.defects if d.requirement == rid and d.owner == "verifier1" and d.confirmed]
        row["ii_weak_check"] = {"requirement": rid, "replaced_with": reg, "defects": [d.to_json() for d in hits], "caught": bool(hits),
                                "ends": [s.id for s in states]}
    else:
        row["ii_weak_check"] = {"skipped": "no gating shadow requirement on this milestone or no regression test"}
    # (iii) a check pinned to the oracle's local name through the code object: a valid alternative fails it
    t3 = seeded("test_crucible_seed_varname", f"""
        def test_crucible_seed_varname():
            import importlib
            assert {var!r} in importlib.import_module({mod!r}).{fn}.__code__.co_varnames
        """)
    rn3 = WindowRunner(t3)
    pos = {"oracle": (await rn3.hidden(t3.oracle_patch()) or {}).get("statuses")}
    pos |= {k: (await rn3.hidden(v) or {}).get("statuses") for k, v in alts.items()}
    ff = ten_competent_people({cid: t3.check_tests[cid]}, pos)
    adm3 = await admit(t3, cid)
    lint3 = lint_checks(t3.checkset["checks"], t3.window["tests_b"], t3.prompt)
    row["iii_false_fail"] = {"ten_people": [f.to_json() for f in ff], "caught": bool(ff) and not adm3["admitted"] and adm3["rows"].get("oracle") is True,
                             "rows": adm3["rows"],
                             "lint_also": [f.kind for f in lint3.findings if f.test.endswith("test_crucible_seed_varname")]}
    return row


def cleanup() -> int:
    r = subprocess.run(["docker", "ps", "-a", "--filter", f"name={GRADE_PREFIX}", "--format", "{{.Names}}"], capture_output=True, text=True, check=False)
    names = [x for x in r.stdout.split() if x.startswith(GRADE_PREFIX)]
    if names:
        subprocess.run(["docker", "rm", "-f", *names], capture_output=True, check=False)
    return len(names)


async def main_async(a: argparse.Namespace) -> None:
    store = WindowStore(Path(a.work) / a.store / "train", split="train")  # the held-out store is never opened here
    repos = {p.name: str(p) for p in Path(a.clones).iterdir() if p.is_dir()}
    lane = WindowLane(store, repo_dirs=repos, sandboxes=sandbox_factory(parallel=a.parallel, name_prefix=GRADE_PREFIX))
    lane.world_prefix = "crucible-lh-p2-world-"
    try:
        if a.cmd == "seed":
            await seed(a, lane)
            return
        ids = [w for w in store.window_ids() if store.window(w).get("stage") in ("dependent", "mutants_valid")]
        if a.only:
            ids = [w for w in ids if w in a.only.split(",")]
        if a.limit:
            ids = ids[: a.limit]
        rec_states = recorded_states([Path(x) for x in a.runs.split(",") if x])
        spend = None if a.no_model else Spend(a.usd, Path(a.out))
        Path(a.out).mkdir(parents=True, exist_ok=True)
        log({"event": "start", "windows": len(ids), "recorded": {w: len(rec_states.get(w, [])) for w in ids if rec_states.get(w)}})
        sem = asyncio.Semaphore(a.windows)

        async def one(w: str) -> None:
            async with sem:
                try:
                    await window(a, lane, spend, w, rec_states.get(w, []))
                except Exception as e:  # noqa: BLE001 - one window's failure is recorded
                    import traceback

                    jsave(Path(a.out) / "windows" / w / "error.json", {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]})
                    log({"event": "window_error", "window": w, "error": f"{type(e).__name__}: {e}"[:300]})

        await asyncio.gather(*[one(w) for w in ids])
        if spend is not None:
            log({"event": "spend", "by_role": SPEND, "total": round(sum(SPEND.values()), 4), "cap": a.usd})
    finally:
        log({"event": "cleanup", "removed": cleanup()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--store", default="store3")
    ap.add_argument("--clones", required=True)
    ap.add_argument("--runs", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("cmd", choices=("run", "seed", "report"))
    ap.add_argument("--usd", type=float, default=5.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--redo", action="store_true", help="recompute results (model outputs stay cached)")
    ap.add_argument("--seed-windows", type=int, default=3)
    ap.add_argument("--retry-rewrites", action="store_true", help="call a rewriter again where it errored, and recompute that window")
    a = ap.parse_args()
    if a.cmd == "report":
        report(a)
    else:
        asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
