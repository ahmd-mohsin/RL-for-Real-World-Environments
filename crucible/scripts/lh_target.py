"""P4 (swarm-pipeline/00 §7, §4 S3/S15, §15): the World fixes P3 reproduced, and failure targeting on store3 windows.

    cd <repo root>   # the OpenRouter key resolves from the nearest .env
    python crucible/scripts/lh_target.py --out <dir> --store <store3/train> --clones <clones> --cap 3 <stage> ...

Stages (each resumable; results under ``<out>``; one USD ledger with ``scripts/lh_run.py``'s guarded clients, persisted
in ``<out>/spend.jsonl``):
- ``world``      every ``world_bug`` P3 left ``reproduced_needs_fix`` (``--p3-proposals``): S3 on the current World, the
                 World rebuilt under recipe 3 (``windows_world.world_needs``), the executed acceptance
                 (``traceaudit.accept_world_bug``: the reproduction fails on the oracle in the current World, passes in the
                 fixed one, S3 passes again with ``--s3-reruns`` clean reruns), an accepted fix as a task version
                 (``<out>/versions.jsonl``);
- ``target``     per window (``--windows``; World of the head version): the prerequisite probe, the full task and its
                 simplified variant (``--kind``) under the target rung (``--model``, ``--attempts``), the counterfactual
                 control, the band and the cause mix, the S15 verdict (``factory.targeting``);
- ``seeded``     the seeded tooling-only failure (``targeting.seeded_tooling``) on one window: both arms in a World whose
                 test command is hidden and whose workspace the tools cannot write; it must come out ``not_attributed``;
- ``paraphrase`` two paraphrases of a window's prompt (author model; ledger, lint and required names re-checked), each run
                 ``--attempts`` times; stability against the full arm's CI;
- ``report``     everything above as ``<out>/p4_report.json``, spend by role;
- ``cleanup``    removes this phase's containers (``crucible-p4-*``, ``crucible-lh-p4-*``) and nothing else.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

GRADE_PREFIX = "crucible-p4-"
WORLD_PREFIX = "crucible-lh-p4-"


def _now() -> float:
    return round(time.time(), 1)


def jload(p: Path, default: Any = None) -> Any:
    return json.loads(p.read_text()) if p.exists() else default


def jsave(p: Path, x: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(x, indent=1, sort_keys=True, default=str))
    tmp.replace(p)


# ---- Part A: the World fixes ------------------------------------------------------------------------------------------------
def _summ_s3(s: Mapping[str, Any]) -> dict[str, Any]:
    return {"ok": s.get("ok"), "why": s.get("why"), "image": s.get("image"), "checks": s.get("checks"),
            "existing_tests": {k: (s.get("existing_tests") or {}).get(k) for k in ("targets", "counts", "infra")},
            "reruns": {k: (s.get("reruns") or {}).get(k) for k in ("n", "clean", "cp_upper")},
            "leak": (s.get("leak") or {}).get("probes"), "grading": s.get("grading")}


async def stage_world(a: argparse.Namespace) -> dict[str, Any]:
    from crucible.factory.lanes.windows import WindowStore, sandbox_factory
    from crucible.factory.lanes.windows_s3 import WindowWorldPort, repair_versions, world_fix, world_s3
    from crucible.factory.lanes.windows_world import ImageBook
    from crucible.factory.traceaudit import Proposal, TaskContext, accept_world_bug, version_accepted
    from crucible.factory.versioning import VersionStore

    out = Path(a.out)
    jdir = out / "world"
    store = WindowStore(Path(a.store), split="train")
    vs = VersionStore(out / "versions.jsonl")
    rows = [json.loads(x) for x in Path(a.p3_proposals).read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r["kind"] == "world_bug" and r["status"] == "reproduced_needs_fix"]
    images = ImageBook(parallel=4)
    sbx = sandbox_factory(parallel=a.parallel, name_prefix=f"{WORLD_PREFIX}s3-", timeout_s=600.0)
    sem = asyncio.Semaphore(2)
    old_stores = [WindowStore(Path(s), split="train") for s in a.p3_stores.split(",") if s]

    async def one(r: dict[str, Any]) -> dict[str, Any]:
        wid = r["task"]
        jp = jdir / f"{wid}.json"
        if (j := jload(jp)) and j.get("done") and not a.redo:
            return j
        async with sem:
            rec = store.window(wid)
            base = store.dir(wid) / "base"
            oracle = {"files": dict(rec["oracle_files"])}
            p = Proposal(kind="world_bug", task=wid, traces=list(r["traces"]), artifact=dict(r["artifact"]), change={"world_fix": None}, source=r["source"],
                         why=r["why"], id=r["id"])
            port = WindowWorldPort(rec, base, s3_reruns=a.s3_reruns, concurrency=4, sandboxes=sbx)
            j: dict[str, Any] = {"window": wid, "proposal": r["id"], "signature": r["artifact"].get("signature"), "command": r["artifact"]["arguments"],
                                 "current_image": port.image}
            # the World P3 reproduced on (the store the trail was mined into), when it differs from store3's
            p3_img = next((s.window(wid)["world"]["image"] for s in old_stores if (s.root / "tasks" / wid / "window.json").exists()), None)
            if p3_img and p3_img != port.image:
                rp = await port.reproduce(r["artifact"], state=oracle, fix={"image": p3_img})
                j["p3_image"] = {"image": p3_img, "repro_on_oracle": {k: rp.get(k) for k in ("failed", "exit", "same_signature")},
                                 "s3": _summ_s3(await world_s3(rec, base, p3_img, reruns=a.s3_reruns, sandboxes=sbx))}
            s3b = await world_s3(rec, base, port.image, reruns=a.s3_reruns, sandboxes=sbx)
            j["s3_before"] = _summ_s3(s3b)
            fix = await world_fix(rec, base, images=images)
            if fix["ok"]:  # the World expert's executed version search when A's own tests fail in the rebuilt World
                rep = await repair_versions(rec, base, fix["image"], sandboxes=sbx)
                fix["repair"] = {k: rep.get(k) for k in ("ok", "why", "failing_at_a", "implicated", "tried", "dist", "version", "from_version", "pin")}
                if rep.get("ok"):
                    fix.update(image=rep["image"], digest=rep["digest"])
            j["fix"] = {k: v for k, v in fix.items() if k != "build_log"}

            async def fixer(_p: Any) -> dict[str, Any] | None:
                return {k: fix.get(k) for k in ("type", "recipe", "image", "digest", "from_image", "needs", "repair")} if fix["ok"] else None

            props = [p]
            failing = (s3b.get("existing_tests") or {}).get("failing") or []
            if failing:  # S3 itself shows a World bug: A's own tests fail in the World at A (upstream released A with them passing)
                cmd = "python -m pytest -q -p no:cacheprovider -o addopts= " + " ".join(failing[:20])
                _rc, txt = await port.run(cmd, state=oracle, image=port.image)
                err = next((m.group(1) for m in re.finditer(r"(?m)^E\s+(\w+(?:Error|Exception)\b.*)$", txt)), "")
                props.append(Proposal(kind="world_bug", task=wid, traces=[], source="s3", why=f"A's own tests fail in the World: {failing[:3]}",
                                      artifact={"type": "repro", "function": "repo.shell", "arguments": {"command": cmd}, "observed": err[:200],
                                                "signature": f"existing tests fail in the World ({err[:80]})"}, change={"world_fix": None}))
            j["proposals"] = []
            for q in props:
                q = await accept_world_bug(q, port, oracle=oracle, fixer=fixer)
                row = {"id": q.id, "source": q.source, "why": q.why, "status": q.status, "reasons": q.reasons,
                       "before_fix_on_oracle": q.evidence.get("before_fix_on_oracle"), "after_fix_on_oracle": q.evidence.get("after_fix_on_oracle")}
                if q.status == "accepted":
                    v = version_accepted(q, vs, TaskContext(wid, prompt=""))
                    row["version"] = v.id if v else None
                j["proposals"].append(row)
            j["status"], j["reasons"], j["version"] = props[0].status, props[0].reasons, j["proposals"][0].get("version")
            j["evidence"] = {k: j["proposals"][0].get(k) for k in ("before_fix_on_oracle", "after_fix_on_oracle")}
            j["s3_after"] = _summ_s3(port.s3_results[-1]) if port.s3_results else None
            j["done"] = True
            jsave(jp, j)
            print(f"world {wid}: " + "; ".join(f"{x['source']}:{x['status']} v={x.get('version')}" for x in j["proposals"]), flush=True)
            return j

    res = await asyncio.gather(*(one(r) for r in rows))
    summary = {"rows": res, "by_status": dict(Counter(x["status"] for r in res for x in r.get("proposals") or [r])), "versions_ok": vs.verify() == []}
    jsave(out / "world_summary.json", summary)
    return summary


# ---- Part B: targeting ----------------------------------------------------------------------------------------------------
def p4_run(a: argparse.Namespace) -> Any:
    """``scripts/lh_run.Run`` (one guarded USD ledger, ``<out>/spend.jsonl``, the run-2 solver clients) with this phase's
    lane: grading containers ``crucible-p4-*``, World containers ``crucible-lh-p4-world-*``."""
    import lh_run

    from crucible.factory.lanes.windows import WindowLane, sandbox_factory

    ns = argparse.Namespace(work=a.out, out=str(Path(a.out) / "lh_out"), cap=a.cap, store=a.store, clones=a.clones, parallel=a.parallel, windows="",
                            routes=False)
    run = lh_run.Run(ns)
    run.lane = WindowLane(run.store, repo_dirs={p.name: str(p) for p in Path(a.clones).iterdir() if p.is_dir()},
                          sandboxes=sandbox_factory(parallel=a.parallel, name_prefix=GRADE_PREFIX))
    run.lane.world_prefix = f"{WORLD_PREFIX}world-"
    return run


def window_task(run: Any, a: argparse.Namespace, wid: str, *, prompt: Mapping[str, Any] | None = None, tag: str = "") -> Any:
    """The window under P2's authored prompt, in the World of its head task version (``<out>/versions.jsonl``)."""
    from crucible.factory.lanes.windows import WindowTask, mined_of, window_checkset
    from crucible.factory.lanes.windows_s3 import world_of_version
    from crucible.factory.versioning import VersionStore

    rec = world_of_version(run.store.window(wid), VersionStore(Path(a.out) / "versions.jsonl"))
    doc = dict(prompt) if prompt is not None else jload(Path(a.prompts) / wid / "prompt.json")
    reqs = list(doc.get("requirements") or [])
    return WindowTask(run.lane, mined_of(rec, run.store.dir(wid) / "base"), rec, task_id=f"{wid}{tag}", prompt=doc["prompt"], requirements=reqs,
                      spans=dict(doc.get("spans") or {}), checkset=dict(doc.get("checkset") or window_checkset(rec, reqs)))


def episode_trail(adir: Path) -> tuple[list[dict[str, Any]], str]:
    """(tool calls ``{function, arguments, result}``, the agent's last message) of one attempt."""
    traj = adir / "trajectory.agent.json"
    calls, last = [], ""
    if traj.exists():
        for st in json.loads(traj.read_text()).get("steps") or []:
            results = [r.get("content") for r in ((st.get("observation") or {}).get("results") or [])]
            for i, tc in enumerate(st.get("tool_calls") or []):
                res = results[i] if i < len(results) else ""
                if isinstance(res, list):
                    res = " ".join(str(x.get("text", "")) for x in res if isinstance(x, dict))
                calls.append({"function": tc.get("function_name"), "arguments": tc.get("arguments") or {}, "result": str(res or "")[-3000:]})
            msg = st.get("message")
            if isinstance(msg, list):
                msg = " ".join(str(x.get("text", "")) for x in msg if isinstance(x, dict))
            if st.get("source") == "agent" and msg:
                last = str(msg)
    return calls, last


def p3_labels(a: argparse.Namespace) -> dict[str, Counter[str]]:
    """Per window, the P3 RLM auditor's labels on the run-1/2 trails (its session ledgers: the ``[stop: agent_done]`` verdict)."""
    root = Path(a.p3_auditor) if a.p3_auditor else None
    out: dict[str, Counter[str]] = {}
    if root is None or not root.exists():
        return out
    tasks = {}
    if a.p3_bundles and Path(a.p3_bundles).exists():
        for x in Path(a.p3_bundles).read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                tasks[r["id"]] = re.split(r"[#~]", str(r["task"]))[0]
    for d in root.iterdir():
        ev = d / "ledger" / "root" / "events.jsonl"
        m = re.match(r"(.+?)-(.+)-([0-9a-f]{16})-(a\d+)-[0-9a-f]+$", d.name)
        if not ev.exists() or not m:
            continue
        bid = f"{m.group(1)}:{m.group(2)}/{m.group(3)}/{m.group(4)}"
        cause = None
        for ln in ev.read_text().splitlines()[::-1]:
            if "[stop: agent_done]" in ln:
                with contextlib.suppress(Exception):
                    cause = json.loads(json.loads(ln)["text"].split("] ", 1)[1]).get("cause")
                break
        if cause and bid in tasks:
            out.setdefault(tasks[bid], Counter())[cause] += 1
    return out


def outcomes(task: Any, eps: list[dict[str, Any]], targets: list[str], compare: list[str]) -> list[Any]:
    from crucible.factory.targeting import outcomes_of

    return outcomes_of(eps, task, targets, compare)


async def stage_target(a: argparse.Namespace) -> dict[str, Any]:
    import lh_run

    from crucible.factory.longtask import ground_truth
    from crucible.factory.targeting import (
        PROBE_PROMPT,
        TargetEvidence,
        band,
        cause_mix,
        counterfactual,
        prerequisite_probe,
        probe_episode,
        scripted_probe,
        simplify,
        target_verdict,
    )

    out = Path(a.out)
    run = p4_run(a)
    wids = [w for w in a.windows.split(",") if w]
    tmap = dict(x.split("=", 1) for x in a.targets.split(",") if "=" in x)
    cfg = lh_run.EpisodeCfg(tool_limit=a.tool_limit)
    pol = lh_run.policy_ref(run, a.model, cfg)
    tdir = out / "target"
    plan: dict[str, Any] = {}
    for wid in wids:
        jp = tdir / f"{wid}.json"
        j = jload(jp, {}) or {}
        task = window_task(run, a, wid)
        targets = tmap.get(wid, "").split("+") if wid in tmap else task.target_milestones()
        simple = simplify(task, a.kind, targets=targets, root=out / "simplified")
        if "s8_simple" not in j:  # the simplified task is a changed task: S8 again before any solver token (00 §4 funnel order)
            gt = await ground_truth(simple)
            j["s8_simple"] = {"ok": gt["ok"], "failed": gt["failed"]}
            j.update(window=wid, targets=targets, kind=a.kind, world=task.window["world"].get("image"), world_version=task.window["world"].get("version"),
                     simplified={"task": simple.task_id, "pre_applied": simple.window["simplified_from"].get("pre_applied"),
                                 "milestones": [m["id"] for m in simple.milestones], "mode": simple.window["simplified_from"].get("mode")})
            jsave(jp, j)
        if not j["s8_simple"]["ok"]:
            print(f"target {wid}: the simplified task fails S8 {j['s8_simple']['failed']}: skipped", flush=True)
            continue
        if "truths" not in (j.get("probe_scripted") or {}):
            j["probe_scripted"] = await scripted_probe(task, out / "probe" / wid)
            jsave(jp, j)
        plan[wid] = (task, simple, targets, j, jp)
    if a.no_solvers:
        return {w: v[3] for w, v in plan.items()}
    # the policy's probe episodes, then both arms of every window in one runner spec (one concurrency pool)
    ptasks = [window_task(run, a, wid, prompt={"prompt": PROBE_PROMPT, "requirements": [], "spans": {}}, tag="~probe") for wid in plan]
    if ptasks:  # idempotent: a finished probe run resumes as done; every stored trail is (re-)judged against the scripted counts
        await lh_run.run_episodes(run, "probe", ptasks, [lh_run.policy_ref(run, a.model, lh_run.EpisodeCfg(tool_limit=12, max_steps=20))], k=1,
                                  concurrency=len(ptasks), stage="probe", role="probe", cfg=lh_run.EpisodeCfg(tool_limit=12, max_steps=20),
                                  checkpoints=False)
        for e in lh_run.episodes_of(run, "probe"):
            wid = str(e["task"]).split("~")[0]
            if wid in plan:
                calls, last = episode_trail(Path(e["src_dir"]))
                plan[wid][3]["probe_episode"] = {**probe_episode(calls, last, plan[wid][3]["probe_scripted"].get("truths")), "usd": e["usd"],
                                                 "stop": e["stop"], "masked": e["masked"]}
                jsave(plan[wid][4], plan[wid][3])
    arm_tasks = [t for wid, (task, simple, *_r) in plan.items() for t in (task, simple)]
    await lh_run.run_episodes(run, "arms", arm_tasks, [pol], k=a.attempts, concurrency=a.concurrency, stage="arms", role="solver", cfg=cfg,
                              checkpoints=False)
    eps = lh_run.episodes_of(run, "arms")
    labels = p3_labels(a)
    rows = {}
    for wid, (task, simple, targets, j, jp) in plan.items():
        # Phi is compared over the target set only: a prefix simplification pre-applies more than the targets' prerequisites, so
        # the rest of the kept milestones get easier too; their gain is reported (``dphi_kept``), never used to attribute
        kept = [m["id"] for m in simple.milestones]
        full_eps = [e for e in eps if e["task"] == task.task_id]
        simple_eps = [e for e in eps if e["task"] == simple.task_id]
        fo, so = outcomes(task, full_eps, targets, targets), outcomes(simple, simple_eps, targets, targets)
        j["arms"] = {"full": [vars(o) for o in fo], "simple": [vars(o) for o in so]}
        j["counterfactual"] = counterfactual(fo, so)
        fk = [o.phi_set for o in outcomes(task, full_eps, targets, kept) if not o.masked]
        sk = [o.phi_set for o in outcomes(simple, simple_eps, targets, kept) if not o.masked]
        j["dphi_kept"] = {"milestones": kept, "full": round(sum(fk) / len(fk), 4) if fk else None, "simple": round(sum(sk) / len(sk), 4) if sk else None,
                          "note": "information only: Phi over every kept milestone, not the target"}
        j["band"] = band([o.reward for o in fo if not o.masked])
        j["cause"] = {**cause_mix(labels.get(wid, Counter())), "source": "P3 RLM auditor labels on the run-1/2 trails of this window"}
        j["probe"] = prerequisite_probe(j["probe_scripted"], j.get("probe_episode"))
        ev = TargetEvidence(task=task.task_id, band=j["band"], cause=j["cause"], counterfactual=j["counterfactual"], probe=j["probe"],
                            paraphrase=j.get("paraphrase_stability"))
        j["verdict"] = target_verdict(ev)
        j["machine"] = await run_s15_machine(out, wid, ev)
        jsave(jp, j)
        rows[wid] = j
        cf = j["counterfactual"]
        print(f"target {wid} {targets}: {cf['verdict']} full {cf.get('success_full')}/{cf.get('n_full')} simple {cf.get('success_simple')}/"
              f"{cf.get('n_simple')} p={cf.get('fisher_p')} dphi={cf.get('dphi')} ci={cf.get('dphi_ci')} power={cf.get('power_at_ref')}; "
              f"probe {j['probe']['ok']}; band {j['band']['in_band']}; S15 ok={j['verdict']['ok']} route={j['verdict']['route']}", flush=True)
    return rows


async def run_s15_machine(out: Path, wid: str, ev: Any) -> dict[str, Any]:
    """S15 through ``factory.machine``: S2 (no automated planner in P4: it hands the same plan back) -> S15; a task routed to
    S2 twice exhausts the loop and escalates (V7), a passing one is accepted, a World-owned probe failure drops it."""
    from crucible.factory.machine import MachineRun, Settle
    from crucible.factory.targeting import target_handlers, target_machine

    async def s2_plan(ctx: Any) -> Any:
        return Settle({"plan": {"window": wid, "findings_seen": list(ctx.findings)[:10], "planner": "not automated in P4"}})

    class Svc:
        def __init__(self) -> None:
            self.evidence, self.results = ev, []

    from dataclasses import asdict

    from crucible.factory.versioning import sha

    svc = Svc()
    rid = f"s15-{wid}-{sha(asdict(ev))[:10]}"  # one machine run per evidence: a journaled run with the same id would resume, not re-decide
    res = await MachineRun(target_machine(), run_dir=out / "machine" / rid, handlers=target_handlers(s2_plan=s2_plan), services=svc, run_id=rid).run()
    return {"run_id": rid, "status": res.status, "why": res.reason, "s15_visits": int(res.entries.get("s15_target", 0))}


async def stage_seeded(a: argparse.Namespace) -> dict[str, Any]:
    """The seeded tooling-only failure on ``--seeded-window``: both arms in a World whose test command is hidden and whose
    workspace is read-only (the grader untouched)."""
    import lh_run

    from crucible.factory.targeting import counterfactual, prerequisite_probe, scripted_probe, seed_image, seeded_tooling, simplify

    out = Path(a.out)
    run = p4_run(a)
    wid = a.seeded_window
    jp = out / "seeded" / f"{wid}.json"
    j = jload(jp, {}) or {}
    task = window_task(run, a, wid)
    tmap = dict(x.split("=", 1) for x in a.targets.split(",") if "=" in x)
    targets = tmap.get(wid, "").split("+") if wid in tmap else task.target_milestones()
    simple = simplify(task, a.kind, targets=targets, root=out / "simplified")
    img = seed_image(task.window["world"]["image"])
    sf, ss = seeded_tooling(task, img), seeded_tooling(simple, img)
    if "probe_scripted" not in j:
        j["probe_scripted"] = await scripted_probe(sf, out / "probe" / f"{wid}~seed")
        j["probe"] = prerequisite_probe(j["probe_scripted"], None)
        j.update(window=wid, targets=targets, seed_image=img, seed={"test_command": "hidden (pytest removed)", "workspace": "read-only"})
        jsave(jp, j)
    cfg = lh_run.EpisodeCfg(tool_limit=a.tool_limit)
    await lh_run.run_episodes(run, "seeded", [sf, ss], [lh_run.policy_ref(run, a.model, cfg)], k=a.attempts, concurrency=a.concurrency, stage="seeded",
                              role="solver_seeded", cfg=cfg, checkpoints=False)
    eps = lh_run.episodes_of(run, "seeded")
    fo = outcomes(task, [e for e in eps if e["task"] == sf.task_id], targets, targets)
    so = outcomes(simple, [e for e in eps if e["task"] == ss.task_id], targets, targets)
    j["arms"] = {"full": [vars(o) for o in fo], "simple": [vars(o) for o in so]}
    j["counterfactual"] = counterfactual(fo, so)
    j["expected"] = "not_attributed"
    j["ok"] = j["counterfactual"]["verdict"] == "not_attributed"
    jsave(jp, j)
    cf = j["counterfactual"]
    print(f"seeded {wid}: {cf['verdict']} (expected not_attributed) full {cf.get('success_full')}/{cf.get('n_full')} simple "
          f"{cf.get('success_simple')}/{cf.get('n_simple')} p_low={cf.get('simple_vs_ref_p')}; probe ok={j['probe']['ok']} ({j['probe']['why']})", flush=True)
    return j


async def stage_paraphrase(a: argparse.Namespace) -> dict[str, Any]:
    """Two checked paraphrases per ``--paraphrase-windows`` window, ``--attempts`` each; stability against the window's full arm."""
    import lh_run

    from crucible.factory.targeting import check_paraphrase, outcomes_of, paraphrase_messages, paraphrased_task, stability, target_verdict

    out = Path(a.out)
    run = p4_run(a)
    author = run.client("author")
    res = {}
    for wid in [w for w in a.paraphrase_windows.split(",") if w]:
        jp = out / "paraphrase" / f"{wid}.json"
        j = jload(jp, {}) or {"window": wid, "paraphrases": []}
        task = window_task(run, a, wid)
        tj = jload(out / "target" / f"{wid}.json", {}) or {}
        ptasks = []
        for k in (1, 2):
            if len(j["paraphrases"]) < k:
                tries = []
                built = None
                for t in range(2):
                    msgs = paraphrase_messages(task, k)
                    if tries:
                        msgs.append({"role": "user", "content": "Your last paraphrase was refused: " + "; ".join(tries[-1]["findings"][:6])
                                     + ". Fix exactly these and keep everything else."})
                    async with run.spend("paraphrase", "author", f"{wid}#{k}.{t}"):
                        para = await author.complete_json(msgs, max_tokens=8000, temperature=0.7)
                    built = check_paraphrase(task, para if isinstance(para, dict) else {})
                    tries.append({"findings": built["findings"]})
                    if built["ok"]:
                        break
                j["paraphrases"].append({"k": k, "tries": tries, "ok": bool(built and built["ok"]), "prompt": (built or {}).get("prompt"),
                                         "requirements": (built or {}).get("requirements"), "spans": (built or {}).get("spans"),
                                         "checkset": (built or {}).get("checkset")})
                jsave(jp, j)
            pr = j["paraphrases"][k - 1]
            if pr["ok"]:
                ptasks.append(paraphrased_task(task, pr, f"para{k}"))
        cfg = lh_run.EpisodeCfg(tool_limit=a.tool_limit)
        if ptasks:
            await lh_run.run_episodes(run, f"para-{wid}", ptasks, [lh_run.policy_ref(run, a.model, cfg)], k=a.attempts, concurrency=a.concurrency,
                                      stage="paraphrase", role="solver_paraphrase", cfg=cfg, checkpoints=False)
        eps = lh_run.episodes_of(run, f"para-{wid}")
        targets = tj.get("targets") or task.target_milestones()
        compare = list(targets)
        from crucible.factory.targeting import Outcome

        orig = [Outcome(**o) for o in (tj.get("arms") or {}).get("full") or []]
        stab = []
        for t in ptasks:
            po = outcomes_of([e for e in eps if e["task"] == t.task_id], task, targets, compare)
            stab.append({"task": t.task_id, **stability(orig, po), "outcomes": [vars(o) for o in po]})
        j["stability"] = stab
        jsave(jp, j)
        if tj:  # the window's S15 verdict with paraphrase stability measured
            from crucible.factory.targeting import TargetEvidence

            tj["paraphrase_stability"] = [{k: v for k, v in s.items() if k != "outcomes"} for s in stab]
            ev = TargetEvidence(task=task.task_id, band=tj.get("band"), cause=tj.get("cause"), counterfactual=tj.get("counterfactual"), probe=tj.get("probe"),
                                paraphrase=tj["paraphrase_stability"])
            tj["verdict"] = target_verdict(ev)
            tj["machine"] = await run_s15_machine(out, wid, ev)
            jsave(out / "target" / f"{wid}.json", tj)
        res[wid] = j
        print(f"paraphrase {wid}: " + "; ".join(f"{s['task']}: stable={s.get('stable')} pass {s.get('pass_rate')} in {s.get('pass_ci')}, "
                                                f"phi {s.get('mean_phi')} in {s.get('phi_ci')}" for s in stab), flush=True)
    return res


def spend_by_role(out: Path) -> dict[str, float]:
    c: Counter[str] = Counter()
    p = out / "spend.jsonl"
    if p.exists():
        for x in p.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                c[f"{r.get('stage')}/{r.get('role')}"] += float(r.get("usd") or 0)
    return {k: round(v, 4) for k, v in sorted(c.items())}


def stage_report(a: argparse.Namespace) -> dict[str, Any]:
    out = Path(a.out)
    rep = {"world": jload(out / "world_summary.json"), "target": {p.stem: jload(p) for p in sorted((out / "target").glob("*.json"))},
           "seeded": {p.stem: jload(p) for p in sorted((out / "seeded").glob("*.json"))},
           "paraphrase": {p.stem: jload(p) for p in sorted((out / "paraphrase").glob("*.json"))}, "spend_by_role": spend_by_role(out)}
    rep["spend_total"] = round(sum(rep["spend_by_role"].values()), 4)
    jsave(out / "p4_report.json", rep)
    return rep


def cleanup() -> None:
    for prefix in (GRADE_PREFIX, WORLD_PREFIX):
        r = subprocess.run(["docker", "ps", "-a", "--filter", f"name={prefix}", "--format", "{{.Names}}"], capture_output=True, text=True, check=False)
        names = [x for x in r.stdout.split() if x.startswith(prefix)]
        if names:
            subprocess.run(["docker", "rm", "-f", *names], capture_output=True, check=False)
        print(f"removed {len(names)} {prefix}* containers")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--store", required=True, help="store3/train (the held-out store is never opened)")
    ap.add_argument("--clones", default="")
    ap.add_argument("--p3-proposals", default="")
    ap.add_argument("--p3-stores", default="", help="the stores P3's trails were mined into (the World P3 reproduced on)")
    ap.add_argument("--prompts", default="", help="dir of <window>/prompt.json (P2's authored prompts)")
    ap.add_argument("--cap", type=float, default=3.0)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--s3-reruns", type=int, default=60)
    ap.add_argument("--windows", default="")
    ap.add_argument("--model", default="qwen/qwen3.7-flash")
    ap.add_argument("--author", default="qwen/qwen3.7-flash")
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--kind", default="prerequisites")
    ap.add_argument("--targets", default="", help="window=milestone,... (default: the deepest milestone with an executed edge)")
    ap.add_argument("--tool-limit", type=int, default=150)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seeded-window", default="")
    ap.add_argument("--paraphrase-windows", default="")
    ap.add_argument("--p3-auditor", default="", help="P3's auditor session dirs (per-window labels for the cause mix)")
    ap.add_argument("--p3-bundles", default="", help="P3's bundles.jsonl (bundle id -> task)")
    ap.add_argument("--no-solvers", action="store_true", help="target: stop before any model call (S8 on the simplified tasks, scripted probes)")
    ap.add_argument("--redo", action="store_true")
    return ap.parse_args(argv)


async def amain(a: argparse.Namespace) -> None:
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for st in a.stages:
        if st == "world":
            s = await stage_world(a)
            print(json.dumps({"by_status": s["by_status"], "versions_ok": s["versions_ok"]}), flush=True)
        elif st == "target":
            await stage_target(a)
        elif st == "seeded":
            await stage_seeded(a)
        elif st == "paraphrase":
            await stage_paraphrase(a)
        elif st == "report":
            r = stage_report(a)
            print(json.dumps({"spend_by_role": r["spend_by_role"], "total": r["spend_total"]}), flush=True)
        elif st == "cleanup":
            cleanup()
        else:
            raise SystemExit(f"unknown stage {st}")


def main() -> None:
    asyncio.run(amain(parse_args()))


if __name__ == "__main__":
    main()
