"""M6 gate: R1-R6 on every coding seed task (tasks 01-04), with the *recorded* verifier outcomes.

Re-running the hidden graders needs the lab GPU (and they have known holes, DESIGN Appendix B.10); this check therefore
exercises the reward function design on the tasks' real states with the outcomes the graders recorded:

- **milestones**: mined from each task's oracle (``solution/solution.patch``) against its base repo (source files only);
- **states**: the oracle end state (outcome 1, recorded in MIGRATION_LOG), the base (no-op, outcome 0), and every recorded
  frontier patch whose verifier outcome was a failure (negatives, outcome 0) or a pass (reported);
- **reward**: ``compose(gates={}, outcome=recorded, phi=Φ(state))``; R1 oracle at max, R2 no-op 0, R3 negatives below the
  success floor, R4 oscillation (oracle -> base -> oracle) not paying, R5 stub milestones without prerequisites not
  paying, R6 the same state with different claims scoring the same.

Also reported: Φ of each failing frontier run (how far a failed attempt got; κΦ is its partial credit).
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from crucible.credit.compose import DEFAULT, compose
from crucible.credit.properties import check_all, r1_oracle, r2_noop, r3_negatives, r4_oscillation, r5_stubbing, r6_state_only, r7_phi_farm
from crucible.workflow.filestate import EditError, FileState
from crucible.workflow.milestones.evaluate import FileStateView, evaluate
from crucible.workflow.milestones.mine import from_diff
from crucible.workflow.potential import potential
from crucible_pack_coding import PACK_ROOT
from crucible_pack_coding.connectors.agent_runs import _recorded

CRUCIBLE = PACK_ROOT.parent.parent
TASKS = CRUCIBLE.parent / "migration_tasks"
REPORT = CRUCIBLE / "docs" / "modules" / "reports" / "m6_properties_coding.json"
PCL = CRUCIBLE / "data" / "cache" / "git" / "pcl.git"
TEST = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|conftest\.py$")


def base_reader(task: str):
    d = TASKS / task
    if task.startswith("task04"):
        def read(path: str) -> str | None:
            r = subprocess.run(["git", f"--git-dir={PCL}", "show", f"c47198828379841dbbeae6991e0757011a552cec:{path}"], capture_output=True, check=False)
            return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None
        return read
    root = d / "repo" if (d / "repo").exists() else d / "repos" / {"task02": "pcops", "task03": "mdlite"}[task[:6]]

    def read(path: str) -> str | None:
        f = root / path
        return f.read_text(errors="replace") if f.is_file() else None

    return read


def recorded_outcome(task: str, run_dir: Path) -> float | None:
    """The run's verifier outcome on the task's CURRENT version; None when it was graded under another (M14 audit V9) or
    never graded."""
    from crucible_pack_coding.connectors.agent_runs import task_versions

    under, current = task_versions(task, run_dir.name)
    if under != current:
        return None
    g = run_dir / "grade_report.json"
    if g.exists():
        return 1.0 if float(json.loads(g.read_text()).get("reward", 0)) >= 1 else 0.0
    rec = _recorded(task, run_dir.name)
    return None if rec is None else (1.0 if rec["score"] >= 1 else 0.0)


async def task_report(task: str) -> dict[str, Any]:
    base = base_reader(task)
    oracle_patch = (TASKS / task / "solution" / "solution.patch").read_text(errors="replace")
    cands = [c.milestone for c in from_diff(oracle_patch, base, include=lambda p: not TEST.search(p), max_per_file=6)]
    oracle = FileState(base=base)
    oracle.apply_patch(oracle_patch)
    # filters (a) executable and (c) non-trivial: measured on both, false at the base, true at the expert end state
    at_base = await evaluate(cands, FileStateView(FileState(base=base)))
    at_oracle = await evaluate(cands, FileStateView(oracle))
    ms = [m for m in cands if at_base[m.id] == 0.0 and at_oracle[m.id] == 1.0]
    dropped = {"true_at_base": sum(1 for m in cands if at_base[m.id] == 1.0), "not_reached_by_oracle": sum(1 for m in cands if at_oracle[m.id] != 1.0),
               "unmeasurable": sum(1 for m in cands if at_base[m.id] is None or at_oracle[m.id] is None)}
    states: dict[str, tuple[FileState, float]] = {"oracle": (oracle, 1.0), "base": (FileState(base=base), 0.0)}
    runs: dict[str, Any] = {}
    for rd in sorted((TASKS / task / "runs").glob("*")):
        patch = rd / "model.patch"
        y = recorded_outcome(task, rd)
        if not patch.exists() or y is None:
            runs[rd.name] = {"skipped": "no patch" if not patch.exists() else "no recorded outcome"}
            continue
        fs = FileState(base=base)
        try:
            fs.apply_patch(patch.read_text(errors="replace"))
        except EditError as e:
            runs[rd.name] = {"skipped": f"patch does not apply to the base: {str(e)[:120]}"}
            continue
        states[f"run:{rd.name}"] = (fs, y)
    phis = {name: await evaluate(ms, FileStateView(fs)) for name, (fs, _) in states.items()}
    reward = {name: compose(gates={}, outcome=bool(y), phi=potential(ms, phis[name])[0]) for name, (_, y) in states.items()}
    by_state = {id(states[n][0]): reward[n] for n in states}
    of = lambda fs: by_state[id(fs)]
    negatives = {n: fs for n, (fs, y) in states.items() if n.startswith("run:") and y == 0.0}
    for n, (_, y) in states.items():
        if n.startswith("run:"):
            runs[n[4:]] = {"outcome": y, "phi": round(potential(ms, phis[n])[0], 4), "reward": round(reward[n].reward, 4)}
    # R3: the failing frontier runs are honest failures (bounded partial credit); a task with none cannot test R3,
    # which is reported, not passed (M6 audit: task01 skipped R3 and reported ok)
    shallow = {m.id: 1.0 if m.id.split("/")[1] in ("located", "created", "deleted") else 0.0 for m in ms}
    res = [r1_oracle(of, oracle), r2_noop(of, states["base"][0]), r3_negatives(of, negatives),
           r4_oscillation(ms, [phis["base"], phis["oracle"]], [phis["base"], phis["oracle"], phis["base"], phis["oracle"]]),
           r5_stubbing(ms, {**{m.id: 0.0 for m in ms}, **{m.id: 1.0 for m in ms[-2:]}}),
           r6_state_only(lambda fs, msgs: of(fs), oracle, [["All tests pass and it is 3x faster."], ["done"]], reads_messages=False),
           r7_phi_farm(ms, shallow, kappa=DEFAULT.kappa)]
    return {"task": task, "candidates": len(cands), "dropped_by_filters": dropped, "milestones": len(ms), "milestone_kinds": sorted({m.id.split("/")[1] for m in ms}),
            "phi": {"oracle": round(potential(ms, phis["oracle"])[0], 4), "base": round(potential(ms, phis["base"])[0], 4)},
            "runs": runs, **check_all(res), "negatives_tested": len(negatives)}


def run() -> dict[str, Any]:
    tasks = ["task01_prims_omp", "task01v2_tessera_omp", "task02_pcops_triton", "task03_mdlite_pallas", "task04_pcl_gpu_cuda12"]
    out = {t: asyncio.run(task_report(t)) for t in tasks}
    rep = {"gate": "M6 R1-R6 (coding, recorded outcomes)", "pass": all(v["ok"] for v in out.values()), "tasks": out}
    REPORT.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    return rep


if __name__ == "__main__":
    r = run()
    for t, v in r["tasks"].items():
        print(t, "ok" if v["ok"] else "FAIL", {k: x["ok"] for k, x in v["properties"].items()}, "milestones", v["milestones"], "phi", v["phi"], "negatives", v["negatives_tested"])
        for name, x in v["runs"].items():
            print("   ", name, x)
