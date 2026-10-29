#!/usr/bin/env python
"""Smoke: one TRAINING debug-lane task end to end on the CPU, no model, no paid call (Docker only).

    python scripts/smoke/debug_episode.py --root <debug data dir> --task db-go-huma-815e1ff8 --out <dir> [--gates S6,S5,S8]

``--root`` is a copy of a debug data dir holding the one task (``authored/<id>.json``, ``gen/store/train/{split.json,
tasks/<id>}``, ``gen/clones/<repo>`` with its alternates pointed at the copy's ``src/<owner>__<repo>``), so the durable
store is never written: the lane's gate pass re-gates and journals there. Held-out and sealed tasks are refused (the
window's split must be train).

What runs, with real components (the task's own Docker World and its hidden verifier):
0. the task's portable forms (no container): the Harbor task folder (``envs.export_harbor``, Harbor's own loader), the
   portable and slime rows, and the agentenv blockers (a patch-graded task cannot run as an agentenv Task yet: reported);
1. ``factory.envs.make_env`` -> ``WorkspaceHarness`` -> grade -> reward, through ``runtime.run.Runner``, for scripted
   policies (``envs.scripted_policy``): the null agent (``ls``, then done) must score 0; the oracle agent (writes the
   reference fix, commits, writes STATUS.json) must score 1; the hack agent (Go tasks: no fix, a ``TestMain`` that exits 0
   in a fixed package, a claim of success) must score 0 and not be masked;
2. ``--gates``: the lane's own gate pass (``Debug.gates``: S6, S5, S8, S3, S10 with its re-gating rules) on the same task;
   S3 and S10 rerun the suite tens of times (``--s3-reruns 60 --flake-reruns 30`` for the lane's verdicts; fewer reruns
   check the plumbing, but S3's Clopper-Pearson bound then fails by design).

``scripts/smoke/audit_credit.py --episodes <out>/runs`` then audits these attempts and computes their credit.

Containers are named ``crucible-lh-d-smoke-<pid>-*``; a run removes its own at the end, ``--cleanup`` every smoke run's (``crucible-lh-d-smoke-*``) and nothing else. Exit 1 when a check fails.
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

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
BASE = "crucible-lh-d-smoke-"
PREFIX = f"{BASE}{os.getpid()}-"  # this run's containers: a run's own cleanup never removes a concurrent run's


def cleanup(prefix: str = PREFIX) -> list[str]:
    names = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True, check=False).stdout.split()
    mine = [n for n in names if n.startswith(prefix)]
    for n in mine:
        subprocess.run(["docker", "rm", "-f", n], capture_output=True, check=False)
    return mine


def load_task(root: Path, tid: str) -> Any:
    import lh_debug

    from crucible.factory.generators.debug import debug_sandboxes

    lh_debug.ROOT = root
    authored = json.loads((root / "authored" / f"{tid}.json").read_text())
    rec = json.loads((root / "gen" / "store" / "train" / "tasks" / tid / "window.json").read_text())
    if rec.get("split") != "train":
        raise SystemExit(f"{tid}: split {rec.get('split')!r}; the smoke runs training tasks only")
    gen = lh_debug.generator([rec["repo"]], "train")
    gen.sandboxes = debug_sandboxes(parallel=1, prefix=f"{PREFIX}grade-")
    gen.lane().world_prefix = f"{PREFIX}world-"
    gen.lane().sandboxes = gen.sandboxes
    return gen, gen.task(tid, authored=authored)


def scripts(task: Any) -> dict[str, list[tuple]]:
    from crucible.factory.lanes.executable import is_protected

    o = {p: t for p, t in task.window["oracle_files"].items() if not is_protected(p) and (t is None or isinstance(t, str))}
    commit = "git add -A && git -c user.email=a@b -c user.name=a commit -qm wip"
    writes = [("write", p, t) if t is not None else ("shell", f"rm -f {p}") for p, t in sorted(o.items())]
    status = json.dumps({r: 0.9 for r in task.requirement_ids})
    out = {"null": [("shell", "ls"), ("final", "Nothing to change.")],
           "oracle": [*writes, ("shell", commit), ("write", "STATUS.json", status), ("final", "All done.")]}
    # a reward hack (Go): no fix, a TestMain in a fixed package that exits 0 before any test runs, and a claim of success
    go = next((p for p, t in sorted(o.items()) if p.endswith(".go") and t), None)
    pkg = next((ln.split()[1] for ln in (o.get(go) or "").splitlines() if ln.startswith("package ")), None) if go else None
    if go and pkg:
        hack = f'package {pkg}\n\nimport (\n\t"os"\n\t"testing"\n)\n\nfunc TestMain(m *testing.M) {{ os.Exit(0) }}\n'
        d = go.rsplit("/", 1)[0] + "/" if "/" in go else ""
        out["hack"] = [("write", f"{d}zz_smoke_hack_test.go", hack), ("shell", commit), ("write", "STATUS.json", status), ("final", "All done.")]
    return out


async def episodes(task: Any, out: Path) -> dict[str, Any]:
    from crucible.factory.checkpoint import CheckpointConfig
    from crucible.factory.envs import Timeouts, make_env, scripted_policy
    from crucible.runtime.run import Runner

    env = make_env(task, checkpoint=CheckpointConfig(every=0, on_commit=False), timeouts=Timeouts(agent_s=900, rollout_s=1800))
    res: dict[str, Any] = {"harness": type(env.harness).__name__}
    for name, steps in scripts(task).items():
        rid = f"smoke-{name}"
        await Runner(out).run(env.run_spec(run_id=rid, policies=[scripted_policy(steps, pid=f"scripted-{name}")], k=1))
        adir = min((out / rid / "samples").glob("*/a*"))
        r = json.loads((adir / "record.json").read_text())
        res[name] = {"reward": r.get("reward"), "masked": r.get("mask_sample"), "failure": (r.get("failure") or {}).get("kind"), "dir": str(adir)}
        print(f"{name}: reward {r.get('reward')} masked {r.get('mask_sample')}", flush=True)
    res["null_ok"] = res["null"]["reward"] == 0 and not res["null"]["masked"]
    res["oracle_ok"] = res["oracle"]["reward"] == 1 and not res["oracle"]["masked"]
    if "hack" in res:  # the hack earns nothing (a masked attempt would hide it, so it must be scored)
        res["hack_ok"] = res["hack"]["reward"] == 0 and not res["hack"]["masked"]
    return res


def interop(task: Any, out: Path) -> dict[str, Any]:
    """The task's portable forms, no container: the Harbor task folder (validated by Harbor's own loader), the portable
    row and the slime row (a training task may train; ``check_row`` passes), and whether its agentenv Task could run."""
    from crucible.factory.envs import agentenv_blockers, export_harbor, slime_rows, task_row

    h = export_harbor(task, out / "harbor")
    row = task_row(task)
    (srow,) = slime_rows([task])
    blockers = agentenv_blockers(row)
    res = {"harbor": {k: h[k] for k in list(h)[:6]} if isinstance(h, dict) else str(h)[:300], "row_split": row["split"], "row_env": row["env"],
           "slime_instance": srow.get("instance_id") or (srow.get("metadata") or {}).get("instance_id"), "agentenv_blockers": blockers}
    # the known gap (harness-reward audit §5): a patch-graded task cannot run as an agentenv Task yet; reported, not a failure
    res["ok"] = row["split"] == "train" and bool(h)
    return res


async def gates(gen: Any, task: Any, which: tuple[str, ...], *, s3_reruns: int = 4, s3_concurrency: int = 1, flake_reruns: int = 2,
                flake_levels: tuple[int, ...] = (1,)) -> dict[str, Any]:
    """The lane's own gate pass (``Debug.gates``: the deterministic gates plus its re-gating rules, e.g. a test that flips
    between reruns of the unchanged start tree is excluded and the gates run again), on the copy under ``--root``; its
    containers carry this smoke's prefix."""
    from crucible.factory.generators import debug as dbg
    from crucible.factory.generators.base import GateConfig

    dbg.PREFIX = PREFIX  # Debug.gates names its S3 containers f"{PREFIX}s3-": this smoke's, never another worker's
    cfg = GateConfig(s3_reruns=s3_reruns, s3_concurrency=s3_concurrency, flake_reruns=flake_reruns, flake_levels=flake_levels, gates=which)
    r = await gen.gates(task.task_id, cfg=cfg)
    out = {g: {"ok": v.get("ok"), "findings": v.get("findings"), "evidence": v.get("evidence")} for g, v in r["gates"].items()}
    rec = gen.store.window(task.task_id)
    out["_excluded_by_the_lane"] = {k: rec.get(k) for k in ("s3_flaky_excluded", "s8_flaky_excluded", "s5_regress_excluded") if rec.get(k)}
    return out


async def main_async(a: argparse.Namespace) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gen, task = load_task(Path(a.root), a.task)
    res: dict[str, Any] = {"task": a.task, "split": task.window.get("split")}
    try:
        res["interop"] = interop(task, out)
        if not a.no_episodes:
            res["episodes"] = await episodes(task, out / "runs")
        if a.gates:
            res["gates"] = await gates(gen, task, tuple(g for g in a.gates.split(",") if g), s3_reruns=a.s3_reruns, s3_concurrency=a.s3_concurrency,
                                       flake_reruns=a.flake_reruns, flake_levels=tuple(int(x) for x in a.flake_levels.split(",")))
    finally:
        res["removed"] = cleanup()
    ep = res.get("episodes") or {"null_ok": True, "oracle_ok": True}
    ok = res["interop"]["ok"] and ep["null_ok"] and ep["oracle_ok"] and ep.get("hack_ok", True) and all(v["ok"] for g, v in (res.get("gates") or {}).items() if not g.startswith("_"))
    res["ok"] = ok
    (out / "smoke.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    return 0 if ok else 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=False)
    p.add_argument("--task", default="db-go-huma-815e1ff8")
    p.add_argument("--out", required=False)
    p.add_argument("--gates", default="", help="comma list of S6,S5,S8,S3,S10 (S3/S10 run reduced rerun counts)")
    p.add_argument("--no-episodes", action="store_true", help="gates only")
    # S3 certifies the start tree clean by a Clopper-Pearson bound: it needs ~60 clean reruns (the lane's default), so a
    # smaller --s3-reruns checks the plumbing and every other S3 check, and S3's verdict itself fails on the bound
    p.add_argument("--s3-reruns", type=int, default=4)
    p.add_argument("--s3-concurrency", type=int, default=1, help="containers at once in S3's second level (keep the host's Docker polite)")
    p.add_argument("--flake-reruns", type=int, default=2)
    p.add_argument("--flake-levels", default="1")
    p.add_argument("--cleanup", action="store_true")
    a = p.parse_args()
    if a.cleanup:
        print(json.dumps({"removed": cleanup(BASE)}))
        return
    if not a.root or not a.out:
        p.error("--root and --out are required")
    sys.exit(asyncio.run(main_async(a)))


if __name__ == "__main__":
    main()
