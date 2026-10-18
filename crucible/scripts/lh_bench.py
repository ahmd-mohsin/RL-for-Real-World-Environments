"""P6 (swarm-pipeline/00 §10, §11, §15 P6): the sealed held-out benchmark, native environments, the slime bridge. $0: no
model is called (scripted solvers only).

    python crucible/scripts/lh_bench.py <stage> [options]

Stages:
- ``build``   the sealed benchmark from ``--heldout`` against ``--train`` (``factory.benchmark.build``): sealing (repos,
              orgs, time split, contamination), the S3/S5/S6/S8/S10 gates in ``crucible-lh-p6-*`` / ``crucible-p6-*``
              containers (journaled per task; a rerun reuses them), ``<out>/manifest.json`` + ``tasks.jsonl`` + ``seal.json``;
- ``verify``  recompute every manifest hash from the store (``factory.benchmark.verify``);
- ``rows``    slime JSONL rows for the trainable training windows with an authored prompt (``--prompts``), refusing every
              id the manifest seals (V9);
- ``equiv``   the exit criterion on a real training window in its Docker World: the scripted pass / fail / hack /
              partial / infra-death solvers through the native path and through the slime generate path; rewards,
              per-turn credit and masking compared; ``<out>/equiv.json``;
- ``plan``    the metrics plan;
- ``cleanup`` removes this phase's containers (``crucible-p6-*``, ``crucible-lh-p6-*``) and nothing else.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PREFIXES = ("crucible-p6-", "crucible-lh-p6-")


def jdump(x: Any) -> str:
    return json.dumps(x, indent=1, sort_keys=True, default=str)


def stage_build(a: argparse.Namespace) -> None:
    from crucible.factory.benchmark import BuildConfig, GateConfig, run

    cfg = BuildConfig(heldout_root=Path(a.heldout), train_root=Path(a.train), out_dir=Path(a.out), train_filter=a.train_filter, time_policy=a.time_policy,
                      model_cutoff=a.model_cutoff, train_prompts=Path(a.prompts) if a.prompts else None, run_gates=not a.no_gates,
                      only=tuple(a.only or ()), gates=GateConfig(reruns=a.reruns, concurrency=a.concurrency, flake_reruns_per_level=a.reruns // 2))
    m = run(cfg, log=lambda s: print(s, flush=True))
    print(jdump({"split_members": m["split_members"], "sealed": m["sealed"], "sealing": m["sealing_summary"],
                 "tasks": {t["id"]: {"in_split": t["in_split"], "excluded": t["excluded"], "gates": t["gates"]} for t in m["tasks"]}}))


def stage_verify(a: argparse.Namespace) -> None:
    from crucible.factory.benchmark import verify

    probs = verify(Path(a.out))
    print(jdump({"ok": not probs, "problems": probs}))
    sys.exit(1 if probs else 0)


def stage_rows(a: argparse.Namespace) -> None:
    from crucible.factory.benchmark import sealed_ids, windows_of
    from crucible.factory.envs import resolve_ref, slime_rows
    from crucible.factory.lanes.windows import WindowLane, WindowStore, WindowTask, mined_of, window_checkset
    from crucible.train.bridges.slime.rows import write_rows

    sealed = sealed_ids(Path(a.manifest)) if a.manifest else set()
    store = WindowStore(Path(a.train), split="train")
    lane = WindowLane(store)
    tasks = []
    for rec in windows_of(Path(a.train), "train"):
        f = Path(a.prompts) / rec["id"] / "prompt.json"
        if not f.exists():
            continue
        doc = json.loads(f.read_text())
        reqs = list(doc.get("requirements") or [])
        t = WindowTask(lane, mined_of(rec, store.dir(rec["id"]) / "base"), rec, task_id=rec["id"], prompt=doc["prompt"], requirements=reqs,
                       spans=dict(doc.get("spans") or {}), checkset=dict(doc.get("checkset") or window_checkset(rec, reqs)))
        resolve_ref(t.ref().to_json(), prompts=Path(a.prompts))  # the row resolves back to the same task (hashes checked)
        tasks.append(t)
    rows = slime_rows(tasks, agent=a.agent, sealed=sealed)
    print(jdump(write_rows(rows, Path(a.out), sealed=sealed)))


# ---- equiv -------------------------------------------------------------------------------------------------------------------
def kill_world() -> None:
    names = subprocess.run(["docker", "ps", "--filter", "name=crucible-lh-p6-world-", "--format", "{{.Names}}"], capture_output=True, text=True,
                           check=False).stdout.split()
    for n in names:
        subprocess.run(["docker", "kill", n], capture_output=True, check=False)


def real_scripts(task: Any) -> dict[str, list[tuple]]:
    from crucible.factory.lanes.executable import is_protected

    o = {p: t for p, t in task.window["oracle_files"].items() if not is_protected(p) and (t is None or isinstance(t, str))}
    commit = "git add -A && git -c user.email=a@b -c user.name=a commit -qm wip"

    def writes(files: dict[str, str | None]) -> list[tuple]:
        return [("write", p, t) if t is not None else ("shell", f"rm -f {p}") for p, t in sorted(files.items())]

    status = json.dumps({r: 0.9 for r in task.requirement_ids})
    mut = next(m for m in task.window.get("mutants") or [] if m.get("status") == "valid")
    mfiles = {p: t for p, t in (mut["solution"].get("files") or {}).items() if not is_protected(p) and (t is None or isinstance(t, str))}
    first = sorted(o)[:1]
    return {
        "pass": [*writes(o), ("shell", commit), ("write", "STATUS.json", status), ("final", "All done.")],
        "fail": [("shell", "ls"), ("final", "Nothing to change.")],
        "hack": [*writes(o), ("write", "conftest.py", "import pytest\n"), ("final", "Done.")],
        "partial": [*writes(mfiles), ("shell", commit), ("write", "STATUS.json", status), ("final", f"GIVE_UP: {mut['milestone']} is beyond me")],
        "infra": [*writes({p: o[p] for p in first}), ("chaos", kill_world), ("shell", "ls"), ("final", "Done.")],
    }


async def equiv(a: argparse.Namespace) -> dict[str, Any]:
    from crucible.core.schema.rollout import RolloutRecord
    from crucible.factory.benchmark import GRADE_PREFIX, WORLD_PREFIX
    from crucible.factory.checkpoint import CheckpointConfig
    from crucible.factory.envs import SlimeProvider, Timeouts, make_env, register, register_script, reset_provider, scripted_policy, slime_rows
    from crucible.factory.honesty import honesty_group_fn
    from crucible.factory.lanes.windows import WindowLane, WindowStore, WindowTask, mined_of, sandbox_factory, window_checkset
    from crucible.runtime.run import Runner
    from crucible.train.bridges.slime import credit as sc
    from crucible.train.bridges.slime.episode import ScriptedModel
    from crucible.train.bridges.slime.generate import generate
    from crucible.train.bridges.slime.types import MirrorSample
    from crucible.train.credit import turn_credit_map

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    store = WindowStore(Path(a.train), split="train")
    lane = WindowLane(store, sandboxes=sandbox_factory(parallel=4, name_prefix=GRADE_PREFIX))
    lane.world_prefix = f"{WORLD_PREFIX}world-"
    rec = store.window(a.window)
    doc = json.loads((Path(a.prompts) / a.window / "prompt.json").read_text())
    reqs = list(doc.get("requirements") or [])
    task = WindowTask(lane, mined_of(rec, store.dir(a.window) / "base"), rec, task_id=a.window, prompt=doc["prompt"], requirements=reqs,
                      spans=dict(doc.get("spans") or {}), checkset=dict(doc.get("checkset") or window_checkset(rec, reqs)))
    cfg = CheckpointConfig(every=a.every, cap=20, concurrency=2)
    env = make_env(task, checkpoint=cfg, timeouts=Timeouts(agent_s=a.agent_timeout, rollout_s=a.rollout_timeout))
    scripts = real_scripts(task)
    native: dict[str, Any] = {}
    for name, steps in scripts.items():
        rid = f"n-{name}"
        await Runner(out / "native").run(env.run_spec(run_id=rid, policies=[scripted_policy(steps, pid=f"scripted-{name}")], k=1))
        adir = min((out / "native" / rid / "samples").glob("*/a*"))
        r = json.loads((adir / "record.json").read_text())
        ro = RolloutRecord.model_validate_json((adir / "rollout.json").read_text()) if (adir / "rollout.json").exists() else None
        native[name] = (r, ro)
        print(f"native {name}: reward {r.get('reward')} masked {r.get('mask_sample')}", flush=True)
    reset_provider(SlimeProvider(checkpoint=cfg))
    register(task)
    (row,) = slime_rows([task], agent="scripted")
    args = SimpleNamespace(crucible_agent_timeout_s=a.agent_timeout, crucible_rollout_timeout_s=a.rollout_timeout, crucible_turn_eta=a.eta,
                           reward_key=None, crucible_workdir=str(out / "slime"))
    (out / "slime").mkdir(exist_ok=True)
    slime: dict[str, Any] = {}
    for i, (name, steps) in enumerate(scripts.items()):
        register_script(name, lambda steps=steps: ScriptedModel(steps))
        (s,) = await generate(args, MirrorSample(index=i, group_index=0, prompt=row["prompt"], metadata={**row["metadata"], "script": name}), {})
        slime[name] = s
        print(f"slime {name}: reward {s.reward} removed {s.remove_sample} {s.metadata.get('abort_reason', '')}", flush=True)
    names = list(scripts)
    recs = [native[n][1] or RolloutRecord.model_validate({"rollout_id": n, "mask_sample": True}) for n in names]
    nat_credit = turn_credit_map(recs, ["g"] * len(recs), eta=a.eta)
    gc = sc.group_credit([slime[n] for n in names], eta=a.eta)
    live = [ro for ro, n in zip(recs, names, strict=True) if not native[n][0]["mask_sample"]]
    for ro in live:
        ro.advantage = None
    honesty_group_fn()("g", live)  # the native group stage: R - mean, honesty term composed
    _, seq = sc.reward_post_process(args, [slime[n] for n in names])
    rows = {}
    for i, n in enumerate(names):
        r, ro = native[n]
        s = slime[n]
        masked = bool(r["mask_sample"]), bool(s.remove_sample)
        row: dict[str, Any] = {"masked": masked, "reward": (None if masked[0] else r["reward"], None if masked[1] else s.reward)}
        if not any(masked):
            nd = {t.call_id: round(float(t.advantage or 0.0), 9) for t in ro.turns}
            sd = {t["call_id"]: t["delta_phi"] for t in s.metadata["crucible"]["turn_credit"]}
            want = [round(nat_credit[(ro.rollout_id, t.call_id)], 9) for t in ro.turns if t.completion_token_ids]
            row.update(delta_phi_equal=nd == sd, delta_phi=list(sd.values()), group_credit_equal=[round(x, 9) for x in gc[("rollout", i)]["turns"]] == want,
                       seq_advantage=(ro.advantage, seq[i]), seq_advantage_equal=abs(float(ro.advantage) - seq[i]) < 1e-9,
                       honesty_term=s.metadata["crucible"]["honesty_term"],
                       phi=s.metadata["crucible"]["phi"], path_violations=s.metadata["crucible"]["path_violations"], give_up=s.metadata["crucible"]["give_up"])
        else:
            row.update(native_failure=(r.get("failure") or {}).get("kind"), slime_reason=s.metadata.get("abort_reason"))
        row["identical"] = masked[0] == masked[1] and (any(masked) or (abs(r["reward"] - s.reward) < 1e-9 and row["delta_phi_equal"] and row["group_credit_equal"]
                                                                       and row["seq_advantage_equal"]))
        rows[n] = row
    res = {"window": a.window, "scripts": {n: [st[0] for st in v] for n, v in scripts.items()}, "results": rows, "all_identical": all(x["identical"] for x in rows.values()),
           "eta": a.eta, "checkpoint_every": a.every}
    (out / "equiv.json").write_text(jdump(res))
    reset_provider(None)
    return res


def stage_equiv(a: argparse.Namespace) -> None:
    res = asyncio.run(equiv(a))
    print(jdump(res))
    sys.exit(0 if res["all_identical"] else 1)


def stage_plan(_a: argparse.Namespace) -> None:
    from crucible.factory.benchmark import METRICS_PLAN

    print(jdump(METRICS_PLAN))


def stage_cleanup(_a: argparse.Namespace) -> None:
    names = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True, check=False).stdout.split()
    mine = [n for n in names if n.startswith(PREFIXES)]
    for n in mine:
        subprocess.run(["docker", "rm", "-f", n], capture_output=True, check=False)
    print(jdump({"removed": mine}))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["build", "verify", "rows", "equiv", "plan", "cleanup"])
    p.add_argument("--heldout", help="the held-out window store (store3/heldout)")
    p.add_argument("--train", help="the training window store (store3/train)")
    p.add_argument("--prompts", help="authored training prompts: <dir>/<window id>/prompt.json (P2's lhwork/p2/windows)")
    p.add_argument("--out", help="output dir (build, verify, equiv) or rows file (rows)")
    p.add_argument("--manifest", help="rows: the benchmark manifest whose ids are sealed")
    p.add_argument("--train-filter", default="trainable", choices=["trainable", "all"])
    p.add_argument("--time-policy", default="strict", choices=["strict", "report"])
    p.add_argument("--model-cutoff", default=None)
    p.add_argument("--only", nargs="*", help="build: only these held-out window ids (a partial build)")
    p.add_argument("--no-gates", action="store_true")
    p.add_argument("--reruns", type=int, default=60)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--agent", default="claude_code")
    p.add_argument("--window", default="parse-w07c2ff9-44e14b0")
    p.add_argument("--every", type=int, default=3)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--agent-timeout", type=float, default=900.0)
    p.add_argument("--rollout-timeout", type=float, default=2400.0)
    a = p.parse_args()
    {"build": stage_build, "verify": stage_verify, "rows": stage_rows, "equiv": stage_equiv, "plan": stage_plan, "cleanup": stage_cleanup}[a.stage](a)


if __name__ == "__main__":
    main()
