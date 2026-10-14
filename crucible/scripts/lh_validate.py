"""M16 validation campaign for the very-long-horizon RLM harness (docs/research/long-horizon-rlm/06-validation.md).

Runs E's ladders (``crucible_pack_toy.longhorizon.ladder``, ``crucible.bench.longctx``) against the M16 harness with its
default config (32k window, Docker kernel, real Qwen2.5 tokens through ``TitoLLM``) and writes one JSON per part:

    python scripts/lh_validate.py ladder   --out v_main.json      # rungs 100..5000 x faithful/forgetful/lazy/forgetful_reread
    python scripts/lh_validate.py window   --out v_window.json    # --enforce-window at 1000 (a served model's 32k window)
    python scripts/lh_validate.py faults   --out v_faults.json    # the 1000 rung with each fault on attempt 1
    python scripts/lh_validate.py longctx  --out v_longctx.json   # CONTEXT delivery through the RLM harness, 8k..512k

Faults: ``overflow_root`` (root calls 40 and 41 raise ContextOverflow), ``silent_root`` (a silent overflow at call 40),
``overflow_summary`` (the first summary attempt overflows), ``kill_mid_cell`` (the kernel is killed at World call 300),
``crash_resume`` (the worker is lost at model call 100; checkpoints every 500 steps), ``cells`` (a pathological regex in a
cell and in ``history.search``, and a 50 MB print, injected at root calls 60, 61 and 62).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

VARIANTS = ["faithful", "forgetful", "lazy", "forgetful_reread"]


def _summary(r: dict[str, Any]) -> dict[str, Any]:
    c = r.get("constraints") or {}
    return {"label": r["label"], "reward": r["reward"], "stop": r["stop"], "mask": r["mask"], "failure_kind": r["failure_kind"],
            "valid": (r.get("valid_for_training") or [None])[0], "world_calls": r["world_calls"], "compactions": r["compactions"],
            "lost": c.get("lost_constraint_rate"), "missed": c.get("missed_awaiting"), "b1": r["b1"], "wall_s": r["wall_s"],
            "checkpoints": r["checkpoints"], "restore": r["restore"], "attempts": r["attempts"], "attempt_failures": r["attempt_failures"],
            "history": r["history_reads_harness"], "faults": r.get("faults")}


async def ladder(a: argparse.Namespace) -> dict[str, Any]:
    from crucible_pack_toy.longhorizon.ladder import LadderConfig, run_ladder

    base = LadderConfig(kernel=a.kernel, enforce_window=a.enforce_window, checkpoint_every=a.checkpoint_every)
    rep = await run_ladder([int(x) for x in a.rungs.split(",")], a.variants.split(","), base=base, out=Path(a.out))
    rep["summary"] = [_summary(r) for r in rep["runs"]]
    return rep


def inject_cells(inner: Any, at: dict[int, str]) -> Any:
    """``inner`` (a scripted model) with fault cells spliced in at the given root-call indices; the log is shared, so the
    ladder still counts every call."""
    from crucible.runtime.scripted_rlm import Reply, ScriptedRLM

    class InjectCells(ScriptedRLM):
        def __init__(self) -> None:
            super().__init__(model=inner.model)
            self.log, self.n, self.fired = inner.log, 0, []

        def decide(self, view: Any) -> Any:
            if view.role == "root":
                k, self.n = self.n, self.n + 1
                if k in at:
                    self.fired.append({"call": k, "cell": at[k][:60]})
                    return Reply("fault_cell", "", at[k])
            return inner.decide(view)

    return InjectCells()


async def faults(a: argparse.Namespace) -> dict[str, Any]:
    from crucible_pack_toy.longhorizon import ladder as L

    from crucible.runtime import scripted_rlm as S
    from crucible.train import tito_scripted as T

    cases: dict[str, dict[str, Any]] = {
        "overflow_root": {"faults": {"overflow_at": [40, 41]}},
        "silent_root": {"faults": {"overflow_at": [40], "silent": True}},
        "overflow_summary": {"faults": {"overflow_summaries": 1}},
        "kill_mid_cell": {"faults": {"kill_at_world_call": 300}},
        "crash_resume": {"faults": {"crash_at_call": 100}, "checkpoint_every": 500},
        "cells": {"inject": {60: S.PATHOLOGICAL_CELL, 61: S.PATHOLOGICAL_HISTORY_CELL, 62: S.BIG_OUTPUT_CELL}},
    }
    only = set(a.cases.split(",")) if a.cases else set(cases)
    rep: dict[str, Any] = {"campaign": "M16 faults at the 1000 rung (faithful)", "runs": {}}
    orig = L.make_model
    for name, spec in cases.items():
        if name not in only:
            continue
        injected: list[Any] = []
        if "inject" in spec:
            def make_model(cfg: Any, codec: Any = None, _at=spec["inject"], injected=injected) -> Any:
                model, _ = orig(cfg, codec)
                inj = inject_cells(model, _at)
                injected.append(inj)
                return model, (lambda: T.TitoLLM(inj, codec))

            L.make_model = make_model
        try:
            cfg = L.LadderConfig(rung=1000, variant="faithful", kernel=a.kernel, faults=spec.get("faults", {}),
                                 checkpoint_every=spec.get("checkpoint_every", 500))
            t0 = time.perf_counter()
            r = await L.run_rung(cfg, codec=T.qwen_codec())
            r["case_s"] = round(time.perf_counter() - t0, 2)
            if injected:
                r["injected"] = injected[0].fired
            rep["runs"][name] = {**_summary(r), "injected": r.get("injected"), "rlm_counters": r.get("rlm_counters")}
        finally:
            L.make_model = orig
        print(f"[faults] {name}: reward {r['reward']} stop {r['stop']} mask {r['mask']} failure {r['failure_kind']} attempts {r['attempts']}")
        Path(a.out).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    return rep


async def longctx(a: argparse.Namespace) -> dict[str, Any]:
    import shutil
    import tempfile

    from crucible.actors.policy import EpisodeLimits, ScriptedPolicy
    from crucible.bench.longctx.oracle import LongCtxSolver
    from crucible.bench.longctx.task import LongCtxTask
    from crucible.core.schema.actions import ToolCall
    from crucible.runtime.harness.native import NativeHarness, PolicyRef
    from crucible.runtime.harness.rlm import RLMConfig, RLMHarness
    from crucible.runtime.run import Runner, RunSpec

    async def run(task: Any, ref: Any, harness: Any) -> dict[str, Any]:
        root = Path(tempfile.mkdtemp(prefix="lc-"))
        t0 = time.perf_counter()
        try:
            s = await Runner(root).run(RunSpec(run_id="lc", tasks=[task], policies=[ref], harnesses=[harness], k=1))
            d = max((root / "lc" / "samples").glob("*/a*"))
            rec = json.loads((d / "record.json").read_text())
            doc = json.loads((d / "rlm.agent.json").read_text()) if (d / "rlm.agent.json").exists() else {}
            return {"reward": s.mean_reward, "steps": rec.get("steps"), "wall_s": round(time.perf_counter() - t0, 2),
                    "failure": (rec.get("failure") or {}).get("kind"), "context_reads": (doc.get("counters") or {}).get("context_reads"),
                    "largest_prompt": (doc.get("context_health") or {}).get("largest_prompt")}
        finally:
            shutil.rmtree(root, ignore_errors=True)

    limits = EpisodeLimits(max_steps=10_000, max_tool_errors=None)
    rep: dict[str, Any] = {"ladder": "long-context via CONTEXT through the M16 RLM harness", "runs": []}
    for size in [int(x) for x in a.sizes.split(",")]:
        for fam in a.families.split(","):
            task = LongCtxTask(fam, a.seed, size, delivery="context")
            try:
                solver = LongCtxSolver(fam)
                h = RLMHarness(RLMConfig(kernel=a.kernel, allow_host=a.kernel == "local", temperature=0.0, cell_timeout_s=600.0), limits=limits)
                out = {"family": fam, "tokens": size, "chars": len(task.item.context),
                       "solver": await run(task, PolicyRef(id="scripted:longctx-solver", make=lambda at, solver=solver: {"agent": solver}), h)}
                gold = [ToolCall(tool_call_id=x["tool_call_id"], function_name=x["function_name"], arguments=x["arguments"]) for x in task.oracle_actions()]
                native = NativeHarness("native:scripted", limits=limits)
                out["gold"] = await run(task, PolicyRef(id="scripted:gold", make=lambda at, gold=gold: {"agent": ScriptedPolicy("agent", list(gold))}), native)
                out["noop"] = await run(task, PolicyRef(id="scripted:noop", make=lambda at: {"agent": ScriptedPolicy("agent", [])}), native)
                out["controls_ok"] = out["solver"]["reward"] == 1.0 and out["gold"]["reward"] == 1.0 and out["noop"]["reward"] == 0.0
            finally:
                task.cleanup()
            rep["runs"].append(out)
            print(f"[longctx] {fam} {size}: solver {out['solver']} gold {out['gold']['reward']} noop {out['noop']['reward']}")
            Path(a.out).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    return rep


async def main(a: argparse.Namespace) -> None:
    fn = {"ladder": ladder, "window": ladder, "faults": faults, "longctx": longctx}[a.part]
    if a.part == "window":
        a.enforce_window = True
    rep = await fn(a)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("part", choices=["ladder", "window", "faults", "longctx"])
    ap.add_argument("--rungs", default="100,500,1000,2500,5000")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--kernel", default="docker", choices=["docker", "local"])
    ap.add_argument("--enforce-window", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--cases", default="")
    ap.add_argument("--families", default="oolong,mrcr,docqa")
    ap.add_argument("--sizes", default="8000,32000,128000,512000")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    asyncio.run(main(ap.parse_args()))
