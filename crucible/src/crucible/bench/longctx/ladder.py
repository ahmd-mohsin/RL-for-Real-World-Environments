"""Run the long-context ladder: per (family, size), the scripted RLM solver (expected 1.0), the gold answer submitted by a
scripted native policy (the grader's positive control, 1.0) and a no-op (0). The solver goes through the RLM harness, so
the rung also measures what reading a document of that size through World calls costs the harness (calls, wall time)."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any


async def _run(task: Any, policy: Any, harness: Any) -> dict[str, Any]:
    from crucible.runtime.run import Runner, RunSpec

    root = Path(tempfile.mkdtemp(prefix="longctx-run-"))
    t0 = time.perf_counter()
    try:
        s = await Runner(root).run(RunSpec(run_id="lc", tasks=[task], policies=[policy], harnesses=[harness], k=1))
        last = max((root / "lc" / "samples").glob("*/a*"))
        rec = json.loads((last / "record.json").read_text())
        return {"reward": s.mean_reward, "steps": rec.get("steps"), "wall_s": round(time.perf_counter() - t0, 2),
                "score": ((rec.get("grade") or {}).get("meta") or {}).get("score"), "failure": (rec.get("failure") or {}).get("kind")}
    finally:
        shutil.rmtree(root, ignore_errors=True)


async def run_rung(family: str, tokens: int, *, seed: int = 0, kernel: str = "docker") -> dict[str, Any]:
    from crucible.actors.policy import EpisodeLimits, ScriptedPolicy
    from crucible.bench.longctx.oracle import LongCtxSolver
    from crucible.bench.longctx.task import LongCtxTask
    from crucible.core.schema.actions import ToolCall
    from crucible.runtime.harness.native import NativeHarness, PolicyRef
    from crucible.runtime.harness.rlm import RLMConfig, RLMHarness

    t0 = time.perf_counter()
    task = LongCtxTask(family, seed, tokens)
    gen_s = round(time.perf_counter() - t0, 2)
    limits = EpisodeLimits(max_steps=10_000, max_tool_errors=None)
    try:
        solver = LongCtxSolver(family)
        rlm = RLMHarness(RLMConfig(kernel=kernel, allow_host=kernel == "local", temperature=0.0, cell_timeout_s=600.0), limits=limits)
        out = {"family": family, "tokens": tokens, "seed": seed, "chars": len(task.item.context), "generate_s": gen_s, "meta": task.item.meta,
               # M10 lab audit R-4: each scripted policy declares what it is (PolicyRef.config); the family names the solver's script
               "solver": await _run(task, PolicyRef(id="scripted:longctx-solver", make=lambda a: {"agent": solver}, config={"script": "longctx solver", "family": family}), rlm)}
        gold = [ToolCall(tool_call_id=a["tool_call_id"], function_name=a["function_name"], arguments=a["arguments"]) for a in task.oracle_actions()]
        native = NativeHarness("native:scripted", limits=limits)
        out["gold"] = await _run(task, PolicyRef(id="scripted:gold", make=lambda a: {"agent": ScriptedPolicy("agent", list(gold))}, config={"script": "gold answer"}), native)
        out["noop"] = await _run(task, PolicyRef(id="scripted:noop", make=lambda a: {"agent": ScriptedPolicy("agent", [])}, config={"script": "noop"}), native)
        out["controls_ok"] = out["solver"]["reward"] == 1.0 and out["gold"]["reward"] == 1.0 and out["noop"]["reward"] == 0.0
        return out
    finally:
        task.cleanup()


async def run_longctx_ladder(families: list[str], sizes: list[int], *, seed: int = 0, kernel: str = "docker", out: Path | None = None,
                             log: Any = print) -> dict[str, Any]:
    rep: dict[str, Any] = {"ladder": "long-context (M16 §4)", "runs": [], "summary": []}
    for size in sizes:
        for fam in families:
            r = await run_rung(fam, size, seed=seed, kernel=kernel)
            rep["runs"].append(r)
            line = {"family": fam, "tokens": size, "solver": r["solver"]["reward"], "gold": r["gold"]["reward"], "noop": r["noop"]["reward"],
                    "solver_world_calls": r["solver"]["steps"], "solver_wall_s": r["solver"]["wall_s"], "controls_ok": r["controls_ok"]}
            rep["summary"].append(line)
            log(f"[longctx] {line}")
            if out is not None:
                Path(out).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    return rep
