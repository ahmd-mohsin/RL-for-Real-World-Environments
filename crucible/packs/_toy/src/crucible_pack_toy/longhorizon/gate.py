"""Long-horizon gate (DESIGN §10.4): one ``xl`` backlog episode end to end through the M10 runner.

The real ``LLMToolPolicy`` (driven API-free by ``ReplayLLM``) works a 1,200-ticket shift with a 128k-token context:
it compacts whenever the context reaches 80% of the budget (the policy writes each summary), the World checkpoints
every 500 steps, the episode is crashed mid-way (a lost worker on attempt 1) and resumes from the last checkpoint, and
the dense progress stream is checked to telescope to the final reward.

    python -m crucible_pack_toy.longhorizon.gate [--n 650] [--budget 128000] [--crash 2500]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

from crucible.actors.policy import ContextConfig, EpisodeLimits, LLMToolPolicy
from crucible.core.schema.rollout import RolloutRecord
from crucible.runtime.harness.native import NativeHarness, PolicyRef
from crucible.runtime.inject import ReplayLLM
from crucible.runtime.run import Runner, RunSpec
from crucible_pack_toy import PACK_ROOT
from crucible_pack_toy.longhorizon.task import BacklogTask
from crucible_pack_toy.longhorizon.world import PRESETS, diligent

REPORT = PACK_ROOT.parent.parent / "docs" / "modules" / "reports" / "longhorizon_xl.json"


async def main(n: int = PRESETS["xl"], budget: int = 128_000, crash: int = 3500, every: int = 500) -> dict:
    task = BacklogTask(0, n)
    acts = diligent(task.b)

    def make(attempt):
        c = (crash, ConnectionError("worker lost")) if crash and attempt.n == 1 else None
        return {"agent": LLMToolPolicy("agent", ReplayLLM(acts, crash_at=c), context=ContextConfig(budget_tokens=budget))}

    root = Path(tempfile.mkdtemp(prefix="lh-gate-"))
    t0 = time.monotonic()
    # M10 lab audit R-4: the replayed script and the fault it crashes at are what this policy is (PolicyRef.config)
    s = await Runner(root).run(RunSpec(run_id="xl", tasks=[task], policies=[PolicyRef(id="replay:diligent", make=make,
                                                                                      config={"script": "diligent replay", "crash_at_call": crash, "budget_tokens": budget})],
                                       harnesses=[NativeHarness(limits=EpisodeLimits(max_steps=1_000_000, max_tool_errors=None), checkpoint_every=every)], k=1))
    secs = round(time.monotonic() - t0, 1)
    adirs = sorted((root / "xl" / "samples").glob("*/a*"))
    last = adirs[-1]
    rec = json.loads((last / "record.json").read_text())
    ro = RolloutRecord.model_validate_json((last / "rollout.json").read_text())
    first = json.loads((adirs[0] / "record.json").read_text())
    comps = [x for x in ro.turns if x.kind == "compaction"]
    traj_tokens = ro.trajectory_tokens or 0
    out = {
        "gate": "long-horizon xl: >1M-token trajectory, compaction, checkpoint + resume, dense progress",
        "tickets": n, "context_budget_tokens": budget, "wall_s": secs, "reward": s.mean_reward, "attempts": s.attempts,
        "attempt1": {"failure": (first.get("failure") or {}).get("kind"), "steps": first.get("steps")},
        "resumed_from": rec["meta"]["artifacts"].get("resumed_from"), "steps_total": rec["steps"],
        "model_calls": len(ro.turns), "compactions": len(comps), "segments": len(ro.segments),
        "max_context_before_compaction": max((sg.context_before or 0) for sg in ro.segments) if ro.segments else None,
        "trajectory_tokens_attempt2": traj_tokens,
        "progress": rec["meta"].get("progress"), "credit_sum": round(sum(x.advantage or 0 for x in ro.turns), 6),
        "trace_events": sum(1 for _ in (last / "trace.jsonl").open()),
    }
    out["trajectory_tokens_episode"] = traj_tokens  # attempt 2 restores attempt 1's counter from the checkpoint: the whole episode
    out["pass"] = bool(s.mean_reward == 1.0 and out["trajectory_tokens_episode"] > 1_000_000 and out["compactions"] > 0
                   and out["max_context_before_compaction"] <= budget and not out["progress"]["drift"]
                   and abs(out["progress"]["sum_delta"] - 1.0) < 1e-6 and (not crash or (s.attempts == 2 and out["resumed_from"])))
    REPORT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=PRESETS["xl"])
    ap.add_argument("--budget", type=int, default=128_000)
    ap.add_argument("--crash", type=int, default=3500)
    a = ap.parse_args()
    asyncio.run(main(a.n, a.budget, a.crash))
