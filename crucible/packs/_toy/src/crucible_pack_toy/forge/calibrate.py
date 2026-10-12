"""Calibrator role for the forged toy tasks (DESIGN §7.2, §7.4): pass rates per (policy, harness), learnability and the
generator-knob advice.

    python -m crucible_pack_toy.forge.calibrate [--model smoke-cheap] [--k 4] [--budget 0.05]

Each forged task is attempted ``k`` times by an LLM tool-calling policy in a fresh World; the binary outcome comes
from the task's own M7 plan (reward 1 = solved). ``learnability.measure`` samples 4 -> 8 -> 16 until the band is
decided (here capped at ``k``), ``advise`` names the knob direction. The rates go to
``docs/modules/reports/m8_toy_calibration.json``; they would go to each card's ``value_proof.frontier``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from crucible.actors.llm import LLMClient
from crucible.actors.policy import EpisodeLimits, LLMToolPolicy, run_episode
from crucible.forge.learnability import BANDS, advise, estimate, learnability
from crucible.forge.schema import CheckSet, TaskSpec
from crucible.verify.runtime.plan import Context
from crucible.verify.runtime.trusted import Trusted
from crucible_pack_toy import PACK_ROOT
from crucible_pack_toy.forge.task import ToyForgedTask

CRUCIBLE = PACK_ROOT.parent.parent


async def attempt(task: ToyForgedTask, llm: LLMClient, k: int, sample: int) -> tuple[bool | None, dict]:
    d = Path(tempfile.mkdtemp(prefix="calib-"))
    w = task._world(d)
    await w.start()
    try:
        await w.reset()
        pol = LLMToolPolicy("agent", llm, temperature=0.7)
        pol.begin_episode(f"{task.task_id}#{sample}")  # separates samples in the record/replay cache
        res = await run_episode(w, {"agent": pol}, limits=EpisodeLimits(max_steps=24))
        rep = await task.plan.arun(Context(Trusted(d / "grade"), params={"view": w.verifier_view()}))
        masked = bool(res.masked) or (rep.infra or {}).get("status") == "masked"
        # a masked episode or grade is no verdict (None), never a failure (M8 audit: it lowered p and steered "easier")
        return (None if masked else rep.reward == 1.0), {"steps": res.steps, "stop": res.stop_condition, "failing": rep.failing_gates(), "masked": masked}
    finally:
        await w.close()


async def calibrate(forge_dir: Path, model: str, k: int, budget: float) -> dict:
    llm = LLMClient(model, budget_usd=budget)
    rows = {}
    for td in sorted((forge_dir / "tasks").iterdir()):
        spec_doc = json.loads((td / "crucible" / "checks.json").read_text())
        cs = CheckSet.model_validate(spec_doc)
        acts = [json.loads(x) for x in (td / "solution" / "actions.jsonl").read_text().splitlines() if x.strip()]
        import yaml

        card = yaml.safe_load((td / "crucible" / "card.yaml").read_text())
        sc = json.loads((forge_dir / "forge" / "scenario.json").read_text())
        spec = TaskSpec(task_id=cs.task_id, title=card.get("title", ""), instruction=(td / "instruction.md").read_text().strip(), properties=[])
        task = ToyForgedTask(spec, cs, acts, [], pack_root=forge_dir, world_file="worlds/helpdesk_v2.yaml", scenario=sc)
        outcomes, logs = [], []
        for i in range(k):
            ok, info = await attempt(task, llm, k, i)
            outcomes.append(ok)
            logs.append(info)
        rate = estimate(outcomes, model=model, harness="crucible.LLMToolPolicy@1", task_hash=task.task_hash)
        moves = advise(rate, BANDS["train"], final=k >= 16)  # the 4 -> 8 -> 16 schedule: act on the point estimate only at 16
        rows[td.name] = {"p": rate.p, "ci": [rate.lo, rate.hi], "n": rate.n, "masked": rate.masked, "learnability": round(learnability(rate.p), 3),
                         "advice": [m.__dict__ for m in moves], "attempts": logs}
    return {"model": model, "harness": "crucible.LLMToolPolicy@1", "k": k, "band": "train p in [0.1, 0.7]", "tasks": rows, "spend_usd": round(llm.spend.usd, 6)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forge-dir", type=Path, default=CRUCIBLE / "data" / "forge" / "toy_helpdesk_v2_s7")
    ap.add_argument("--model", default="smoke-cheap")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--budget", type=float, default=0.05)
    a = ap.parse_args()
    rep = asyncio.run(calibrate(a.forge_dir, a.model, a.k, a.budget))
    (CRUCIBLE / "docs" / "modules" / "reports" / "m8_toy_calibration.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps({"model": rep["model"], "spend_usd": rep["spend_usd"],
                      "tasks": {k: (v["p"], v["ci"], v["learnability"], [m["direction"] for m in v["advice"]], [x["failing"] for x in v["attempts"]])
                                for k, v in rep["tasks"].items()}}, indent=1, default=str))


if __name__ == "__main__":
    main()
