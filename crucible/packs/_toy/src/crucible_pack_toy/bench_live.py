"""M13 exit gate with real models, on the toy pack: a benchmark report for 2 frontier models and 1 open model with every
step metric (the coding tasks 01–04 part needs GPU Worlds; this is the toy part with paid models).

    python -m crucible_pack_toy.bench_live [--k 2] [--budget 2.5] [--models implementer-strong,reviewer-strong,open-default]
                                           [--rlm implementer-strong] [--only forged|helpdesk|backlog]

Tasks: the 5 forged ``toy/helpdesk_v2`` tasks (the build pipeline's, scripted roles), the helpdesk ``all_tickets`` task
and one 12-ticket backlog (long horizon, Φ per step, stage reach). Each model is an ``LLMToolPolicy`` under the native
harness. ``--rlm`` also runs that model under the RLM harness on the backlog (sandboxed kernel). Every client has a hard
USD budget, so the run stops spending at the cap. The report goes to ``docs/modules/reports/m13_toy_live/``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from crucible_pack_toy import PACK_ROOT

CRUCIBLE = PACK_ROOT.parent.parent
OUT = CRUCIBLE / "docs" / "modules" / "reports" / "m13_toy_live"


async def forged_tasks(work: Path) -> list:
    from crucible.studio.label import ingest
    from crucible.studio.pack import find_pack, load_manifest
    from crucible_pack_toy.studio import forge

    d = find_pack("toy")
    ingest(d, load_manifest(d), work / "store")
    out = []
    for t, _ in await forge(work / "forge", work / "store"):
        rt = t.runtime_task()
        rt.oracle = t.oracle
        out.append(rt)
    return out


async def main(a: argparse.Namespace) -> dict:
    from crucible.actors.llm import LLMClient
    from crucible.actors.policy import EpisodeLimits, LLMToolPolicy
    from crucible.bench.run import run_bench
    from crucible.runtime.harness.native import NativeHarness, PolicyRef
    from crucible.runtime.harness.rlm import RLMConfig, RLMHarness
    from crucible_pack_toy.longhorizon.task import BacklogTask
    from crucible_pack_toy.runtime import HelpdeskTask
    from crucible_pack_toy.studio import stages as toy_stages

    work = Path(tempfile.mkdtemp(prefix="m13-live-"))
    tasks: list = []
    if a.only in (None, "forged"):
        tasks += await forged_tasks(work)
    if a.only in (None, "helpdesk"):
        tasks.append(HelpdeskTask())
    backlog = None
    if a.only in (None, "backlog"):
        from crucible_pack_toy import studio

        backlog = BacklogTask(seed=a.seed, n=12)
        studio._MADE[backlog.id] = backlog
        tasks.append(backlog)
    clients = {m: LLMClient(m, budget_usd=a.budget, cache_mode="off") for m in a.models.split(",")}
    policies = [PolicyRef(id=f"llm:{c.model}", make=lambda att, c=c: {"agent": LLMToolPolicy("agent", c, temperature=a.temperature)},
                          spend=lambda c=c: (c.spend.usd, c.spend.uncertain_usd)) for c in clients.values()]
    try:
        res = await run_bench("toy-live", root=work / "runs", tasks=tasks, policies=policies,
                              harnesses=[NativeHarness(limits=EpisodeLimits(max_steps=a.max_steps))], k=a.k, ks=(1,), stages_of=toy_stages,
                              run_id="toy-live-native")
        out = {"native": res["paths"], "spend": {m: round(c.spend.usd, 4) for m, c in clients.items()}}
        if a.rlm and backlog is not None:
            rc = LLMClient(a.rlm, budget_usd=a.budget, cache_mode="off")
            h = RLMHarness(RLMConfig(temperature=a.temperature, budget_tokens=24_000, keep_recent_turns=2, max_turns=80), limits=EpisodeLimits(max_steps=a.max_steps))
            r2 = await run_bench("toy-live-rlm", root=work / "runs", tasks=[backlog], policies=[PolicyRef(id=f"llm:{rc.model}", make=lambda att: {"agent": rc},
                                                                                                          spend=lambda: (rc.spend.usd, 0.0))],
                                 harnesses=[h], k=1, stages_of=toy_stages, run_id="toy-live-rlm")
            out["rlm"] = r2["paths"]
            out["spend"]["rlm:" + a.rlm] = round(rc.spend.usd, 4)
    finally:
        if backlog is not None:
            backlog.cleanup()
    OUT.mkdir(parents=True, exist_ok=True)
    import shutil

    for key in ("native", "rlm"):
        if key in out:
            for f in ("summary.json", "report.html"):
                shutil.copy(out[key][f.split(".")[0] if f == "summary.json" else "report"], OUT / f"{key}_{f}")
    out["total_spend_usd"] = round(sum(out["spend"].values()), 4)
    (OUT / "run.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="implementer-strong,reviewer-strong,open-default")
    ap.add_argument("--rlm", default=None)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--budget", type=float, default=2.5, help="hard USD cap per model client")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--only", choices=["forged", "helpdesk", "backlog"], default=None)
    print(json.dumps(asyncio.run(main(ap.parse_args())), indent=1, default=str))
