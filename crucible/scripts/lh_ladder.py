"""M16 validation ladders (docs/research/long-horizon-rlm/04-design.md §4).

Step ladder: the toy backlog at ~100 / 500 / 1,000 / 2,500 / 5,000 World calls, worked by scripted RLM models with no
hidden memory (``crucible.runtime.scripted_rlm``), one JSON report per invocation
(``crucible_pack_toy.longhorizon.ladder``). Long-context ladder (``--longctx``): the oolong / MRCR / long-document
families at the given rungs, each run with its scripted oracle and a no-op through the World ``docs`` app.

    python scripts/lh_ladder.py --rungs 100,500,1000 --variants faithful,forgetful --budget 32000 --out ladder.json
    python scripts/lh_ladder.py --rungs 1000 --variants faithful --rlm window_tokens=32000,tail_tokens=8000 --out new.json
    python scripts/lh_ladder.py --rungs 1000 --variants faithful --faults crash_at_call=100 --checkpoint-every 500 --out crash.json
    python scripts/lh_ladder.py --longctx --families oolong,mrcr,docqa --sizes 8000,128000 --out longctx.json

``--rlm k=v,...`` sets ``RLMConfig`` fields by name (fields the installed harness lacks are listed as ``ignored`` in the
report). ``--kernel local`` runs cells on the host (trusted scripted code only); the default is the Docker kernel.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


def _kv(s: str) -> dict:
    out: dict = {}
    for part in filter(None, (s or "").split(",")):
        k, v = part.split("=", 1)
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


async def main(a: argparse.Namespace) -> None:
    if a.longctx:
        from crucible.bench.longctx.ladder import run_longctx_ladder

        rep = await run_longctx_ladder([f for f in a.families.split(",") if f], [int(x) for x in a.sizes.split(",") if x], seed=a.seed,
                                       kernel=a.kernel, out=Path(a.out) if a.out else None)
        print(json.dumps(rep["summary"], indent=1))
        return
    from crucible_pack_toy.longhorizon.ladder import LadderConfig, run_ladder

    base = LadderConfig(seed=a.seed, harness=a.harness, tito=None if a.tito == "none" else a.tito, kernel=a.kernel, budget_tokens=a.budget,
                        trigger=a.trigger, keep_recent_turns=a.keep, summary_max_tokens=a.summary_max_tokens, enforce_window=a.enforce_window,
                        checkpoint_every=a.checkpoint_every, restore_probe=not a.no_restore_probe, model_kwargs=_kv(a.model), rlm=_kv(a.rlm),
                        faults=_kv(a.faults))
    rep = await run_ladder([int(x) for x in a.rungs.split(",") if x], [v for v in a.variants.split(",") if v], base=base,
                           out=Path(a.out) if a.out else None)
    if not a.out:
        print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rungs", default="100,500,1000")
    ap.add_argument("--variants", default="faithful,forgetful")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--harness", default="rlm", choices=["rlm", "native"])
    ap.add_argument("--tito", default="qwen", choices=["qwen", "byte", "none"])
    ap.add_argument("--kernel", default="docker", choices=["docker", "local"])
    ap.add_argument("--budget", type=int, default=32_000)
    ap.add_argument("--trigger", type=float, default=0.8)
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--summary-max-tokens", type=int, default=2_000)
    ap.add_argument("--enforce-window", action="store_true", help="the engine refuses prompts longer than --budget (a served model's window)")
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--no-restore-probe", action="store_true")
    ap.add_argument("--model", default="", help="BacklogRLM knobs, k=v,... (per_cell, print_chars, replies, max_waits)")
    ap.add_argument("--rlm", default="", help="RLMConfig fields, k=v,...")
    ap.add_argument("--faults", default="", help="attempt-1 faults, k=v,... (crash_at_call, overflow_summaries, silent, kill_at_world_call)")
    ap.add_argument("--longctx", action="store_true")
    ap.add_argument("--families", default="oolong,mrcr,docqa")
    ap.add_argument("--sizes", default="8000,32000,128000")
    ap.add_argument("--out", default="")
    asyncio.run(main(ap.parse_args()))
