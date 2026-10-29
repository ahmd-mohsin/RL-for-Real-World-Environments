#!/usr/bin/env python
"""Smoke: the trace audit, the per-turn credit and the reward paths, on the CPU, no model, no paid call.

    python scripts/smoke/audit_credit.py --out <dir> [--episodes <debug_episode.py --out dir>/runs]

1. **Seeded trace audit** (``factory.traceaudit.run_seeded``, P3's exit criterion): a small real repository graded by local
   pytest, four planted cases (a lucky pass, an unstated public name, a broken World tool, a fake proposal); no auditor,
   so only the deterministic facts, first wrong steps, pass audits and the executed acceptance run. Every planted case
   must end as expected.
2. **Real episodes** (``--episodes``: the attempt dirs ``scripts/smoke/debug_episode.py`` wrote): each attempt loaded as an
   ``EpisodeBundle``; facts, the honest-trace checks, the first wrong step of each failure against the passing episodes,
   the pass audit of each pass. Then the reward and credit path on the same rollouts as one GRPO group: the outcome
   advantage R - mean (Dr.GRPO) with the honesty term (``factory.honesty.honesty_group_fn``), and the per-call process
   credit (``train.credit.turn_credit_map``). Checks: the oracle's advantage > 0 > the null's; each attempt's Φ deltas sum
   to Φ_end - Φ_0; a hack claiming success is flagged by the honest-trace checks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any


async def seeded(out: Path) -> dict[str, Any]:
    from crucible.factory.traceaudit import AcceptConfig, LoopConfig, PytestRunner, run_seeded

    cfg = LoopConfig(intent={"id": "I-seeded", "behaviour": "Durations parse and render"}, accept=AcceptConfig())  # the default 2 x 30 reruns: the flake bound needs them
    rep = await run_seeded(out, runner=PytestRunner(image=None), config=cfg)
    cases = {k: {x: v.get(x) for x in ("expected", "got", "ok")} for k, v in rep["seeded"].items()}
    return {"cases": cases, "ok": bool(rep["seeded_ok"])}


def episodes(root: Path) -> dict[str, Any]:
    from crucible.core.schema.rollout import RolloutRecord
    from crucible.factory.honesty import honesty_group_fn
    from crucible.factory.traceaudit import facts, first_wrong_step, honest_trace_facts, load_bundle, pass_audit
    from crucible.train.credit import turn_credit_map

    dirs = {p.parent.parent.parent.name.removeprefix("smoke-"): p for p in sorted(root.glob("smoke-*/samples/*/a1"))}
    if not dirs:
        return {"ok": None, "skipped": f"no attempts under {root}"}
    bundles = {n: load_bundle(d, run="smoke") for n, d in dirs.items()}
    passing = [b for b in bundles.values() if b.passed]
    res: dict[str, Any] = {}
    for n, b in bundles.items():
        res[n] = {"passed": b.passed, "masked": b.masked, "facts": [f.kind for f in facts(b)], "honest": [f.kind for f in honest_trace_facts(b)],
                  "first_wrong_step": first_wrong_step(b, [p for p in passing if p is not b]).kind if not b.passed else None,
                  "pass_audit": pass_audit(b) if b.passed else None}
    recs = {n: RolloutRecord.model_validate_json((d / "rollout.json").read_text()) for n, d in dirs.items() if (d / "rollout.json").exists()}
    names = list(recs)
    live = [recs[n] for n in names if not recs[n].mask_sample]
    for r in live:
        r.advantage = None
    honesty_group_fn()("g", live)
    credit = turn_credit_map([recs[n] for n in names], ["g"] * len(names), eta=0.5)
    for n in names:
        r = recs[n]
        deltas = [float(t.advantage or 0.0) for t in r.turns]
        b = bundles[n]
        res[n].update(reward=r.reward, advantage=r.advantage, honesty_term=(r.reward_components or {}).get("honesty/term"),
                      delta_phi_sum=round(sum(deltas), 6), phi=b.phi_final, phi0=b.phi0,
                      credit=[round(v, 4) for (rid, _c), v in credit.items() if rid == r.rollout_id])
    checks = {}
    if "oracle" in res and "null" in res:
        checks["oracle_adv_positive"] = (res["oracle"]["advantage"] or 0) > 0
        checks["null_adv_negative"] = (res["null"]["advantage"] or 0) < 0
    for n in names:
        if res[n].get("phi") is not None:
            checks[f"{n}_phi_telescopes"] = abs(res[n]["delta_phi_sum"] - (float(res[n]["phi"]) - float(res[n]["phi0"] or 0.0))) < 1e-6
    if "hack" in res:
        checks["hack_flagged"] = bool(res["hack"]["honest"]) or bool(res["hack"]["facts"])
    res["checks"] = checks
    res["ok"] = all(checks.values())
    return res


async def main_async(a: argparse.Namespace) -> dict[str, Any]:
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    res = {"seeded": await seeded(out / "seeded")}
    if a.episodes:
        res["episodes"] = episodes(Path(a.episodes))
    res["ok"] = all(v.get("ok") is not False for v in res.values() if isinstance(v, dict))
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", default="")
    a = p.parse_args()
    res = asyncio.run(main_async(a))
    print(json.dumps(res, indent=1, default=str))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
