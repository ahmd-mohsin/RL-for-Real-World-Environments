"""M9 exit gate G1 (research m9-m10/01 §15.9): the toy tasks pass the *full* contract.

The 5 forged ``toy/helpdesk_v2`` tasks and the 6 G-hindsight tasks are accepted with every check (A1-A13, S1):
mutation harness on the oracle's tool calls (A3), 8 identical full verdicts (A4), the containment audit and
out-of-contract tamper attempts (A6), two fresh starts (A9), frontier calibration runs through the M10 runner (A10,
paid, small budget), the B.5 injections on each task's runtime path (A11), and the policy-cheat attacks with controls (A13).

    python -m crucible_pack_toy.accept_gate [--budget 0.02] [--no-paid]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from crucible.accept.model import Need
from crucible.accept.registry.model import load_registry
from crucible.accept.registry.run import power_tests
from crucible.accept.run import accept_task
from crucible_pack_toy import PACK_ROOT

CRUCIBLE = PACK_ROOT.parent.parent
REPORT = CRUCIBLE / "docs" / "modules" / "reports" / "m9_toy_g1.json"


async def tasks() -> list[tuple[Any, Any]]:
    from crucible_pack_toy.forge.hindsight import forge_hindsight
    from crucible_pack_toy.forge.world_synth import forge

    col: list = []
    await forge(collect=col)
    await forge_hindsight(collect=col)
    return col


async def main(budget: float = 0.02, paid: bool = True) -> dict:
    needs = [Need.DOCKER, *([Need.PAID] if paid else [])]
    rows, spend = {}, 0.0
    for t, resolve in await tasks():
        rep = await accept_task(t, needs=needs, cache_root=CRUCIBLE / "data" / "accept", resolve=resolve, budget_usd=budget)
        r = rep.results
        mut = r["A3"].measured.get("mutation", {}) if "A3" in r else {}
        a10 = r["A10"].measured if "A10" in r else {}
        rows[t.task_id] = {"accepted": rep.accepted, "blocking": rep.blocking(), "key": rep.key_digest,
                           "statuses": {k: str(v.status) for k, v in sorted(r.items())},
                           "reasons": {k: v.reasons for k, v in r.items() if v.reasons},
                           "mutation": {k: mut.get(k) for k in ("score", "killed", "equivalent", "per_family")} | {"survivors": mut.get("survivors")},
                           "attacks": r["A13"].measured.get("attacks") if "A13" in r else None,
                           "tamper": r["A6"].measured.get("tamper") if "A6" in r else None,
                           "a10": {k: a10.get(k) for k in ("split", "band", "p", "interval", "n")}, "new_grades": rep.stats.get("new_grades")}
    reg = load_registry()
    pt = await power_tests(reg)
    out = {"gate": "M9 G1: toy tasks pass the full contract (A1-A13, S1)", "registry": reg.version(),
           "power_tests": {"proven": sum(r["status"] == "proven" for r in pt), "per_kit": sum(r["status"] == "per_kit" for r in pt),
                           "not_proven": [r["id"] for r in pt if r["status"] not in ("proven", "per_kit")]},
           "tasks": rows, "accepted": sum(v["accepted"] for v in rows.values()), "total": len(rows), "budget_per_task_usd": budget, "spend_usd": spend}
    out["pass"] = out["accepted"] == out["total"]
    REPORT.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({"pass": out["pass"], "accepted": f"{out['accepted']}/{out['total']}", "power_tests": out["power_tests"],
                      "blocking": {k: v["blocking"] for k, v in rows.items() if v["blocking"]}}, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.02)
    ap.add_argument("--no-paid", action="store_true")
    a = ap.parse_args()
    asyncio.run(main(a.budget, paid=not a.no_paid))
