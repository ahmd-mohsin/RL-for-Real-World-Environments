"""LLM-judge baseline for M6 gate S2 on SWE-rebench: judge each held-out attempt at the same midpoint where Φ is
measured (the prefix up to the middle tool call), 3 passes, budget-capped. Writes one JSONL row per attempt.

``python -m crucible_pack_coding.m56.rebench_judge <phi.jsonl> <out.jsonl> [budget_usd] [model_alias] [fraction]``
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from crucible.actors.llm import BudgetExceeded, LLMClient
from crucible.credit.judge import judge, render_prefix
from crucible.credit.validity import time_split
from crucible_pack_coding.corpora import swe_rebench as S
from crucible_pack_coding.m56.rebench_s2 import load


def prefix_at(trajectory: list[dict], frac: float) -> list[dict]:
    """Messages up to and including the tool result of the call at ``frac`` of the attempt's tool calls."""
    call_idx = [i for i, m in enumerate(trajectory) if m.get("role") == "assistant" and m.get("tool_calls")]
    if not call_idx:
        return trajectory
    cut_call = call_idx[min(len(call_idx) - 1, max(0, round(frac * len(call_idx)) - 1))]
    end = cut_call + 1
    while end < len(trajectory) and trajectory[end].get("role") == "tool":
        end += 1
    return trajectory[:end]


async def main(phi_path: Path, out: Path, budget: float, model: str, frac: float = 0.5) -> None:
    rows = [r for r in load(phi_path) if r["replay_agent_state"] and r["phi"]]
    created = {r["instance_id"]: r["created_at"] for r in rows}
    _, hold_inst = time_split(sorted(created), time=lambda i: created[i], holdout=0.3)
    hold = [r for r in rows if r["instance_id"] in set(hold_inst)]
    done = {json.loads(x)["trajectory_id"] for x in out.read_text().splitlines()} if out.exists() else set()
    todo = [r for r in hold if r["trajectory_id"] not in done]
    idx = {r["trajectory_id"]: r for r in S.index()}
    client = LLMClient(model, budget_usd=budget)
    print(f"{len(todo)} attempts to judge ({len(done)} done), model {client.model}, budget ${budget}", flush=True)
    sem = asyncio.Semaphore(8)
    stop = asyncio.Event()

    async def one(rec: dict) -> dict | None:
        async with sem:
            if stop.is_set():
                return None
            try:
                j = await judge(client, render_prefix(prefix_at(rec["trajectory"], frac)), key=f"{rec['trajectory_id']}@{frac}" if frac != 0.5 else rec["trajectory_id"])
            except BudgetExceeded as e:
                print("budget reached:", e)
                stop.set()
                return None
            return {"trajectory_id": rec["trajectory_id"], "p_success": j.p_success, "passes": j.passes, "spread": j.spread,
                    "cost_usd": round(j.cost_usd, 6), "resolved": rec["resolved"], "error": j.error, "model": client.model}

    recs = list(S.load([idx[r["trajectory_id"]] for r in todo], columns=("trajectory_id", "trajectory", "resolved")))
    with out.open("a") as f:
        for k, coro in enumerate(asyncio.as_completed([one(r) for r in recs])):
            row = await coro
            if row is not None:
                f.write(json.dumps(row) + "\n")
                f.flush()
            if k % 20 == 0:
                print(f"{k}/{len(recs)} judged, spent ${client.spend.usd:.3f}", flush=True)
    print(f"done; spent ${client.spend.usd:.3f} over {client.spend.calls} calls ({client.spend.cached_calls} cached)")


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 5.0,
                     sys.argv[4] if len(sys.argv) > 4 else "judge-default", float(sys.argv[5]) if len(sys.argv) > 5 else 0.5))
