"""Lab-campaign phases without an existing CLI gate.

    python -m crucible_pack_toy.labstack.phases probe  OUT.json   # the served model returns a tool call
    python -m crucible_pack_toy.labstack.phases m10    OUT.json   # Runner: k samples of the LLM policy on the toy tasks
    python -m crucible_pack_toy.labstack.phases m11    OUT.json   # an LLM trajectory replayed → the live grade
    python -m crucible_pack_toy.labstack.phases m3     OUT.json   # M3 toy ingest (the trace store M4 / M8 / M9 read)
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path


async def probe() -> dict:
    from crucible.actors.llm import LLMClient

    llm = LLMClient("policy-default", cache_mode="off")
    tools = [{"type": "function", "function": {"name": "get_ticket", "description": "Fetch a ticket by id.",
                                               "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}}}]
    t0 = time.monotonic()
    r = await llm.complete([{"role": "user", "content": "Fetch ticket 2."}], tools=tools, max_tokens=256, temperature=0.0)
    return {"model": llm.model, "tool_call": bool(r.tool_calls), "tool_calls": r.tool_calls, "text": (r.text or "")[:400],
            "token_record": bool(r.tokens and r.tokens.get("completion_token_ids")), "latency_s": round(time.monotonic() - t0, 2)}


async def m10() -> dict:
    from crucible.actors.harness import BashLoopPolicy  # noqa: F401  (import check: the native harness stack loads)
    from crucible.actors.llm import LLMClient
    from crucible.runtime.harness.native import NativeHarness
    from crucible.runtime.run import Runner, RunSpec
    from crucible_pack_toy.runtime import FibTask, HelpdeskTask, llm_policy

    out: dict = {}
    root = Path(tempfile.mkdtemp(prefix="labstack-m10-"))
    llm = LLMClient("policy-default", cache_mode="off")
    for mode in ("eval", "train"):
        t0 = time.monotonic()
        spec = RunSpec(run_id=f"lab-m10-{mode}", tasks=[HelpdeskTask()], policies=[llm_policy(llm, temperature=0.7)], harnesses=[NativeHarness()],
                       k=4, concurrency=4, rollout_mode=mode, sampling={"temperature": 0.7})
        s = await Runner(root / mode).run(spec)
        rolls = list((root / mode).rglob("rollout.json"))
        valid = 0
        for p in rolls:
            d = json.loads(p.read_text())
            valid += int(not d.get("mask_sample") and d.get("reward") is not None and all(t.get("completion_token_ids") for t in d.get("turns", [])))
        out[mode] = {"planned": s.planned, "terminal": s.terminal, "scored": s.scored, "masked": s.masked_terminal, "mean_reward": s.mean_reward,
                     "pass_at_1": s.pass_at_1, "stopped": s.stopped, "by_failure": s.by_failure, "rollouts": len(rolls), "train_valid_rollouts": valid,
                     "seconds": round(time.monotonic() - t0, 1)}
    _ = FibTask
    # rc=0 is not a pass: run 4 "passed" with every rollout masked (agent_run_error). Both modes must score, and train
    # mode must yield token-complete rollouts.
    out["pass"] = all(out[m]["scored"] > 0 for m in ("eval", "train")) and out["train"]["train_valid_rollouts"] > 0
    return out


async def m11() -> dict:
    """The runtime-independence contract on an LLM trajectory: replaying the live episode's actions in a fresh World
    reproduces the live grade and the final state fingerprint."""
    from crucible.actors.llm import LLMClient
    from crucible.actors.policy import EpisodeLimits, LLMToolPolicy, run_episode
    from crucible.interop.replay import from_world_trace, replay
    from crucible_pack_toy.runtime import HelpdeskTask

    task = HelpdeskTask()
    d = Path(tempfile.mkdtemp(prefix="labstack-m11-"))
    w = await task.open(d / "live")
    try:
        llm = LLMClient("policy-default", cache_mode="off")
        await run_episode(w, {"agent": LLMToolPolicy("agent", llm, temperature=0.7)}, limits=EpisodeLimits(max_steps=40), reset=False)
        live = await task.grade(w, d / "g")
        trace = w.to_trace()
        from crucible.interop.replay import _fingerprint

        live_fp = await _fingerprint(w)
    finally:
        await w.close()
    acts = from_world_trace(trace)
    rep = await replay(task, acts, d / "replay", episode_key=f"{task.id}|lab")
    return {"actions": len(acts), "live_reward": live.reward, "replay_reward": rep["reward"], "fingerprint_equal": rep["fingerprint"] == live_fp,
            "steps_rows": rep["steps"], "pass": rep["reward"] == live.reward and rep["fingerprint"] == live_fp}


async def m3_toy_ingest() -> dict:
    """M3 on the lab: ingest the toy ticket logs (file-drop connector, redaction on) into the trace store the M4 signals,
    M8 world synthesis and M9 acceptance read (``data/m3/store``); twice, to check idempotency."""
    from crucible.connectors.base import ingest
    from crucible.connectors.file_drop import FieldMap, FileDropConnector
    from crucible.connectors.redact import RedactionPolicy
    from crucible.connectors.trace_store import TraceStore
    from crucible.core.store import Store
    from crucible_pack_toy import PACK_ROOT

    root = PACK_ROOT.parent.parent / "data" / "m3" / "store"
    ts = TraceStore(Store(root))
    conn = FileDropConnector(FieldMap(family="toy/support_ticket", extra=["team"]))
    refs = list(conn.discover({"glob": str(PACK_ROOT / "data" / "ticket_logs" / "*.jsonl")}))
    keys = ingest(conn, refs, ts, policy=RedactionPolicy())
    again = ingest(conn, refs, ts, policy=RedactionPolicy())
    toy = [i for i in ts.ids() if i.startswith("toy/support_ticket:")]
    return {"refs": len(refs), "traces": len(set(keys)), "toy_traces": len(toy), "idempotent": sorted(keys) == sorted(again), "pass": bool(toy) and sorted(keys) == sorted(again)}


def main() -> None:
    which, out = sys.argv[1], Path(sys.argv[2])
    res = asyncio.run({"probe": probe, "m10": m10, "m11": m11, "m3": m3_toy_ingest}[which]())
    out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, default=str)[:2000])


if __name__ == "__main__":
    main()
