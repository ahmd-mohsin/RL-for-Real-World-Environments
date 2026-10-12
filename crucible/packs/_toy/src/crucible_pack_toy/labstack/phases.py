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
        from crucible.core.schema.rollout import RolloutRecord

        valid, why = 0, {}
        for p in rolls:  # the trainer's own test, not a looser one (eval-mode records are checked the same way)
            ok, reason = RolloutRecord.model_validate_json(p.read_text()).valid_for_training()
            valid += int(ok)
            if not ok:
                why[reason] = why.get(reason, 0) + 1
        out[mode] = {"planned": s.planned, "terminal": s.terminal, "scored": s.scored, "masked": s.masked_terminal, "mean_reward": s.mean_reward,
                     "pass_at_1": s.pass_at_1, "stopped": s.stopped, "by_failure": s.by_failure, "rollouts": len(rolls), "train_valid_rollouts": valid, "not_train_valid": why,
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
    from crucible.connectors.file_drop import FileDropConnector
    from crucible.connectors.redact import RedactionPolicy
    from crucible.connectors.trace_store import TraceStore
    from crucible.core.store import Store
    from crucible_pack_toy import PACK_ROOT

    root = PACK_ROOT.parent.parent / "data" / "m3" / "store"
    ts = TraceStore(Store(root))
    from crucible_pack_toy.m4.gate import TICKET_FIELDS

    conn = FileDropConnector(TICKET_FIELDS)
    refs = list(conn.discover({"glob": str(PACK_ROOT / "data" / "ticket_logs" / "*.jsonl")}))
    keys = ingest(conn, refs, ts, policy=RedactionPolicy())
    again = ingest(conn, refs, ts, policy=RedactionPolicy())
    toy = [i for i in ts.ids() if i.startswith("toy/support_ticket:")]
    return {"refs": len(refs), "traces": len(set(keys)), "toy_traces": len(toy), "idempotent": sorted(keys) == sorted(again), "pass": bool(toy) and sorted(keys) == sorted(again)}


async def tito() -> dict:
    """Token-in-token-out drift through the chat API, on M10's own path (Runner + NativeHarness + the pack's LLM
    policy, eval mode so nothing is masked): for each call after the first, does the engine's prompt extend the previous
    prompt + completion? On the first break, decode both sides around the divergence (what the server's chat template
    re-rendered vs what the model sampled)."""
    from transformers import AutoTokenizer

    from crucible.actors.llm import LLMClient
    from crucible.core.schema.rollout import RolloutRecord
    from crucible.runtime.harness.native import NativeHarness
    from crucible.runtime.run import Runner, RunSpec
    from crucible_pack_toy.runtime import HelpdeskTask, llm_policy

    root = Path(tempfile.mkdtemp(prefix="labstack-tito-"))
    llm = LLMClient("policy-default", cache_mode="off")
    await Runner(root).run(RunSpec(run_id="lab-tito", tasks=[HelpdeskTask()], policies=[llm_policy(llm, temperature=0.7)], harnesses=[NativeHarness()],
                                   k=1, concurrency=1, rollout_mode="eval", sampling={"temperature": 0.7}))
    tok = AutoTokenizer.from_pretrained(llm.model)
    out = {"rollouts": 0, "calls": 0, "breaks": 0, "first_break": None}
    for f in root.rglob("rollout.json"):
        ro = RolloutRecord.model_validate_json(f.read_text())
        out["rollouts"] += 1
        prev: dict = {}
        for t in ro.turns:
            out["calls"] += 1
            key = (t.actor, t.segment)
            full = (prev[key] + list(t.prompt_token_ids or [])) if (t.prompt_is_delta and key in prev) else list(t.prompt_token_ids or [])
            if t.prefix_break and key in prev:
                out["breaks"] += 1
                if out["first_break"] is None:
                    pv = prev[key]
                    k = next((j for j in range(min(len(pv), len(full))) if pv[j] != full[j]), min(len(pv), len(full)))
                    out["first_break"] = {"call": t.call_id, "diverge_token": k, "prev_len": len(pv), "next_len": len(full),
                                          "sampled_side": tok.decode(pv[max(0, k - 40): k + 40]), "rerendered_side": tok.decode(full[max(0, k - 40): k + 40]),
                                          "sampled_ids": pv[k: k + 8], "rerendered_ids": full[k: k + 8]}
            prev[key] = full + list(t.completion_token_ids or [])
    out["pass"] = out["calls"] > 1 and out["breaks"] == 0
    return out


def main() -> None:
    which, out = sys.argv[1], Path(sys.argv[2])
    res = asyncio.run({"probe": probe, "m10": m10, "m11": m11, "m3": m3_toy_ingest, "tito": tito}[which]())
    out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, default=str)[:2000])


if __name__ == "__main__":
    main()
