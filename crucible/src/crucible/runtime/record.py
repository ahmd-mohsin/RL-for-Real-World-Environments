"""The producer side of ``crucible.rollout/v1`` (schema: ``crucible.train.rollout``). M10 writes one record per attempt
from the harness result, the policy's model calls, the ATIF trajectory and the dense ``steps.jsonl`` rows.

``mode="train"``: a rollout whose token record is incomplete is masked as ``crucible:capture_incomplete`` (infra) —
the trainer must never guess logprobs. ``mode="eval"``: token records are optional (hosted APIs return none) and
their absence changes nothing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from harbor.models.trajectories import Trajectory

from crucible.core.schema.reward import StepRow
from crucible.core.schema.run import RunRecord
from crucible.train.rollout import RolloutRecord, Segment, Turn


def credit_from_progress(calls: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]], actor: str) -> tuple[dict[str, float], float]:
    """Per-call credit from the dense progress stream: the k-th action this actor executed is the k-th progress row of the
    actor (one World step per action), and a call is credited with the ΔΦ of the actions it produced. Returns the credit
    per call id and the ΔΦ no call produced (end-of-episode drift corrections)."""
    acts = [c["call_id"] for c in calls for _ in (c.get("actions") or [])]
    mine = [r for r in rows if r.get("actor") == actor and r.get("step", 0) >= 0]
    out: dict[str, float] = {}
    for cid, r in zip(acts, mine, strict=False):
        out[cid] = out.get(cid, 0.0) + float(r.get("delta_phi") or 0.0)
    rest = sum(float(r.get("delta_phi") or 0.0) for r in rows if r.get("step", 0) < 0) + sum(float(r.get("delta_phi") or 0.0) for r in mine[len(acts):])
    return out, rest


def build(rec: RunRecord, *, calls: dict[str, Sequence[dict[str, Any]]], trajectories: dict[str, Trajectory] | None = None,
          rows: Sequence[StepRow] = (), sampling: dict[str, Any] | None = None, group: str | None = None, mode: str = "eval",
          timings: dict[str, float] | None = None, trace_id: str | None = None, progress: Path | None = None,
          segments: dict[str, list[dict[str, Any]]] | None = None) -> RolloutRecord:
    adv: dict[str, float] = {}
    for r in rows:  # a call's credit = the sum of the rows attached to it
        if r.call_id is not None:
            adv[r.call_id] = adv.get(r.call_id, 0.0) + r.delta_phi
    prog_rows = [json.loads(x) for x in progress.read_text().splitlines() if x.strip()] if progress else []
    unattributed = 0.0
    if prog_rows:
        for actor, cs in calls.items():
            got, rest = credit_from_progress(cs, prog_rows, actor)
            for k, v in got.items():
                adv[k] = adv.get(k, 0.0) + v
            unattributed += rest
    steps_of: dict[str, list[int]] = {}
    for traj in (trajectories or {}).values():
        for s in traj.steps:
            cid = (s.extra or {}).get("call_id")
            if cid:
                steps_of.setdefault(cid, []).append(s.step_id)
    turns = []
    for actor, cs in calls.items():
        prev_full: list[int] | None = None
        prev_seg = None
        for c in cs:
            tok = c.get("tokens") or {}
            full = tok.get("prompt_token_ids")
            seg = int(c.get("segment") or 0)
            delta, is_delta, brk = full, False, False
            if full is not None and prev_full is not None and seg == prev_seg:
                if full[: len(prev_full)] == prev_full:
                    delta, is_delta = full[len(prev_full):], True
                else:
                    brk = True  # the engine re-tokenized: this call's prompt does not extend the last one
            if full is not None:
                prev_full = list(full) + list(tok.get("completion_token_ids") or [])
            prev_seg = seg
            turns.append(Turn(call_id=c["call_id"], actor=actor, kind=c.get("kind") or "turn", segment=seg, world_step=c.get("step"),
                              prompt_is_delta=is_delta, prefix_break=brk, prompt_token_ids=delta,
                              completion_token_ids=tok.get("completion_token_ids"), logprobs=tok.get("logprobs"), policy_version=tok.get("policy_version"),
                              version_spans=[tuple(x) for x in tok["version_spans"]] if tok.get("version_spans") else None,
                              sampling=dict(sampling or {}), finish=c.get("finish"), advantage=adv.get(c["call_id"]),
                              atif_step_ids=steps_of.get(c["call_id"], [])))
    segs = [Segment(index=sg["index"], summary_call=sg.get("summary_call"), context_before=sg.get("context_before"), context_after=sg.get("context_after"))
            for sgs in (segments or {}).values() for sg in sgs]
    failure, m = rec.failure, rec.meta
    sid = (rec.attempt or rec.run_id).rsplit("-a", 1)[0]
    out = RolloutRecord(rollout_id=rec.attempt or rec.run_id, sample=sid, task=rec.task or "", harness=m.get("harness") or "", policy=rec.policy,
                        group=group, turns=turns, reward=rec.scored_reward(), reward_components=dict(m.get("components") or {}),
                        gates=dict(m.get("gates") or {}), truncated=bool(m.get("truncated")), stop=m.get("stop"),
                        failure_kind=failure.kind if failure else None, blame=failure.who if failure else None, mask_sample=rec.mask_sample,
                        timings=dict(timings or m.get("timings") or {}), trace_id=trace_id, segments=segs,
                        trajectory_tokens=sum((m.get("trajectory_tokens") or {}).values()) or None,
                        progress={**(m.get("progress") or {}), **({"unattributed_delta": round(unattributed, 9)} if prog_rows else {})})
    if mode == "train" and not out.mask_sample:
        ok, why = out.valid_for_training()
        if not ok:
            out = out.model_copy(update={"mask_sample": True, "reward": None, "failure_kind": "crucible:capture_incomplete", "blame": "infra",
                                         "stop": f"capture_incomplete: {why}"})
    return out
