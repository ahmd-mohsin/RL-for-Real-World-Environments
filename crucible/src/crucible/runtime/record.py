"""The producer side of ``crucible.rollout/v1`` (schema: ``crucible.train.rollout``). M10 writes one record per attempt
from the harness result, the policy's model calls, the ATIF trajectory and the dense ``steps.jsonl`` rows.

``mode="train"``: a rollout whose token record is incomplete is masked as ``crucible:capture_incomplete`` (infra) —
the trainer must never guess logprobs. ``mode="eval"``: token records are optional (hosted APIs return none) and
their absence changes nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from harbor.models.trajectories import Trajectory

from crucible.core.schema.reward import StepRow
from crucible.core.schema.run import RunRecord
from crucible.train.rollout import RolloutRecord, Turn


def build(rec: RunRecord, *, calls: dict[str, Sequence[dict[str, Any]]], trajectories: dict[str, Trajectory] | None = None,
          rows: Sequence[StepRow] = (), sampling: dict[str, Any] | None = None, group: str | None = None, mode: str = "eval",
          timings: dict[str, float] | None = None, trace_id: str | None = None) -> RolloutRecord:
    adv: dict[str, float] = {}
    for r in rows:  # a call's credit = the sum of the rows attached to it
        if r.call_id is not None:
            adv[r.call_id] = adv.get(r.call_id, 0.0) + r.delta_phi
    steps_of: dict[str, list[int]] = {}
    for traj in (trajectories or {}).values():
        for s in traj.steps:
            cid = (s.extra or {}).get("call_id")
            if cid:
                steps_of.setdefault(cid, []).append(s.step_id)
    turns = []
    for actor, cs in calls.items():
        for c in cs:
            tok = c.get("tokens") or {}
            turns.append(Turn(call_id=c["call_id"], actor=actor, world_step=c.get("step"), prompt_token_ids=tok.get("prompt_token_ids"),
                              completion_token_ids=tok.get("completion_token_ids"), logprobs=tok.get("logprobs"), policy_version=tok.get("policy_version"),
                              sampling=dict(sampling or {}), finish=c.get("finish"), advantage=adv.get(c["call_id"]),
                              atif_step_ids=steps_of.get(c["call_id"], [])))
    failure, m = rec.failure, rec.meta
    sid = (rec.attempt or rec.run_id).rsplit("-a", 1)[0]
    out = RolloutRecord(rollout_id=rec.attempt or rec.run_id, sample=sid, task=rec.task or "", harness=m.get("harness") or "", policy=rec.policy,
                        group=group, turns=turns, reward=rec.scored_reward(), reward_components=dict(m.get("components") or {}),
                        gates=dict(m.get("gates") or {}), truncated=bool(m.get("truncated")), stop=m.get("stop"),
                        failure_kind=failure.kind if failure else None, blame=failure.who if failure else None, mask_sample=rec.mask_sample,
                        timings=dict(timings or m.get("timings") or {}), trace_id=trace_id)
    if mode == "train" and not out.mask_sample:
        ok, why = out.valid_for_training()
        if not ok:
            out = out.model_copy(update={"mask_sample": True, "reward": None, "failure_kind": "crucible:capture_incomplete", "blame": "infra",
                                         "stop": f"capture_incomplete: {why}"})
    return out
