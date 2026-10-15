"""The producer side of ``crucible.rollout/v1`` (schema: ``crucible.core.schema.rollout``). M10 writes one record per attempt
from the harness result, the policy's model calls, the ATIF trajectory and the dense ``steps.jsonl`` rows.

``mode="train"``: a rollout whose token record is incomplete is masked as ``crucible:capture_incomplete`` (infra) —
the trainer must never guess logprobs. ``mode="eval"``: token records are optional (hosted APIs return none) and
their absence changes nothing.

M16 (very-long-horizon RLM, design §3.5): ``model_calls`` entries may carry ``agent`` ("" = root, a child's path),
``depth``, ``edge``, ``accepted`` (summary attempts), ``trainable``, ``window`` and ``ledger_index``; absent = the old
behaviour. Linear chains are keyed (actor, agent, segment); sub-calls and summary attempts are tree nodes; a child's calls
share the credit of the root turn that spawned it. Segments may carry ``agent``, ``boundary_reason``, ``attempts``,
``failed`` and ``tokens_source``.

``Turn.sampling`` is what each call sent (its call record's ``sampling``, R-3); the ``sampling`` argument fills it only for
calls that recorded none, from a caller that knows what its engine was sent (the training environments).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from crucible.core.schema.reward import StepRow
from crucible.core.schema.rollout import RolloutRecord, Segment, Turn
from crucible.core.schema.run import RunRecord

if TYPE_CHECKING:  # annotation only: in-process rollouts (M12) must not require Harbor installed
    from harbor.models.trajectories import Trajectory


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


# legacy segment ``strategy`` (actors/policy.py, actors/rlm/policy.py) -> ``Segment.boundary_reason`` (H13); every other
# strategy (summary, keep_recent, discard_all, tiered:*, hybrid:*) rebuilt the context: a compaction
_REASON = {"rerender": "rerender", "resume": "resume", "restore": "restore"}


def _ancestors(calls: dict[str, Sequence[dict[str, Any]]], adv: dict[str, float]) -> tuple[dict[str, str], dict[str, str]]:
    """Per call, the root-chain call whose credit it shares (M16 §3.5): itself for a root turn; the issuing turn for a
    sub-call (m14 G1 D4); for a child agent's call the root turn that spawned the child (its first call's ``parent``),
    through any depth. A child that acted on the World (``world="shared"``) carries ΔΦ of its own: it is folded into that
    ancestor, so Φ still telescopes along the root chain and the spawning decision is credited with what the child did.
    Also returns each tree node's direct parent (``Turn.depends_on``): a child's later calls inherit its spawning call."""
    anc: dict[str, str] = {}
    direct: dict[str, str] = {}
    for cs in calls.values():
        spawn: dict[str, str] = {}
        for c in cs:
            cid, ag, par = c["call_id"], str(c.get("agent") or ""), c.get("parent")
            if ag and par and ag not in spawn:
                spawn[ag] = par
            if c.get("kind") == "sub":
                anc[cid] = anc.get(par, par) if par else cid
                if par:
                    direct[cid] = par
            elif ag:
                p = par or spawn.get(ag)
                anc[cid] = anc.get(p, p) if p else cid
                if p:
                    direct[cid] = p
                if anc[cid] != cid and cid in adv:
                    adv[anc[cid]] = adv.get(anc[cid], 0.0) + adv.pop(cid)
            else:
                anc[cid] = cid
                if par:
                    direct[cid] = par
    return anc, direct


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
    anc, direct = _ancestors(calls, adv)
    turns = []
    frozen: list[Turn] = []
    for actor, cs in calls.items():
        # one linear chain per (agent, segment) (M16 §3.5): the root's, each child agent's; sub-calls and summary attempts
        # are nodes. M16 stress B7: a child's turn between two root turns no longer breaks the root's chain
        prev_full: dict[str, list[int] | None] = {}
        prev_seg: dict[str, int | None] = {}
        for c in cs:
            tok = c.get("tokens") or {}
            full = tok.get("prompt_token_ids")
            seg = int(c.get("segment") or 0)
            ag = str(c.get("agent") or "")
            kind = c.get("kind") or "turn"
            parent = direct.get(c["call_id"])
            # a summary attempt the loop did not resume from (``accepted`` False) is recorded, never trained (verifiers
            # trace.py:515-553: trainable only when a compaction edge shows the run resumed from it); a call answered by
            # another model (an RLM sub_llm, a child on another model: M14 audit G1 D1) is never trained either
            trainable = not (c.get("trainable") is False or (kind == "compaction" and c.get("accepted") is False))
            pf, ps = prev_full.get(ag), prev_seg.get(ag)
            chained = pf is not None and seg == ps and len(pf) == tok.get("prompt_prefix_len")
            if full is None and tok.get("prompt_delta") is not None and chained:  # linear capture (policy._compress): rebuild from the chain
                full = pf + list(tok["prompt_delta"])
            probe = Turn(call_id=c["call_id"], actor=actor, kind=kind, edge=c.get("edge"))
            if probe.node():
                # an RLM sub-LLM call (m14) or a summary attempt (M16) is its own sequence: a tree node, not a link of the
                # agent's chain. It shares its ancestor root turn's ΔΦ (the cell that issued it) and keeps its version
                # spans (G1 D4, D14)
                if not trainable:
                    # not the policy's to train: recorded, never trained. No tokens are kept; it goes after the chains
                    frozen.append(Turn(call_id=c["call_id"], actor=actor, kind=kind, segment=seg, world_step=c.get("step"), model=c.get("model"),
                                       agent=ag, edge=c.get("edge"), trainable=False, sampling=c.get("sampling") or dict(sampling or {}), finish=c.get("finish"),
                                       depends_on=[parent] if parent else None, atif_step_ids=[]))
                    continue
                turns.append(Turn(call_id=c["call_id"], actor=actor, kind=kind, segment=seg, world_step=c.get("step"), prompt_is_delta=False,
                                  prefix_break=False, prompt_token_ids=full, completion_token_ids=tok.get("completion_token_ids"),
                                  logprobs=tok.get("logprobs"), policy_version=tok.get("policy_version"),
                                  version_spans=[tuple(x) for x in tok["version_spans"]] if tok.get("version_spans") else None,
                                  sampling=c.get("sampling") or dict(sampling or {}), finish=c.get("finish"),
                                  advantage=adv.get(anc.get(c["call_id"], c["call_id"])) if kind == "sub" else adv.get(c["call_id"]),
                                  depends_on=[parent] if parent else None, model=None, agent=ag, edge=c.get("edge"), atif_step_ids=[]))
                continue
            delta, is_delta, brk = full, False, False
            if full is not None and pf is not None and seg == ps:
                if full[: len(pf)] == pf:
                    delta, is_delta = full[len(pf):], True
                else:
                    brk = True  # the engine re-tokenized: this call's prompt does not extend the last one (masked, never stitched)
            if full is not None:
                prev_full[ag] = list(full) + list(tok.get("completion_token_ids") or [])
            prev_seg[ag] = seg
            # a non-trainable chain call keeps its tokens: later calls of the chain store deltas over them (context, mask 0)
            turns.append(Turn(call_id=c["call_id"], actor=actor, kind=kind, segment=seg, world_step=c.get("step"),
                              prompt_is_delta=is_delta, prefix_break=brk, prompt_token_ids=delta,
                              completion_token_ids=tok.get("completion_token_ids"), logprobs=tok.get("logprobs"), policy_version=tok.get("policy_version"),
                              version_spans=[tuple(x) for x in tok["version_spans"]] if tok.get("version_spans") else None,
                              sampling=c.get("sampling") or dict(sampling or {}), finish=c.get("finish"), advantage=adv.get(anc.get(c["call_id"], c["call_id"])),
                              depends_on=[parent] if ag and parent else None, model=None if trainable else c.get("model"),
                              trainable=trainable, agent=ag, edge=c.get("edge"), atif_step_ids=steps_of.get(c["call_id"], [])))
    turns += frozen
    segs = [Segment(index=sg["index"], start_call=sg.get("start_call"), end_call=sg.get("end_call"), summary_call=sg.get("summary_call"),
                    context_before=sg.get("context_before"), context_after=sg.get("context_after"), agent=str(sg.get("agent") or ""),
                    boundary_reason=sg.get("boundary_reason") or _REASON.get(sg.get("strategy") or "", "compaction" if sg.get("strategy") else None),
                    attempts=sg.get("attempts"), failed=bool(sg.get("failed")), tokens_source=sg.get("tokens_source"))
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
