"""Rollout records -> Tinker datums with per-token advantages: Crucible's credit, applied once (swarm-pipeline/09 §5).

Tinker trains what it is sent: ``importance_sampling`` / ``ppo`` / ``cispo`` take ``target_tokens``, the sampling
``logprobs`` and one ``advantages`` value per target token, and sum the loss over tokens (Tinker docs, losses). So the
advantage leaving here is final, and it is built by the same rules every other Crucible trainer path uses:

1. **The outcome.** R is the task's own reward (0.5·Φ_final + 0.5·[every gate] for long-horizon tasks: the grader's
   ``reward``, never recomputed here). The group advantage is R − mean R over the group's live rollouts
   (``train.credit.outcome_advantages``: Dr.GRPO, no std normalisation, as the slime path and the torch trainer's default).
2. **Honesty** enters the advantage, never the reward: Â' = ``compose_advantage(Â, term)`` with the STATUS.json
   Brier term from the checkpoint sidecar (``progress["honesty"]["term"]``, else ``reward_components["honesty/term"]``):
   a failure never turns positive, a success is floored at 0 (``factory.honesty.compose_advantage``; the slime bridge's
   mirror is reused, pinned equal by the P6 test).
3. **Turn credit** (``turn_eta`` > 0): M6's one rule (R1-R3, commit 64a30ba): ``train.credit.credited_records`` turns each
   call's ΔΦ into the rule's credit (centred within the episode, bounded by the outcome advantage, none on hard-rule or
   hacked episodes, none in a group without outcome spread); ``require_turn_credit`` refuses a run that claims turn
   credit while no live rollout recorded progress.
4. **Sequences.** ``train.bridges.common.flatten`` (the one TITO linearizer: one sequence per (actor, agent, segment),
   observations at mask 0, a token-incomplete rollout yields nothing) with ``seq_adv`` = Â' and the credited turns.
   Each sequence becomes one ``TokenDatum``: ``input_ids`` = full[:-1], targets / logprobs / advantages / mask = the
   per-position values shifted by one (tinker-cookbook ``rl/data_processing.trajectory_to_data``, PATTERN). Tokens
   outside the policy's own completions carry advantage 0 and no gradient (Tinker's RL losses take no mask).
5. **Scale.** ``loss_scale="sum"`` sends the advantages as they are (Tinker's sum); ``"token_mean"`` divides them by the
   batch's trained-token count, so the step size does not grow with episode length.

Masked rollouts never reach here (their group is dropped whole: ``trainer``). Per datum, ``meta`` names the rollout,
its group, version range and the composed sequence advantage, so the equivalence test can compare the slime path turn by
turn.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from crucible.core.schema.rollout import RolloutRecord
from crucible.train.bridges.common import TrainSeq, flatten
from crucible.train.bridges.slime.credit import compose_advantage
from crucible.train.bridges.tinker.backend import TokenDatum
from crucible.train.credit import credited_records, outcome_advantages, require_turn_credit


def honesty_term(ro: RolloutRecord) -> float:
    """The episode's honesty term (≤ 0), where ``factory.honesty.honesty_group_fn`` reads it."""
    h = (ro.progress or {}).get("honesty") or {}
    if "term" in h and h["term"] is not None:
        return min(0.0, float(h["term"]))
    return min(0.0, float((ro.reward_components or {}).get("honesty/term", 0.0)))


def sequence_advantages(records: Sequence[RolloutRecord], groups: Sequence[str], *, honesty: bool = True) -> dict[str, float]:
    """Rollout id -> the honesty-composed outcome advantage (0 for a masked rollout)."""
    out = outcome_advantages(records, groups)
    return {r.rollout_id: (compose_advantage(out[r.rollout_id], honesty_term(r)) if honesty else out[r.rollout_id])
            if not r.mask_sample and r.reward is not None else 0.0 for r in records}


def seq_datum(s: TrainSeq, meta: dict[str, Any] | None = None) -> TokenDatum:
    full = list(s.prompt_ids) + list(s.response_ids)
    p = len(s.prompt_ids)
    mask = [0] * p + [int(m) for m in s.response_mask]
    lps = [0.0] * p + [float(x) for x in s.logprobs]
    adv = [0.0] * p + [float(a) if m else 0.0 for a, m in zip(s.advantages, s.response_mask, strict=True)]
    return TokenDatum(input_ids=full[:-1], target_ids=full[1:], logprobs=lps[1:], advantages=adv[1:], mask=mask[1:], meta=dict(meta or {}))


def build_datums(records: Sequence[RolloutRecord], groups: Sequence[str], *, turn_eta: float = 0.0, honesty: bool = True,
                 frozen_actors: Iterable[str] = ()) -> list[TokenDatum]:
    """Datums for the live rollouts of ``records`` (module doc). ``groups[i]`` is record i's GRPO group."""
    frozen = tuple(frozen_actors)
    recs, keys = list(records), list(groups)
    seq = sequence_advantages(recs, keys, honesty=honesty)
    if turn_eta > 0:
        require_turn_credit(recs, frozen, where="tinker.build_datums")
        recs = credited_records(recs, keys, turn_weight=turn_eta, frozen_actors=frozen)
    out: list[TokenDatum] = []
    for ro, g in zip(recs, keys, strict=True):
        for s in flatten(ro, seq[ro.rollout_id], turn_weight=turn_eta, frozen_actors=frozen):
            if not any(s.response_mask):
                continue
            vs = [v for v in s.versions_seen]
            out.append(seq_datum(s, {"rollout": ro.rollout_id, "group": g, "seq_adv": seq[ro.rollout_id], "reward": ro.reward, "segment": s.segment,
                                     "actor": s.actor, "call_spans": [list(x) for x in s.call_spans], "prompt_len": len(s.prompt_ids),
                                     "min_version": min(vs) if vs else None, "max_version": max(vs) if vs else None}))
    return out


def scale(datums: Sequence[TokenDatum], how: str) -> list[TokenDatum]:
    if how == "sum":
        return list(datums)
    if how != "token_mean":
        raise ValueError(f"loss_scale is sum or token_mean, got {how!r}")
    n = sum(d.n_train for d in datums) or 1
    return [TokenDatum(d.input_ids, d.target_ids, d.logprobs, [a / n for a in d.advantages], d.mask, {**d.meta, "scale": 1.0 / n}) for d in datums]


def call_advantages(d: TokenDatum) -> dict[str, list[float]]:
    """Per call of the datum, the advantages on its completion tokens (target coordinates: shifted by one)."""
    off = d.meta["prompt_len"] - 1
    return {cid: d.advantages[off + a: off + b] for cid, a, b in d.meta["call_spans"]}
