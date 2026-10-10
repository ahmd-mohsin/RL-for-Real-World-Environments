"""M6 credit → token advantages (DESIGN §10.3 point 6; ARCHITECTURE M12 ``credit.py``).

A rollout record (``crucible.rollout/v1``) has per-call ``Turn``s; the trainer needs one advantage per completion token
and a loss mask. Rules:

- the **sequence advantage** Â (group-relative or critic) is broadcast to every completion token of every trainable call
  (including compaction summary calls: they share the episode's advantage, CompactionRL);
- **turn credit** (M6 ΔΦ per call, ``Turn.advantage``) is added on that call's tokens with weight ``turn_weight``
  (``credit_unit = "turn"``), centred within the group by the caller (``objectives``) so it adds no net mass;
- observation / tool-output / prompt tokens never carry loss (they are prompt tokens of the next call: only completion
  tokens are trainable);
- masked rollouts (``mask_sample``) contribute nothing; truncated rollouts are masked or bootstrapped by the caller
  (critic mode), never scored 0 here;
- frozen sub-agents (``frozen_actors``): their calls get loss mask 0 (Kimi PARL).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from crucible.train.rollout import RolloutRecord


@dataclass
class TokenCredit:
    call_id: str
    completion_ids: list[int]
    logprobs: list[float]
    advantages: list[float]
    loss_mask: list[int]
    policy_version: int | str | None


def token_credit(ro: RolloutRecord, seq_adv: float, *, turn_weight: float = 0.0, frozen_actors: Iterable[str] = (),
                 train_compaction: bool = True) -> list[TokenCredit]:
    frozen = set(frozen_actors)
    out = []
    for t in ro.turns:
        ids = list(t.completion_token_ids or [])
        trainable = not ro.mask_sample and t.actor not in frozen and (train_compaction or t.kind != "compaction")
        a = seq_adv + (turn_weight * (t.advantage or 0.0) if trainable else 0.0)
        out.append(TokenCredit(call_id=t.call_id, completion_ids=ids, logprobs=list(t.logprobs or []),
                               advantages=[a if trainable else 0.0] * len(ids), loss_mask=[1 if trainable else 0] * len(ids),
                               policy_version=t.policy_version))
    return out
