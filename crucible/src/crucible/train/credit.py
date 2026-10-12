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

from crucible.core.schema.rollout import RolloutRecord


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


def critical_steps(ro: RolloutRecord, main_actor: str | None = None) -> int:
    """Kimi K2.5 PARL's Critical Steps: Σ over stages of (main-agent steps + the slowest sub-agent's steps), a stage being
    the stretch from one main-agent call to the next. Dispatching more sub-agents without shortening the longest branch
    gains nothing. ``main_actor`` defaults to the actor of the first call."""
    turns = [t for t in ro.turns if t.kind != "compaction"]
    if not turns:
        return 0
    main = main_actor or turns[0].actor
    total, subs, in_stage = 0, {}, False
    for t in turns:
        if t.actor == main:
            if in_stage:
                total += 1 + max(subs.values(), default=0)
            subs, in_stage = {}, True
        else:
            subs[t.actor] = subs.get(t.actor, 0) + 1
    return total + (1 if in_stage else 0) + max(subs.values(), default=0)


def dag_critical_path(ro: RolloutRecord, *, weight: str = "calls") -> float:
    """The longest dependency chain through the calls (``Turn.depends_on``; DeepSeek-V4.1 Agent Team's derived latency):
    the length in calls, or in completion tokens (``weight="tokens"``). A call without ``depends_on`` follows the
    previous call of its own actor."""
    best: dict[str, float] = {}
    last_of: dict[str, str] = {}
    for t in ro.turns:
        w = float(len(t.completion_token_ids or [])) if weight == "tokens" else 1.0
        deps = t.depends_on if t.depends_on is not None else ([last_of[t.actor]] if t.actor in last_of else [])
        best[t.call_id] = w + max((best.get(d, 0.0) for d in deps), default=0.0)
        last_of[t.actor] = t.call_id
    return max(best.values(), default=0.0)
