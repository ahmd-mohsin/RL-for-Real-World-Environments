"""M6 credit → token advantages (DESIGN §10.3 point 6; ARCHITECTURE M12 ``credit.py``).

A rollout record (``crucible.rollout/v1``) has per-call ``Turn``s; the trainer needs one advantage per completion token
and a loss mask. Rules:

- the **sequence advantage** Â (group-relative or critic) is broadcast to every completion token of every trainable call
  (including compaction summary calls: they share the episode's advantage, CompactionRL);
- **turn credit** (M6 ΔΦ per call, ``Turn.advantage``) is added on that call's tokens with weight ``turn_weight``
  (``credit_unit = "turn"``) after ``turn_credit_map`` has passed it through M6's group rule (centred within each
  episode, bounded by the outcome advantage, none on gate-failed or hacked episodes, none in a group without outcome
  spread). M6 audit: nothing centred it, so raw ΔΦ reached the tokens: a gate-failed rollout gained credit and a
  group's net advantage was +9.0;
- observation / tool-output / prompt tokens never carry loss (they are prompt tokens of the next call: only completion
  tokens are trainable);
- masked rollouts (``mask_sample``) contribute nothing; truncated rollouts are masked or bootstrapped by the caller
  (critic mode), never scored 0 here;
- frozen sub-agents (``frozen_actors``): their calls get loss mask 0 (Kimi PARL).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
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
        trainable = not ro.mask_sample and t.trainable and t.actor not in frozen and (train_compaction or t.kind != "compaction")
        a = seq_adv + (turn_weight * (t.advantage or 0.0) if trainable else 0.0)
        out.append(TokenCredit(call_id=t.call_id, completion_ids=ids, logprobs=list(t.logprobs or []),
                               advantages=[a if trainable else 0.0] * len(ids), loss_mask=[1 if trainable else 0] * len(ids),
                               policy_version=t.policy_version))
    return out


def turn_credit_map(records: Sequence[RolloutRecord], groups: Sequence[str], *, eta: float, frozen_actors: Iterable[str] = ()) -> dict[tuple[str, str], float]:
    """(rollout id, call id) -> the process credit to add on that call's tokens: each call's ΔΦ (``Turn.advantage``)
    through ``credit.compose.turn_credit`` per group, ``eta`` its weight.

    RLM sub-calls (``kind="sub"``) are tree nodes of the turn whose cell issued them, not steps of the Φ chain: they are
    left out of the chain and get their parent's credit (M14 audit G1 D4: copying the parent's ΔΦ onto each sub turn
    inflated Φ and diluted the root turn's credit with every sub-call its cell made). The same holds for summary attempts
    and for the calls of a child agent of an RLM tree (M16 §3.5): only the root chain (``agent`` "", not a node) is Φ's
    chain; a child's calls take the credit of the call they depend on (their spawning root turn, through any depth)."""
    import itertools

    from crucible.credit.compose import Episode, hard_gate_failed, turn_credit

    frozen = set(frozen_actors)
    by: dict[str, list[RolloutRecord]] = {}
    for r, g in zip(records, groups, strict=True):
        by.setdefault(g, []).append(r)
    out: dict[tuple[str, str], float] = {}
    for rs in by.values():
        eps, keys = [], []
        for r in rs:
            ts = [t for t in r.turns if t.completion_token_ids and t.actor not in frozen and t.trainable and not t.node() and not t.agent]
            masked = r.mask_sample or r.reward is None
            eps.append(Episode(reward=None if masked else float(r.reward), phis=list(itertools.accumulate(float(t.advantage or 0.0) for t in ts)),
                               gated=hard_gate_failed(r.gates, r.gate_kinds), hack=is_hack(r)))
            keys.append([(r.rollout_id, t.call_id) for t in ts])
        for ks, row in zip(keys, turn_credit(eps, eta=eta), strict=True):
            out.update(zip(ks, row, strict=True))
        for r in rs:
            for t in r.turns:
                if (t.node() or t.agent) and t.trainable and t.completion_token_ids and t.actor not in frozen:
                    parent = (t.depends_on or [None])[0]
                    out[(r.rollout_id, t.call_id)] = out.get((r.rollout_id, parent), 0.0) if parent else 0.0
    return out


def outcome_advantages(records: Sequence[RolloutRecord], groups: Sequence[str]) -> dict[str, float]:
    """Rollout id -> R − mean R over the live (scored, unmasked) rollouts of its group; a masked rollout gets 0. The same
    numbers as ``objectives.group_advantages(estimator="mean")`` (Dr.GRPO, the trainer's default), without torch, for
    bridges that hand a framework a finished per-token advantage."""
    import statistics

    live = lambda r: not r.mask_sample and r.reward is not None
    by: dict[str, list[float]] = {}
    for r, g in zip(records, groups, strict=True):
        if live(r):
            by.setdefault(g, []).append(float(r.reward))  # type: ignore[arg-type]
    return {r.rollout_id: float(r.reward) - statistics.fmean(by[g]) if live(r) else 0.0  # type: ignore[arg-type]
            for r, g in zip(records, groups, strict=True)}


def has_turn_credit(records: Sequence[RolloutRecord], frozen_actors: Iterable[str] = ()) -> bool:
    """Whether any live rollout carries per-turn progress (``Turn.advantage`` set on a trained call). ``None`` means no
    progress was recorded; 0.0 is progress that did not move."""
    frozen = set(frozen_actors)
    return any(t.advantage is not None for r in records if not r.mask_sample and r.reward is not None
               for t in r.turns if t.trainable and t.actor not in frozen and t.completion_token_ids)


def require_turn_credit(records: Sequence[RolloutRecord], frozen_actors: Iterable[str] = (), *, where: str) -> None:
    """Refuse dense credit that would silently be absent (M6 R3): ``turn_weight > 0`` and live rollouts, none of which
    recorded per-turn progress, would train on the outcome alone while the run claims turn credit. A batch with no live
    rollout trains nothing either way and passes."""
    if not any(not r.mask_sample and r.reward is not None for r in records) or has_turn_credit(records, frozen_actors):
        return
    raise ValueError(f"{where}: turn_weight > 0 but no live rollout carries per-turn progress (Turn.advantage is None on every "
                     "trained call): the tasks declare no subgoals, or their rollouts recorded no progress. Train on tasks with "
                     "subgoals, or set turn_weight=0 to train on the outcome only.")


def credited_records(records: Sequence[RolloutRecord], groups: Sequence[str], *, turn_weight: float,
                     frozen_actors: Iterable[str] = ()) -> list[RolloutRecord]:
    """Copies of ``records`` whose ``Turn.advantage`` is M6's per-call credit (``turn_credit_map``) divided by
    ``turn_weight``, so a path that adds ``turn_weight · Turn.advantage`` to the sequence advantage (``common.flatten``,
    ``token_credit``) trains exactly the rule's credit. The one place raw ΔΦ becomes trainable credit: the torch trainer
    and every bridge call it (M6 R1: the rLLM bridge added raw ΔΦ)."""
    if turn_weight <= 0:
        raise ValueError("credited_records needs turn_weight > 0")
    tc = turn_credit_map(records, groups, eta=turn_weight, frozen_actors=frozen_actors)
    return [r.model_copy(update={"turns": [t.model_copy(update={"advantage": tc.get((r.rollout_id, t.call_id), 0.0) / turn_weight})
                                           for t in r.turns]}) for r in records]


def is_hack(r: RolloutRecord) -> bool:
    """A hack for per-turn credit: confirmed by the level-2 judge (``hack/confirmed`` > 0), or zeroed by a group stage:
    ``hack/zeroed`` (``group_ops.zero_hacks``, holding the reward it replaced, which may itself be 0: its presence is the
    flag) or ``gar/hack_zeroed`` (``credit.gar.gar_group_fn`` with a custom detector). M6 D2: the rule read only
    ``hack/confirmed``, so a zeroed hack's Φ-raising call kept positive credit."""
    comps = r.reward_components or {}
    return float(comps.get("hack/confirmed", 0.0)) > 0 or "hack/zeroed" in comps or float(comps.get("gar/hack_zeroed", 0.0)) > 0


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
