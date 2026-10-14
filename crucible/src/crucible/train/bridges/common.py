"""``crucible.rollout/v1`` → training sequences, the shape every trainer consumes (research m11-m12/01 verl
``AgentLoopOutput``, /03 SkyRL ``GeneratorOutput`` and prime-rl samples, /04 TRL ``rollout_func`` and rLLM ``Step``).

One rollout becomes **one sequence per (actor, compaction segment)**: within a segment every call's prompt extends the
previous prompt + completion (exact TITO, linear storage), so the segment is one causal sequence

    prompt_ids = the segment's first full prompt
    response   = completion_1 · obs_1 · completion_2 · obs_2 · … · completion_n

with ``response_mask`` = 1 exactly on sampled tokens (observations / tool outputs are never trained), behaviour
``logprobs`` aligned (0.0 where masked: never read), per-token ``advantages`` (the rollout's sequence advantage on every
sampled token + M6 turn credit on its own call's span), and the per-token ``versions`` (from ``version_spans`` when a call
crossed a publish). A segment whose calls do not chain (``prefix_break``) is split at the break: never stitched.

Chains are keyed (actor, agent, segment) (M16 §3.5): the root's turns of one segment form one sequence even when RLM
sub-calls and child-agent turns interleave with them in the record; each sub-call and each summary attempt is a sequence
of its own, each child agent's segment one more. M16 stress B7: the single open chain broke at every interleaved sub
turn, so the next root turn re-opened a sequence with its whole prompt: the trainer forwarded 7.0x the entered tokens
(19.7x with 256 sub-calls). The rollout's final root sequence is emitted last (``last``: the outcome closes there).

A rollout yields **no** sequences when it is masked (``mask_sample`` or reward None: never trained as 0) or when a call of
a trained actor is not token-complete (``Turn.missing()``: no completion, logprobs missing or of the wrong length, no
policy version, a prefix break, impure version spans, top-p without candidates). M12 audit G4 §5: the behaviour logprobs
of such a call used to be zero-filled, i.e. trained as if π_rollout = 1. Frozen actors' calls are context only (mask 0),
so their logprobs are never read and need not exist. Each bridge then keeps the masked rollout *aligned* in its own
framework's terms (a placeholder row with no trainable token), never as a reward of 0.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

from crucible.core.schema.rollout import RolloutRecord, full_prompts


@dataclass
class TrainSeq:
    rollout_id: str
    actor: str
    segment: int
    prompt_ids: list[int]
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    versions: list[int | None] = field(default_factory=list)
    call_spans: list[tuple[str, int, int]] = field(default_factory=list)  # (call_id, start, end) in response coordinates
    reward: float | None = None
    num_turns: int = 0
    truncated: bool = False  # the rollout's own truncation flag (a turn/token limit stop)
    finish: str | None = None  # the finish reason of this sequence's last call (stop | length | abort | None)
    last: bool = True  # the rollout's final sequence (its outcome closes here)
    versions_seen: list[int] = field(default_factory=list)  # every policy version that produced a sampled token
    agent: str = ""  # the RLM tree agent ("" = root, a child's path): M16
    node: bool = False  # a stand-alone tree node (a sub-call or a summary attempt), not an agent's chain


def _call_versions(t, n: int) -> list[int | None]:
    if t.version_spans:
        out: list[int | None] = []
        spans = list(t.version_spans) + [(n, None)]
        for (s, v), (e, _) in pairwise(spans):
            out += [v] * (e - s)
        return out[:n]
    return [t.policy_version if isinstance(t.policy_version, int) else None] * n


def untrainable(ro: RolloutRecord, frozen_actors: Sequence[str] = ()) -> str | None:
    """Why this rollout yields no training sequence (None: it trains). ``RolloutRecord.valid_for_training`` with frozen
    actors' calls exempt from the token-record checks (they are context, mask 0)."""
    if ro.mask_sample or ro.reward is None:
        return f"masked ({ro.failure_kind or 'no reward'})"
    frozen = set(frozen_actors)
    live = [t for t in ro.turns if t.actor not in frozen and t.trainable]  # another model's calls (Turn.trainable False) are context
    if not live:
        return "no trained call"
    bad = [(t.call_id, t.missing()) for t in live if t.missing()]
    if bad:
        return f"token record incomplete for {len(bad)}/{len(live)} calls (first: {bad[0][0]}: {', '.join(bad[0][1])})"
    return None


def flatten(ro: RolloutRecord, seq_adv: float = 0.0, *, turn_weight: float = 0.0, frozen_actors: Sequence[str] = ()) -> list[TrainSeq]:
    if untrainable(ro, frozen_actors) is not None:
        return []
    prompts = full_prompts(ro)
    out: list[TrainSeq] = []
    open_: dict[tuple, TrainSeq] = {}  # the open sequence of each chain (actor, agent, segment); a node is its own chain
    final: TrainSeq | None = None  # the sequence holding the last root-chain turn: where the outcome closes
    frozen = set(frozen_actors)
    for t in ro.turns:
        key = t.chain()
        cur = open_.get(key)
        full = prompts[t.call_id]
        comp = list(t.completion_token_ids or [])
        n = len(cur.prompt_ids) + len(cur.response_ids) if cur is not None else 0
        chains = cur is not None and not t.prefix_break and len(full) >= n and full[: len(cur.prompt_ids)] == cur.prompt_ids and \
            full[len(cur.prompt_ids): n] == cur.response_ids
        if not chains:
            cur = TrainSeq(ro.rollout_id, t.actor, t.segment, list(full), reward=ro.reward, truncated=bool(ro.truncated), agent=t.agent, node=t.node())
            out.append(cur)
            open_[key] = cur
        else:
            obs = full[n:]
            cur.response_ids += obs
            cur.response_mask += [0] * len(obs)
            cur.logprobs += [0.0] * len(obs)
            cur.advantages += [0.0] * len(obs)
            cur.versions += [None] * len(obs)
        trainable = 0 if (t.actor in frozen or not t.trainable) else 1
        a = seq_adv + turn_weight * (t.advantage or 0.0)
        start = len(cur.response_ids)
        cur.response_ids += comp
        cur.response_mask += [trainable] * len(comp)
        lps = list(t.logprobs or [])
        cur.logprobs += (lps + [0.0] * len(comp))[: len(comp)]
        cur.advantages += [a * trainable] * len(comp)
        cur.versions += _call_versions(t, len(comp))
        cur.call_spans.append((t.call_id, start, start + len(comp)))
        cur.num_turns += 1
        cur.finish = t.finish
        if trainable:
            cur.versions_seen = sorted(set(cur.versions_seen) | {int(v) for v in t.versions()})
        if not cur.node and not cur.agent:
            final = cur
    if final is not None and out and out[-1] is not final:  # an episode can end on a sub-call or a child's turn: the root closes it
        out = [s for s in out if s is not final] + [final]  # by identity: two sequences may compare equal
    for s in out[:-1]:
        s.last = False
    return out
