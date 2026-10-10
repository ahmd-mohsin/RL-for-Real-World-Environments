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
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

from crucible.train.rollout import RolloutRecord, full_prompts


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


def _call_versions(t, n: int) -> list[int | None]:
    if t.version_spans:
        out: list[int | None] = []
        spans = list(t.version_spans) + [(n, None)]
        for (s, v), (e, _) in pairwise(spans):
            out += [v] * (e - s)
        return out[:n]
    return [t.policy_version if isinstance(t.policy_version, int) else None] * n


def flatten(ro: RolloutRecord, seq_adv: float = 0.0, *, turn_weight: float = 0.0, frozen_actors: Sequence[str] = ()) -> list[TrainSeq]:
    if ro.mask_sample or ro.reward is None:
        return []
    prompts = full_prompts(ro)
    out: list[TrainSeq] = []
    cur: TrainSeq | None = None
    frozen = set(frozen_actors)
    for t in ro.turns:
        full = prompts[t.call_id]
        comp = list(t.completion_token_ids or [])
        chains = cur is not None and cur.actor == t.actor and cur.segment == t.segment and not t.prefix_break and \
            full[: len(cur.prompt_ids) + len(cur.response_ids)] == cur.prompt_ids + cur.response_ids
        if not chains:
            cur = TrainSeq(ro.rollout_id, t.actor, t.segment, list(full), reward=ro.reward)
            out.append(cur)
        else:
            obs = full[len(cur.prompt_ids) + len(cur.response_ids):]
            cur.response_ids += obs
            cur.response_mask += [0] * len(obs)
            cur.logprobs += [0.0] * len(obs)
            cur.advantages += [0.0] * len(obs)
            cur.versions += [None] * len(obs)
        trainable = 0 if t.actor in frozen else 1
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
    return out
