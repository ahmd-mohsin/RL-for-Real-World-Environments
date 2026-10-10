"""Training-side batch granularity (research m11-m12/06 §3.4; m11-m12/01–04 for the framework mechanisms).

Definitions used everywhere in M12 (the frameworks disagree on names; these are ours):

- **global batch**: the rollouts one optimizer step consumes (B groups × n samples, or B rollouts in critic mode);
- **mini batch**: a slice of the global batch for one optimizer update (``updates_per_batch`` > 1 = several PPO epochs /
  mini-batches per global batch); the default is one update per global batch;
- **micro batch**: what one forward/backward holds on one device, bounded by a **token budget** (dynamic batching:
  verl ``use_dynamic_bsz`` / ``ppo_max_token_len_per_gpu``), gradients accumulated across micro batches;
- **DP rank**: one data-parallel replica. Before each update the global batch is partitioned across ranks so the
  summed tokens are balanced (verl ``balance_batch``, MiMo ``_balance_batch``): each rank does about equal work, which
  removes the straggler rank at the all-reduce.

``balance_partitions`` is the Karmarkar–Karp-style largest-first greedy (sequences sorted by length, each to the
lightest rank); ``micro_batches`` packs a rank's sequences first-fit-decreasing into token budgets. Loss scaling across
micro batches must use the *global* token count (``objectives.aggregate`` over the whole batch), so the gradient is the
same as with one giant batch: ``loss_scale`` returns the factor per micro batch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def balance_partitions(lengths: Sequence[int], k: int) -> list[list[int]]:
    """Indices of ``lengths`` split into ``k`` parts with near-equal summed length (largest-first to the lightest part)."""
    if k <= 0:
        raise ValueError("k must be positive")
    parts: list[list[int]] = [[] for _ in range(k)]
    load = [0] * k
    for i in sorted(range(len(lengths)), key=lambda j: (-lengths[j], j)):
        r = min(range(k), key=lambda x: (load[x], len(parts[x]), x))
        parts[r].append(i)
        load[r] += lengths[i]
    return [sorted(p) for p in parts]


def imbalance(lengths: Sequence[int], parts: Sequence[Sequence[int]]) -> float:
    loads = [sum(lengths[i] for i in p) for p in parts]
    return max(loads) / max(min(loads), 1)


@dataclass
class MicroBatch:
    indices: list[int]
    tokens: int


def micro_batches(lengths: Sequence[int], indices: Sequence[int], token_budget: int) -> list[MicroBatch]:
    """First-fit-decreasing packing of ``indices`` into micro batches of at most ``token_budget`` tokens (a sequence
    longer than the budget gets its own micro batch: it is never split, splitting would cut the causal context)."""
    bins: list[MicroBatch] = []
    for i in sorted(indices, key=lambda j: (-lengths[j], j)):
        for b in bins:
            if b.tokens + lengths[i] <= token_budget:
                b.indices.append(i)
                b.tokens += lengths[i]
                break
        else:
            bins.append(MicroBatch([i], lengths[i]))
    for b in bins:
        b.indices.sort()
    return bins


def loss_scale(micro_tokens: int, global_tokens: int) -> float:
    """Gradient-accumulation factor for ``token_mean_global``: each micro batch's token-sum loss divided by the global
    token count, so Σ over micro batches equals the full-batch token mean exactly."""
    return micro_tokens / max(global_tokens, 1)


def plan(lengths: Sequence[int], *, dp: int, token_budget: int) -> dict:
    """The complete placement of one global batch: per DP rank, its micro batches."""
    parts = balance_partitions(lengths, dp)
    return {"ranks": [[m.__dict__ for m in micro_batches(lengths, p, token_budget)] for p in parts], "imbalance": round(imbalance(lengths, parts), 4),
            "global_tokens": int(sum(lengths))}
