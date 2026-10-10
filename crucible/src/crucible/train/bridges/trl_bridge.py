"""The TRL bridge (research m11-m12/04 §C.4, §C.7: ``GRPOTrainer`` colocate + ``rollout_func`` is the single-GPU path TRL
supports as-is). ``make_rollout_func`` returns ``rollout_func(prompts, trainer) -> dict`` that runs one Crucible World
episode per prompt × ``num_generations`` (concurrently: per-rollout dispatch, environments off the GPU), with the
engine call bound to TRL's colocated vLLM (``trainer.vllm_generation.generate``: token ids in; ids + logprobs out) or
any ``gen(prompt_ids, max_tokens, temperature)``. The return carries TRL's fields (``prompt_ids``, ``completion_ids``,
``logprobs``, ``env_mask``) plus ``crucible_reward`` for a reward function that returns it. A masked episode yields
``None`` reward: TRL treats None as unscorable (excluded from the baseline), never 0.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from crucible.train.bridges.adapters import trl_rollout
from crucible.train.bridges.common import flatten
from crucible.train.engine import GenResult


def trl_gen(trainer: Any) -> Callable:
    """``gen`` over TRL's colocated vLLM generation (``vllm_generation.generate(prompt_ids, images, num_generations)``)."""
    async def gen(ids: list[int], max_tokens: int, temperature: float) -> GenResult:
        out = await asyncio.to_thread(trainer.vllm_generation.generate, [ids], None, 1)
        _, comp, lps = out[0][0], out[1][0], out[2][0]
        return GenResult(list(comp), [float(x) for x in lps], int(getattr(trainer.state, "global_step", 0)), "stop", None, len(ids))
    return gen


def make_rollout_func(rollout: Callable, *, num_generations: int, gen_factory: Callable[[Any], Callable] = trl_gen, temperature: float = 1.0):
    """``rollout``: a ``world_env.world_rollout(...)`` function (rid, prompt, seed, gen, temperature=…) -> RolloutRecord | None."""
    def rollout_func(prompts: list[Any], trainer: Any) -> dict[str, Any]:
        gen = gen_factory(trainer)

        async def all_():
            jobs = [rollout(f"trl-{i}-{k}", str(p), k, gen, temperature=temperature) for i, p in enumerate(prompts) for k in range(num_generations)]
            return await asyncio.gather(*jobs, return_exceptions=True)

        recs = asyncio.run(all_())
        seqs, rewards = [], []
        for r in recs:
            if isinstance(r, Exception) or r is None:  # masked: an empty sample with reward None (unscorable in TRL)
                seqs.append(None)
                rewards.append(None)
                continue
            s = flatten(r)
            seqs.append(s[0] if s else None)  # TRL rows are single sequences: the first segment (compaction → use the native runtime)
            rewards.append(r.reward)
        out = trl_rollout([s for s in seqs if s is not None])
        out["crucible_reward"] = rewards
        return out

    return rollout_func
