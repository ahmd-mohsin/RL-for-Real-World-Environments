"""Per-trainer field mappings over ``common.TrainSeq`` (research m11-m12/01, /03, /04). Pure functions: they produce the
framework's own field names and shapes; the framework-side registration (verl agent loop, prime-rl algorithm, SkyRL
estimator) imports the framework lazily in ``plugins``.

- verl ``AgentLoopOutput`` (``verl/experimental/agent_loop/agent_loop.py``): prompt_ids, response_ids, response_mask,
  response_logprobs, reward_score, num_turns, extra_fields (Crucible token advantages for a registered estimator);
- SkyRL ``GeneratorOutput`` (``skyrl/train/generators/base.py``): prompt_token_ids, response_ids, rewards (per-token lists:
  token-level estimators honour them; GRPO sums them, so the per-token reward vector puts the outcome on the last
  sampled token), loss_masks, stop_reasons, rollout_logprobs;
- TRL ``rollout_func`` return (``trl/trainer/grpo_trainer.py``): prompt_ids, completion_ids, logprobs, env_mask + extras;
- rLLM ``Step`` with ``advantage`` as a per-token list (``use_precomputed_advantage``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from crucible.train.bridges.common import TrainSeq


def verl_agent_loop_output(s: TrainSeq) -> dict[str, Any]:
    return {"prompt_ids": s.prompt_ids, "response_ids": s.response_ids, "response_mask": s.response_mask, "response_logprobs": s.logprobs,
            "reward_score": s.reward, "num_turns": s.num_turns,
            "extra_fields": {"crucible_token_advantages": s.advantages, "crucible_versions": s.versions, "crucible_rollout": s.rollout_id,
                             "crucible_segment": s.segment}}


def skyrl_generator_output(seqs: Sequence[TrainSeq], *, token_rewards: bool = True) -> dict[str, Any]:
    rewards: list[Any] = []
    for s in seqs:
        if token_rewards:
            r = [0.0] * len(s.response_ids)
            last = max((i for i, m in enumerate(s.response_mask) if m), default=None)
            if last is not None:
                r[last] = float(s.reward or 0.0)
            rewards.append(r)
        else:
            rewards.append(float(s.reward or 0.0))
    return {"prompt_token_ids": [s.prompt_ids for s in seqs], "response_ids": [s.response_ids for s in seqs], "rewards": rewards,
            "loss_masks": [s.response_mask for s in seqs], "stop_reasons": ["stop"] * len(seqs), "rollout_logprobs": [s.logprobs for s in seqs]}


def trl_rollout(seqs: Sequence[TrainSeq]) -> dict[str, Any]:
    return {"prompt_ids": [s.prompt_ids for s in seqs], "completion_ids": [s.response_ids for s in seqs], "logprobs": [s.logprobs for s in seqs],
            "env_mask": [s.response_mask for s in seqs], "crucible_reward": [s.reward for s in seqs],
            "crucible_versions": [s.versions for s in seqs]}


def rllm_steps(s: TrainSeq) -> list[dict[str, Any]]:
    """One rLLM ``Step`` per call, ``advantage`` a per-token list on that call's completion (precomputed advantages)."""
    out = []
    for cid, a, b in s.call_spans:
        out.append({"id": cid, "prompt_ids": s.prompt_ids + s.response_ids[:a], "response_ids": s.response_ids[a:b], "logprobs": s.logprobs[a:b],
                    "advantage": s.advantages[a:b], "reward": s.reward if b == s.call_spans[-1][2] else 0.0})
    return out
