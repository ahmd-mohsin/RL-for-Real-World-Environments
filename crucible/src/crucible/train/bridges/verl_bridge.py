"""The verl bridge (research m11-m12/01 §G: DEPEND on verl v1 for scale). Pinned upstream: verl 5eb4826
(``RL-Environments/06-frameworks-and-env-code/infra-volcengine-verl``); every ``V/`` path below is relative to it.

How Crucible's credit reaches verl's trainer (M12 audit H4 V20/V21/V28, G2 H6/H7, pick 8):

1. **Registration in every process.** ``import verl`` loads every ``verl.plugins`` entry point (``V/verl/__init__.py:55-75``),
   in the driver and in each Ray actor (the trainer runs in ``TaskRunnerV1``, ``V/verl/trainer/main_ppo.py:102``).
   Crucible's ``pyproject.toml`` declares ``crucible = "crucible.train.bridges.verl_plugin"``; that module registers the
   ``crucible_delta`` estimator (``register_adv_est``, ``V/verl/trainer/ppo/core_algos.py:116-133``) and the
   ``crucible_agent`` agent loop (``@register``, ``V/verl/experimental/agent_loop/agent_loop.py:484-495``) at import.
2. **The only per-token channel is ``rm_scores``.** verl v1 (``trainer.use_v1: true``, the default,
   ``V/verl/trainer/config/ppo_trainer.yaml:232``) stores ``AgentLoopOutput.as_dict()`` in the TransferQueue
   (``V/verl/trainer/ppo/v1/agent_loop_tq.py:197-207``) and computes advantages from exactly ``uid``, ``response_mask``
   and ``rm_scores`` (``V/verl/trainer/ppo/v1/trainer_base.py:1743-1760``: ``token_level_scores = rm_scores``) through
   ``compute_advantage`` (``V/verl/trainer/ppo/ray_trainer.py:253-272``: a registered estimator receives
   ``token_level_rewards``, ``response_mask``, ``config`` and ``index`` only). ``extra_fields`` never reach an estimator.
   So ``CrucibleAgentLoopOutput.as_dict`` (``verl_agent_loop.py``) writes the per-token encoding below as ``rm_scores``.
3. **Encoding** (``token_rewards``), on a response of length L whose last sampled token is at ``l``:

       r_t      = c_k                    every sampled token t < l of call k (c_k: M6 turn credit, group rule applied)
       r_{L-1} += R − Σ_t r_t           so the row sums to the outcome R

   The row sum is R, which is what verl's own outcome estimators and metrics read (``core_algos.py:304``
   ``scores = token_level_rewards.sum(dim=-1)``; ``trainer_base.py:1121``), wherever verl put the scalar: the last
   response position (``agent_loop.py:143-148``), the last attended position (legacy ``agent_loop.py:1110-1116``) or
   ``valid_response_length - 1`` (``V/verl/experimental/reward_loop/reward_manager/base.py:78-80``). The final sampled
   token carries no credit of its own (one token per trajectory).
4. **Decoding** (``crucible_delta``): ``R`` = the row sum; ``A_t = (R − mean_group R) + r_t`` on sampled tokens before the
   last one, ``R − mean`` on the last one, 0 on observation tokens (Dr.GRPO: no std). Rows with no sampled token (masked
   rollouts: a placeholder row) and rows whose R equals ``config.invalid_reward_value`` (MiMo-verl's sentinel) get A = 0
   and are left out of the group mean: never trained as 0. ``algorithm.use_kl_in_reward`` is refused: it adds −β·kl_t to
   every sampled token (``ray_trainer.py:93-107``), which this encoding would read as turn credit (keep KL in the loss,
   ``actor.use_kl_loss``). A group of one has A = 0.

Limits: the legacy trainer (``use_v1: false``, deprecated) pads ``_InternalAgentLoopOutput`` from the scalar
``reward_score`` (``agent_loop.py:1110-1116``), so it trains the outcome only (R is still exact: decoding sums the row).
A rollout split by compaction gives one row per segment, and each row counts once in its uid's mean (verl's own
multi-output semantics for non-GRPO estimators, ``V/verl/trainer/ppo/v1/utils.py:166-176``).
"""

from __future__ import annotations

from typing import Any

import torch

from crucible.train.bridges.adapters import VERL_ESTIMATOR, cfg_get
from crucible.train.bridges.common import TrainSeq

ESTIMATOR = VERL_ESTIMATOR
AGENT_LOOP = "crucible_agent"


def token_rewards(s: TrainSeq, *, turn_credit: dict[str, float] | None = None, length: int | None = None) -> list[float]:
    """The ``rm_scores`` row for ``s`` (encoding in the module docstring). ``length``: the response length verl keeps
    (``rollout.response_length``; the row is cut there, and R still closes it)."""
    n = len(s.response_ids) if length is None else min(length, len(s.response_ids))
    mask = s.response_mask[:n]
    r = [0.0] * n
    if not n:
        return r
    last = max((i for i, m in enumerate(mask) if m), default=None)
    for cid, a, b in s.call_spans:
        c = float((turn_credit or {}).get(cid, 0.0))
        for i in range(a, min(b, n)):
            if mask[i] and last is not None and i < last:
                r[i] = c
    r[n - 1] += float(s.reward or 0.0) - sum(r)
    return r


_cfg_get = cfg_get


def crucible_delta(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: Any = None, config: Any = None, **_: Any):
    """verl advantage estimator (``compute_advantage`` call shape, ``ray_trainer.py:253-272``). ``token_level_rewards``,
    ``response_mask``: [B, T]; ``index``: the group uid per row. Returns ``(advantages, returns)``."""
    if _cfg_get(config, "use_kl_in_reward", False):
        raise ValueError("crucible_delta reads per-token rewards as turn credit; algorithm.use_kl_in_reward would be read as credit. "
                         "Use actor.use_kl_loss instead.")
    m = response_mask.bool()
    B, T = m.shape
    pos = torch.arange(T, device=m.device).expand(B, T)
    last = torch.where(m, pos, torch.full_like(pos, -1)).max(dim=1).values if T else torch.full((B,), -1, device=m.device)
    R = token_level_rewards.sum(dim=-1)
    valid = last >= 0
    invalid = _cfg_get(config, "invalid_reward_value", None)
    if invalid is not None:
        valid = valid & ~torch.isclose(R, torch.full_like(R, float(invalid)))
    groups = [str(x) for x in index] if index is not None else ["0"] * B
    base = torch.zeros_like(R)
    for g in set(groups):
        rows = torch.tensor([i for i, x in enumerate(groups) if x == g and bool(valid[i])], dtype=torch.long, device=m.device)
        if len(rows):
            base[rows] = R[rows].mean()
    credit = torch.where(m & (pos < last[:, None]), token_level_rewards, torch.zeros_like(token_level_rewards))
    adv = ((R - base)[:, None] + credit) * m * valid[:, None]
    return adv, adv


def register_verl() -> None:
    """Register the estimator and the agent loop in the current process (what the ``verl.plugins`` entry point does)."""
    from crucible.train.bridges.verl_plugin import register

    register()
