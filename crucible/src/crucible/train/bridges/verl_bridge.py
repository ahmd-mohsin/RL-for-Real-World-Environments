"""The verl bridge (research m11-m12/01 §G: DEPEND on verl v1 for scale; plug points ``register_adv_est``
``core_algos.py:116``, ``AgentLoopOutput`` ``agent_loop.py:88-115``, ``@register`` ``:484-495``).

verl calls an advantage estimator with ``token_level_rewards``, ``response_mask``, ``index`` (group uid) and ``config``
only, so Crucible's dense credit travels **inside the token rewards** with a fixed encoding:

    every sampled token of call k  ← c_k  (M6 turn credit of call k, centred within the group by Crucible)
    the last sampled token         ← R    (the outcome; that token's own c is dropped: one token of the trajectory)

and ``crucible_delta`` returns A_t = (R − mean_group R) + c_t on sampled tokens (A = R − mean on the last one); masked
rollouts (no sampled token) are excluded from the group mean. With no turn credit this is outcome GRPO (Dr.GRPO: no
std). ``token_rewards(seq)`` builds the encoding from a flattened ``TrainSeq``; ``register_verl()`` registers the
estimator in verl's own registry when verl is installed (lazy import).
"""

from __future__ import annotations

import torch

from crucible.train.bridges.common import TrainSeq


def token_rewards(s: TrainSeq, *, turn_credit: dict[str, float] | None = None) -> list[float]:
    r = [0.0] * len(s.response_ids)
    for cid, a, b in s.call_spans:
        c = (turn_credit or {}).get(cid, 0.0)
        for i in range(a, b):
            if s.response_mask[i]:
                r[i] = c
    last = max((i for i, m in enumerate(s.response_mask) if m), default=None)
    if last is not None:
        r[last] = float(s.reward or 0.0)
    return r


def crucible_delta(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index=None, config=None, **_):
    """``token_level_rewards``/``response_mask``: [B, T]; ``index``: group ids per row. Returns (advantages, returns)."""
    m = response_mask.bool()
    B, T = m.shape
    pos = torch.arange(T, device=m.device).expand(B, T)
    last = torch.where(m, pos, torch.full_like(pos, -1)).max(dim=1).values  # last sampled token per row (-1: none)
    valid = last >= 0
    R = torch.zeros(B, dtype=token_level_rewards.dtype, device=m.device)
    R[valid] = token_level_rewards[torch.arange(B, device=m.device)[valid], last[valid]]
    groups = list(index) if index is not None else [0] * B
    base = torch.zeros(B, dtype=R.dtype, device=m.device)
    for g in set(groups):
        rows = torch.tensor([i for i, x in enumerate(groups) if x == g and bool(valid[i])], dtype=torch.long, device=m.device)
        if len(rows):
            base[rows] = R[rows].mean()
    outcome = (R - base)[:, None].expand(B, T)
    credit = torch.where(pos == last[:, None], torch.zeros_like(token_level_rewards), token_level_rewards)
    adv = (outcome + credit) * m
    return adv, adv


def register_verl() -> None:
    from verl.trainer.ppo.core_algos import register_adv_est

    register_adv_est("crucible_delta")(crucible_delta)
