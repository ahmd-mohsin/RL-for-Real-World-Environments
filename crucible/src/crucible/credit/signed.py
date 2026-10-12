# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Crucible contributors (port to pure Python; Crucible mark sources)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``adv_signed``: sign-aware token penalty with per-sign mass conservation (PORT of Xiaomi's verl fork,
``verl/trainer/ppo/signed_rebalance.py:46-172``; chosen in the M5-M7 sweep as the best M6 credit operator).

A marked call's tokens are treated by the sign of their rollout's outcome advantage:

  positive rollout, marked token -> 0 (stop reinforcing the span)
  negative rollout, marked token -> x kappa (kappa >= 1: punish it harder than the rest of its trajectory)

Each sign then keeps its total mass over the batch: what the positive marked tokens lose is handed back to the clean
tokens of positive rollouts by one factor alpha >= 1 (capped at ``max_scale``); what the negative marked tokens gain is
taken back from the clean negative tokens by one factor beta <= 1 (floored at ``min_scale``). Subtracting a penalty
(Crucible's ``tool_format_penalty`` reduction) instead adds net negative mass to the batch, which flattens the policy
and lengthens outputs (-5.0 on a toy batch where this keeps +10 and -10).

Marks are per model call: ``kappa`` of the strongest rule that hit it (format errors, violations, confirmed hack
flags, M4 negative step signals), 0 = unmarked. Rows with a zero outcome advantage are left untouched.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

logger = logging.getLogger(__name__)


def solve_signed_factors(removed: float, pos_base: float, added: float, neg_base: float, *, min_scale: float,
                         max_scale: float) -> tuple[float, bool, float, bool]:
    """``(alpha, alpha_clamped, beta, beta_clamped)``. A clamped side is not conserved; a side with no clean tokens
    keeps factor 1 and is flagged clamped."""
    if not (0 < min_scale <= 1.0 and max_scale >= 1.0):
        raise ValueError(f"need 0 < min_scale <= 1 <= max_scale, got {min_scale}, {max_scale}")
    alpha, a_cl = 1.0, False
    if removed != 0.0:
        if pos_base > 0.0:
            alpha = 1.0 + removed / pos_base
            if alpha > max_scale:
                alpha, a_cl = max_scale, True
            elif alpha < 0.0:
                alpha, a_cl = 1.0, True
        else:
            a_cl = True
    beta, b_cl = 1.0, False
    if added > 0.0:
        if neg_base > 0.0:
            beta = 1.0 - added / neg_base
            if beta < min_scale:
                beta, b_cl = min_scale, True
        else:
            b_cl = True
    if a_cl or b_cl:
        logger.warning("[adv_signed] rebalance clamped (alpha=%.4f, beta=%.4f): the penalty mass is comparable to the outcome mass", alpha, beta)
    return alpha, a_cl, beta, b_cl


def signed_rebalance(adv: Sequence[float], hit: Sequence[float], seq_sign: Sequence[int], *, min_scale: float = 0.5, max_scale: float = 2.0,
                     weight: Sequence[float] | None = None) -> tuple[list[float], dict[str, float]]:
    """Flat form over every trained token of the batch. ``hit``: kappa >= 1, or 0 = unmarked; ``seq_sign``: the sign of
    the token's rollout outcome advantage (0 = untouched); ``weight``: per-token loss weight (mass = weight x adv)."""
    n = len(adv)
    if not (len(hit) == n == len(seq_sign)) or (weight is not None and len(weight) != n):
        raise ValueError("adv, hit, seq_sign (and weight) must have one length")
    if any(not (h == 0 or h >= 1.0) for h in hit):
        raise ValueError("hit values must be 0 (no hit) or a multiplier >= 1")
    a = [float(x) for x in adv]
    w = [1.0] * n if weight is None else [float(x) for x in weight]
    pos_hit = [i for i in range(n) if seq_sign[i] > 0 and hit[i] > 0]
    pos_clean = [i for i in range(n) if seq_sign[i] > 0 and hit[i] == 0]
    neg_hit = [i for i in range(n) if seq_sign[i] < 0 and hit[i] > 0]
    neg_clean = [i for i in range(n) if seq_sign[i] < 0 and hit[i] == 0]

    def mass(idx: list[int]) -> float:
        return sum(w[i] * a[i] for i in idx)

    pos_pre, neg_pre = mass(pos_hit + pos_clean), mass(neg_hit + neg_clean)
    removed, pos_base = mass(pos_hit), mass(pos_clean)
    added = -sum((hit[i] - 1.0) * w[i] * a[i] for i in neg_hit)  # (kappa - 1)|A|, A < 0
    neg_base = -mass(neg_clean)
    alpha, a_cl, beta, b_cl = solve_signed_factors(removed, pos_base, added, neg_base, min_scale=min_scale, max_scale=max_scale)
    for i in pos_hit:
        a[i] = 0.0
    for i in pos_clean:
        a[i] *= alpha
    for i in neg_hit:
        a[i] *= hit[i]
    for i in neg_clean:
        a[i] *= beta
    pos_post, neg_post = mass(pos_hit + pos_clean), mass(neg_hit + neg_clean)
    if not a_cl and abs(pos_post - pos_pre) > 1e-6 * max(1.0, abs(pos_pre)):
        raise AssertionError(f"positive mass not conserved: {pos_pre} -> {pos_post}")
    if not b_cl and abs(neg_post - neg_pre) > 1e-6 * max(1.0, abs(neg_pre)):
        raise AssertionError(f"negative mass not conserved: {neg_pre} -> {neg_post}")
    return a, {"penalty/signed/pos_hit_tokens": len(pos_hit), "penalty/signed/pos_mass_removed": removed, "penalty/signed/pos_scale": alpha,
               "penalty/signed/pos_scale_clamped": int(a_cl), "penalty/signed/neg_hit_tokens": len(neg_hit), "penalty/signed/neg_mass_added": added,
               "penalty/signed/neg_scale": beta, "penalty/signed/neg_scale_clamped": int(b_cl), "penalty/signed/adv_pos_sum_pre": pos_pre,
               "penalty/signed/adv_pos_sum_post": pos_post, "penalty/signed/adv_neg_sum_pre": neg_pre, "penalty/signed/adv_neg_sum_post": neg_post}
