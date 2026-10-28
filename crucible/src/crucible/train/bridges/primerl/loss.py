"""The policy-gradient losses Crucible trains with, as a prime-rl custom loss (``trainer.loss.type = "custom"``).

prime-rl (Apache-2.0, pinned v0.10.0) calls ``crucible_loss(inputs, **kwargs)`` once per packed sequence with
``LossInputs`` (``prime_rl/trainer/rl/loss.py:21-37``): the learner's and the sampler's per-token logprobs, the
per-token advantages Crucible sent (``wire.to_sample``), and the loss mask of the sampled tokens. Returned: a
``LossOutputs``-shaped object (``loss`` and ``metrics``), prime-rl's own class when importable.

The objectives are the Tinker ones the Tinker bridge trains with (Tinker docs, losses; ``bridges/tinker/fake.py``), so a
run means the same thing on both backends. With r = exp(log p_θ − log q) per sampled token and A the advantage:

| ``kind`` | per-token loss |
|---|---|
| ``importance_sampling`` | −r·A (r capped at ``max_importance_ratio``, the cap's gradient kept straight-through) |
| ``ppo`` | −min(r·A, clip(r, ``ratio_low``, ``ratio_high``)·A) |
| ``cispo`` | −sg(clip(r, ``ratio_low``, ``ratio_high``))·A·log p_θ |

plus, when ``kl_coeff`` β > 0, the sampler–learner anchor of Tinker's ``stable_reinforce``: β·(log q − log p_θ), the k1
estimate of KL(q‖p_θ) on the sampled tokens, whose gradient −β∇log p_θ raises every sampled token regardless of its
advantage and so keeps the learner near the policy that produced the data (no reference model).

The sum is over the mask. prime-rl divides the rl component by the batch's trainable-token count; the bridge sends
advantages pre-multiplied by that count (``wire.undo_token_mean``), so the trained objective is exactly the one Crucible's
``loss_scale`` chose (``sum`` or ``token_mean``), as on Tinker. Metrics: the mean ratio, the mismatch KL
(E[r − 1 − log r]), the anchor's KL estimate, and the share of clipped tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

KINDS = ("importance_sampling", "ppo", "cispo")


@dataclass
class LossOutputsMirror:
    """``prime_rl.trainer.rl.loss.LossOutputs`` (``loss``: a scalar tensor; ``metrics``: name -> tensor), used when
    prime-rl is not importable (tests, the Mac)."""

    loss: Any
    metrics: dict[str, Any] = field(default_factory=dict)


def _outputs(loss: Any, metrics: dict[str, Any]) -> Any:
    try:
        from prime_rl.trainer.rl.loss import LossOutputs  # type: ignore[import-not-found]
    except ImportError:
        return LossOutputsMirror(loss, metrics)
    return LossOutputs(loss=loss, metrics=metrics)


def token_objective(lp_train: Any, lp_sample: Any, adv: Any, mask: Any, *, kind: str = "importance_sampling", kl_coeff: float = 0.0,
                    ratio_low: float = 0.8, ratio_high: float = 1.2, max_importance_ratio: float = 1e4) -> tuple[Any, dict[str, Any]]:
    """The summed loss over ``mask`` and its metrics (module doc). Tensors of one shape; ``mask`` boolean."""
    import torch

    if kind not in KINDS:
        raise ValueError(f"kind is one of {KINDS}, got {kind!r}")
    if kl_coeff < 0:
        raise ValueError("kl_coeff >= 0")
    m = mask.bool()
    lp, lq, a = lp_train[m], lp_sample[m].detach(), adv[m].detach()
    log_r = lp - lq
    zero = lp_train.sum() * 0.0  # keeps the graph when no token is trainable
    if lp.numel() == 0:
        return zero, {"ratio/mean": zero.detach(), "kl/mismatch": zero.detach(), "kl/anchor": zero.detach(), "clip/frac": zero.detach()}
    if kind == "importance_sampling":
        # a Python float bound: a torch.tensor here lives on the CPU and fails beside CUDA logprobs (smoke attempt 10)
        capped = log_r.detach().clamp(max=math.log(max_importance_ratio))
        r = torch.exp(capped + (log_r - log_r.detach()))
        per = -r * a
        clipped = (log_r.detach() > capped).float()
    else:
        r = torch.exp(log_r)
        rc = r.clamp(min=ratio_low, max=ratio_high)
        clipped = ((r.detach() < ratio_low) | (r.detach() > ratio_high)).float()
        per = -torch.minimum(r * a, rc * a) if kind == "ppo" else -rc.detach() * a * lp
    loss = per.sum()
    anchor = (lq - lp)
    if kl_coeff > 0:
        loss = loss + kl_coeff * anchor.sum()
    rd = torch.exp(log_r.detach())
    metrics = {"ratio/mean": rd.mean(), "kl/mismatch": (rd - 1.0 - log_r.detach()).mean(), "kl/anchor": anchor.detach().mean(),
               "clip/frac": clipped.mean()}
    return loss, metrics


def crucible_loss(inputs: Any, **kwargs: Any) -> Any:
    """prime-rl's custom-loss entry point (``trainer.loss.import_path = "crucible.train.bridges.primerl.loss.crucible_loss"``,
    ``kwargs`` from ``trainer.loss.kwargs``). ``loss_weights`` (prime-rl's rl weight stream) scales each token when present."""
    adv = inputs.advantages
    if getattr(inputs, "loss_weights", None) is not None:
        adv = adv * inputs.loss_weights
    loss, metrics = token_objective(inputs.trainer_logprobs, inputs.inference_logprobs, adv, inputs.loss_mask, **kwargs)
    return _outputs(loss, metrics)
