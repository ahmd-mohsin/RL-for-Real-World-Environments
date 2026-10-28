"""The policy-gradient losses Crucible trains with, as a prime-rl custom loss (``trainer.loss.type = "custom"``).

prime-rl (Apache-2.0, pinned v0.10.0) calls ``crucible_loss(inputs, **kwargs)`` once per packed sequence with
``LossInputs`` (``prime_rl/trainer/rl/loss.py:21-37``): the learner's and the sampler's per-token logprobs, the
per-token advantages Crucible sent (``wire.to_sample``), and the loss mask of the sampled tokens. Returned: a
``LossOutputs``-shaped object (``loss`` and ``metrics``), prime-rl's own class when importable.

**The objective has two axes.** ``kind`` is the Tinker objective, so a run means the same thing on both backends (Tinker
docs, losses; ``bridges/tinker/fake.py``). ``pg_correction`` is the off-policy correction of score centering's grid
(arXiv 2609.20807; reference code martin-marek/score-centering, MIT). With r = exp(log p_θ − log q) per sampled token,
A the advantage, w a detached weight and sg the stop-gradient:

| ``pg_correction`` | per-token loss | ``kind`` |
|---|---|---|
| ``none`` | the ``kind`` objective below, unchanged | any |
| ``tis`` / ``mis`` | −A·sg(w(r))·log p_θ, w = TIS min(r, 2) / MIS r·𝟙[0.5 ≤ r ≤ 5] | ``importance_sampling`` |
| ``sc`` / ``sc+tis`` / ``sc+mis`` (**default ``sc+mis``**) | −A·(sg(w(r))·log p_θ − Σ_{v∈H} sg(q_v·w(p_v/q_v) − α·p_v)·log p_v), w = 1 / TIS / MIS | ``importance_sampling`` |

| ``kind`` (``pg_correction = none``) | per-token loss |
|---|---|
| ``importance_sampling`` | −r·A (naive IS; r capped at ``max_importance_ratio``, the cap's gradient kept straight-through) |
| ``ppo`` | −min(r·A, clip(r, ``ratio_low``, ``ratio_high``)·A) |
| ``cispo`` | −sg(clip(r, ``ratio_low``, ``ratio_high``))·A·log p_θ |

**Score centering** removes the drift E_q[R]·s̄ of the policy gradient under a training–inference mismatch, where
s̄ = Σ_v q_v ∇log p_v is the score's mean under the sampler q (zero only on-policy). Subtracting the score's expectation
under the effective sampling mass q·w leaves the signal Cov_q(R, s) and nothing that depends on the mean reward.
The sampler reports its top-k head H per token (``topk``). The tail is modeled as ρ·p_v with
ρ = (1 − Σ_H q)/(1 − Σ_H p), and α = ρ·w(1/ρ), so the tail's centering term is α·Σ_tail p_v∇log p_v
= −α·Σ_H p_v∇log p_v: only the k head logprobs carry gradient. As the reference code says: float32, the head is not
renormalized, and the sampled token's own logprob is kept when it lies outside the head. With exact IS weights
(w = r) the centering term vanishes, and on-policy (q = p) SC is plain policy gradient (tests).

**No extra ratio clip.** The paper uses SC alone or composed with TIS / MIS, never with a PPO clip; TRL's async GRPO trainer
replaces its clipped objective when score centering is on. So ``pg_correction != "none"`` requires ``kind =
"importance_sampling"`` and ignores ``ratio_low`` / ``ratio_high``. PPO and CISPO stay available with ``pg_correction =
"none"`` (swarm-pipeline/14 §7.3).

**The sampler anchor** (``kl_coeff`` β > 0, Tinker's ``stable_reinforce``): β·(log q − log p_θ), the k1 estimate of
KL(q‖p_θ) on the sampled tokens. Its gradient −β∇log p_θ raises every sampled token regardless of its advantage. In
expectation it is β·∇KL(q‖p_θ), a deliberate pull toward the sampler of known strength, independent of the reward, so it
composes with every row above. Default 0.

**Values.** The SC / TIS / MIS surrogates are written so their value is the IS surrogate's, −A·w (−A·sg(w)·e^(log p −
sg(log p)); the centering term adds 0), and only their gradients differ. On-policy every row has the value −A, as
``importance_sampling`` does.

**Missing heads.** When a sequence has no heads (``sc_head`` delivers none: the bridge sent none for the step, because
one of its sequences lacked a head at a trained position), a centered correction falls back to its importance part for
the whole step: ``sc`` and ``sc+mis`` to MIS, ``sc+tis`` to TIS, logged as ``sc/fallback`` = 1. A sequence with heads at
some trained positions but not all is an error, never a partial centering.

The sum is over the mask. prime-rl divides the rl component by the batch's trainable-token count; the bridge sends
advantages pre-multiplied by that count (``wire.undo_token_mean``), so the trained objective is exactly the one Crucible's
``loss_scale`` chose (``sum`` or ``token_mean``), as on Tinker. The advantages are the loop's own (Dr.GRPO, honesty, M6
turn credit) and unchanged here.

Metrics: the mean ratio, the mismatch KL (E[r − 1 − log r]), the anchor's KL estimate, the share of corrected tokens
(``clip/frac``), MIS's dropped share (``is/masked_frac``) and, when centered: head coverage Σ_H q (``sc/head_mass``,
``sc/head_mass_min``), ρ (``sc/rho``, ``sc/rho_min``, ``sc/rho_max``), the drift proxy ``sc/score_mean_l1`` (the L1 norm of
the logit-space score mean, Σ_H |q_v − p_v| + |q_tail − p_tail|, a lower bound of 2·TV(q, p): s̄ in logit space is q − p),
the head KL(q‖p) estimate (``sc/kl_head``), the share of tails MIS dropped (``sc/tail_dropped``) and ``sc/align_err``
(the largest gap between the sampled token's logprob and its own head entry: 0 when the heads are aligned).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Any

KINDS = ("importance_sampling", "ppo", "cispo")
CORRECTIONS = ("none", "tis", "mis", "sc", "sc+tis", "sc+mis")
DEFAULT_CORRECTION = "sc+mis"
MIS_LOW, MIS_HIGH = 0.5, 5.0  # the paper's MIS band (and TRL's vllm_importance_sampling_clip_min / _max)
TIS_CAP = 2.0  # the paper's TIS cap
TAIL_EPS = 1e-6  # the reference code's floor on the tail masses
LOG_RATIO_CLAMP = 20.0  # |log r| bound before exp (the reference code and crucible.train.mismatch)


def parse_correction(pg_correction: str) -> tuple[bool, str]:
    """``(centered, weight)``: weight is ``one`` (no IS), ``tis`` or ``mis``; ``none`` is ``(False, "kind")``."""
    if pg_correction not in CORRECTIONS:
        raise ValueError(f"pg_correction is one of {CORRECTIONS}, got {pg_correction!r}")
    if pg_correction == "none":
        return False, "kind"
    if pg_correction.startswith("sc"):
        return True, pg_correction.partition("+")[2] or "one"
    return False, pg_correction


def fallback_weight(pg_correction: str) -> str:
    """The importance part a centered correction keeps when the step has no heads (module doc)."""
    centered, w = parse_correction(pg_correction)
    return ("tis" if w == "tis" else "mis") if centered else w


def check_combination(kind: str, pg_correction: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"kind is one of {KINDS}, got {kind!r}")
    parse_correction(pg_correction)
    if pg_correction != "none" and kind != "importance_sampling":
        raise ValueError(f"pg_correction={pg_correction!r} replaces the ratio clip of {kind!r}: score centering and TIS/MIS run with no PPO clip "
                         f"(the paper's evidence). Use kind='importance_sampling' with it, or pg_correction='none' for {kind}")


@dataclass
class Head:
    """One sequence's sampler head at its sampled-token positions (``sc_head`` builds it in the trainer).

    ``lp``: the learner's logprobs of the head ids ``[T, k]`` (with gradient); ``lq``: the sampler's; ``valid``: real
    entries; ``present``: positions with a head; ``lq_label``: the sampled token's logprob read off its own head entry
    (NaN where it is not in the head), for ``sc/align_err``."""

    lp: Any
    lq: Any
    valid: Any
    present: Any
    lq_label: Any = None


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


def mass(lp: Any, lq: Any, weight: str, *, mis_low: float = MIS_LOW, mis_high: float = MIS_HIGH, tis_cap: float = TIS_CAP) -> Any:
    """q·w(p/q) from log p and log q, exactly (no division): the effective sampling mass of the score-centering grid. The
    same function gives the token weight w(r) as ``mass(log r, 0)`` and the tail factor α = ρ·w(1/ρ) as ``mass(0, log ρ)``
    (the reference code's ``log_mass``). MIS's band edges are inclusive and tested in log space."""
    import torch

    lp, lq = torch.as_tensor(lp), torch.as_tensor(lq)
    lp, lq = torch.broadcast_tensors(lp, lq.to(lp.dtype))
    if weight == "one":
        return torch.exp(lq)
    if weight == "is":
        return torch.exp(lp)
    if weight == "tis":
        return torch.exp(torch.minimum(lp, lq + math.log(tis_cap)))
    if weight == "mis":
        d = lp - lq
        band = (d >= math.log(mis_low)) & (d <= math.log(mis_high))  # float bounds: no CPU tensor beside CUDA (smoke attempt 10)
        return torch.where(band, torch.exp(lp), torch.zeros_like(lp))
    raise ValueError(f"unknown weight {weight!r}")


def centering(head: Head, m: Any, weight: str, **band: float) -> tuple[Any, dict[str, Any]]:
    """The centering term per trained token, Σ_H sg(q_v·w(p_v/q_v) − α·p_v)·(log p_v − sg(log p_v)) (value 0, the
    gradient of the module doc's formula), and its diagnostics. ``m`` selects the trained positions; ``band``: ``mass``'s
    ``mis_low`` / ``mis_high`` / ``tis_cap``."""
    import torch

    dt = torch.promote_types(head.lp.dtype, torch.float32)
    lp_h = head.lp[m].to(dt)
    lq_h = head.lq[m].to(dt).detach()
    valid = head.valid[m].bool()
    lp_d = lp_h.detach()
    p_h = torch.where(valid, torch.exp(lp_d), torch.zeros_like(lp_d))
    q_h = torch.where(valid, torch.exp(lq_h), torch.zeros_like(lq_h))
    p_tail = (1.0 - p_h.sum(-1)).clamp_min(TAIL_EPS)
    q_tail = (1.0 - q_h.sum(-1)).clamp_min(TAIL_EPS)
    log_rho = torch.log(q_tail) - torch.log(p_tail)
    alpha = mass(torch.zeros_like(log_rho), log_rho, weight, **band)
    hm = torch.where(valid, mass(lp_d, lq_h, weight, **band), torch.zeros_like(lp_d))
    residual = torch.where(valid, hm - alpha.unsqueeze(-1) * p_h, torch.zeros_like(lp_d))
    lp_safe = torch.where(valid, lp_h, torch.zeros_like(lp_h))
    term = (residual.detach() * (lp_safe - lp_safe.detach())).sum(-1)
    rho = torch.exp(log_rho)
    head_q = q_h.sum(-1)
    kl_head = torch.where(valid, q_h * (lq_h - lp_d), torch.zeros_like(lp_d)).sum(-1) + q_tail * log_rho
    metrics = {"sc/head_mass": head_q.mean(), "sc/head_mass_min": head_q.min(), "sc/rho": rho.mean(), "sc/rho_min": rho.min(),
               "sc/rho_max": rho.max(), "sc/score_mean_l1": ((q_h - p_h).abs().sum(-1) + (q_tail - p_tail).abs()).mean(),
               "sc/kl_head": kl_head.mean(), "sc/tail_dropped": (alpha == 0).to(dt).mean(), "sc/fallback": torch.zeros((), dtype=dt, device=lp_d.device)}
    return term, metrics


def token_objective(lp_train: Any, lp_sample: Any, adv: Any, mask: Any, *, kind: str = "importance_sampling", kl_coeff: float = 0.0,
                    ratio_low: float = 0.8, ratio_high: float = 1.2, max_importance_ratio: float = 1e4, pg_correction: str = "none",
                    head: Head | None = None, mis_low: float = MIS_LOW, mis_high: float = MIS_HIGH, tis_cap: float = TIS_CAP) -> tuple[Any, dict[str, Any]]:
    """The summed loss over ``mask`` and its metrics (module doc). Tensors of one shape; ``mask`` boolean; ``head`` the
    sequence's sampler head (centered corrections; None falls back, module doc); ``mis_low`` / ``mis_high`` / ``tis_cap``
    the MIS band and the TIS cap (the paper's 0.5, 5 and 2)."""
    import torch

    band = {"mis_low": mis_low, "mis_high": mis_high, "tis_cap": tis_cap}
    if not 0.0 < mis_low <= 1.0 <= mis_high or tis_cap < 1.0:
        raise ValueError("the MIS band holds 1 (0 < mis_low <= 1 <= mis_high) and tis_cap >= 1")

    check_combination(kind, pg_correction)
    if kl_coeff < 0:
        raise ValueError("kl_coeff >= 0")
    centered, weight = parse_correction(pg_correction)
    m = mask.bool()
    lp, lq, a = lp_train[m], lp_sample[m].detach(), adv[m].detach()
    log_r = lp - lq
    zero = lp_train.sum() * 0.0  # keeps the graph when no token is trainable
    if lp.numel() == 0:
        return zero, {"ratio/mean": zero.detach(), "kl/mismatch": zero.detach(), "kl/anchor": zero.detach(), "clip/frac": zero.detach()}
    extra: dict[str, Any] = {}
    if weight == "kind":
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
    else:
        if centered and head is not None:
            present = head.present[m].bool()
            if not bool(present.all()):
                raise ValueError(f"{int((~present).sum())} of {present.numel()} trained tokens have no sampler head: a step centers all or none")
        elif centered:
            weight = fallback_weight(pg_correction)
            extra["sc/fallback"] = torch.ones((), dtype=torch.float32, device=log_r.device)
        lr_d = log_r.detach().clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
        w = mass(lr_d, torch.zeros_like(lr_d), weight, **band)
        per = -a * w * torch.exp(lp - lp.detach())  # value −A·w, gradient −A·w·∇log p (module doc)
        clipped = (w != torch.exp(lr_d)).float() if weight != "one" else torch.zeros_like(w)
        extra["is/masked_frac"] = (w == 0).float().mean()
        if centered and head is not None:
            term, sc_metrics = centering(head, m, weight, **band)
            per = per + a * term.to(per.dtype)
            extra.update(sc_metrics)
            if head.lq_label is not None:
                gap = (head.lq_label[m].to(lq.dtype) - lq).abs()
                gap = gap[~torch.isnan(gap)]
                extra["sc/align_err"] = gap.max() if gap.numel() else torch.zeros((), dtype=lq.dtype, device=lq.device)
    loss = per.sum()
    anchor = (lq - lp)
    if kl_coeff > 0:
        loss = loss + kl_coeff * anchor.sum()
    rd = torch.exp(log_r.detach())
    metrics = {"ratio/mean": rd.mean(), "kl/mismatch": (rd - 1.0 - log_r.detach()).mean(), "kl/anchor": anchor.detach().mean(),
               "clip/frac": clipped.mean(), **{k: v.detach() for k, v in extra.items()}}
    return loss, metrics


def crucible_loss(inputs: Any, **kwargs: Any) -> Any:
    """prime-rl's custom-loss entry point (``trainer.loss.import_path = "crucible.train.bridges.primerl.loss.crucible_loss"``,
    ``kwargs`` from ``trainer.loss.kwargs``; ``pg_correction`` defaults to ``sc+mis``). ``loss_weights`` (prime-rl's rl
    weight stream) scales each token when present. The sampler head of the sequence comes from ``sc_head`` (the trainer
    hook), keyed by the ``trainer_logprobs`` tensor prime-rl passes; outside the trainer there is none."""
    kwargs.setdefault("pg_correction", DEFAULT_CORRECTION)
    adv = inputs.advantages
    if getattr(inputs, "loss_weights", None) is not None:
        adv = adv * inputs.loss_weights
    head = None
    if kwargs["pg_correction"].startswith("sc") and "crucible.train.bridges.primerl.sc_head" in sys.modules:
        head = sys.modules["crucible.train.bridges.primerl.sc_head"].lookup(inputs.trainer_logprobs)
    loss, metrics = token_objective(inputs.trainer_logprobs, inputs.inference_logprobs, adv, inputs.loss_mask, head=head, **kwargs)
    return _outputs(loss, metrics)


def _install_trainer_hook() -> None:
    """In prime-rl's trainer process (its ``train`` module loaded: the loss is imported by ``setup_rl_loss_fn``), install
    the LM-head hook that delivers the sampler heads (``sc_head``). Elsewhere (the loop, tests, the Mac) nothing."""
    if any(getattr(mod, "setup_rl_loss_fn", None) is not None and getattr(mod, "compute_loss", None) is not None and hasattr(mod, "gather_for_cp")
           for mod in list(sys.modules.values()) if mod is not None and getattr(mod, "__name__", "") in ("prime_rl.trainer.rl.train", "__main__")):
        from crucible.train.bridges.primerl import sc_head

        sc_head.install()


_install_trainer_hook()
