"""Train–inference mismatch and staleness corrections (DESIGN §10.3 point 4; research m11-m12/06 §5).

Notation (HisPO's factorization, art. §3.3.3.3):

    π_θ        current policy on the training engine          -> ``logp`` (requires grad)
    π_θold     the behaviour version, on the training engine   -> ``old_logp`` (optional)
    π_rollout  the behaviour version, on the inference engine  -> ``rollout_logp`` (recorded at sampling, TITO)

    r = π_θ / π_rollout = r_dis · r_stale,  r_dis = π_θold / π_rollout (engine mismatch),  r_stale = π_θ / π_θold (drift)

Every correction is a pure function returning the **per-token objective J_t to maximize** (the loss is
``-aggregate(J, mask)``, ``objectives.aggregate``) and its metrics. They differ only in how they weight / mask / clip:

| name     | weight / gate                                              | needs π_θold | source |
|----------|------------------------------------------------------------|--------------|--------|
| ppo      | min(r_s Â, clip(r_s, 1−ε_l, 1+ε_h) Â), dual clip c = 3     | yes          | PPO / DAPO clip-higher, verl |
| tis      | sg(min(r_dis, C)) · ppo                                    | yes          | Yao et al.; slime, verl, OpenRLHF |
| icepop   | sg(r_dis)·𝟙[α ≤ r_dis ≤ β] · ppo                            | yes          | Ring-1T (α=0.5, β=5) |
| kpop     | sg(r_dis)·𝟙[D_B(p‖q) ≤ φ]·𝟙[D_B(q‖p) ≤ φ] · ppo (binary KL) | yes          | Ling 2.6 (IcePop's skeleton) |
| dis      | sg(r)·𝟙[1−ε_l < r < 1+ε_h] · Â · log π_θ                   | **no**       | GLM / SAO (coding ε_l=0.8, ε_h=3.0) |
| mimo4    | sg(r)·𝟙[bounds by sign of Â] · Â · log π_θ                  | **no**       | MiMo-V2.6 ([0.2, 5] both signs) |
| mispo    | 𝟙[x_t ∈ tok]·𝟙[geomean x ∈ traj] · Â · log π_θ, x = r_dis    | yes          | StepFun (tok [0.5,2], traj [0.996,1.001]) |
| hispo    | seq + token gates on r_dis; three-bound clip on full r     | yes          | LongCat |
| cispo    | sg(clip(r_s, 1−ε_l, 1+ε_h)) · Â · log π_θ                  | yes          | MiniMax M1 (M2: lower bound 0) |
| gspo     | sequence ratio s_i = exp(mean log r_s), token form, clip   | yes          | Qwen (ε = 3e−4 / 4e−4) |
| ipo      | 𝟙[|π_θ − π_rollout| ≤ ε] · sg(min(r, cap)) · Â · log π_θ    | **no**       | prime-rl default (ε = 0.3) |

``tis`` / ``icepop`` with ``level="sequence"``: one weight per rollout, the product ρ = Π_t r_dis (TRL sequence_truncate /
sequence_mask; NeMo-RL's generation-IS weight under ``sequence_level_importance_ratios=True``). Not NeMo-RL's
``seq-mask-tis``: that gates on the geometric mean exp(mean_t log r_dis) and keeps token-level weights. Under rollout
sampling E[log r_dis] ≤ 0, so the product of a long rollout typically drifts down (``icepop`` masks it below α, ``tis``
weights it toward 0).
``cispo`` with ``mask=True``: MiniMax-M1's unified token mask (PPO's trust region).
Name map to other stacks: prime-rl's ``icepop`` (a [0.2, 5] band on π_θ/π_rollout, no PPO term) is Crucible's ``mimo4``;
Crucible's ``icepop`` is Ring-1T's (r_dis gate around a PPO term), as AReaL implements it.

Corrections that need π_θold refuse to run without it unless ``old_logprob_source`` is set: ``"rollout"`` (use
π_rollout, i.e. assume no engine mismatch) or ``"recompute_current"`` (π_θ detached, valid when K = 1 and on-policy).
DIS and mimo4 need only the recorded rollout logprobs, so they are valid under multi-version async with call-level
versions — hence the default ``dis(0.8, 3.0)`` (research m11-m12/06 §5.4).

All tensors are [B, T] (padded); ``mask`` marks trainable completion tokens; ``adv`` is per token (sequence
advantages broadcast, or M6 turn advantages on their call's tokens). ``seq_ids`` [B] names the sequence (rollout) of each
row: GSPO's ratio and the MIS-PO / HisPO trajectory gates are per rollout, however many rows (calls, segments) it has.

Numerics (M12 audit): every log ratio is clamped to ±20 before ``exp`` and gated-out tokens multiply a finite zero weight,
so one extreme behaviour logprob cannot turn the gradient into NaN; a non-finite behaviour or old logprob drops its token
(``nonfinite_tokens``). The PPO family applies the dual clip c = 3 by default (verl); ``dual_clip=0`` turns it off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

NEEDS_OLD = {"ppo", "tis", "icepop", "kpop", "mispo", "hispo", "cispo", "gspo"}


@dataclass
class Correction:
    name: str = "dis"
    params: dict[str, float] = field(default_factory=dict)
    old_logprob_source: str | None = None  # None | "rollout" | "recompute_current"

    def p(self, k: str, default: float) -> float:
        return float(self.params.get(k, default))


@dataclass
class TokenTerms:
    objective: torch.Tensor  # J_t, [B, T]; zero where mask == 0
    metrics: dict[str, float]
    kept: torch.Tensor  # [B, T] bool: tokens that carry gradient after masking


def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> float:
    n = m.sum()
    return float((x * m).sum() / n) if n > 0 else 0.0


def _quantiles(x: torch.Tensor, m: torch.Tensor, qs=(0.01, 0.5, 0.99)) -> dict[str, float]:
    v = x[m.bool()]
    if v.numel() == 0:
        return {}
    q = torch.quantile(v.float(), torch.tensor(qs, dtype=torch.float32, device=v.device))  # on the data's device (CUDA / MPS trainers)
    return {f"r_dis/p{round(a * 100)}": float(b) for a, b in zip(qs, q, strict=True)}


def binary_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """D_B(p‖q) = p log(p/q) + (1−p) log((1−p)/(1−q)) for Bernoulli(p) vs Bernoulli(q) (KPop)."""
    p, q = p.clamp(eps, 1 - eps), q.clamp(eps, 1 - eps)
    return p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()


LOG_RATIO_CLAMP = 20.0  # |log r| bound before exp (SkyRL torch_utils, prime-rl): exp(±20) is finite in fp32 and fp64
SEQ_LEVEL = {"gspo", "mispo", "hispo"}  # corrections with a per-sequence statistic: the rows of one sequence stay together


def needs_sequence(c: Correction) -> bool:
    """Whether the correction reads a per-sequence statistic (then a rollout's rows share one micro batch)."""
    return c.name in SEQ_LEVEL or (c.name in ("tis", "icepop") and c.params.get("level") == "sequence")


def _exp(x: torch.Tensor) -> torch.Tensor:
    """exp of a log ratio, clamped: an extreme behaviour logprob (−9999 from an OpenAI-style server, < −88 in fp32) gives
    a large but finite weight instead of inf, so a gated-out token back-propagates 0·w = 0, never 0·inf = NaN."""
    return x.clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP).exp()


def _seq_mean(x: torch.Tensor, m: torch.Tensor, seq_ids: torch.Tensor | None) -> torch.Tensor:
    """Per row, the masked mean of ``x`` over every row of its sequence (``seq_ids``: one id per row; None = a row is a
    sequence). A rollout split into several rows (calls, compaction segments, actors) is one sequence for GSPO's ratio and
    the MIS-PO / HisPO trajectory gates (verl, MiMo: one row per trajectory)."""
    num, den = (x * m).sum(-1), m.sum(-1)
    if seq_ids is not None:
        sid = torch.as_tensor(seq_ids, device=x.device).long()
        k = int(sid.max()) + 1 if sid.numel() else 0
        num = torch.zeros(k, dtype=num.dtype, device=x.device).index_add(0, sid, num)[sid]
        den = torch.zeros(k, dtype=den.dtype, device=x.device).index_add(0, sid, den)[sid]
    return num / den.clamp(min=1)


def _seq_count(m: torch.Tensor, seq_ids: torch.Tensor | None) -> torch.Tensor:
    """Per row, the trained-token count of its whole sequence."""
    den = m.sum(-1)
    if seq_ids is not None:
        sid = torch.as_tensor(seq_ids, device=m.device).long()
        k = int(sid.max()) + 1 if sid.numel() else 0
        den = torch.zeros(k, dtype=den.dtype, device=m.device).index_add(0, sid, den)[sid]
    return den


def ppo_term(r: torch.Tensor, adv: torch.Tensor, eps_l: float, eps_h: float, dual_clip: float | None = None) -> torch.Tensor:
    surr = torch.minimum(r * adv, r.clamp(1 - eps_l, 1 + eps_h) * adv)
    if dual_clip is not None and dual_clip > 1:  # bound the negative-advantage branch (r → ∞ with Â < 0)
        surr = torch.where(adv < 0, torch.maximum(surr, dual_clip * adv), surr)
    return surr


def correct(c: Correction, *, logp: torch.Tensor, rollout_logp: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor,
            old_logp: torch.Tensor | None = None, seq_ids: torch.Tensor | None = None) -> TokenTerms:
    """``seq_ids`` [B]: the sequence each row belongs to (rows of one rollout share an id); None = every row its own."""
    m = mask.to(logp.dtype)
    if c.name in NEEDS_OLD and old_logp is None:
        if c.old_logprob_source == "rollout":
            old_logp = rollout_logp
        elif c.old_logprob_source == "recompute_current":
            old_logp = logp.detach()
        else:
            raise ValueError(f"correction {c.name!r} needs π_θold; set old_logprob_source='rollout'|'recompute_current' or pass old_logp")
    # a non-finite behaviour / old logprob on a trained token: the token is dropped (counted), never trained on NaN
    bad = ~torch.isfinite(rollout_logp)
    if old_logp is not None:
        bad = bad | ~torch.isfinite(old_logp)
    bad = bad & mask.bool()
    if bad.any():
        safe = logp.detach()
        rollout_logp = torch.where(bad, safe, rollout_logp)
        old_logp = torch.where(bad, safe, old_logp) if old_logp is not None else None
        m = m * (~bad).to(m.dtype)
        mask = mask * (~bad).to(mask.dtype)
    ref = (old_logp if old_logp is not None else logp.detach())
    log_rdis = (ref - rollout_logp).detach()
    r_dis = _exp(log_rdis)
    # with old_logp: the engine gap at the behaviour version (r_dis); without it the reference is the current policy, so the
    # metric also contains the policy drift since sampling (staleness) — log both regimes under the same name, knowingly
    metrics: dict[str, float] = {"rollout_probs_diff": _masked_mean((ref.exp() - rollout_logp.exp()).abs().detach(), m), **_quantiles(r_dis, m)}
    metrics["nonfinite_tokens"] = float(bad.sum())
    keep = mask.bool().clone()
    clipped = torch.zeros_like(keep)
    name = c.name
    zero = torch.zeros_like(logp)

    if name in ("ppo", "tis", "icepop", "kpop"):
        r_s = _exp(logp - old_logp)
        eps_l, eps_h = c.p("eps_low", 0.2), c.p("eps_high", 0.28)
        dc = c.params.get("dual_clip", 3.0)  # verl's clip_ratio_c = 3 by default; 0 / None disables it
        base = ppo_term(r_s, adv, eps_l, eps_h, float(dc) if dc else None)
        clipped = ((r_s < 1 - eps_l) | (r_s > 1 + eps_h)) & keep
        if name == "ppo":
            obj = base
        elif name in ("tis", "icepop") and c.params.get("level") == "sequence":
            # sequence-level IS on the engine gap (TRL vllm_importance_sampling_mode="sequence_truncate"/"sequence_mask"; the
            # product form, not NeMo-RL's geometric-mean seq-mask-tis): ρ = Π_t r_dis over the rollout (a sum of log ratios,
            # clamped), one weight per rollout
            log_rho = (_seq_mean(log_rdis, m, seq_ids) * _seq_count(m, seq_ids)).clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
            rho = log_rho.exp()[:, None]
            if name == "tis":
                obj = rho.clamp(max=c.p("C", 2.0)) * base
            else:
                gate = ((rho >= c.p("alpha", 0.5)) & (rho <= c.p("beta", 5.0))).expand_as(base)
                keep &= gate
                obj = torch.where(gate, rho.expand_as(base), zero) * base
        elif name == "tis":
            obj = r_dis.clamp(max=c.p("C", 2.0)) * base
        elif name == "icepop":
            gate = (r_dis >= c.p("alpha", 0.5)) & (r_dis <= c.p("beta", 5.0))
            keep &= gate
            obj = torch.where(gate, r_dis, zero) * base  # M(k) = k inside [α, β] (calibration), 0 outside (masking)
        else:  # kpop: IcePop's skeleton, the fixed ratio band replaced by a symmetric binary-KL gate (Ling 2.6; AReaL weights by r_dis too)
            p, q = ref.exp().detach(), rollout_logp.exp().detach()
            phi = c.p("phi", 0.01)
            gate = (binary_kl(p, q) <= phi) & (binary_kl(q, p) <= phi)
            keep &= gate
            obj = torch.where(gate, r_dis, zero) * base
    elif name == "dis":
        r = _exp(logp - rollout_logp).detach()
        gate = (r > 1 - c.p("eps_low", 0.8)) & (r < 1 + c.p("eps_high", 3.0))
        keep &= gate
        obj = torch.where(gate, r, zero) * adv * logp
    elif name == "mimo4":
        r = _exp(logp - rollout_logp).detach()
        pos = (adv >= 0) & (r >= c.p("pos_low", 0.2)) & (r <= c.p("pos_high", 5.0))
        neg = (adv < 0) & (r >= c.p("neg_low", 0.2)) & (r <= c.p("neg_high", 5.0))
        gate = pos | neg
        keep &= gate
        obj = torch.where(gate, r, zero) * adv * logp
    elif name == "mispo":
        tok = (r_dis >= c.p("tok_low", 0.5)) & (r_dis <= c.p("tok_high", 2.0))
        geo = _seq_mean(log_rdis, m, seq_ids).exp()  # trajectory geometric mean of x_t
        traj = (geo >= c.p("traj_low", 0.996)) & (geo <= c.p("traj_high", 1.001))
        gate = tok & traj[:, None]
        keep &= gate
        obj = gate.to(logp.dtype) * adv * logp
        metrics["mispo/traj_rejected"] = float((~traj).float().mean())
    elif name == "hispo":
        seq_gate = (_seq_mean(log_rdis, m, seq_ids).exp() - 1).abs() < c.p("delta_seq", 0.01)
        tok_gate = (r_dis - 1).abs() < c.p("delta_tok", 0.5)
        gate = seq_gate[:, None] & tok_gate
        keep &= gate
        r = _exp(logp - rollout_logp)
        lo, nh, ph = c.p("eps_neg_low", 0.2), c.p("eps_neg_high", 0.28), c.p("eps_pos_high", 0.28)
        pos_obj = torch.minimum(r * adv, r.clamp(1 - lo, 1 + ph) * adv)
        neg_obj = torch.maximum(torch.minimum(r * adv, r.clamp(1 - lo, 1 + nh) * adv), (1 + nh) * adv)
        base = torch.where(adv >= 0, pos_obj, neg_obj)
        obj = gate.to(logp.dtype) * base
        clipped = ((r < 1 - lo) | (r > 1 + torch.where(adv >= 0, torch.full_like(r, ph), torch.full_like(r, nh)))) & keep
    elif name == "cispo":
        r_s = _exp(logp - old_logp)
        lo = 0.0 if c.params.get("m2") else 1 - c.p("eps_low", 0.2)
        r_hat = r_s.clamp(lo, 1 + c.p("eps_high", 0.28)).detach()
        clipped = ((r_s < lo) | (r_s > 1 + c.p("eps_high", 0.28))) & keep
        obj = r_hat * adv * logp
        if c.params.get("mask"):  # MiniMax-M1's unified form: PPO's trust region as an explicit token mask (M ≡ 1 is pure CISPO)
            hi, lo_m = 1 + c.p("eps_high", 0.28), 1 - c.p("eps_low", 0.2)
            gate = ~(((adv > 0) & (r_s.detach() > hi)) | ((adv < 0) & (r_s.detach() < lo_m)))
            keep &= gate
            obj = gate.to(logp.dtype) * obj
    elif name == "ipo":
        # prime-rl's default loss (loss.py IPOLoss): a symmetric trust region in probability space — a token whose
        # probability moved more than ε in absolute terms since sampling is masked — and a capped, gradient-preserving
        # importance weight: gradient min(r, cap)·Â·∇log π_θ (value differs from prime-rl's r·Â; the gradient is equal)
        r = _exp(logp - rollout_logp).detach()
        diff = (logp.detach().exp() - rollout_logp.exp()).abs()
        gate = diff <= c.p("eps", 0.3)
        keep &= gate
        obj = torch.where(gate, r.clamp(max=c.p("max_ratio", 1e4)), zero) * c.p("adv_tau", 1.0) * adv * logp
    elif name == "gspo":
        log_s = _seq_mean((logp - old_logp).detach(), m, seq_ids).clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)  # log sequence ratio (length-normalized)
        s_tok = (log_s[:, None] + logp - logp.detach()).exp()  # GSPO-token: value = s_i, gradient per token
        eps_l, eps_h = c.p("eps_low", 3e-4), c.p("eps_high", 4e-4)
        obj = torch.minimum(s_tok * adv, s_tok.clamp(1 - eps_l, 1 + eps_h) * adv)
        s = log_s.exp()
        clipped = (((s < 1 - eps_l) | (s > 1 + eps_h))[:, None]) & keep
    else:
        raise ValueError(f"unknown correction {name!r}")

    obj = obj * m
    metrics["masked_token_share"] = 1.0 - _masked_mean(keep.to(logp.dtype), mask.to(logp.dtype) + bad.to(logp.dtype))
    metrics["clipped_token_share"] = _masked_mean(clipped.to(logp.dtype), m)
    return TokenTerms(objective=obj, metrics=metrics, kept=keep)


def describe(c: Correction) -> dict[str, Any]:
    return {"correction": c.name, "params": dict(c.params), "old_logprob_source": c.old_logprob_source, "needs_old": c.name in NEEDS_OLD}
