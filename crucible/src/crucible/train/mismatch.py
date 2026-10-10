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
| ppo      | min(r_s Â, clip(r_s, 1−ε_l, 1+ε_h) Â) (+ dual clip c)      | yes          | PPO / DAPO clip-higher |
| tis      | sg(min(r_dis, C)) · ppo                                    | yes          | Yao et al.; slime, verl, OpenRLHF |
| icepop   | sg(r_dis)·𝟙[α ≤ r_dis ≤ β] · ppo                            | yes          | Ring-1T (α=0.5, β=5) |
| kpop     | 𝟙[D_B(p‖q) ≤ φ]·𝟙[D_B(q‖p) ≤ φ] · ppo (binary KL)           | yes          | Ling 2.6 |
| dis      | sg(r)·𝟙[1−ε_l < r < 1+ε_h] · Â · log π_θ                   | **no**       | GLM / SAO (coding ε_l=0.8, ε_h=3.0) |
| mimo4    | sg(r)·𝟙[bounds by sign of Â] · Â · log π_θ                  | **no**       | MiMo-V2.6 ([0.2, 5] both signs) |
| mispo    | 𝟙[x_t ∈ tok]·𝟙[geomean x ∈ traj] · Â · log π_θ, x = r_dis    | yes          | StepFun (tok [0.5,2], traj [0.996,1.001]) |
| hispo    | seq + token gates on r_dis; three-bound clip on full r     | yes          | LongCat |
| cispo    | sg(clip(r_s, 1−ε_l, 1+ε_h)) · Â · log π_θ                  | yes          | MiniMax M1 (M2: lower bound 0) |
| gspo     | sequence ratio s_i = exp(mean log r_s), token form, clip   | yes          | Qwen (ε = 3e−4 / 4e−4) |

Corrections that need π_θold refuse to run without it unless ``old_logprob_source`` is set: ``"rollout"`` (use
π_rollout, i.e. assume no engine mismatch) or ``"recompute_current"`` (π_θ detached, valid when K = 1 and on-policy).
DIS and mimo4 need only the recorded rollout logprobs, so they are valid under multi-version async with call-level
versions — hence the default ``dis(0.8, 3.0)`` (research m11-m12/06 §5.4).

All tensors are [B, T] (padded); ``mask`` marks trainable completion tokens; ``adv`` is per token (sequence
advantages broadcast, or M6 turn advantages on their call's tokens). ``seq_ids`` is unused here (sequence = row).
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
    q = torch.quantile(v.float(), torch.tensor(qs, dtype=torch.float32))
    return {f"r_dis/p{round(a * 100)}": float(b) for a, b in zip(qs, q, strict=True)}


def binary_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """D_B(p‖q) = p log(p/q) + (1−p) log((1−p)/(1−q)) for Bernoulli(p) vs Bernoulli(q) (KPop)."""
    p, q = p.clamp(eps, 1 - eps), q.clamp(eps, 1 - eps)
    return p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()


def ppo_term(r: torch.Tensor, adv: torch.Tensor, eps_l: float, eps_h: float, dual_clip: float | None = None) -> torch.Tensor:
    surr = torch.minimum(r * adv, r.clamp(1 - eps_l, 1 + eps_h) * adv)
    if dual_clip is not None:  # bound the negative-advantage branch (r → ∞ with Â < 0)
        surr = torch.where(adv < 0, torch.maximum(surr, dual_clip * adv), surr)
    return surr


def correct(c: Correction, *, logp: torch.Tensor, rollout_logp: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor,
            old_logp: torch.Tensor | None = None) -> TokenTerms:
    m = mask.to(logp.dtype)
    if c.name in NEEDS_OLD and old_logp is None:
        if c.old_logprob_source == "rollout":
            old_logp = rollout_logp
        elif c.old_logprob_source == "recompute_current":
            old_logp = logp.detach()
        else:
            raise ValueError(f"correction {c.name!r} needs π_θold; set old_logprob_source='rollout'|'recompute_current' or pass old_logp")
    ref = (old_logp if old_logp is not None else logp.detach())
    log_rdis = (ref - rollout_logp).detach()
    r_dis = log_rdis.exp()
    metrics: dict[str, float] = {"rollout_probs_diff": _masked_mean((ref.exp() - rollout_logp.exp()).abs().detach(), m), **_quantiles(r_dis, m)}
    keep = mask.bool().clone()
    clipped = torch.zeros_like(keep)
    name = c.name

    if name in ("ppo", "tis", "icepop", "kpop"):
        r_s = (logp - old_logp).exp()
        eps_l, eps_h = c.p("eps_low", 0.2), c.p("eps_high", 0.28)
        dc = c.params.get("dual_clip")
        base = ppo_term(r_s, adv, eps_l, eps_h, float(dc) if dc is not None else None)
        clipped = ((r_s < 1 - eps_l) | (r_s > 1 + eps_h)) & keep
        if name == "ppo":
            obj = base
        elif name == "tis":
            obj = r_dis.clamp(max=c.p("C", 2.0)) * base
        elif name == "icepop":
            gate = (r_dis >= c.p("alpha", 0.5)) & (r_dis <= c.p("beta", 5.0))
            keep &= gate
            obj = torch.where(gate, r_dis * base, torch.zeros_like(base))
        else:  # kpop
            p, q = ref.exp().detach(), rollout_logp.exp().detach()
            phi = c.p("phi", 0.01)
            gate = (binary_kl(p, q) <= phi) & (binary_kl(q, p) <= phi)
            keep &= gate
            obj = torch.where(gate, base, torch.zeros_like(base))
    elif name == "dis":
        r = (logp - rollout_logp).exp().detach()
        gate = (r > 1 - c.p("eps_low", 0.8)) & (r < 1 + c.p("eps_high", 3.0))
        keep &= gate
        obj = torch.where(gate, r * adv * logp, torch.zeros_like(logp))
    elif name == "mimo4":
        r = (logp - rollout_logp).exp().detach()
        pos = (adv >= 0) & (r >= c.p("pos_low", 0.2)) & (r <= c.p("pos_high", 5.0))
        neg = (adv < 0) & (r >= c.p("neg_low", 0.2)) & (r <= c.p("neg_high", 5.0))
        gate = pos | neg
        keep &= gate
        obj = torch.where(gate, r * adv * logp, torch.zeros_like(logp))
    elif name == "mispo":
        tok = (r_dis >= c.p("tok_low", 0.5)) & (r_dis <= c.p("tok_high", 2.0))
        n = m.sum(-1).clamp(min=1)
        geo = ((log_rdis * m).sum(-1) / n).exp()  # trajectory geometric mean of x_t
        traj = (geo >= c.p("traj_low", 0.996)) & (geo <= c.p("traj_high", 1.001))
        gate = tok & traj[:, None]
        keep &= gate
        obj = torch.where(gate, adv * logp, torch.zeros_like(logp))
        metrics["mispo/traj_rejected"] = float((~traj).float().mean())
    elif name == "hispo":
        n = m.sum(-1).clamp(min=1)
        seq_gate = (((log_rdis * m).sum(-1) / n).exp() - 1).abs() < c.p("delta_seq", 0.01)
        tok_gate = (r_dis - 1).abs() < c.p("delta_tok", 0.5)
        gate = seq_gate[:, None] & tok_gate
        keep &= gate
        r = (logp - rollout_logp).exp()
        lo, nh, ph = c.p("eps_neg_low", 0.2), c.p("eps_neg_high", 0.28), c.p("eps_pos_high", 0.28)
        pos_obj = torch.minimum(r * adv, r.clamp(1 - lo, 1 + ph) * adv)
        neg_obj = torch.maximum(torch.minimum(r * adv, r.clamp(1 - lo, 1 + nh) * adv), (1 + nh) * adv)
        base = torch.where(adv >= 0, pos_obj, neg_obj)
        obj = torch.where(gate, base, torch.zeros_like(base))
        clipped = ((r < 1 - lo) | (r > 1 + torch.where(adv >= 0, torch.full_like(r, ph), torch.full_like(r, nh)))) & keep
    elif name == "cispo":
        r_s = (logp - old_logp).exp()
        lo = 0.0 if c.params.get("m2") else 1 - c.p("eps_low", 0.2)
        r_hat = r_s.clamp(lo, 1 + c.p("eps_high", 0.28)).detach()
        clipped = ((r_s < lo) | (r_s > 1 + c.p("eps_high", 0.28))) & keep
        obj = r_hat * adv * logp
    elif name == "gspo":
        n = m.sum(-1).clamp(min=1)
        log_s = ((logp - old_logp) * m).sum(-1) / n  # log sequence ratio (length-normalized)
        s_tok = (log_s.detach()[:, None] + logp - logp.detach()).exp()  # GSPO-token: value = s_i, gradient per token
        eps_l, eps_h = c.p("eps_low", 3e-4), c.p("eps_high", 4e-4)
        obj = torch.minimum(s_tok * adv, s_tok.clamp(1 - eps_l, 1 + eps_h) * adv)
        s = log_s.exp()
        clipped = (((s < 1 - eps_l) | (s > 1 + eps_h))[:, None]) & keep
    else:
        raise ValueError(f"unknown correction {name!r}")

    obj = obj * m
    metrics["masked_token_share"] = 1.0 - _masked_mean(keep.to(logp.dtype), m)
    metrics["clipped_token_share"] = _masked_mean(clipped.to(logp.dtype), m)
    return TokenTerms(objective=obj, metrics=metrics, kept=keep)


def describe(c: Correction) -> dict[str, Any]:
    return {"correction": c.name, "params": dict(c.params), "old_logprob_source": c.old_logprob_source, "needs_old": c.name in NEEDS_OLD}
