"""Critic mode (M12 checklist #16, #46–#48; research m11-m12/06 §5; article §3.5 SAO / MIS-PO / CompactionRL / Seed).

One enterprise case gives one trajectory and one outcome: a group-relative baseline does not exist (n = 1). The
critic route supplies the baseline with a value model V_φ and assigns credit per token with GAE:

- **value models**: ``TinyValue`` (CPU, mirrors ``TinyPolicy``) and ``HFValue`` (an HF causal LM's backbone + a scalar
  head; ``freeze_attention`` trains only the MLP/MoE projections — SAO found the critic's instability comes from the
  attention layers). Contract: ``token_values(prompt, completion) -> [n]``, the value of the state *before* each
  completion token (positions len(prompt)−1 … len(prompt)−1+n−1), with gradient.
- **advantages** (``objectives.gae``, per rollout over its trainable tokens only): skip-observation (observation tokens
  never enter the Bellman chain: only completion tokens are on it), compaction segments discounted by (γλ)^{N_{>s}}
  with the shared outcome, truncation bootstrap R̂ = V(s_T) (StepFun), policy λ = 1 − 1/(α·l) (VAPO length-adaptive,
  α = 1.5) and critic λ = 1 (Monte-Carlo returns).
- **TTUR**: ``updates_per_policy_update`` critic steps per policy step (SAO K = 2; StepFun 3×).
- **health**: explained variance 1 − Var(R − V)/Var(R) over trainable tokens (SAO's critic metric).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CriticConfig:
    lr: float = 5e-3
    updates_per_policy_update: int = 2  # TTUR
    gamma: float = 1.0
    lam_policy: float | str = "adaptive"  # a float, or "adaptive" = 1 − 1/(alpha · trainable length)
    alpha: float = 1.5
    lam_critic: float = 1.0
    grad_clip: float = 1.0


class TinyValue(torch.nn.Module):
    def __init__(self, vocab: int = 32, dim: int = 32, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed + 7)
        self.emb = torch.nn.Embedding(vocab, dim)
        self.rnn = torch.nn.GRU(dim, dim, batch_first=True)
        self.head = torch.nn.Linear(dim, 1)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
            self.head.weight.zero_()
            self.head.bias.zero_()

    def token_values(self, prompt: list[int], completion: list[int]) -> torch.Tensor:
        if not completion:
            return torch.zeros(0)
        dev = self.emb.weight.device
        h, _ = self.rnn(self.emb(torch.tensor([prompt + completion[:-1]], device=dev)))
        return self.head(h[0, len(prompt) - 1: len(prompt) - 1 + len(completion)]).squeeze(-1)


class HFValue(torch.nn.Module):
    """An HF causal LM's transformer backbone + a zero-initialized scalar head (value pretraining / init-from-policy:
    pass the policy's own model). ``freeze_attention``: parameters under ``self_attn`` get no gradient."""

    def __init__(self, causal_lm: torch.nn.Module, *, freeze_attention: bool = True):
        super().__init__()
        self.backbone = getattr(causal_lm, "model", causal_lm)
        hidden = causal_lm.config.hidden_size
        self.head = torch.nn.Linear(hidden, 1)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()
        if freeze_attention:
            for n, p in self.backbone.named_parameters():
                if "self_attn" in n:
                    p.requires_grad_(False)

    def token_values(self, prompt: list[int], completion: list[int]) -> torch.Tensor:
        if not completion:
            return torch.zeros(0)
        dev = self.head.weight.device
        h = self.backbone(input_ids=torch.tensor([prompt + completion[:-1]], device=dev)).last_hidden_state
        return self.head(h[0, len(prompt) - 1: len(prompt) - 1 + len(completion)].float()).squeeze(-1)


def explained_variance(returns: torch.Tensor, values: torch.Tensor) -> float:
    v = float(returns.var(unbiased=False))
    return 1.0 - float((returns - values).var(unbiased=False)) / v if v > 1e-12 else 0.0
