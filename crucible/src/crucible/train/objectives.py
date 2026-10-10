"""Objectives: loss aggregation, group-relative advantages, critic-mode GAE (DESIGN §10.3 point 5; research m11-m12/06
§3.2, §5, §6.3, §7.1).

**Aggregation** (``loss_agg``) — the training unit should match the reward unit (GSPO's principle):

| mode                  | J = …                                                   | who |
|-----------------------|---------------------------------------------------------|-----|
| token_mean_global     | Σ_{i,t} J_it / Σ_i |o_i|        (default)                 | DAPO, CISPO |
| seq_mean_token_mean   | (1/G) Σ_i (1/|o_i|) Σ_t J_it                            | GRPO, IcePop |
| prompt_mean           | mean over prompts of the token-mean inside each prompt  | MiMo |
| seq_ratio             | same arithmetic as seq_mean_token_mean (GSPO's unit)     | GSPO |
| const_max_len         | Σ_{i,t} J_it / (G · max_i |o_i|)                         | HisPO |

**Group-relative advantages** (``group_advantages``): Â_i = R_i − mean_group (Dr.GRPO, Kimi K3, MiMo: no std), optionally
/ std (GRPO), or leave-one-out (RLOO). Masked rollouts (reward None) are excluded from every statistic and get Â = 0.
All-equal groups give Â = 0 (dynamic sampling decides elsewhere whether to keep them).

**Critic mode** (``gae``): token-level GAE over **trainable tokens only** (skip-observation: observation / tool tokens
are removed from the Bellman chain, SAO), with **cross-segment discounting** for compacted rollouts,
Â_{s,i} = (γλ)^{N_{>s}} · A^loc_{s,i} (CompactionRL), truncation bootstrap (R̂ = V(s_T) for truncated rollouts,
StepFun), and the length-adaptive λ_policy = 1 − 1/(α·l) (VAPO/SAO, α = 1.5).

**DAPO overlong shaping** (``overlong_penalty``): 0 if |y| ≤ L_max − L_cache; ((L_max − L_cache) − |y|)/L_cache in
between; −1 beyond L_max (L_max 20,480, L_cache 4,096).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

LOSS_AGG = ("token_mean_global", "seq_mean_token_mean", "prompt_mean", "seq_ratio", "const_max_len")


def aggregate(J: torch.Tensor, mask: torch.Tensor, mode: str = "token_mean_global", *, prompt_ids: Sequence[int] | torch.Tensor | None = None,
              max_len: int | None = None) -> torch.Tensor:
    """Scalar objective from per-token terms. ``J`` and ``mask`` are [B, T]; ``prompt_ids`` [B] for ``prompt_mean``."""
    m = mask.to(J.dtype)
    n_seq = m.sum(-1)
    if mode == "token_mean_global":
        return (J * m).sum() / m.sum().clamp(min=1)
    if mode in ("seq_mean_token_mean", "seq_ratio"):
        per = (J * m).sum(-1) / n_seq.clamp(min=1)
        valid = n_seq > 0
        return per[valid].mean() if valid.any() else J.sum() * 0
    if mode == "prompt_mean":
        if prompt_ids is None:
            raise ValueError("prompt_mean needs prompt_ids")
        pid = torch.as_tensor(prompt_ids)
        out = []
        for p in pid.unique():
            sel = pid == p
            tok = m[sel].sum()
            if tok > 0:
                out.append((J[sel] * m[sel]).sum() / tok)
        return torch.stack(out).mean() if out else J.sum() * 0
    if mode == "const_max_len":
        L = max_len or int(n_seq.max().item() or 1)
        return (J * m).sum() / (J.shape[0] * max(L, 1))
    raise ValueError(f"unknown loss_agg {mode!r}")


def group_advantages(rewards: Sequence[float | None], groups: Sequence[str], *, std_normalize: bool = False, estimator: str = "mean",
                     eps: float = 1e-6) -> list[float]:
    """Per-rollout advantages. ``estimator``: ``mean`` (R − mean), ``rloo`` (R − mean of the others). Masked (None) rollouts
    get 0 and are excluded from their group's statistics."""
    out = [0.0] * len(rewards)
    idx: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        if rewards[i] is not None:
            idx.setdefault(g, []).append(i)
    for members in idx.values():
        vals = torch.tensor([float(rewards[i]) for i in members], dtype=torch.float64)
        n = len(members)
        if estimator == "rloo" and n > 1:
            base = (vals.sum() - vals) / (n - 1)
        else:
            base = vals.mean().expand(n)
        adv = vals - base
        if std_normalize and n > 1:
            adv = adv / (vals.std(unbiased=False) + eps)
        for j, i in enumerate(members):
            out[i] = float(adv[j])
    return out


def overlong_penalty(length: int, *, l_max: int = 20_480, l_cache: int = 4_096) -> float:
    if length <= l_max - l_cache:
        return 0.0
    if length <= l_max:
        return ((l_max - l_cache) - length) / l_cache
    return -1.0


def length_adaptive_lambda(length: int, alpha: float = 1.5) -> float:
    return 1.0 - 1.0 / (alpha * max(length, 1))


def gae(rewards: torch.Tensor, values: torch.Tensor, trainable: torch.Tensor, *, gamma: float = 1.0, lam: float = 1.0,
        segments: torch.Tensor | None = None, truncated: bool = False, bootstrap_value: float = 0.0,
        shared_terminal: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-level GAE for ONE rollout (1-D tensors over its token positions).

    - Only ``trainable`` tokens (model-generated) are on the Bellman chain; observation tokens are skipped entirely, so
      the TD step links an action's last token directly to the next action's first token (SAO skip-observation GAE).
      Rewards on non-trainable positions are carried to the next trainable token.
    - ``segments`` (int per position, compaction segment index): local GAE is computed per segment with the segment's
      own terminal (value 0 beyond its end). With ``shared_terminal`` = R, every segment's last trainable token receives
      the shared task reward R (segments are trained as samples sharing one outcome), and the local advantages are
      scaled by (γλ)^{N_{>s}}, N_{>s} = trainable tokens generated after segment s (CompactionRL), so an early segment
      does not believe the outcome is near. With one segment and R on the last token this is standard GAE.
    - ``truncated``: the terminal bootstrap is V(s_T) = ``bootstrap_value`` instead of 0 (StepFun).
    Returns (advantages, returns) on every position (0 on non-trainable positions)."""
    T = rewards.shape[0]
    adv = torch.zeros(T, dtype=torch.float64)
    ret = torch.zeros(T, dtype=torch.float64)
    pos = [int(i) for i in torch.nonzero(trainable, as_tuple=False).flatten()]
    if not pos:
        return adv.float(), ret.float()
    # carry rewards from skipped (observation) positions onto the next trainable token (or the last one)
    r_t = {p: 0.0 for p in pos}
    nxt = 0
    for i in range(T):
        while nxt < len(pos) and pos[nxt] < i:
            nxt += 1
        target = pos[nxt] if nxt < len(pos) else pos[-1]
        r_t[target] += float(rewards[i])
    seg = [int(segments[p]) for p in pos] if segments is not None else [0] * len(pos)
    last_seg = seg[-1]

    def run(idxs: list[int], terminal_v: float) -> list[float]:
        a, out = 0.0, [0.0] * len(idxs)
        for k in range(len(idxs) - 1, -1, -1):
            p = pos[idxs[k]]
            v_next = float(values[pos[idxs[k + 1]]]) if k + 1 < len(idxs) else terminal_v
            delta = r_t[p] + gamma * v_next - float(values[p])
            a = delta + gamma * lam * a
            out[k] = a
        return out

    by_seg: dict[int, list[int]] = {}
    for k, s in enumerate(seg):
        by_seg.setdefault(s, []).append(k)
    if shared_terminal is not None:
        for idxs in by_seg.values():
            r_t[pos[idxs[-1]]] += float(shared_terminal)
    order = sorted(by_seg)
    after = {s: sum(len(by_seg[x]) for x in order if x > s) for s in order}
    for s in order:
        idxs = by_seg[s]
        terminal = bootstrap_value if (truncated and s == last_seg) else 0.0
        local = run(idxs, terminal)
        scale = (gamma * lam) ** after[s]
        for k, a in zip(idxs, local, strict=True):
            p = pos[k]
            adv[p] = scale * a
            ret[p] = adv[p] + float(values[p])
    return adv.float(), ret.float()
