"""Objectives: loss aggregation, group-relative advantages, critic-mode GAE (DESIGN §10.3 point 5; research m11-m12/06
§3.2, §5, §6.3, §7.1).

**Aggregation** (``loss_agg``) — the training unit should match the reward unit (GSPO's principle):

| mode                  | J = …                                                   | who |
|-----------------------|---------------------------------------------------------|-----|
| token_mean_global     | Σ_{i,t} J_it / Σ_i |o_i|        (default)                 | DAPO, CISPO |
| seq_mean_token_mean   | (1/G) Σ_i (1/|o_i|) Σ_t J_it                            | GRPO, IcePop |
| prompt_mean           | mean over prompts of the token-mean inside each prompt  | MiMo |
| seq_ratio             | same arithmetic as seq_mean_token_mean (GSPO's unit)     | GSPO |
| const_max_len         | per group Σ_{i,t} J_it / (G · max_i |o_i|), mean over groups | HisPO |

The trainer applies these through ``row_weights``: per row a weight w with J = Σ w · Σ_t J_t, so every mode micro-batches
and splits across DP ranks exactly. The sequence is the **rollout**, however many rows it becomes. ``loss_agg="auto"``
(the trainer default) picks the correction's own unit (``NATURAL_AGG``: GSPO → sequence mean, HisPO → const_max_len).

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

import math
from collections.abc import Sequence
from typing import Any

import torch

LOSS_AGG = ("token_mean_global", "seq_mean_token_mean", "prompt_mean", "seq_ratio", "const_max_len")  # + "auto" (``resolve_agg``)


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


# the aggregation each correction is defined with (GSPO: sequence mean of the sequence ratio; HisPO: G·max|y|); "auto"
# picks it, every other correction trains under the global token mean (DAPO, CISPO, MiMo's verl default)
NATURAL_AGG = {"gspo": "seq_mean_token_mean", "hispo": "const_max_len"}
REQUIRED_AGG = {"gspo": ("seq_mean_token_mean", "seq_ratio")}  # GSPO's clip and value are per sequence: token mean is a different objective


def resolve_agg(mode: str, correction: str) -> str:
    if mode == "auto":
        return NATURAL_AGG.get(correction, "token_mean_global")
    if mode not in LOSS_AGG:
        raise ValueError(f"unknown loss_agg {mode!r}")
    need = REQUIRED_AGG.get(correction)
    if need and mode not in need:
        raise ValueError(f"correction {correction!r} is defined with loss_agg in {need} (got {mode!r}); use loss_agg='auto'")
    return mode


def row_weights(lengths: Sequence[int], seqs: Sequence[Any], prompts: Sequence[Any], mode: str, *, max_len: int | None = None) -> list[float]:
    """Per row (trainer item), the weight w such that the aggregated objective is J = Σ_rows w · Σ_t J_t over its trained
    tokens: every aggregation is linear in the token terms, so any micro-batching and any DP split of the rows gives the
    full-batch value and gradient exactly (verl ``global_batch_info``, SkyRL / NeMo-RL global normalisers).

    ``lengths``: trained tokens per row; ``seqs``: the sequence (rollout) of each row — a rollout split into calls,
    compaction segments or actors is ONE sequence; ``prompts``: the prompt group of each row.

    - token_mean_global: 1 / Σ tokens;
    - seq_mean_token_mean, seq_ratio: 1 / (n_seq · T_seq) — each rollout weighs the same;
    - prompt_mean: 1 / (n_prompt · T_prompt) — the token mean inside each prompt group, averaged over groups (MiMo);
    - const_max_len: 1 / (n_prompt · G_p · L_p) — HisPO's per-group 1/(G·max|y|), L_p = ``max_len`` (a constant) or the
      group's longest rollout, averaged over groups."""
    n = len(lengths)
    if mode == "token_mean_global":
        tot = sum(lengths)
        return [1.0 / tot if tot else 0.0] * n
    t_seq: dict[Any, int] = {}
    for L, s in zip(lengths, seqs, strict=True):
        t_seq[s] = t_seq.get(s, 0) + L
    live = {s for s, t in t_seq.items() if t > 0}
    if mode in ("seq_mean_token_mean", "seq_ratio"):
        return [1.0 / (len(live) * t_seq[s]) if s in live else 0.0 for s in seqs]
    if mode == "prompt_mean":
        t_p: dict[Any, int] = {}
        for L, p in zip(lengths, prompts, strict=True):
            t_p[p] = t_p.get(p, 0) + L
        n_p = sum(1 for t in t_p.values() if t > 0)
        return [1.0 / (n_p * t_p[p]) if t_p[p] else 0.0 for p in prompts]
    if mode == "const_max_len":
        members: dict[Any, set] = {}
        for s, p in zip(seqs, prompts, strict=True):
            if s in live:
                members.setdefault(p, set()).add(s)
        n_p = len(members)
        out = []
        for s, p in zip(seqs, prompts, strict=True):
            if s not in live:
                out.append(0.0)
                continue
            L = max_len or max(t_seq[x] for x in members[p])
            out.append(1.0 / (n_p * len(members[p]) * max(L, 1)))
        return out
    raise ValueError(f"unknown loss_agg {mode!r}")


ESTIMATORS = ("mean", "rloo", "remax", "reinforce_pp", "reinforce_pp_baseline")


def group_advantages(rewards: Sequence[float | None], groups: Sequence[str], *, std_normalize: bool = False, estimator: str = "mean",
                     eps: float = 1e-6, baselines: dict[str, float] | None = None, std_ddof: int = 0) -> list[float]:
    """Per-rollout advantages. ``estimator``:

    - ``mean``: R − mean of the group (GRPO; Dr.GRPO without std);
    - ``rloo``: R − mean of the other members;
    - ``remax``: R − b_g, the reward of the group's greedy rollout (``baselines[group]``; ReMax);
    - ``reinforce_pp``: (R − mean of the global batch) / std of the global batch (REINFORCE++: no per-prompt baseline);
    - ``reinforce_pp_baseline``: R − mean of the group, then whitened by the global batch std (OpenRLHF
      ``reinforce_baseline``).

    ``std_ddof``: 0 = population std (Crucible's default), 1 = Bessel (TRL, verl, OpenRLHF: GRPO advantages there are
    √(n/(n−1)) smaller, 1.15× at n = 4). Masked (None) rollouts get 0 and are excluded from every statistic."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}")
    out = [0.0] * len(rewards)
    if estimator == "reinforce_pp_baseline":
        centred = group_advantages(rewards, groups, estimator="mean")
        live = [i for i, r in enumerate(rewards) if r is not None]
        if len(live) > 1:
            vals = torch.tensor([centred[i] for i in live], dtype=torch.float64)
            sd = float(vals.std(correction=std_ddof)) + eps
            for i in live:
                out[i] = centred[i] / sd
        return out
    if estimator == "reinforce_pp":
        live = [i for i, r in enumerate(rewards) if r is not None]
        if not live:
            return out
        vals = torch.tensor([float(rewards[i]) for i in live], dtype=torch.float64)
        adv = (vals - vals.mean()) / (vals.std(correction=std_ddof) + eps) if len(live) > 1 else vals - vals.mean()
        for j, i in enumerate(live):
            out[i] = float(adv[j])
        return out
    idx: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        if rewards[i] is not None:
            idx.setdefault(g, []).append(i)
    for gid, members in idx.items():
        vals = torch.tensor([float(rewards[i]) for i in members], dtype=torch.float64)
        n = len(members)
        if estimator == "remax":
            if not baselines or gid not in baselines:
                raise ValueError(f"remax needs the greedy baseline reward of group {gid!r}")
            base = torch.full((n,), float(baselines[gid]), dtype=torch.float64)
        elif estimator == "rloo" and n > 1:
            base = (vals.sum() - vals) / (n - 1)
        else:
            base = vals.mean().expand(n)
        adv = vals - base
        if std_normalize and n > 1:
            adv = adv / (vals.std(correction=std_ddof) + eps)
        for j, i in enumerate(members):
            out[i] = float(adv[j])
    return out


def reward_to_go(step_rewards: Sequence[float | None], outcome: float) -> list[float]:
    """G_t = Σ_{p ≥ t} r_p, with the outcome added at the last step (MiniMax M2: (s_t, a_t) atomic units, reward-to-go
    minus a trajectory-level baseline). Missing step rewards count 0."""
    out, acc = [0.0] * len(step_rewards), float(outcome)
    for t in range(len(step_rewards) - 1, -1, -1):
        acc += float(step_rewards[t] or 0.0)
        out[t] = acc
    return out


def speed_bonus(t: float, t_base: float, *, kind: str = "linear") -> float:
    """h(T / T_baseline), monotone decreasing (MiniMax M2.5 r_speed): ``linear`` = 1 − min(x, 2) ∈ [−1, 1] (0 at the
    baseline); ``log`` = −log2(x) clipped to [−1, 1]."""
    if t_base <= 0 or t < 0:
        return 0.0
    x = t / t_base
    if kind == "linear":
        return 1.0 - min(x, 2.0)
    if kind == "log":
        return max(-1.0, min(1.0, -math.log2(max(x, 1e-9))))
    raise ValueError("speed kind: linear | log")


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
