"""Truncated sampling and the matching training logprobs (M12 checklist #37; research m11-m12/06 §6 "top-p
mismatch").

With nucleus sampling (top_p < 1) the engine samples from π restricted to the nucleus and renormalized, and the
behaviour logprob it reports is log π_trunc(token). Every correction compares the trainer's π_θ with that logprob, so
the trainer must evaluate π_θ renormalized over the **same candidate set**, else the ratio is biased by the dropped mass
(log Σ_{c ∈ nucleus} π(c), different for every token). The engine records the candidate set per token
(``Turn.candidates``); the trainer renormalizes over it. Order, as in vLLM: temperature → top_p.

``nucleus(probs, top_p)``: the smallest set of tokens, by descending probability (ties: lower id first), whose mass
reaches ``top_p``; always at least the most likely token. Engines and trainers share this one definition.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def nucleus(probs: torch.Tensor, top_p: float) -> list[int]:
    if top_p >= 1.0:
        return list(range(probs.shape[-1]))
    p = probs.double()
    order = sorted(range(p.shape[-1]), key=lambda i: (-float(p[i]), i))
    out, mass = [], 0.0
    for i in order:
        out.append(i)
        mass += float(p[i])
        if mass >= top_p:
            break
    return sorted(out)


def renormalized(logprob_rows: torch.Tensor, tokens: Sequence[int], candidates: Sequence[Sequence[int]]) -> torch.Tensor:
    """log π(token_t) − log Σ_{c ∈ candidates_t} π(c), per position; ``logprob_rows`` [T, V] are full log-softmax rows
    (temperature applied). Differentiable."""
    out = []
    for t, (tok, cands) in enumerate(zip(tokens, candidates, strict=True)):
        idx = torch.tensor(list(cands), device=logprob_rows.device)
        out.append(logprob_rows[t, tok] - torch.logsumexp(logprob_rows[t, idx], -1))
    return torch.stack(out) if out else logprob_rows[:0, 0]
