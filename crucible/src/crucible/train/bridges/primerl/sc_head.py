"""The trainer-side hook for score centering: the learner's logprobs of the sampler's head ids, gathered in prime-rl's
chunked LM head (swarm-pipeline/14 §7.2).

**Why a hook.** prime-rl v0.10.0's custom loss gets ``LossInputs`` (``trainer/rl/loss.py:21-37``): the sampled token's
learner and sampler logprobs, advantages, mask, weights. Score centering also needs log p_v for the k head ids and Σ_H p_v
at every trained position. The logprobs come from ``FusedOutputLinear`` (``trainer/models/layers/lm_head.py``), which
runs ``_SequenceChunkedLogProbEntropyFn`` over token chunks × vocab chunks of 8192 with an online logsumexp, so [T, V]
(262K × 248K) is never materialized. That function already gathers logits at per-position ids (its top-p/top-k
sampling-mask replay) and is the narrowest place to gather the head: the same loop, k extra gathers per vocab chunk.

**What is patched** (``install``, run when prime-rl's trainer imports ``loss``; nothing else in prime-rl changes):

1. ``FusedOutputLinear.forward``: when its ``sampling_mask`` argument is a Crucible payload (``topk.SC_MAGIC`` in column
   0; prime-rl has already shifted it to the labels and sharded it under context parallel), it runs ``HeadLogProbFn``:
   the stock function's full-vocab logprob and entropy (no renormalization over the head) plus ``head_lp`` [N, k] with
   its gradient, and stashes the head of this micro batch. A genuine sampling mask (no marker) takes the stock path.
2. ``VanillaOutputLinear.forward`` refuses a payload: prime-rl's train loop would read it as a sampling mask and
   renormalize the sampled logprobs over the head (score centering needs the fused head, prime-rl's default:
   ``model.fused_lm_head_token_chunk_size = 8192``).
3. The train module's ``gather_for_cp`` records the context-parallel group, and its ``compute_loss`` gathers the stashed
   head across it (prime-rl's own differentiable all-gather, as for the logprobs), shifts it to the sampled-token
   positions (``shift_tensor_right``'s convention), splits it per sequence, and keys each piece by the
   ``trainer_logprobs`` tensor prime-rl hands that sequence's ``LossInputs``; ``loss.crucible_loss`` looks it up.

Memory: ``head_lp`` is 4·k bytes per token (33.5 MB at 262K tokens and k = 32, plus the same for its gradient); the
extra compute is k gathers and a scatter per token per vocab chunk, small against the head's matmuls.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from crucible.train.bridges.primerl.loss import Head
from crucible.train.bridges.primerl.topk import SC_MAGIC

VOCAB_CHUNK = 8192  # prime-rl's vocab chunk in _SequenceChunkedLogProbEntropyFn


def decode_payload(sampling_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """A ``[..., W]`` int32 sampling-mask tensor -> ``(ids [..., k] long, -1 where absent; lq [..., k] float32; present
    [...])``, or None when it carries no Crucible row (a genuine mask, or all padding). Rows are ``[SC_MAGIC, ids,
    bits(logprobs)]`` or all -1 (``topk.encode_payload``); anything else is refused."""
    col0 = sampling_mask[..., 0]
    present = col0 == SC_MAGIC
    if not bool(present.any()):
        return None
    if not bool(((col0 == SC_MAGIC) | (col0 == -1)).all()):
        raise ValueError("a sampling mask mixes Crucible's score-centering rows with prime-rl sampling-mask rows")
    w = sampling_mask.shape[-1]
    if w < 3 or (w - 1) % 2:
        raise ValueError(f"a score-centering payload row has width {w}, not 1 + 2k")
    k = (w - 1) // 2
    sm = sampling_mask.to(torch.int32)
    ids = torch.where(present.unsqueeze(-1), sm[..., 1:1 + k], torch.full_like(sm[..., 1:1 + k], -1)).long()
    lq = sm[..., 1 + k:].contiguous().view(torch.float32)
    lq = torch.where(present.unsqueeze(-1), lq, torch.full_like(lq, float("-inf")))
    return ids, lq, present


def _online(m: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """prime-rl's ``_online_logsumexp_and_weighted_update``."""
    cm = torch.amax(x, dim=-1)
    mn = torch.maximum(m, cm)
    e_old = torch.exp(m - mn)
    ex = torch.exp(x - mn.unsqueeze(-1))
    return mn, s * e_old + ex.sum(-1), t * e_old + (ex * x).sum(-1)


class HeadLogProbFn(torch.autograd.Function):
    """prime-rl's ``_SequenceChunkedLogProbEntropyFn`` without sampling-mask replay, plus the logprobs of ``head_ids``.

    Forward, per token chunk and vocab chunk: the temperature-scaled logits, the online logsumexp, the label's logit and
    the head ids' logits. Returns ``logprobs`` [N], ``entropy`` [N] and ``head_lp`` [N, k] (0 at -1 ids). Backward:
    d/dz of Σ g_tok·log p_y + Σ_v g_v·log p_v is g_tok·e_y + Σ_v g_v·e_v − (g_tok + Σ_v g_v)·softmax(z), recomputed
    chunk by chunk as prime-rl does."""

    @staticmethod
    def forward(ctx, hidden, weight, labels, inv_t, chunk_size, head_ids, vocab_chunk=VOCAB_CHUNK):  # type: ignore[override]
        n, vocab = hidden.shape[0], weight.shape[0]
        k = head_ids.shape[1]
        dev = hidden.device
        logprobs = torch.empty((n,), device=dev, dtype=torch.float32)
        entropy = torch.empty((n,), device=dev, dtype=torch.float32)
        logz = torch.empty((n,), device=dev, dtype=torch.float32)
        head_lp = torch.zeros((n, k), device=dev, dtype=torch.float32)
        valid = head_ids >= 0
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            h, lab, it = hidden[start:end], labels[start:end], inv_t[start:end].to(torch.float32).unsqueeze(-1)
            ids, ok = head_ids[start:end], valid[start:end]
            c = end - start
            m = torch.full((c,), float("-inf"), device=dev, dtype=torch.float32)
            s = torch.zeros((c,), device=dev, dtype=torch.float32)
            t = torch.zeros((c,), device=dev, dtype=torch.float32)
            target = torch.zeros((c,), device=dev, dtype=torch.float32)
            hl = torch.zeros((c, k), device=dev, dtype=torch.float32)
            for v0 in range(0, vocab, vocab_chunk):
                v1 = min(v0 + vocab_chunk, vocab)
                z = (h @ weight[v0:v1].t()).to(torch.float32) * it
                m, s, t = _online(m, s, t, z)
                inr = (lab >= v0) & (lab < v1)
                loc = (lab - v0).clamp(0, v1 - v0 - 1).long()
                target = torch.where(inr, z.gather(1, loc.unsqueeze(1)).squeeze(1), target)
                hin = ok & (ids >= v0) & (ids < v1)
                hloc = (ids - v0).clamp(0, v1 - v0 - 1)
                hl = torch.where(hin, z.gather(1, hloc), hl)
            lz = m + torch.log(s)
            logz[start:end] = lz
            logprobs[start:end] = target - lz
            entropy[start:end] = lz - t / s
            head_lp[start:end] = torch.where(ok, hl - lz.unsqueeze(-1), torch.zeros_like(hl))
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(hidden, weight, labels, inv_t, logz, head_ids)
        ctx.chunk_size, ctx.vocab_chunk = chunk_size, vocab_chunk
        return logprobs, entropy, head_lp

    @staticmethod
    def backward(ctx, g_lp, g_ent, g_head):  # type: ignore[override]
        if g_ent is not None:
            raise RuntimeError("no backward through the entropy (prime-rl's fused head has none either)")
        hidden, weight, labels, inv_t, logz, head_ids = ctx.saved_tensors
        n, vocab = hidden.shape[0], weight.shape[0]
        need_h, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        gh = torch.zeros_like(hidden) if need_h else None
        gw = torch.zeros_like(weight) if need_w else None
        valid = head_ids >= 0
        for start in range(0, n, ctx.chunk_size):
            end = min(start + ctx.chunk_size, n)
            h, lab, it = hidden[start:end], labels[start:end], inv_t[start:end].to(torch.float32).unsqueeze(-1)
            ids, ok, lz = head_ids[start:end], valid[start:end], logz[start:end]
            gt = g_lp[start:end].to(torch.float32) if g_lp is not None else torch.zeros((end - start,), device=hidden.device, dtype=torch.float32)
            gv = (torch.where(ok, g_head[start:end].to(torch.float32), torch.zeros_like(g_head[start:end], dtype=torch.float32))
                  if g_head is not None else torch.zeros(ids.shape, device=hidden.device, dtype=torch.float32))
            total = gt + gv.sum(-1)
            for v0 in range(0, vocab, ctx.vocab_chunk):
                v1 = min(v0 + ctx.vocab_chunk, vocab)
                wc = weight[v0:v1]
                z = (h @ wc.t()).to(torch.float32) * it
                gz = (-total).unsqueeze(-1) * torch.exp(z - lz.unsqueeze(-1))
                inr = (lab >= v0) & (lab < v1)
                loc = (lab - v0).clamp(0, v1 - v0 - 1).long()
                gz.scatter_add_(1, loc.unsqueeze(1), (gt * inr).unsqueeze(1))
                hin = ok & (ids >= v0) & (ids < v1)
                gz.scatter_add_(1, (ids - v0).clamp(0, v1 - v0 - 1), gv * hin)
                gz = gz * it
                if need_h:
                    gh[start:end].add_(gz.to(hidden.dtype) @ wc)
                if need_w:
                    gw[v0:v1].add_(gz.to(weight.dtype).t() @ h)
        return gh, gw, None, None, None, None, None


@dataclass
class _Pending:
    """One micro batch's head, as the LM head computed it: labels-aligned, this context-parallel rank's shard."""

    lp: torch.Tensor  # [B, S, k]
    lq: torch.Tensor  # [B, S, k]
    valid: torch.Tensor  # [B, S, k] bool
    present: torch.Tensor  # [B, S] bool
    lq_label: torch.Tensor  # [B, S]: the label's logprob off its own head entry, NaN when it is not in the head


@dataclass
class _Ctx:
    pending: _Pending | None = None  # the last forward's head, consumed by the next compute_loss
    cp_group: Any = None  # the context-parallel group train.py gathers the logprobs over
    by_sample: dict[int, Head] = field(default_factory=dict)  # id(trainer_logprobs) -> head, during one compute_loss
    installed: bool = False


CTX = _Ctx()


def head_forward(lm_head: Any, hidden_states: torch.Tensor, labels: torch.Tensor, temperature: torch.Tensor, payload: tuple) -> dict[str, torch.Tensor]:
    """``FusedOutputLinear.forward`` for a micro batch that carries heads: logprobs and entropy as prime-rl computes them
    (full vocabulary), the head stashed for ``compute_loss``."""
    ids, lq, present = payload
    b, s, hdim = hidden_states.shape
    k = ids.shape[-1]
    hid = hidden_states.reshape(b * s, hdim).contiguous()
    lab = labels.reshape(b * s).contiguous()
    inv_t = (1.0 / temperature.reshape(b * s)).contiguous()
    ids_f = ids.reshape(b * s, k)
    lp, ent, head_lp = HeadLogProbFn.apply(hid, lm_head.weight, lab, inv_t, lm_head.chunk_size, ids_f)
    valid = ids >= 0
    hit = (ids == labels.unsqueeze(-1)) & valid
    lq_label = torch.where(hit.any(-1), torch.where(hit, lq, torch.zeros_like(lq)).sum(-1), torch.full_like(lq[..., 0], float("nan")))
    CTX.pending = _Pending(lp=head_lp.reshape(b, s, k), lq=lq, valid=valid, present=present, lq_label=lq_label)
    return {"logprobs": lp.reshape(b, s), "entropy": ent.reshape(b, s)}


def _shift_right(t: torch.Tensor, pad: float | bool) -> torch.Tensor:
    """prime-rl's ``shift_tensor_right`` along dim 1 for [B, S, ...] tensors."""
    return torch.cat([torch.full_like(t[:, :1], pad), t[:, :-1]], dim=1)


def split_pending(p: _Pending, lengths: list[int], *, gather: Callable[[torch.Tensor], torch.Tensor] | None = None) -> list[Head]:
    """The micro batch's head -> one ``loss.Head`` per sequence at sampled-token positions (module doc, point 3)."""
    lp, lq, valid, present, lq_label = p.lp, p.lq, p.valid, p.present, p.lq_label
    if gather is not None:  # context parallel: prime-rl's differentiable all-gather over the cp group, as for the logprobs
        lp, lq, lq_label = gather(lp), gather(lq), gather(lq_label)
        valid, present = gather(valid.to(torch.int8)).bool(), gather(present.to(torch.int8)).bool()
    if lp.shape[1] != sum(lengths):
        raise ValueError(f"the head covers {lp.shape[1]} positions, the micro batch {sum(lengths)}")
    lp, lq, lq_label = _shift_right(lp, 0.0), _shift_right(lq, float("-inf")), _shift_right(lq_label, float("nan"))
    valid, present = _shift_right(valid, False), _shift_right(present, False)
    parts = [x[0].split(lengths) for x in (lp, lq, valid, present, lq_label)]
    return [Head(lp=a, lq=b, valid=c, present=d, lq_label=e) for a, b, c, d, e in zip(*parts, strict=True)]


def compute_loss_wrapper(orig: Callable[..., Any], *, gather_for_cp: Callable[..., Any] | None = None) -> Callable[..., Any]:
    """prime-rl's ``compute_loss`` with the micro batch's heads keyed by each sequence's ``trainer_logprobs`` tensor."""

    def compute_loss(*args: Any, **kwargs: Any) -> Any:
        p, CTX.pending = CTX.pending, None
        if p is None:
            return orig(*args, **kwargs)
        tl = kwargs["trainer_logprobs"] if "trainer_logprobs" in kwargs else args[0]
        lengths = [int(t.shape[0]) for t in tl]
        gather = None
        if p.lp.shape[1] != sum(lengths):
            if CTX.cp_group is None or gather_for_cp is None:
                raise RuntimeError(f"the head covers {p.lp.shape[1]} of {sum(lengths)} positions and no context-parallel group was seen")
            group = CTX.cp_group
            gather = lambda x: gather_for_cp(x, group)
        heads = split_pending(p, lengths, gather=gather)
        CTX.by_sample = {id(t): h for t, h in zip(tl, heads, strict=True)}
        try:
            return orig(*args, **kwargs)
        finally:
            CTX.by_sample = {}

    compute_loss.__crucible_sc__ = True  # type: ignore[attr-defined]
    return compute_loss


def lookup(trainer_logprobs: Any) -> Head | None:
    """The head of the sequence whose ``LossInputs.trainer_logprobs`` this is (None: the step has no heads)."""
    return CTX.by_sample.get(id(trainer_logprobs))


def install() -> None:
    """Patch prime-rl's trainer (module doc). Idempotent; raises when the interfaces it hooks are not where v0.10.0 has
    them, so a changed prime-rl fails at startup, never with silently uncentered or renormalized logprobs."""
    if CTX.installed:
        return
    from prime_rl.trainer.models.layers import lm_head as lmh  # type: ignore[import-not-found]
    from prime_rl.trainer.rl import loss as prl_loss  # type: ignore[import-not-found]

    fused, vanilla = lmh.FusedOutputLinear, lmh.VanillaOutputLinear
    orig_fused, orig_vanilla = fused.forward, vanilla.forward

    def fused_forward(self, hidden_states, labels=None, temperature=None, sampling_mask=None):
        payload = decode_payload(sampling_mask) if sampling_mask is not None and temperature is not None else None
        if payload is None:
            return orig_fused(self, hidden_states, labels, temperature=temperature, sampling_mask=sampling_mask)
        return lmh.PrimeLmOutput(**head_forward(self, hidden_states, labels, temperature, payload))

    def vanilla_forward(self, hidden_states, labels=None, temperature=None, sampling_mask=None):
        if sampling_mask is not None and decode_payload(sampling_mask) is not None:
            raise RuntimeError("score centering needs prime-rl's fused LM head (model.fused_lm_head_token_chunk_size, default 8192): "
                               "the vanilla head's train loop would renormalize the sampled logprobs over the sampler's head")
        return orig_vanilla(self, hidden_states, labels, temperature=temperature, sampling_mask=sampling_mask)

    # each module's own namespace, never ``getattr``: a lazy module (transformers' ``_LazyModule``, which every process that
    # imported transformers holds in ``sys.modules``) imports a submodule on attribute access, and one whose optional
    # dependency is missing (``image_processing_aria_fast`` without torchvision) raised out of this scan. It made the hook
    # depend on what else the process had imported (the unit test failed only after a test that imported transformers)
    def own(m: Any, name: str) -> Any:
        d = getattr(m, "__dict__", None)
        return d.get(name) if isinstance(d, dict) else None

    trains = [m for m in list(sys.modules.values()) if m is not None and own(m, "compute_loss") is prl_loss.compute_loss
              and own(m, "gather_for_cp") is not None and own(m, "setup_rl_loss_fn") is not None]
    if not trains:
        raise RuntimeError("prime-rl's train module (compute_loss, gather_for_cp) is not loaded: the score-centering hook cannot attach")
    fused.forward, vanilla.forward = fused_forward, vanilla_forward
    for mod in trains:
        orig_gather = mod.gather_for_cp

        def gather_for_cp(t, cp_group, _orig=orig_gather):
            CTX.cp_group = cp_group
            return _orig(t, cp_group)

        mod.gather_for_cp = gather_for_cp
        mod.compute_loss = compute_loss_wrapper(prl_loss.compute_loss, gather_for_cp=orig_gather)
    CTX.installed = True
