"""Hugging Face causal LMs as M12 policies (the trainer side of a real model).

``HFPolicy`` exposes what ``TorchTrainer`` needs: ``token_logprobs(prompt, completion, temperature)`` (with gradient)
and a batched ``batch_token_logprobs`` (right-padded, attention-masked: one forward per micro batch), plus HF-format
``state_dict`` names, which is exactly what vLLM's ``load_weights`` consumes (``CheckpointPublisher`` →
``vllm_ext.CrucibleWorkerExtension``).

``lm_head_fp32`` (default on): the LM head is applied in FP32 on the final hidden states (BF16 input, FP32 output), the
"highest-value single change" against train–inference mismatch (MiniMax M1, Ling 2.6; research m11-m12/06 §5.3). The
engine side must match it where configurable; the run record states both.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


class HFPolicy(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, lm_head_fp32: bool = True, gradient_checkpointing: bool = False,
                 autocast_dtype: torch.dtype | None = None):
        """``autocast_dtype=torch.bfloat16`` with an fp32 model: fp32 master weights (small learning rates survive the update;
        bf16 weights silently drop updates below their precision) and a bf16 forward."""
        super().__init__()
        self.model = model
        self.lm_head_fp32 = lm_head_fp32
        self.autocast_dtype = autocast_dtype
        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    @classmethod
    def from_pretrained(cls, name: str, *, dtype: torch.dtype = torch.bfloat16, device: str = "cpu", **kw) -> HFPolicy:
        from transformers import AutoModelForCausalLM

        return cls(AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device), **kw)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _logits(self, ids: torch.Tensor, attn: torch.Tensor | None = None, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        if self.autocast_dtype is not None:
            with torch.autocast(device_type=ids.device.type, dtype=self.autocast_dtype):
                return self._logits_inner(ids, attn, position_ids)
        return self._logits_inner(ids, attn, position_ids)

    def _logits_inner(self, ids: torch.Tensor, attn: torch.Tensor | None = None, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        kw = {"position_ids": position_ids} if position_ids is not None else {}
        if not self.lm_head_fp32:
            return self.model(input_ids=ids, attention_mask=attn, **kw).logits.float()
        out = self.model(input_ids=ids, attention_mask=attn, output_hidden_states=True, **kw)
        h = out.hidden_states[-1]
        head = self.model.get_output_embeddings()
        with torch.autocast(device_type=ids.device.type, enabled=False):  # BF16 input, FP32 output (the LM head itself in FP32)
            return torch.nn.functional.linear(h.float(), head.weight.float(), None if head.bias is None else head.bias.float())

    def token_logprobs(self, prompt: list[int], completion: list[int], temperature: float = 1.0, candidates: list[list[int]] | None = None) -> torch.Tensor:
        return self.batch_token_logprobs([(prompt, completion)], temperature, candidates=[candidates])[0]

    def batch_token_logprobs(self, pairs: Sequence[tuple[list[int], list[int]]], temperature: float = 1.0,
                             candidates: Sequence[list[list[int]] | None] | None = None) -> list[torch.Tensor]:
        seqs = [p + c[:-1] if c else p for p, c in pairs]
        L = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), L, dtype=torch.long, device=self.device)
        attn = torch.zeros(len(seqs), L, dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, device=self.device)
            attn[i, : len(s)] = 1
        logits = self._logits(ids, attn) / temperature
        lp = torch.log_softmax(logits, -1)
        out = []
        for i, (p, c) in enumerate(pairs):
            if not c:
                out.append(lp[i, :0, 0])
                continue
            rows = lp[i, len(p) - 1: len(p) - 1 + len(c)]
            if candidates is not None and candidates[i] is not None:  # top-p: renormalize over the engine's nucleus
                from crucible.train.sampling import renormalized

                out.append(renormalized(rows, c, candidates[i]))
                continue
            out.append(rows[torch.arange(len(c), device=self.device), torch.tensor(c, device=self.device)])
        return out

    def shared_prefix_logprobs(self, prompt: list[int], completions: list[list[int]], temperature: float = 1.0) -> list[torch.Tensor]:
        """Prefix sharing (checklist #50; MiniMax Forge packing): one sequence ``prompt + c₁[:-1] + c₂[:-1] + …`` with position
        ids restarting after the prompt for every completion and a 4D additive mask (each completion sees the prompt and its
        own earlier tokens only). The prompt is encoded once; logprobs equal separate passes (a float mask: a boolean 4D
        mask is silently ignored by eager attention)."""
        P = len(prompt)
        seq, pos, seg, starts = list(prompt), list(range(P)), [-1] * P, []
        for i, c in enumerate(completions):
            starts.append(len(seq))
            body = c[:-1]
            seq += body
            pos += list(range(P, P + len(body)))
            seg += [i] * len(body)
        L, dev = len(seq), self.device
        segt = torch.tensor(seg, device=dev)
        q = torch.arange(L, device=dev)
        allow = (q[None, :] <= q[:, None]) & ((segt[None, :] == -1) | (segt[None, :] == segt[:, None]))
        mask = torch.zeros(1, 1, L, L, device=dev).masked_fill(~allow, torch.finfo(torch.float32).min)
        lp = torch.log_softmax(self._logits(torch.tensor([seq], device=dev), mask, torch.tensor([pos], device=dev))[0] / temperature, -1)
        out = []
        for i, c in enumerate(completions):
            if not c:
                out.append(lp[:0, 0])
                continue
            rows = torch.cat([lp[P - 1: P], lp[starts[i]: starts[i] + len(c) - 1]])
            out.append(rows[torch.arange(len(c), device=dev), torch.tensor(c, device=dev)])
        return out

    def state_dict(self, *a, **kw):  # HF names (what vLLM load_weights expects), without the wrapper prefix
        return self.model.state_dict(*a, **kw)

    def load_state_dict(self, sd, *a, **kw):
        return self.model.load_state_dict(sd, *a, **kw)
