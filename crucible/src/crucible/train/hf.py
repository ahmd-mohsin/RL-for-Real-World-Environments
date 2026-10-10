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
                 autocast_dtype: torch.dtype | None = None, logit_chunk: int = 1024):
        """``autocast_dtype=torch.bfloat16`` with an fp32 model: fp32 master weights (small learning rates survive the update;
        bf16 weights silently drop updates below their precision) and a bf16 forward."""
        super().__init__()
        self.model = model
        self.lm_head_fp32 = lm_head_fp32
        self.autocast_dtype = autocast_dtype
        self.logit_chunk = logit_chunk  # LM-head rows per checkpointed chunk (m13 B-2)
        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            # HF recomputes layers only when ``self.training`` (``from_pretrained`` returns eval mode, so checkpointing was
            # silently off and an 8k-token segment OOMed a 32 GB GPU on a 0.6B model). Train mode with every dropout at 0:
            # the forward stays deterministic and equal to the engine's.
            for m in model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.p = 0.0
            cfg = getattr(model, "config", None)
            for k in [k for k in vars(cfg) if k.endswith("dropout")] if cfg is not None else []:
                if isinstance(getattr(cfg, k), float):
                    setattr(cfg, k, 0.0)
            for m in model.modules():
                if hasattr(m, "attention_dropout") and isinstance(m.attention_dropout, float):
                    m.attention_dropout = 0.0
            model.train()

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

    # ---- memory-bounded log-probs (m13 B-2) ------------------------------------------------------------------------
    # The LM head runs only on the positions that predict trained tokens (never on the prompt), in checkpointed chunks of
    # ``logit_chunk`` rows: peak memory is chunk × vocab instead of batch × length × vocab in fp32, and backward recomputes
    # each chunk instead of storing it (prime-rl / AReaL chunked log-probs, the pattern).
    def _hidden(self, ids: torch.Tensor, attn: torch.Tensor | None = None, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        kw = {"position_ids": position_ids} if position_ids is not None else {}
        base = getattr(self.model, getattr(self.model, "base_model_prefix", "model"), None)

        def run() -> torch.Tensor:
            if base is not None and base is not self.model:  # the decoder without the LM head: no full-vocabulary logits at all
                return base(input_ids=ids, attention_mask=attn, **kw).last_hidden_state
            return self.model(input_ids=ids, attention_mask=attn, output_hidden_states=True, **kw).hidden_states[-1]

        if self.autocast_dtype is not None:
            with torch.autocast(device_type=ids.device.type, dtype=self.autocast_dtype):
                return run()
        return run()

    def _head_rows(self, h: torch.Tensor, temperature: float) -> torch.Tensor:
        head = self.model.get_output_embeddings()
        with torch.autocast(device_type=h.device.type, enabled=False):
            if self.lm_head_fp32:  # BF16 input, FP32 output (the LM head itself in FP32)
                return torch.nn.functional.linear(h.float(), head.weight.float(), None if head.bias is None else head.bias.float()) / temperature
            return head(h).float() / temperature

    def _head_logprobs(self, h: torch.Tensor, targets: torch.Tensor, temperature: float) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint

        def chunk(hc: torch.Tensor, tc: torch.Tensor) -> torch.Tensor:
            logits = self._head_rows(hc, temperature)
            return logits.gather(-1, tc[:, None])[:, 0] - torch.logsumexp(logits, -1)

        if h.shape[0] == 0:
            return h.new_zeros(0, dtype=torch.float32)
        C = max(1, self.logit_chunk)
        parts = [checkpoint(chunk, h[i:i + C], targets[i:i + C], use_reentrant=False) if torch.is_grad_enabled() else chunk(h[i:i + C], targets[i:i + C])
                 for i in range(0, h.shape[0], C)]
        return torch.cat(parts)

    def completion_logprob_rows(self, prompt: list[int], completion: list[int], temperature: float = 1.0) -> torch.Tensor:
        """[len(completion), V] full log-softmax rows at the completion positions (full-vocabulary OPD, m13 C-5): the one
        place a full-vocabulary fp32 row per position is materialized, so keep segments short in this mode."""
        ids = torch.tensor([prompt + completion[:-1]], device=self.device)
        h = self._hidden(ids)[0, len(prompt) - 1: len(prompt) - 1 + len(completion)]
        return torch.log_softmax(self._head_rows(h, temperature), -1)

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
        h = self._hidden(ids, attn)
        out: list[torch.Tensor | None] = [None] * len(pairs)
        rows, tgts, spans = [], [], []
        for i, (p, c) in enumerate(pairs):
            hi = h[i, len(p) - 1: len(p) - 1 + len(c)]  # the positions predicting the completion tokens
            if not c:
                out[i] = h.new_zeros(0, dtype=torch.float32)
            elif candidates is not None and candidates[i] is not None:  # top-p: renormalize over the engine's nucleus (full rows, small)
                from crucible.train.sampling import renormalized

                out[i] = renormalized(torch.log_softmax(self._head_rows(hi, temperature), -1), c, candidates[i])
            else:
                spans.append((i, sum(len(r) for r in rows), len(c)))
                rows.append(hi)
                tgts.append(torch.tensor(c, device=self.device))
        if rows:
            lp = self._head_logprobs(torch.cat(rows), torch.cat(tgts), temperature)
            for i, s0, n in spans:
                out[i] = lp[s0: s0 + n]
        return out  # type: ignore[return-value]

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
        h = self._hidden(torch.tensor([seq], device=dev), mask, torch.tensor([pos], device=dev))[0]
        rows, tgts, spans = [], [], []
        for i, c in enumerate(completions):
            if c:
                spans.append((i, sum(len(t) for t in tgts), len(c)))
                rows.append(torch.cat([h[P - 1: P], h[starts[i]: starts[i] + len(c) - 1]]))
                tgts.append(torch.tensor(c, device=dev))
        out = [h.new_zeros(0, dtype=torch.float32) for _ in completions]
        if rows:
            lp = self._head_logprobs(torch.cat(rows), torch.cat(tgts), temperature)
            for i, s0, n in spans:
                out[i] = lp[s0: s0 + n]
        return out

    def state_dict(self, *a, **kw):  # HF names (what vLLM load_weights expects), without the wrapper prefix
        return self.model.state_dict(*a, **kw)

    def load_state_dict(self, sd, *a, **kw):
        return self.model.load_state_dict(sd, *a, **kw)
