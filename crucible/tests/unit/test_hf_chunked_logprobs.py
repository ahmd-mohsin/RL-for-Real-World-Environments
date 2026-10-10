"""Memory-bounded log-probs (m13 B-2): HFPolicy runs the LM head only on positions predicting trained tokens, in
checkpointed chunks. Log-probs and gradients equal the full-logits computation; on an accelerator the memory held for
backward drops by more than an order of magnitude (the full fp32 logits are never materialized)."""

from __future__ import annotations

import copy

import pytest
import torch


def _qwen(vocab: int):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=8, max_position_embeddings=4096, tie_word_embeddings=True)
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg)


def _reference(model, pairs, temperature=1.0):
    out = []
    dev = next(model.parameters()).device
    for p, c in pairs:
        logits = model(input_ids=torch.tensor([p + c[:-1]], device=dev)).logits[0].float() / temperature
        lp = torch.log_softmax(logits, -1)[len(p) - 1: len(p) - 1 + len(c)]
        out.append(lp[torch.arange(len(c), device=dev), torch.tensor(c, device=dev)])
    return out


@pytest.mark.parametrize("chunk", [1, 7, 4096])
def test_chunked_equals_full_logits_values_and_gradients(chunk):
    from crucible.train.hf import HFPolicy

    base = _qwen(512)
    g = torch.Generator().manual_seed(1)
    pairs = [(torch.randint(0, 512, (20,), generator=g).tolist(), torch.randint(0, 512, (n,), generator=g).tolist()) for n in (5, 11, 3)]
    ref_m, pol = copy.deepcopy(base), HFPolicy(copy.deepcopy(base), lm_head_fp32=True, logit_chunk=chunk)
    ref = _reference(ref_m, pairs, 0.8)
    got = pol.batch_token_logprobs(pairs, 0.8)
    assert all(torch.allclose(a, b.float(), atol=1e-5) for a, b in zip(got, ref, strict=True))
    sum(x.sum() for x in ref).backward()
    sum(x.sum() for x in got).backward()
    grads = {n: q.grad for n, q in ref_m.named_parameters() if q.grad is not None}
    for n, q in pol.model.named_parameters():
        if q.grad is not None:
            assert torch.allclose(q.grad, grads[n], atol=1e-5), n


@pytest.mark.skipif(not torch.backends.mps.is_available() and not torch.cuda.is_available(), reason="needs an accelerator to measure memory")
def test_chunked_logprobs_hold_far_less_memory_for_backward():
    from crucible.train.hf import HFPolicy

    dev = "cuda" if torch.cuda.is_available() else "mps"
    alloc = (lambda: torch.cuda.memory_allocated()) if dev == "cuda" else (lambda: torch.mps.current_allocated_memory())
    sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: torch.mps.synchronize())
    model = _qwen(32_000).to(dev)
    p, c = list(range(1500)), list(range(1, 1500))
    sync()
    base = alloc()
    full = _reference(model, [(p, c)])  # holds batch × length × vocab fp32 logits for backward
    sync()
    full_held = alloc() - base
    del full
    pol = HFPolicy(model, lm_head_fp32=True, logit_chunk=128)
    sync()
    base = alloc()
    out = pol.batch_token_logprobs([(p, c)])
    sync()
    chunk_held = alloc() - base
    assert out[0].shape[0] == len(c)
    logits_bytes = len(c) * 32_000 * 4  # one fp32 copy of the completion positions' logits
    # the attention activations (L² in this tiny-hidden model) are the same in both paths; the chunked path saves at least
    # the full logits (the reference also keeps the log-softmax copy)
    assert full_held - chunk_held >= logits_bytes, (chunk_held, full_held, logits_bytes)


def test_gradient_checkpointing_engages_and_stays_deterministic():
    """``from_pretrained`` returns eval mode, where HF never recomputes layers: checkpointing was silently off (lab M12
    OOM). With ``gradient_checkpointing=True`` each decoder layer runs again during backward, dropout is zero, and the
    log-probs equal the eval-mode model's."""
    from crucible.train.hf import HFPolicy

    base = _qwen(256)
    base.config.attention_dropout = 0.3  # would make a train-mode forward stochastic
    base.eval()
    ref_m = copy.deepcopy(base)
    pol = HFPolicy(copy.deepcopy(base), gradient_checkpointing=True)
    assert pol.model.training and pol.model.config.attention_dropout == 0.0
    calls = [0]
    for layer in pol.model.model.layers:
        layer.register_forward_pre_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
    pairs = [(list(range(10)), [3, 4, 5, 6])]
    lp = pol.batch_token_logprobs(pairs)[0]
    fwd = calls[0]
    lp.sum().backward()
    assert fwd == len(pol.model.model.layers) and calls[0] == 2 * fwd  # recomputed in backward
    torch.testing.assert_close(lp.detach(), _reference(ref_m, pairs)[0].detach(), atol=1e-5, rtol=1e-5)
