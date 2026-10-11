"""A real Hugging Face causal LM as an M12 generation engine (m13 F-2), plus the engine capability protocol (F-1).

``HFGenerateEngine`` samples from an HF model on CPU / MPS / CUDA with the same contract as ``TorchEngine``: token ids in,
token ids + behaviour logprobs out (the sampled token's log-prob under the temperature- and top-p-processed
distribution, i.e. the distribution actually sampled from), the policy version pinned at call start, token-level resume
(``resume_completion``; ``version_switch="token"`` records ``version_spans``), top-p candidate sets, abort of every
call or of one call (``abort(call_id)``), pause / resume, and a KV cache, so decoding costs O(1) forward positions
per token. It makes end-to-end M12 runs with real models possible without vLLM (laptops, CI), and it is the reference
the vLLM / SGLang engines are measured against.

Weights: ``load_weights(state | checkpoint dir, version)`` loads into a copy while calls are running, so a publish never
changes the weights under a running call, and in place when the engine is idle.

``capabilities()`` (F-1) states what an engine supports, so the runtime and the tests can check instead of assuming:
``token_ids``, ``logprobs`` ("processed" | "raw"), ``abort`` ("all" | "per_call"), ``pause`` modes, ``sleep``,
``top_p_candidates``, ``routing``, ``weight_load`` ("state_dict" | "checkpoint_dir"). ``probe_logprobs_mode(engine,
reference)`` measures the logprob convention by sampling at temperature 2 and comparing with a reference model's
log-probs at temperature 2 and at 1. The off-policy corrections depend on the convention, and vLLM's default
(``raw_logprobs``) differs from the processed one.
"""

from __future__ import annotations

import asyncio
import copy
import zlib
from pathlib import Path
from typing import Any

import torch

from crucible.train.engine import GenRequest, GenResult


class HFGenerateEngine:
    def __init__(self, model: Any, *, capacity: int = 8, device: str | None = None, version_switch: str = "call", decode_delay_s: float = 0.0):
        if isinstance(model, str):
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.capacity, self.version_switch, self.decode_delay_s = capacity, version_switch, decode_delay_s
        self.version = 0
        self._abort_epoch = 0
        self._aborted: set[str] = set()
        self._unpaused = asyncio.Event()
        self._unpaused.set()
        self.inflight = 0

    def capabilities(self) -> dict[str, Any]:
        return {"token_ids": True, "logprobs": "processed", "abort": "per_call", "pause": ["keep"], "sleep": False, "top_p_candidates": True,
                "routing": False, "weight_load": "state_dict|checkpoint_dir"}

    def _forward(self, model: Any, ids: list[int], past: Any) -> tuple[torch.Tensor, Any]:
        with torch.no_grad():  # each forward its own scope: grad mode must never span an await
            x = torch.tensor([ids], device=self.device)
            out = model(input_ids=x, past_key_values=past, use_cache=True)
            return out.logits[0, -1].float().cpu(), out.past_key_values

    async def generate(self, req: GenRequest) -> GenResult:
        await self._unpaused.wait()
        self.inflight += 1
        epoch, version, model = self._abort_epoch, self.version, self.model  # pinned at call start
        out, lps = list(req.resume_completion), list(req.resume_logprobs)
        cands: list[list[int]] = list(req.resume_candidates)
        spans = list(req.resume_spans) or ([(0, version)] if not out else [])
        if out and (not spans or spans[-1][1] != version):
            spans.append((len(out), version))
        g = torch.Generator().manual_seed(req.seed if req.seed is not None else zlib.crc32(f"{req.call_id}|{len(out)}".encode()))

        def result(finish: str) -> GenResult:
            sp = spans if (self.version_switch == "token" or len(spans) > 1) and finish == "abort" else (spans if len(spans) > 1 else None)
            return GenResult(out, lps, version, finish, sp, len(req.prompt_ids), cands or None, None)

        try:
            logits, past = self._forward(model, list(req.prompt_ids) + out, None)
            for _ in range(req.max_tokens - len(out)):
                if self._abort_epoch != epoch or req.call_id in self._aborted:
                    self._aborted.discard(req.call_id)
                    return result("abort")
                p = torch.softmax(logits / max(req.temperature, 1e-6), -1)
                if req.top_p < 1.0:
                    from crucible.train.sampling import nucleus

                    c = nucleus(p, req.top_p)
                    q = torch.zeros_like(p)
                    q[c] = p[c] / p[c].sum()
                    p = q
                    cands.append(c)
                tok = int(torch.multinomial(p, 1, generator=g))
                out.append(tok)
                lps.append(float(torch.log(p[tok])))
                if tok in req.stop_ids:
                    return result("stop")
                await asyncio.sleep(self.decode_delay_s)
                logits, past = self._forward(model, [tok], past)
            return result("length")
        finally:
            self.inflight -= 1

    async def abort_all(self) -> int:
        n = self.inflight
        self._abort_epoch += 1
        return n

    async def abort(self, call_id: str) -> None:
        self._aborted.add(call_id)

    async def pause(self) -> None:
        self._unpaused.clear()

    async def resume(self) -> None:
        self._unpaused.set()

    async def load_weights(self, state: Any, version: int) -> None:
        if isinstance(state, str | Path):
            from safetensors.torch import load_file

            state = load_file(str(Path(state) / "model.safetensors"))
        target = self.model if self.inflight == 0 else copy.deepcopy(self.model)  # running calls keep the weights they pinned
        missing, unexpected = target.load_state_dict({k.removeprefix("model."): v for k, v in state.items()} if _wrapped(state, target) else state,
                                                     strict=False)
        tied = getattr(getattr(target, "config", None), "tie_word_embeddings", False)
        missing = [m for m in missing if not (tied and m == "lm_head.weight")]
        if unexpected or missing:  # a partial load would serve weights the trainer never published
            raise ValueError(f"weights do not match this engine: unexpected {unexpected[:5]}, missing {missing[:5]}")
        self.model = target.eval()
        self.version = version


def _wrapped(state: dict, model: Any) -> bool:
    """State dicts from ``HFPolicy`` are prefixed ``model.``; the engine holds the bare HF model."""
    keys = set(model.state_dict())
    return bool(state) and all(k.startswith("model.") for k in state) and not any(k in keys for k in list(state)[:5])


def capabilities(engine: Any) -> dict[str, Any]:
    """An engine's declared capabilities; engines that predate the protocol get a conservative default."""
    fn = getattr(engine, "capabilities", None)
    if callable(fn):
        return fn()
    return {"token_ids": True, "logprobs": "unknown", "abort": "all", "pause": ["keep"], "sleep": False, "top_p_candidates": False, "routing": False}


async def probe_logprobs_mode(engine: Any, reference: Any, prompt: list[int], *, temperature: float = 2.0, tokens: int = 8, tol: float = 1e-3) -> str:
    """"processed" when the engine's logprobs equal the reference's log-probs at the sampling temperature, "raw" when they
    equal temperature 1, else "unknown". ``reference``: a policy with ``token_logprobs(prompt, completion, temperature)``
    and the engine's weights."""
    res = await engine.generate(GenRequest(rid="probe", call_id="probe:c1", prompt_ids=list(prompt), max_tokens=tokens, temperature=temperature, seed=7))
    got = torch.tensor(res.logprobs, dtype=torch.float64)
    with torch.no_grad():
        at_t = reference.token_logprobs(list(prompt), list(res.completion_ids), temperature).double()
        at_1 = reference.token_logprobs(list(prompt), list(res.completion_ids), 1.0).double()
    if torch.allclose(got, at_t, atol=tol):
        return "processed"
    if torch.allclose(got, at_1, atol=tol):
        return "raw"
    return "unknown"
