"""vLLM API-server side of score centering: prime-rl's ``/inference/v1/generate`` also returns the sampler's top-k head
as packed arrays (swarm-pipeline/14 §7.1).

**What vLLM 0.31.0 and prime-rl v0.10.0 return.** With ``sampling_params.logprobs = k`` vLLM's sampler gathers k + 1
entries per generated token, the sampled token first and then the top k by logprob (``Sampler.gather_logprobs``), from
the temperature-processed distribution (prime-rl sets ``logprobs_mode = "processed_logprobs"``); the API server keeps
them as ``FlatLogprobs`` (flat Python lists, ``flat_logprobs = True``, which prime-rl turns on). Upstream's generate
handler would answer with ``logprobs.content``: one JSON object per token with a ``top_logprobs`` list of
``{"token": "token_id:<n>", "logprob": x}``, keyed by token id, so a sampled token outside the head pushes the k-th head
entry out of the list. prime-rl's handler (``inference/vllm/serving_tokens.py``, ``_PackedOutputs``) replaces that
with ``completion_logprobs``, the sampled entries only as a base64 float32 array, and drops the rest: on prime-rl's
endpoint the head is not returned at all. vLLM PR 62446 (open, not merged) adds packed top-k arrays upstream; this module
does not depend on it.

**The patch** (``install``): ``_PackedOutputs`` also packs entries 1..k of every position as ``topk_ids`` (int32) and
``topk_logprobs`` (float32), both ``[n, k]`` in prime-rl's ``{data, shape, dtype}`` base64 form, and the response models
gain the two fields. Nothing else changes; ``logprobs = 0`` requests return nothing new.

**Loading.** vLLM imports every ``--middleware`` import path in each API-server process while it builds the app, before
prime-rl's ``init_app_state`` swaps in its handler; ``deploy.inference_toml`` sets ``vllm.middleware = [MIDDLEWARE]``
(prime-rl passes extra ``vllm`` keys through to vLLM's namespace) and ``vllm.max_logprobs = k`` (vLLM's default cap is
20). ``CrucibleTopK`` is a pass-through ASGI middleware; importing this module installs the patch.

**Cost** (§7.1): the GPU part is one ``torch.topk`` over the vocabulary per sampled token. The API server holds k + 1
Python floats, ints and ranks per token in ``FlatLogprobs`` until the call ends (about 2.8 KB per token at k = 32, so
~23 MB for an 8192-token reply), and converts them with numpy once per call; the JSON carries 8·k bytes per token as
base64 (~350 B at k = 32) instead of ~60 B per head entry as ``top_logprobs`` objects.
"""

from __future__ import annotations

import sys
from typing import Any


def pack_topk(flat: Any, k: int) -> dict[str, Any] | None:
    """``FlatLogprobs`` -> ``(ids [n, k] int32, logprobs [n, k] float32)`` from entries 1..k of each position (entry 0
    is the sampled token); positions with fewer entries are padded with -1 / -inf. None when k < 1."""
    import numpy as np

    if k < 1:
        return None
    starts = np.asarray(flat.start_indices, dtype=np.int64)
    ends = np.asarray(flat.end_indices, dtype=np.int64)
    tok = np.asarray(flat.token_ids, dtype=np.int32)
    lps = np.asarray(flat.logprobs, dtype=np.float32)
    n = len(starts)
    ids = np.full((n, k), -1, dtype=np.int32)
    out = np.full((n, k), -np.inf, dtype=np.float32)
    if n and bool((ends - starts == k + 1).all()):  # the regular case: one reshape
        idx = starts[:, None] + 1 + np.arange(k)[None, :]
        ids[:], out[:] = tok[idx], lps[idx]
    else:
        for i, (s, e) in enumerate(zip(starts.tolist(), ends.tolist(), strict=True)):
            m = max(0, min(k, e - s - 1))
            ids[i, :m], out[i, :m] = tok[s + 1:s + 1 + m], lps[s + 1:s + 1 + m]
    return {"ids": ids, "logprobs": np.maximum(out, -9999.0)}  # vLLM's clamp for -inf (prime-rl LOGPROB_SENTINEL)


def install() -> None:
    """Patch prime-rl's packed generate handler (module doc). Idempotent."""
    from prime_rl.inference.vllm import serving_tokens as st  # type: ignore[import-not-found]
    from vllm.logprobs import FlatLogprobs  # type: ignore[import-not-found]

    if getattr(st, "__crucible_topk__", False):
        return

    class CrucibleChoice(st.PrimeRlGenerateResponseChoice):
        topk_ids: dict[str, Any] | None = None
        topk_logprobs: dict[str, Any] | None = None

    class CrucibleResponse(st.PrimeRlGenerateResponse):
        choices: list[CrucibleChoice]  # type: ignore[assignment]

    orig_init = st._PackedOutputs.__init__

    def init(self, generator, request):
        k = int(getattr(request.sampling_params, "logprobs", None) or 0)

        async def with_topk(gen=generator):
            async for request_output in gen:
                for output in request_output.outputs:
                    if k > 0 and isinstance(output.logprobs, FlatLogprobs):
                        packed = pack_topk(output.logprobs, k)
                        fields = self.fields.setdefault(output.index, {})
                        fields["topk_ids"] = st.encode_array(packed["ids"])
                        fields["topk_logprobs"] = st.encode_array(packed["logprobs"])
                yield request_output

        orig_init(self, with_topk(), request)

    st._PackedOutputs.__init__ = init
    # post_process builds these module globals by name: the extra fields survive validation and serialization
    st.PrimeRlGenerateResponseChoice, st.PrimeRlGenerateResponse = CrucibleChoice, CrucibleResponse
    st.__crucible_topk__ = True


class CrucibleTopK:
    """Pass-through ASGI middleware (``deploy.MIDDLEWARE``): its import in vLLM's API server installs the patch, and so
    does building the app's middleware stack, before the first request."""

    def __init__(self, app: Any):
        self.app = app
        install()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await self.app(scope, receive, send)


if "prime_rl.inference.vllm.server" in sys.modules:  # imported by vLLM's build_app in prime-rl's API server
    install()
