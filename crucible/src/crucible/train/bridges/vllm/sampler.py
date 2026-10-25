"""The ``vllm`` sampler backend for ``factory.models`` (registered ``final=True``: our own engine, numerics recorded).

``make(model, role=..., thinking=...)`` is a ``TinkerChatClient`` over a ``TinkerSampler`` whose service is a
``VLLMBackend`` (``CRUCIBLE_VLLM_URL``) and whose row is ``vllm_model(model)``: the factory's roles and the student's
episodes use the same renderer and the same engine side as on Tinker, at no per-token price.
"""

from __future__ import annotations

import os
from typing import Any

from crucible.train.bridges.tinker.sampler import TinkerChatClient, TinkerSampler
from crucible.train.bridges.tinker.spend import SpendMeter
from crucible.train.bridges.vllm.client import DEFAULT_CONTEXT, VLLMBackend, vllm_model

_SHARED: dict[str, Any] = {}
UNCAPPED = 1e12  # no per-token price: the meter only counts tokens (the GPU-hours are the Slurm account's)


def shared_backend() -> VLLMBackend:
    if "backend" not in _SHARED:
        _SHARED["backend"] = VLLMBackend(os.environ.get("CRUCIBLE_VLLM_URL"))
    return _SHARED["backend"]


def meter() -> SpendMeter:
    if "meter" not in _SHARED:
        _SHARED["meter"] = SpendMeter(UNCAPPED)
    return _SHARED["meter"]


def make(model: str, *, role: str | None = None, thinking: bool | None = None, backend: Any = None, codec: Any = None,
         context: int | None = None, **_: Any) -> TinkerChatClient:
    from crucible.train.bridges.tinker.codec import TinkerChatCodec, load_tokenizer

    m = vllm_model(model, context=int(context or os.environ.get("CRUCIBLE_VLLM_CONTEXT") or DEFAULT_CONTEXT))
    if codec is None:
        codec = TinkerChatCodec(load_tokenizer(m), m, enable_thinking=bool(thinking))
    return TinkerChatClient(TinkerSampler(backend or shared_backend(), m, meter=meter(), codec=codec), role=role)
