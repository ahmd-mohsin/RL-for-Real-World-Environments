"""Exact token-in/token-out for the scripted RLM models (M16 validation): ``TitoLLM`` serves a scripted model's reply through
``EngineLLM`` over a real chat codec, ``LockedTitoLLM`` pairs each reply with its call for concurrent agents, ``qwen_codec`` is the
offline Qwen2.5 codec. They live in ``train`` because they use the engine and the codec; the scripted models themselves stay in
``runtime.scripted_rlm`` / ``runtime.scripted_rlm_tree`` with no upward import (the layer contract, M0 audit D3)."""

from __future__ import annotations

import asyncio
from typing import Any

from crucible.actors.llm import LLMResponse
from crucible.runtime.scripted_rlm import ScriptedRLM


class TitoLLM:
    """Exact TITO for a scripted model: ``EngineLLM`` over ``codec`` serves the model's reply as the sampled completion."""

    def __init__(self, model: ScriptedRLM, codec: Any, *, name: str | None = None):
        from crucible.train.world_env import EngineLLM

        self.model, self.codec = model, codec
        self._next: list[int] = []
        self.engine = EngineLLM(codec, self._gen, model=name or model.model)
        self.spend = self.engine.spend

    def _encode(self, text: str) -> list[int]:
        enc = getattr(self.codec, "encode_text", None)
        ids = enc(text) if callable(enc) else self.codec.tok.encode(text, add_special_tokens=False)
        return list(ids) + [self.codec.eos_ids[0]]

    async def _gen(self, prompt: list[int], max_tokens: int, temperature: float, *, seed: int | None = None, stop_ids: Any = ()) -> Any:
        from crucible.train.engine import GenResult

        ids = self._next
        if self.model.log:
            self.model.log[-1].update(prompt_tokens=len(prompt), completion_tokens=len(ids), usage_source="tito")
        return GenResult(completion_ids=ids, logprobs=[-0.1] * len(ids), version=0, finish="stop", prompt_tokens=len(prompt))

    async def complete(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, **kw: Any) -> LLMResponse:
        r = self.model.reply_for(messages, tools)
        self._next = self._encode(self.model.hermes(r))
        return await self.engine.complete(messages, tools=tools, **kw)


def qwen_codec(name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> Any:
    """The real Qwen2.5 chat codec from the local Hugging Face cache (offline)."""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    from crucible.train.chat import HFChatCodec

    return HFChatCodec(AutoTokenizer.from_pretrained(name))


class LockedTitoLLM(TitoLLM):
    """``TitoLLM`` for a tree of concurrent agents: the decision and the served completion are paired under one lock."""

    def __init__(self, model: Any, codec: Any, *, name: str | None = None):
        super().__init__(model, codec, name=name)
        self._lock: asyncio.Lock | None = None

    async def complete(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, **kw: Any) -> Any:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            return await super().complete(messages, tools=tools, **kw)
