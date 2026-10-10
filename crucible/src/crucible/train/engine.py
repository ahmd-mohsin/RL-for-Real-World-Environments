"""The inference side of M12: the engine contract, the router, and a tiny real torch policy for CPU tests.

**Engine contract** (research m11-m12/06 §4.4, checklist 2–3; m11-m12/01 verl ``llm_server`` / slime router):

- ``generate(GenRequest) -> GenResult``: token ids in, token ids + **behaviour logprobs** out (token-in-token-out:
  never detokenize→retokenize), and the **policy version that served it**. With ``version_switch="call"`` (default) a
  call is served by one version; an abort ends it with ``finish="abort"`` and the caller re-issues the whole call.
  With ``"token"`` the partial output is kept and the call resumes on the next version (AReaL), recorded as
  ``version_spans``.
- ``abort_all()``: every in-flight request returns ``finish="abort"`` now (colocated sample barrier).
- ``pause()`` / ``resume()``: new requests wait while paused (weights are being replaced).
- ``load_weights(state, version)``: install a new weight set and its version (atomically between requests).
- ``capacity``: concurrent sequences (continuous batching slots).

**Router** (``EnginePool``): requests of one rollout stick to one engine (prefix / KV reuse across turns: verl and slime
route by session; we hash with crc32, not Python's salted ``hash()`` — verl pitfall), falling back to the least-loaded
engine when the sticky one is full.

**``TinyPolicy`` / ``TorchEngine``**: a real (small) autoregressive torch model and an engine that samples from it
token by token with real logprobs, cooperative aborts, an optional decode delay, and an optional **numerics gap**
(the engine samples from a bf16-rounded copy of the trainer's weights) so train–inference mismatch is real, not
assumed. This is what the CPU end-to-end loop trains.
"""

from __future__ import annotations

import asyncio
import copy
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch


@dataclass
class GenRequest:
    rid: str  # rollout id (sticky routing key)
    call_id: str
    prompt_ids: list[int]
    max_tokens: int = 64
    temperature: float = 1.0
    stop_ids: Sequence[int] = ()
    seed: int | None = None
    resume_completion: list[int] = field(default_factory=list)  # token-level resume: output kept from an aborted call
    resume_logprobs: list[float] = field(default_factory=list)
    resume_spans: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class GenResult:
    completion_ids: list[int]
    logprobs: list[float]
    version: int
    finish: str  # stop | length | abort
    version_spans: list[tuple[int, int]] | None = None
    prompt_tokens: int = 0
    reprefill_tokens: int = 0


class Engine(Protocol):
    capacity: int
    version: int

    async def generate(self, req: GenRequest) -> GenResult: ...
    async def abort_all(self) -> int: ...
    async def pause(self) -> None: ...
    async def resume(self) -> None: ...
    async def load_weights(self, state: Any, version: int) -> None: ...


class EnginePool:
    def __init__(self, engines: Sequence[Engine]):
        self.engines = list(engines)
        self.inflight = [0] * len(self.engines)

    def route(self, rid: str) -> int:
        i = zlib.crc32(rid.encode()) % len(self.engines)
        if self.inflight[i] < self.engines[i].capacity:
            return i
        return min(range(len(self.engines)), key=lambda j: (self.inflight[j] / max(self.engines[j].capacity, 1), j))

    async def generate(self, req: GenRequest) -> GenResult:
        i = self.route(req.rid)
        self.inflight[i] += 1
        try:
            return await self.engines[i].generate(req)
        finally:
            self.inflight[i] -= 1

    async def abort_all(self) -> int:
        return sum([await e.abort_all() for e in self.engines])

    async def pause(self) -> None:
        for e in self.engines:
            await e.pause()

    async def resume(self) -> None:
        for e in self.engines:
            await e.resume()

    async def load_weights(self, state: Any, version: int) -> None:
        for e in self.engines:
            await e.load_weights(state, version)

    @property
    def version(self) -> int:
        return min(e.version for e in self.engines)


class TinyPolicy(torch.nn.Module):
    """A small causal LM (embedding → GRU → logits) over a toy vocabulary: enough for real RL dynamics on CPU."""

    def __init__(self, vocab: int = 32, dim: int = 32, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.vocab = vocab
        self.emb = torch.nn.Embedding(vocab, dim)
        self.rnn = torch.nn.GRU(dim, dim, batch_first=True)
        self.head = torch.nn.Linear(dim, vocab)
        for p in self.parameters():
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:  # [B, T] -> logits [B, T, V]
        h, _ = self.rnn(self.emb(ids))
        return self.head(h)

    def token_logprobs(self, prompt: list[int], completion: list[int], temperature: float = 1.0) -> torch.Tensor:
        """log π(completion_t | prompt + completion_<t), with gradient."""
        ids = torch.tensor([prompt + completion[:-1]]) if completion else torch.tensor([prompt])
        logits = self(ids)[0, len(prompt) - 1:] / temperature
        lp = torch.log_softmax(logits, -1)
        return lp[torch.arange(len(completion)), torch.tensor(completion)] if completion else lp[:0, 0]


def bf16_copy(model: torch.nn.Module) -> torch.nn.Module:
    m = copy.deepcopy(model)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(p.to(torch.bfloat16).to(p.dtype))
    return m


class TorchEngine:
    """Samples from a ``TinyPolicy`` copy. ``numerics="bf16"`` rounds the served weights (a real train–inference gap)."""

    def __init__(self, model: TinyPolicy, *, capacity: int = 16, decode_delay_s: float = 0.0, numerics: str = "fp32",
                 version_switch: str = "call"):
        self.capacity, self.decode_delay_s, self.numerics, self.version_switch = capacity, decode_delay_s, numerics, version_switch
        self.version = 0
        self._serve = bf16_copy(model) if numerics == "bf16" else copy.deepcopy(model)
        self._abort_epoch = 0
        self._unpaused = asyncio.Event()
        self._unpaused.set()
        self.inflight = 0

    async def generate(self, req: GenRequest) -> GenResult:
        await self._unpaused.wait()
        self.inflight += 1
        epoch = self._abort_epoch
        version, serve = self.version, self._serve  # pinned at call start: a publish never changes weights under a running call
        out, lps = list(req.resume_completion), list(req.resume_logprobs)
        spans = list(req.resume_spans) or ([(0, version)] if not out else [])
        if out and (not spans or spans[-1][1] != version):
            spans.append((len(out), version))
        g = torch.Generator().manual_seed(req.seed if req.seed is not None else zlib.crc32(f"{req.call_id}|{len(out)}".encode()))
        def forward(seq: list[int]) -> torch.Tensor:
            # grad mode is thread-local: a no_grad block must never span an ``await`` (it would leak into the trainer's
            # coroutine on the same event-loop thread), so each forward gets its own scope
            with torch.no_grad():
                return serve(torch.tensor([seq]))[0, -1]

        try:
            ids = list(req.prompt_ids) + out
            logits_all = forward(ids)
            for _ in range(req.max_tokens - len(out)):
                if self._abort_epoch != epoch:
                    return GenResult(out, lps, version, "abort", spans if self.version_switch == "token" else None, len(req.prompt_ids))
                p = torch.softmax(logits_all / req.temperature, -1)
                tok = int(torch.multinomial(p, 1, generator=g))
                out.append(tok)
                lps.append(float(torch.log(p[tok])))
                if tok in req.stop_ids:
                    return GenResult(out, lps, version, "stop", spans if len(spans) > 1 else None, len(req.prompt_ids))
                await asyncio.sleep(self.decode_delay_s)
                ids = ids + [tok]
                logits_all = forward(ids)
            return GenResult(out, lps, version, "length", spans if len(spans) > 1 else None, len(req.prompt_ids))
        finally:
            self.inflight -= 1

    async def abort_all(self) -> int:
        n = self.inflight
        self._abort_epoch += 1
        return n

    async def pause(self) -> None:
        self._unpaused.clear()

    async def resume(self) -> None:
        self._unpaused.set()

    async def load_weights(self, state: dict[str, torch.Tensor], version: int) -> None:
        new = copy.deepcopy(self._serve)  # a new object: running calls keep the one they pinned
        new.load_state_dict(state)
        self._serve = bf16_copy(new) if self.numerics == "bf16" else new
        self.version = version


class ScriptedEngine:
    """An engine that emits scripted completions (``script(rid, call_index, prompt_ids) -> token ids``) with fixed
    logprobs, honouring versions, pause and abort: the pipeline test double for real chat models."""

    def __init__(self, script, *, capacity: int = 64, logprob: float = -0.1, delay_s: float = 0.0, stop_ids: Sequence[int] = ()):
        self.script, self.capacity, self.logprob, self.delay_s, self.stop_ids = script, capacity, logprob, delay_s, tuple(stop_ids)
        self.version = 0
        self._unpaused = asyncio.Event()
        self._unpaused.set()
        self._abort_epoch = 0
        self.calls: dict[str, int] = {}

    async def generate(self, req: GenRequest) -> GenResult:
        await self._unpaused.wait()
        epoch, version = self._abort_epoch, self.version
        k = self.calls.get(req.rid, 0)
        await asyncio.sleep(self.delay_s)
        if self._abort_epoch != epoch:
            return GenResult([], [], version, "abort")
        self.calls[req.rid] = k + 1
        out = list(self.script(req.rid, k, req.prompt_ids))[: req.max_tokens]
        fin = "stop" if out and out[-1] in self.stop_ids else "length"
        return GenResult(out, [self.logprob] * len(out), version, fin, None, len(req.prompt_ids))

    async def abort_all(self) -> int:
        self._abort_epoch += 1
        return 0

    async def pause(self) -> None:
        self._unpaused.clear()

    async def resume(self) -> None:
        self._unpaused.set()

    async def load_weights(self, state: Any, version: int) -> None:
        self.version = version

