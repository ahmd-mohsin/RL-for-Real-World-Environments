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
    top_p: float = 1.0  # nucleus sampling; < 1 makes the engine report each token's candidate set (checklist #37)
    resume_candidates: list[list[int]] = field(default_factory=list)
    resume_routing: list[list[int]] = field(default_factory=list)


@dataclass
class GenResult:
    completion_ids: list[int]
    logprobs: list[float]
    version: int
    finish: str  # stop | length | abort
    version_spans: list[tuple[int, int]] | None = None
    prompt_tokens: int = 0
    candidates: list[list[int]] | None = None  # per completion token, the nucleus it was sampled from (top_p < 1)
    routing: list[list[int]] | None = None  # MoE: per completion token, the experts that produced it (R3 replay)
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
        # wall time with ≥ 1 call in flight anywhere in the pool and the pool not paused (bubble accounting, checklist #9):
        # calls re-issued at the colocated barrier wait on the paused engines, and that wait is idle time, not busy time
        # (found on the lab: paused waits were counted as ~100 % utilization during training)
        self.busy_s = 0.0
        self._active, self._busy_t0, self._paused = 0, 0.0, False

    def _running(self) -> bool:
        return self._active > 0 and not self._paused

    def busy_until_now(self) -> float:
        import time

        return self.busy_s + (time.monotonic() - self._busy_t0 if self._running() else 0.0)

    def _set(self, *, active: int | None = None, paused: bool | None = None) -> None:
        import time

        was = self._running()
        if active is not None:
            self._active = active
        if paused is not None:
            self._paused = paused
        now_running = self._running()
        if was and not now_running:
            self.busy_s += time.monotonic() - self._busy_t0
        elif now_running and not was:
            self._busy_t0 = time.monotonic()

    def _enter(self) -> None:
        self._set(active=self._active + 1)

    def _exit(self) -> None:
        self._set(active=self._active - 1)

    def route(self, rid: str) -> int:
        i = zlib.crc32(rid.encode()) % len(self.engines)
        if self.inflight[i] < self.engines[i].capacity:
            return i
        return min(range(len(self.engines)), key=lambda j: (self.inflight[j] / max(self.engines[j].capacity, 1), j))

    async def generate(self, req: GenRequest) -> GenResult:
        i = self.route(req.rid)
        self.inflight[i] += 1
        self._enter()
        try:
            return await self.engines[i].generate(req)
        finally:
            self.inflight[i] -= 1
            self._exit()

    async def abort_all(self) -> int:
        return sum([await e.abort_all() for e in self.engines])

    def capabilities(self) -> dict[str, Any]:
        """What every engine in the pool supports (m13 F-1): a capability is offered only if all engines offer it."""
        from crucible.train.hf_engine import capabilities

        caps = [capabilities(e) for e in self.engines]
        if not caps:
            return {}
        out = dict(caps[0])
        for c in caps[1:]:
            for k, v in list(out.items()):
                w = c.get(k)
                if isinstance(v, bool):
                    out[k] = v and bool(w)
                elif isinstance(v, list):
                    out[k] = [x for x in v if x in (w or [])]
                elif v != w:
                    out[k] = "mixed"
        return out

    async def pause(self) -> None:
        self._set(paused=True)
        for e in self.engines:
            await e.pause()

    async def resume(self) -> None:
        for e in self.engines:
            await e.resume()
        self._set(paused=False)

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

    def token_logprobs(self, prompt: list[int], completion: list[int], temperature: float = 1.0, candidates: list[list[int]] | None = None) -> torch.Tensor:
        """log π(completion_t | prompt + completion_<t), with gradient; renormalized over ``candidates`` (top-p) if given."""
        dev = self.emb.weight.device
        ids = torch.tensor([prompt + completion[:-1]] if completion else [prompt], device=dev)
        logits = self(ids)[0, len(prompt) - 1:] / temperature
        lp = torch.log_softmax(logits, -1)
        if not completion:
            return lp[:0, 0]
        if candidates is not None:
            from crucible.train.sampling import renormalized

            return renormalized(lp[: len(completion)], completion, candidates)
        return lp[torch.arange(len(completion), device=dev), torch.tensor(completion, device=dev)]

    def completion_logprob_rows(self, prompt: list[int], completion: list[int], temperature: float = 1.0) -> torch.Tensor:
        """[len(completion), V] full log-softmax rows at the completion positions (full-vocabulary OPD, m13 C-5)."""
        dev = self.emb.weight.device
        ids = torch.tensor([prompt + completion[:-1]] if completion else [prompt], device=dev)
        return torch.log_softmax(self(ids)[0, len(prompt) - 1: len(prompt) - 1 + len(completion)] / temperature, -1)

    def shared_prefix_logprobs(self, prompt: list[int], completions: list[list[int]], temperature: float = 1.0) -> list[torch.Tensor]:
        """Prefix sharing (checklist #50): the prompt runs once; every completion continues from its final GRU state. The
        gradient through the shared prompt is the sum over the group, exactly as in separate passes."""
        dev = self.emb.weight.device
        h_prompt, state = self.rnn(self.emb(torch.tensor([prompt], device=dev)))
        first = torch.log_softmax(self.head(h_prompt[0, -1]) / temperature, -1)  # every completion's first-token distribution
        out = []
        for c in completions:
            if not c:
                out.append(first[:0])
                continue
            lps = [first[c[0]]]
            if len(c) > 1:
                hc, _ = self.rnn(self.emb(torch.tensor([c[:-1]], device=dev)), state)
                rows = torch.log_softmax(self.head(hc[0]) / temperature, -1)
                lps.extend(rows[torch.arange(len(c) - 1, device=dev), torch.tensor(c[1:], device=dev)])
            out.append(torch.stack(lps))
        return out


class TinyMoEPolicy(TinyPolicy):
    """``TinyPolicy`` + one top-k mixture-of-experts layer after the GRU (checklist #36): the CPU stand-in for MoE
    policies, where the engine's numerics (bf16) can flip the router's choice and the trainer then evaluates a different
    expert path than the one that sampled the token (the MoE train–inference mismatch R3 removes).

    ``forward(ids, routing=None)``: ``routing`` [T][k] expert ids per position replaces the router's top-k (R3: the
    *choice* is replayed, the gate weights are the trainer's softmax over the replayed experts, so they get gradient).
    After every forward ``last_routes`` holds the experts used per position and ``last_confidence`` the routing
    confidence Σ_k (mean over positions of the top-k router probability mass)."""

    shared_prefix_logprobs = None  # the GRU-state shortcut does not hold through the MoE layer: the trainer falls back

    def __init__(self, vocab: int = 32, dim: int = 32, seed: int = 0, experts: int = 4, k: int = 1, router_scale: float = 0.05):
        super().__init__(vocab, dim, seed)
        g = torch.Generator().manual_seed(seed + 1)
        self.k = k
        self.router = torch.nn.Linear(dim, experts, bias=False)
        self.experts = torch.nn.ModuleList(torch.nn.Linear(dim, dim) for _ in range(experts))
        with torch.no_grad():
            self.router.weight.copy_(torch.randn(self.router.weight.shape, generator=g) * router_scale)  # low margins: flips happen
            for e in self.experts:
                for q in e.parameters():
                    q.copy_(torch.randn(q.shape, generator=g) * 0.5)
        self.last_routes: list[list[int]] = []
        self.last_confidence: float = 1.0

    def forward(self, ids: torch.Tensor, routing: list[list[int]] | None = None) -> torch.Tensor:
        h, _ = self.rnn(self.emb(ids))  # [B, T, D]
        probs = torch.softmax(self.router(h), -1)  # [B, T, E]
        top = probs.topk(self.k, -1)
        self.last_confidence = float(top.values.detach().sum(-1).mean())
        idx = top.indices
        if routing is not None:  # R3: replay the engine's expert choice for the positions given
            idx = idx.clone()
            for t, r in enumerate(routing):
                idx[0, t] = torch.tensor(r[: self.k])
        self.last_routes = idx[0].tolist()
        gate = torch.softmax(torch.log(probs.gather(-1, idx) + 1e-12), -1)  # renormalized over the chosen experts
        outs = torch.stack([e(h) for e in self.experts], -2)  # [B, T, E, D]
        mixed = (outs.gather(-2, idx.unsqueeze(-1).expand(*idx.shape, h.shape[-1])) * gate.unsqueeze(-1)).sum(-2)
        return self.head(h + torch.tanh(mixed))

    def token_logprobs(self, prompt: list[int], completion: list[int], temperature: float = 1.0, candidates: list[list[int]] | None = None,
                       routing: list[list[int]] | None = None) -> torch.Tensor:
        if not completion:
            return torch.zeros(0)
        ids = torch.tensor([prompt + completion[:-1]])
        full = None
        if routing is not None:  # routes for the positions that predicted each completion token; prompt positions keep their own
            with torch.no_grad():
                self(ids)
            full = [list(r) for r in self.last_routes]
            for t, r in enumerate(routing):
                full[len(prompt) - 1 + t] = list(r)
        logits = self(ids, routing=full)[0, len(prompt) - 1:] / temperature
        lp = torch.log_softmax(logits, -1)
        if candidates is not None:
            from crucible.train.sampling import renormalized

            return renormalized(lp[: len(completion)], completion, candidates)
        return lp[torch.arange(len(completion)), torch.tensor(completion)]


def bf16_copy(model: torch.nn.Module) -> torch.nn.Module:
    m = copy.deepcopy(model)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(p.to(torch.bfloat16).to(p.dtype))
    return m


def serve_copy(model: torch.nn.Module, numerics: str) -> torch.nn.Module:
    """The engine's copy: ``fp32`` exact; ``bf16`` weights rounded, fp32 compute; ``bf16_full`` weights *and* activations
    in bf16 (what an inference engine computes: enough to flip MoE router choices, checklist #36)."""
    if numerics == "bf16":
        return bf16_copy(model)
    if numerics == "bf16_full":
        return copy.deepcopy(model).to(torch.bfloat16)
    return copy.deepcopy(model)


class TorchEngine:
    """Samples from a ``TinyPolicy`` copy. ``numerics``: ``fp32`` | ``bf16`` (served weights rounded) | ``bf16_full``
    (weights and activations in bf16): a real train–inference gap."""

    def __init__(self, model: TinyPolicy, *, capacity: int = 16, decode_delay_s: float = 0.0, numerics: str = "fp32",
                 version_switch: str = "call"):
        self.capacity, self.decode_delay_s, self.numerics, self.version_switch = capacity, decode_delay_s, numerics, version_switch
        self.version = 0
        self._serve = serve_copy(model, numerics)
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
        cands: list[list[int]] = list(req.resume_candidates)
        routes: list[list[int]] = list(req.resume_routing)
        spans = list(req.resume_spans) or ([(0, version)] if not out else [])
        if out and (not spans or spans[-1][1] != version):
            spans.append((len(out), version))
        g = torch.Generator().manual_seed(req.seed if req.seed is not None else zlib.crc32(f"{req.call_id}|{len(out)}".encode()))
        moe = hasattr(serve, "last_routes")
        pending_route: list[list[int]] = []

        def forward(seq: list[int]) -> torch.Tensor:
            # grad mode is thread-local: a no_grad block must never span an ``await`` (it would leak into the trainer's
            # coroutine on the same event-loop thread), so each forward gets its own scope
            with torch.no_grad():
                out = serve(torch.tensor([seq]))[0, -1].float()
            if moe:  # the experts at the last position: the ones that produce the next token (checklist #36)
                pending_route[:] = [list(serve.last_routes[-1])]
            return out

        try:
            ids = list(req.prompt_ids) + out
            logits_all = forward(ids)
            for _ in range(req.max_tokens - len(out)):
                if self._abort_epoch != epoch:
                    return GenResult(out, lps, version, "abort", spans if self.version_switch == "token" else None, len(req.prompt_ids), cands or None, routes or None)
                p = torch.softmax(logits_all / req.temperature, -1)
                if req.top_p < 1.0:  # sample from the renormalized nucleus; report its logprob and the candidate set
                    from crucible.train.sampling import nucleus

                    c = nucleus(p, req.top_p)
                    q = torch.zeros_like(p)
                    q[c] = p[c] / p[c].sum()
                    p = q
                    cands.append(c)
                tok = int(torch.multinomial(p, 1, generator=g))
                out.append(tok)
                lps.append(float(torch.log(p[tok])))
                if moe:
                    routes.append(pending_route[0])
                if tok in req.stop_ids:
                    return GenResult(out, lps, version, "stop", spans if len(spans) > 1 else None, len(req.prompt_ids), cands or None, routes or None)
                await asyncio.sleep(self.decode_delay_s)
                ids = ids + [tok]
                logits_all = forward(ids)
            return GenResult(out, lps, version, "length", spans if len(spans) > 1 else None, len(req.prompt_ids), cands or None, routes or None)
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

    def capabilities(self) -> dict[str, Any]:
        return {"token_ids": True, "logprobs": "processed", "abort": "all", "pause": ["keep"], "sleep": False, "top_p_candidates": True,
                "routing": hasattr(self._serve, "last_routes"), "weight_load": "state_dict"}

    async def load_weights(self, state: dict[str, torch.Tensor], version: int) -> None:
        new = copy.deepcopy(self._serve).float()  # a new object: running calls keep the one they pinned
        new.load_state_dict(state)
        self._serve = serve_copy(new, self.numerics)
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

    def capabilities(self) -> dict[str, Any]:
        return {"token_ids": True, "logprobs": "fixed", "abort": "all", "pause": ["keep"], "sleep": False, "top_p_candidates": False, "routing": False,
                "weight_load": "version_only"}

    async def pause(self) -> None:
        self._unpaused.clear()

    async def resume(self) -> None:
        self._unpaused.set()

    async def load_weights(self, state: Any, version: int) -> None:
        self.version = version

