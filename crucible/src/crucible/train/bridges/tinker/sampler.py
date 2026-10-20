"""Sampling any Tinker model token-in/token-out: the student's policy, a sample-only tester, a factory role (09 §3).

**``TinkerSampler``** is the engine side: ``gen(prompt_ids, max_tokens, temperature, seed=, stop_ids=)`` is the
``train.world_env.EngineLLM`` generate function, so the chat side is the one Crucible already trusts (exact TITO: the
model's own tokens are never re-tokenized; conversations keyed by the episode's ``sample_key``; a template that
re-renders history opens a new segment). Our renderer is ``codec.TinkerChatCodec`` (the model's HF chat template, the
family's tool parser, ``enable_thinking`` explicit and recorded). Each call:

1. refuses a prompt that, with ``max_tokens``, overflows the model's context (``ContextOverflow``: the policy's episode
   ends truncated, scored, never masked);
2. holds its worst case on the run's ``SpendMeter`` (``TinkerCapReached`` past the cap), samples, books the measured
   prefill (cached / uncached as the service billed it) and sampled tokens;
3. returns the sampled ids and their logprobs with ``version`` = the sampler's weight version (0 = the base model,
   n = the weights saved after optimizer step n): ``Turn.policy_version``. The version -> ``tinker://`` path table is
   ``versions`` (the trainer writes it to the run directory), and every call is logged (``log``: version, path,
   latency, tokens, USD, episode key), so each turn's sampler path is recoverable;
4. Tinker errors split into infra (retried with backoff, then ``LLMError``: masked) and the request's fault
   (``backend.classify``).

``refresh(path, version)`` swaps in newer sampler weights between optimizer steps; calls already in flight finish on the
weights they started with (call-level switching: one version per call, exactly what ``Turn.policy_version`` records).

**``SamplerLLM``** (``llm()``) is the ``LLMClient``-shaped object a ``LLMToolPolicy`` drives (``complete``, ``spend``,
``lock_settings``): every booked dollar is attributed to the episode (``sample_key``) that spent it (``per_episode``).
``policy_ref(...)`` wraps it as the runtime's ``PolicyRef`` with its declared config (the run identity).

**``TinkerChatClient``** is a factory role on a Tinker model: ``complete`` and ``complete_json`` (JSON parsed from the
reply, then checked by an optional ``schema``: a callable, a pydantic model or a list of required keys; a malformed
reply is retried with the error fed back, ``json_retries`` times, then ``LLMOutputError``), plus ``spec`` /
``family`` (``factory.models.ModelSpec`` fields, so the factory's same-family refusal sees the model's family).
"""

from __future__ import annotations

import contextvars
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from crucible.actors.llm import ContextOverflow, LLMOutputError, LLMResponse
from crucible.train.bridges.tinker.models import TinkerModel, resolve
from crucible.train.bridges.tinker.spend import SpendMeter, sample_estimate
from crucible.train.engine import GenResult

_EPISODE: contextvars.ContextVar[str | None] = contextvars.ContextVar("crucible_tinker_episode", default=None)
_CALL: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("crucible_tinker_call", default=None)  # the caller's slot for this call's log row


@dataclass
class EpisodeSpend:
    usd: float = 0.0
    calls: int = 0
    prefill: int = 0
    prefill_cached: int = 0
    sample: int = 0
    latency_s: float = 0.0


class TinkerSampler:
    def __init__(self, backend: Any, model: TinkerModel | str, *, meter: SpendMeter, codec: Any, path: str | None = None, version: int = 0,
                 log_limit: int = 100_000):
        self.backend, self.meter, self.codec = backend, meter, codec
        self.model = model if isinstance(model, TinkerModel) else resolve(model)
        self.path, self.version = path, int(version)
        self.versions: dict[int, str | None] = {self.version: path}
        self.handle: Any = None
        self.log: list[dict[str, Any]] = []
        self.log_limit = log_limit
        self.per_episode: dict[str, EpisodeSpend] = {}
        self.calls = 0
        self.usd = 0.0  # this sampler's own spend (the meter holds the run's total)

    async def start(self) -> TinkerSampler:
        if self.handle is None:
            self.handle = await self.backend.sampler(self.model.name, self.path)
        return self

    async def refresh(self, path: str, version: int) -> None:
        """Sample from ``path`` (weights saved after optimizer step ``version``) from the next call on."""
        h = await self.backend.sampler(self.model.name, path)
        self.handle, self.path, self.version = h, path, int(version)
        self.versions[self.version] = path

    async def gen(self, prompt: list[int], max_tokens: int, temperature: float, *, seed: int | None = None, stop_ids: Sequence[int] = ()) -> GenResult:
        if self.handle is None:
            await self.start()
        n = len(prompt)
        if n + int(max_tokens) > self.model.context:
            raise ContextOverflow(f"{self.model.name}: {n} prompt tokens + {max_tokens} > the {self.model.context}-token context")
        handle, version, path = self.handle, self.version, self.path  # one version per call
        key = _EPISODE.get()
        t0 = time.monotonic()
        with self.meter.hold(sample_estimate(self.model, n, int(max_tokens)), f"sample {self.model.name} ({n} prompt tokens)"):
            out = await handle.sample(prompt, max_tokens=int(max_tokens), temperature=float(temperature), stop=list(stop_ids) or None, seed=seed)
            dt = time.monotonic() - t0
            cached = max(0, min(n, int(out.cached_prompt_tokens))) if out.cached_prompt_tokens is not None else 0
            usd = self.meter.book("sample", self.model, prefill=n - cached, prefill_cached=cached, sample=len(out.tokens), version=version,
                                  episode=key, latency_s=round(dt, 3), stop=out.stop_reason)
        self.calls += 1
        self.usd += usd
        if key is not None:
            e = self.per_episode.setdefault(key, EpisodeSpend())
            e.usd += usd
            e.calls += 1
            e.prefill += n - cached
            e.prefill_cached += cached
            e.sample += len(out.tokens)
            e.latency_s += dt
        row = {"version": version, "path": path, "episode": key, "prompt": n, "cached": cached, "completion": len(out.tokens),
               "latency_s": round(dt, 3), "usd": round(usd, 7), "stop": out.stop_reason, "cache_reported": out.cached_prompt_tokens is not None}
        if len(self.log) < self.log_limit:
            self.log.append(row)
        slot = _CALL.get()
        if slot is not None:
            slot.update(row)
        return GenResult(completion_ids=list(out.tokens), logprobs=list(out.logprobs), version=version,
                         finish="length" if out.stop_reason == "length" else "stop", prompt_tokens=n)

    def llm(self, name: str | None = None) -> SamplerLLM:
        return SamplerLLM(self, name=name)

    def settings(self) -> dict[str, Any]:
        return {"provider": "tinker", "model": self.model.name, "family": self.model.family, "context": self.model.context,
                "codec": self.codec.settings() if hasattr(self.codec, "settings") else type(self.codec).__name__}

    def policy_ref(self, *, pid: str | None = None, temperature: float = 1.0, max_tokens: int = 1024, system: str | None = None,
                   llm: SamplerLLM | None = None, **policy_kw: Any) -> Any:
        """A ``PolicyRef`` whose agent is ``LLMToolPolicy`` on this sampler (one shared ``SamplerLLM``: conversations are
        keyed per episode). ``policy_kw``: the env's ``policy_kwargs()`` (tool limit, its warning)."""
        from crucible.actors.policy import SYSTEM, LLMToolPolicy
        from crucible.runtime.harness.native import PolicyRef

        shared = llm or self.llm()
        kw = {"temperature": temperature, "max_tokens": max_tokens, **policy_kw}
        cfg = {"sampler": self.settings(), "version": self.version, "path": self.path, "policy": {k: v for k, v in kw.items()},
               "system": system or "default"}
        return PolicyRef(id=pid or f"tinker-{self.model.name.replace('/', '_').replace(':', '_')}-v{self.version}",
                         make=lambda _a: {"agent": LLMToolPolicy("agent", shared, system=system or SYSTEM, **kw)},
                         spend=lambda: (self.meter.spent, 0.0), settings=self.settings(), config=cfg, endpoint=f"tinker:{self.model.name}")


class SamplerLLM:
    """``EngineLLM`` over a ``TinkerSampler``, with per-episode spend attribution (module doc)."""

    def __init__(self, sampler: TinkerSampler, *, name: str | None = None):
        from crucible.train.world_env import EngineLLM

        self.sampler = sampler
        self.model = name or sampler.model.name
        self.engine = EngineLLM(sampler.codec, sampler.gen, model=self.model)

    @property
    def spend(self) -> Any:
        s = self.engine.spend
        s.usd = self.sampler.usd  # this client's own calls (a ModelBook sums clients; the shared meter holds the cap)
        return s

    def lock_settings(self) -> dict[str, Any]:
        return self.sampler.settings()

    async def complete(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, temperature: float = 1.0,
                       max_tokens: int = 1024, sample_key: str | None = None, seed: int | None = None, **kw: Any) -> LLMResponse:
        tok, slot = _EPISODE.set(sample_key), _CALL.set({})
        try:
            r = await self.engine.complete(messages, tools=tools, temperature=temperature, max_tokens=max_tokens, sample_key=sample_key, seed=seed)
            last = _CALL.get() or {}
        finally:
            _EPISODE.reset(tok)
            _CALL.reset(slot)
        if last:
            r.cost_usd = float(last["usd"])
            r.usage = {**(r.usage or {}), "cache_read_tokens": last["cached"], "source": "engine"}
            if r.tokens is not None:
                r.tokens["sampler_path"] = last["path"]  # the call's own weights (also in the version table)
        r.sampling = {"temperature": temperature, "max_tokens": max_tokens, "seed": seed}
        return r


@dataclass
class TinkerChatClient:
    """A factory role served by a Tinker model (module doc). ``complete`` keeps no conversation state between calls
    unless the caller passes a ``sample_key``."""

    sampler: TinkerSampler
    role: str | None = None
    json_retries: int = 2
    default_max_tokens: int = 2048
    _llm: SamplerLLM | None = field(default=None, repr=False)

    @property
    def llm(self) -> SamplerLLM:
        if self._llm is None:
            self._llm = self.sampler.llm()
        return self._llm

    @property
    def model(self) -> str:
        return self.sampler.model.name

    @property
    def family(self) -> str:
        return self.sampler.model.family

    def spec(self) -> dict[str, str]:
        return self.sampler.model.spec()

    def lock_settings(self) -> dict[str, Any]:
        return {**self.sampler.settings(), "role": self.role}

    @property
    def spend(self) -> Any:
        return self.llm.spend

    async def complete(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, temperature: float = 0.0,
                       max_tokens: int | None = None, json_mode: bool = False, seed: int | None = 0, sample_key: str | None = None, **kw: Any) -> LLMResponse:
        msgs = list(messages)
        if json_mode:  # no provider JSON mode on a token endpoint: asked for in the prompt, parsed and checked by complete_json
            msgs = [*msgs[:-1], {**msgs[-1], "content": f"{msgs[-1].get('content') or ''}\n\nReply with one JSON object only."}] if msgs else msgs
        return await self.llm.complete(msgs, tools=tools, temperature=temperature, max_tokens=max_tokens or self.default_max_tokens,
                                       sample_key=sample_key, seed=seed)

    async def complete_json(self, messages: list[dict[str, Any]], *, schema: Any = None, **kw: Any) -> dict[str, Any]:
        kw.setdefault("json_mode", True)
        msgs = list(messages)
        last_err = ""
        for attempt in range(self.json_retries + 1):
            r = await self.complete(msgs, **kw)
            try:
                doc = parse_json_reply(r.text)
                check_schema(doc, schema)
                return doc
            except ValueError as e:
                last_err = str(e)
                msgs = [*msgs, {"role": "assistant", "content": r.text or ""},
                        {"role": "user", "content": f"That reply was not usable: {last_err[:300]}. Reply again with one JSON object only."}]
                kw = {**kw, "seed": None if kw.get("seed") is None else int(kw["seed"]) + attempt + 1}
        raise LLMOutputError(f"{self.model}: no valid JSON after {self.json_retries + 1} attempts: {last_err[:300]}")


def parse_json_reply(text: str) -> dict[str, Any]:
    t = (text or "").strip()
    if not t:
        raise ValueError("empty reply")
    if t.startswith("```"):
        t = t.strip("`").removeprefix("json").strip()
    for cand in (t, t[t.find("{"): t.rfind("}") + 1] if "{" in t and "}" in t else ""):
        if not cand:
            continue
        try:
            d = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict):
            return d
    raise ValueError(f"not a JSON object: {t[:200]}")


def check_schema(doc: dict[str, Any], schema: Any) -> None:
    """``schema``: None, a callable raising or returning False, a pydantic model class, or a sequence of required keys."""
    if schema is None:
        return
    if isinstance(schema, (list, tuple, set, frozenset)):
        miss = [k for k in schema if k not in doc]
        if miss:
            raise ValueError(f"missing keys {miss}")
        return
    if hasattr(schema, "model_validate"):
        try:
            schema.model_validate(doc)
        except Exception as e:
            raise ValueError(f"schema: {str(e)[:300]}") from e
        return
    if callable(schema):
        if schema(doc) is False:
            raise ValueError("the reply failed the schema check")
        return
    raise TypeError(f"unsupported schema {type(schema).__name__}")


async def make_sampler(backend: Any, model: str, *, meter: SpendMeter, enable_thinking: bool = False, path: str | None = None, version: int = 0,
                       codec: Any = None, tokenizer: Any = None, episode_tokens: int | None = None, max_tokens: int = 1024) -> TinkerSampler:
    """A started sampler for ``model`` with our renderer for it (``codec.TinkerChatCodec`` on the model's tokenizer);
    ``episode_tokens`` (the episode's context cap) is checked against the model's context first."""
    from crucible.train.bridges.tinker.codec import TinkerChatCodec, load_tokenizer
    from crucible.train.bridges.tinker.models import check_context

    m = resolve(model)
    if episode_tokens is not None:
        check_context(m, episode_tokens=episode_tokens, max_tokens=max_tokens)
    if codec is None:
        codec = TinkerChatCodec(tokenizer or load_tokenizer(m), m, enable_thinking=enable_thinking)
    return await TinkerSampler(backend, m, meter=meter, codec=codec, path=path, version=version).start()


# ---- the factory's sampler backend (``factory.models.register_backend("tinker", make, final=True)``) ----------------------------
_SHARED: dict[str, Any] = {}


def shared_meter() -> SpendMeter:
    """The process's meter for factory roles on Tinker: one hard cap over every role client (``CRUCIBLE_TINKER_CAP_USD``,
    default 1.0 USD; ``CRUCIBLE_TINKER_SPEND`` names its receipt file, default none)."""
    import os
    from pathlib import Path

    if "meter" not in _SHARED:
        cap = float(os.environ.get("CRUCIBLE_TINKER_CAP_USD") or 1.0)
        path = os.environ.get("CRUCIBLE_TINKER_SPEND")
        _SHARED["meter"] = SpendMeter(cap, path=Path(path) if path else None)
    return _SHARED["meter"]


def make(model: str, *, role: str | None = None, thinking: bool | None = None, meter: SpendMeter | None = None, backend: Any = None,
         codec: Any = None, **_: Any) -> TinkerChatClient:
    """``factory.models`` sampler backend: a ``TinkerChatClient`` for ``model`` (any Tinker id with a price row) serving
    ``role``. ``thinking`` None = the bridge's default (off); a reasoning-only model (GLM-5.3, gpt-oss) always reasons.
    The sampling client is created on the first call; the tokenizer is loaded here (Hugging Face cache or hub)."""
    from crucible.train.bridges.tinker.backend import RealTinker
    from crucible.train.bridges.tinker.codec import TinkerChatCodec, load_tokenizer

    m = resolve(model)
    if codec is None:
        codec = TinkerChatCodec(load_tokenizer(m), m, enable_thinking=bool(thinking))
    if backend is None:
        backend = _SHARED.setdefault("backend", RealTinker())
    return TinkerChatClient(TinkerSampler(backend, m, meter=meter or shared_meter(), codec=codec), role=role)
