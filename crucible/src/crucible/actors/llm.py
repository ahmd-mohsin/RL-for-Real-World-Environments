"""LLM client for actors and judges: any OpenAI-compatible endpoint (OpenRouter, vLLM, SGLang).

- **Budget**: every client has a hard USD cap; exceeding it raises ``BudgetExceeded`` (failure kind
  ``crucible:credits_exhausted`` — an infrastructure failure, masked, never a 0 score).
- **Cache** (record / replay): responses are keyed by sha256(model, messages, tools, params) and stored
  under ``<data>/llm_cache``. ``CRUCIBLE_LLM_CACHE=replay`` makes a missing entry an error (CI);
  ``record`` (default) reads then writes; ``off`` disables. Replay makes simulated actors deterministic.
- **Accounting**: provider-reported cost when present (OpenRouter ``usage.cost``), else config prices, else (a hosted
  model with no price) ``UNPRICED_USD_PER_M`` or ``prices["*"]``, booked as spend and as uncertain: an unpriced model
  used to cost 0 USD, so its hard budget could never trip (M2 audit). Self-hosted engines (localhost, ``tito``, a
  ``policy_version``) cost 0. Usage is ``core.schema.usage.Usage`` whatever the provider's shape, so cache reads and
  writes are priced at their own rates (R-19: every prompt token was charged at the input rate, 2.29x a cached call).
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from crucible.core.config import Config, get_config, resolve_secret
from crucible.core.ids import sha256_text
from crucible.core.schema.rollout import SamplingParams
from crucible.core.schema.usage import FinishReason, Usage, finish_fields, finish_reason

# Spend attribution per asyncio task (M8 audit V3): every booked cost is also added to the innermost open scope, so a
# caller that runs many roles concurrently (the forge orchestrator) charges each node its own calls, not the global delta.
_SPEND_SCOPE: contextvars.ContextVar[list[float] | None] = contextvars.ContextVar("crucible_spend_scope", default=None)


def book_scoped(usd: float) -> None:
    """Add ``usd`` to the spend scope of the running task (no-op outside one). Non-LLM spenders (sandboxed trials) call it."""
    acc = _SPEND_SCOPE.get()
    if acc is not None:
        acc[0] += float(usd)
        acc[1] += 1


class SpendScope:
    """``with SpendScope() as s: ...; s.usd``: the spend booked by code running in this task (and tasks it starts)
    inside the block. ``s.booked`` is False when nothing booked into it (the caller then has no attribution)."""

    def __enter__(self) -> SpendScope:  # noqa: PYI034
        self._acc = [0.0, 0]
        self._tok = _SPEND_SCOPE.set(self._acc)
        return self

    def __exit__(self, *exc: object) -> None:
        _SPEND_SCOPE.reset(self._tok)

    @property
    def usd(self) -> float:
        return self._acc[0]

    @property
    def booked(self) -> bool:
        return self._acc[1] > 0


class BudgetExceeded(RuntimeError):
    failure_kind = "crucible:credits_exhausted"


class LLMError(RuntimeError):
    failure_kind: str | None = "provider_unavailable"
    # M10 lab audit R-5 item 3: the provider's Retry-After (seconds) of the answer that ended the retries; None when it gave none.
    # ``LLMToolPolicy`` carries it to the episode's failure, where ``RetryRule.backoff`` reads it
    retry_after: float | None = None

    def __init__(self, *args: object):
        # provider errors can echo the request: a key, a signed URL (M10 audit V12: a credential reached 14 record and
        # trace files). Scrubbed here, at the source, so no caller can store it
        from crucible.core.secrets import scrub_secrets

        super().__init__(*(scrub_secrets(a) if isinstance(a, str) else a for a in args))


class LLMOutputError(LLMError):
    """The model answered, but not in the requested form (empty, not JSON, truncated): a content failure the caller may
    retry with feedback, not a provider outage."""

    failure_kind = "malformed_output"


class RequestRejected(LLMError):
    """A 4xx the provider will keep returning for this request (bad params, auth, model): never retried."""

    failure_kind = "agent_request_failed"


class UsageBlank(LLMError):
    """The provider answered without usage: spend cannot be accounted, so the hard budget could never trip."""

    failure_kind = "crucible:usage_blank"


class ContextOverflow(LLMError):
    """The conversation no longer fits the model's context: the policy's episode ends *truncated* (scored, not
    masked) -- an infra mask would hide every long-horizon failure from the data."""

    failure_kind = None


# Ported from verifiers v1 (MIT) harnesses/utils/compaction.py: provider wordings of a context overflow.
CONTEXT_OVERFLOW_MARKERS = (
    "context_length_exceeded", "exceeds the context window", "reduce the length of the messages", "maximum context length",
    "prompt is too long", "request_too_large", "request entity too large", "exceeds the maximum number of tokens", "maximum prompt length is",
)


def is_context_overflow(e: Exception) -> bool:
    status = getattr(e, "status_code", None)
    details = f"{e} {getattr(e, 'body', '') or ''}".casefold()
    return status in (400, 413) and any(m in details for m in CONTEXT_OVERFLOW_MARKERS)


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    """Tool-call arguments as a dict; anything else is kept raw for the World to answer with a usable error."""
    try:
        val = json.loads(raw or "{}")
        if isinstance(val, str):  # double-encoded JSON objects are common; accept them
            val = json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return {"__unparseable__": raw}
    return val if isinstance(val, dict) else {"__unparseable__": raw, "__type__": type(val).__name__}


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # [{id, name, arguments(dict)}]
    usage: dict[str, Any] = field(default_factory=dict)  # a dumped core.schema.usage.Usage (R-19); {} = not reported
    cost_usd: float = 0.0
    model: str = ""
    cached: bool = False
    raw_finish: str | None = None
    raw_message: dict[str, Any] = field(default_factory=dict)  # provider fields (reasoning, signatures) to replay verbatim
    latency_s: float = 0.0
    response_id: str | None = None
    tokens: dict[str, Any] | None = None  # token-in-token-out record when the endpoint returns it (prompt/completion ids, logprobs, policy_version)
    sampling: dict[str, Any] = field(default_factory=dict)  # the sampling parameters the call sent (core.schema.rollout.SamplingParams, R-3)

    def __post_init__(self) -> None:
        # one shape whoever answered (R-19): a provider's usage dict, a Usage, or a cache entry written before R-19
        if isinstance(self.usage, Usage):
            self.usage = self.usage.model_dump()
        elif self.usage:
            self.usage = Usage.model_validate(self.usage).model_dump()

    @property
    def finish(self) -> FinishReason | None:
        return finish_reason(self.raw_finish)

    def finish_fields(self) -> dict[str, Any]:
        """A call record's finish: the normalized reason, and the provider's own word when it differs (R-19)."""
        return finish_fields(self.raw_finish)


@dataclass
class Spend:
    usd: float = 0.0
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    uncertain_usd: float = 0.0  # calls whose cost could not be read (blank usage in tolerant mode): an upper-bound estimate
    blank_usage_calls: int = 0
    unpriced_calls: int = 0  # calls booked at the fallback price (no provider cost, no configured price)
    budget_waits: int = 0  # calls that waited for money held by calls in flight before they could start (M2 R-2)


UNPRICED_USD_PER_M = (3.0, 15.0)  # [input, output] USD per 1M tokens for a hosted model with no price anywhere
# fields of a /models entry that give the context window: vLLM and SGLang, OpenRouter, others (verifiers' discover_threshold)
WINDOW_FIELDS = ("max_model_len", "context_length", "context_window", "max_context_length")
WINDOW_DISCOVERY_S = 15.0  # one /models read per client; past this the window stays unknown
RATE_LIMIT_ATTEMPTS = 6  # a 429 gets at least this many attempts, on a slower curve: 3 tries over 4.5 s gave up mid-burst


def _retry_after(e: Exception) -> float | None:
    """M10 lab audit R-5 item 3: the provider's Retry-After of the error's response in seconds (the HTTP-date form is not
    read), or None when it is absent, unreadable, not finite or negative."""
    import math

    headers = getattr(getattr(e, "response", None), "headers", None) or {}
    try:
        v = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v >= 0 else None


def _retry_delay(e: Exception, attempt: int, cap: float = 60.0) -> float:
    """Provider's Retry-After (seconds) when given, else capped exponential backoff with jitter (Karotte's agent
    loop; verifiers' ``backoff`` curve). Rate limits back off from 5 s instead of 1.5 s."""
    import random

    ra = _retry_after(e)
    if ra is not None:
        return min(ra, cap)
    base = 5.0 if getattr(e, "status_code", None) == 429 else 1.5
    return min(min(base * 2**attempt, cap) * (0.5 + random.random()), cap)


def _approx_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(len(json.dumps(m, default=str)) for m in messages) // 4


def _usage_dict(u: Any) -> Any:
    """A provider's usage object (the SDK's model, or any object with attributes) as plain dicts, extras included."""
    if isinstance(u, dict) or u is None:
        return u
    if hasattr(u, "model_dump"):
        return u.model_dump()
    return {k: _usage_dict(v) if hasattr(v, "__dict__") else v for k, v in vars(u).items()}


def token_record(resp: Any, choice: Any, policy_version: int | str | None) -> dict[str, Any] | None:
    """The token-in-token-out record of one call, as the engine sampled it (vLLM/SGLang ``return_token_ids`` and
    ``logprobs``); ``None`` when the endpoint returns neither (hosted APIs). Never re-tokenized here."""
    extra = getattr(choice, "model_extra", None) or {}
    rextra = getattr(resp, "model_extra", None) or {}
    lp = getattr(choice, "logprobs", None)
    content = getattr(lp, "content", None) or []
    logprobs = [c.logprob for c in content] or None
    completion_ids = extra.get("token_ids")
    prompt_ids = rextra.get("prompt_token_ids") or extra.get("prompt_token_ids")
    if logprobs is None and completion_ids is None:
        return None
    return {"prompt_token_ids": prompt_ids, "completion_token_ids": completion_ids, "logprobs": logprobs, "policy_version": policy_version}


class _NoChoices(RuntimeError):
    pass


class LLMClient:
    def __init__(self, model: str = "sim-default", *, config: Config | None = None, budget_usd: float | None = None,
                 cache_dir: Path | None = None, cache_mode: str | None = None, timeout_s: float | None = None, max_retries: int = 3,
                 require_usage: bool = True):
        self.config = config or get_config()
        self.provider, self.model = self.config.model(model)
        self.budget_usd = self.config.llm_budget_usd if budget_usd is None else budget_usd
        from crucible.core import env as knobs

        self.cache_mode = cache_mode or knobs.get("CRUCIBLE_LLM_CACHE")
        if self.cache_mode not in knobs.KNOBS["CRUCIBLE_LLM_CACHE"].choices:  # R-15: "reply" used to turn the cache off silently
            raise ValueError(f"cache_mode {self.cache_mode!r} is not one of {knobs.KNOBS['CRUCIBLE_LLM_CACHE'].choices}")
        self.cache_dir = cache_dir or (self.config.data_path() / "llm_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # read timeout per call (CRUCIBLE_LLM_TIMEOUT_S, default 600 s): reasoning models routinely think past the old fixed
        # 120 s, which failed the call after paying for it (M2 audit); connecting still fails fast
        self.timeout_s = float(timeout_s if timeout_s is not None else knobs.get("CRUCIBLE_LLM_TIMEOUT_S"))
        self.max_retries = max_retries
        self.require_usage = require_usage  # False: tolerate blank usage, booking an upper-bound estimate as uncertain spend
        # the served checkpoint's version (provider config), replaced by the trainer bridge on every weight update; stamped on each call's token record
        self.policy_version: int | str | None = self.provider.policy_version
        self.spend = Spend()
        self._reserved = 0.0  # USD held by calls in flight (the hard budget counts them: see ``_reserve``)
        self._holds = 0  # calls in flight holding money: a call waits on them only while there are some
        self._max_seen = 0.0  # the largest single-call cost so far: the reservation when no price is configured
        self._overrun = 1.0  # the largest cost / priced worst case seen: holds scale by it (a stale price, a long prompt)
        self._settled = asyncio.Event()  # set (and replaced) whenever a call releases its hold: calls waiting on money wake
        self._client = None
        self.window_tokens: int | None = None  # the model's context window once resolved (``context_window``)
        self.window_source: str | None = None  # where it came from: "config", "models" or "unknown"; None = not resolved yet

    def lock_settings(self) -> dict[str, Any]:
        """What this client is for a run lock (runtime.lock, R-1): the resolved model, the endpoint and every provider
        setting that shapes a reply, and the cache mode (a replayed reply is not a live one). Keys and headers never."""
        p = self.provider
        return {"model": self.model, "base_url": p.base_url, "request_defaults": p.request_defaults, "tito": p.tito,
                "template_kwargs": p.template_kwargs, "policy_version": self.policy_version, "cache_mode": self.cache_mode}

    def _request(self, messages, tools, *, temperature, max_tokens, json_mode, seed, extra) -> dict[str, Any]:
        """The chat request's keyword arguments: the call's parameters, then the provider's ``request_defaults`` for keys the
        call left unset, then the call's extras (``extra_body`` merged, call-level keys winning)."""
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if seed is not None:
            kwargs["seed"] = seed
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        defaults = dict(self.provider.request_defaults or {})  # provider-level defaults (token capture on vLLM / SGLang)
        body = {**(defaults.pop("extra_body", None) or {}), **(extra.get("extra_body") or {})}  # call-level keys win
        for k, v in defaults.items():
            kwargs.setdefault(k, v)
        kwargs.update({k: v for k, v in extra.items() if k != "extra_body"})
        if body:
            kwargs["extra_body"] = body
        return kwargs

    def _sent(self, *, temperature, max_tokens, json_mode, seed, extra) -> dict[str, Any]:
        """The sampling parameters a call with these arguments sends (R-3): after the request_defaults merge for a chat
        endpoint; temperature, max_tokens and seed for a token-in-token-out engine, which takes nothing else."""
        if self.provider.tito:
            return SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed).model_dump()
        return SamplingParams.from_request(self._request([], None, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
                                                         seed=seed, extra=extra)).model_dump()

    def _openai(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(base_url=self.provider.base_url, api_key=resolve_secret(self.provider.api_key),
                                       timeout=httpx.Timeout(self.timeout_s, connect=min(30.0, self.timeout_s)),
                                       default_headers=self.provider.resolved_headers() or None, max_retries=0)
        return self._client

    def _key(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, params: dict[str, Any]) -> str:
        return sha256_text(json.dumps({"model": self.model, "messages": messages, "tools": tools, "params": params}, sort_keys=True, ensure_ascii=False))

    def _self_hosted(self) -> bool:
        from urllib.parse import urlparse

        host = (urlparse(self.provider.base_url).hostname or "").casefold()
        return self.provider.tito or self.provider.policy_version is not None or host in ("localhost", "127.0.0.1", "::1", "0.0.0.0") or host.endswith(".local")

    async def context_window(self) -> int | None:
        """The model's context window in tokens (prompt plus reply), resolved once (M2 R-14; PATTERN: verifiers
        ``discover_threshold`` reads the /models card, MIT): ``crucible.toml``'s ``context_windows`` for this model id, else
        a self-hosted engine's /models entry (vLLM and SGLang give ``max_model_len``), else None (unknown: callers keep
        their defaults). Compaction budgets defaulted to 128,000 tokens whatever the served window, so a 32,768-token
        engine overflowed before compacting once. Hosted APIs are read from config only: a cached run stays offline."""
        if self.window_source is None:
            n, src = self.config.context_windows.get(self.model), "config"
            if n is None and self._self_hosted() and self.cache_mode != "replay":
                n, src = await self._discover_window(), "models"
            self.window_tokens, self.window_source = (n, src) if n else (None, "unknown")
        return self.window_tokens

    async def _discover_window(self) -> int | None:
        try:
            page = await asyncio.wait_for(self._openai().models.list(), timeout=WINDOW_DISCOVERY_S)
            for m in getattr(page, "data", None) or []:
                card = m.model_dump() if hasattr(m, "model_dump") else dict(vars(m))
                if card.get("id") == self.model:
                    return next((int(v) for f in WINDOW_FIELDS if isinstance(v := card.get(f), int) and v > 0), None)
        except Exception:  # noqa: BLE001 -- an engine without /models, or one down: the window stays unknown, never an error here
            return None
        return None

    def _prices(self) -> tuple[list[float] | tuple[float, float] | None, bool]:
        """(USD per 1M [input, output], is_fallback). No price for a self-hosted engine; a hosted model with none
        configured gets ``prices["*"]`` or ``UNPRICED_USD_PER_M``."""
        p = self.config.prices.get(self.model)
        if p:
            return p, False
        if self._self_hosted():
            return None, False
        return self.config.prices.get("*") or UNPRICED_USD_PER_M, True

    def _price(self, usage: Usage) -> float:
        """USD from the configured prices: cache reads and writes at their own rates when given, else the input rate."""
        p, fallback = self._prices()
        if not p:
            return 0.0
        if fallback:
            self.spend.unpriced_calls += 1
        read, write = (p[2] if len(p) > 2 else p[0]), (p[3] if len(p) > 3 else p[0])
        cost = (usage.input_tokens * p[0] + usage.cache_read_tokens * read + usage.cache_write_tokens * write + usage.output_tokens * p[1]) / 1e6
        if fallback:
            self.spend.uncertain_usd += cost
        return cost

    async def complete(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, temperature: float = 0.0,
                       max_tokens: int = 800, json_mode: bool = False, seed: int | None = 0, sample_key: str | None = None, **extra: Any) -> LLMResponse:
        """``sample_key`` (e.g. episode id / sample index) separates rollouts in the record/replay cache: without it,
        N samples of one task would all replay the first completion (zero variance). Simulators and judges omit it."""
        params: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens, "json": json_mode, "seed": seed, **extra}
        key = self._key(messages, tools, {**params, **({"sample": sample_key} if sample_key else {})})
        path = self.cache_dir / key[:2] / f"{key}.json"
        d = self._cached(path) if self.cache_mode in ("record", "replay") else None
        if d is not None:
            self.spend.cached_calls += 1
            return LLMResponse(text=d["text"], tool_calls=d.get("tool_calls", []), usage=d.get("usage", {}), cost_usd=0.0, model=d.get("model", self.model), cached=True,
                               raw_finish=d.get("finish"), raw_message=d.get("raw_message", {}),
                               sampling=d.get("sampling") or self._sent(temperature=temperature, max_tokens=max_tokens, json_mode=json_mode, seed=seed, extra=extra))
        if self.cache_mode == "replay":
            raise LLMError(f"cache miss in replay mode for {self.model} (key {key[:12]})")
        hold, priced = await self._reserve(messages, max_tokens)
        try:
            r = await self._complete_paid(messages, tools, params, path, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
                                          seed=seed, extra=extra, sample_key=sample_key)
            if priced > 0 and r.cost_usd > priced:
                self._overrun = max(self._overrun, r.cost_usd / priced)
            return r
        finally:
            self._reserved -= hold
            self._holds -= 1
            settled, self._settled = self._settled, asyncio.Event()
            settled.set()

    @staticmethod
    def _cached(path: Path) -> dict[str, Any] | None:
        """A cache entry, or ``None`` (a miss) when absent or unreadable: one corrupt file (a crash mid-write, before
        writes were atomic) raised JSONDecodeError on every later run (M2 audit)."""
        try:
            d = json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            return None
        return d if isinstance(d, dict) and isinstance(d.get("text"), str) else None

    async def _reserve(self, messages: list[dict[str, Any]], max_tokens: int) -> tuple[float, float]:
        """Check the budget and hold this call's worst-case cost until it settles; returns (hold, priced worst case). A
        check of money already spent let 40 concurrent calls through a 0.05 USD budget and spend 0.40 (M2 audit): calls in
        flight count. The hold is the configured price of the prompt plus ``max_tokens``, times the largest overrun seen
        (``_overrun``), else, unpriced, the largest single call seen so far.

        M2 R-2 (PATTERN: repo2rlenv's budget waits only on pressure from holds that can still settle, Apache-2.0): only a
        call whose own estimate does not fit beside the money already spent is refused (``credits_exhausted`` stops the
        run). One that fits but for what calls in flight hold waits for them to settle: the dearest call's cost floored
        every hold, so after one 0.20 USD call 5 of 16 cheap calls stopped the run with 91% of the budget unused."""
        p, _ = self._prices()
        priced = (_approx_tokens(messages) / 1e6 * p[0] + max_tokens / 1e6 * p[1]) if p else 0.0
        est = priced * self._overrun if p else self._max_seen
        waited = False
        while True:
            if self.spend.usd >= self.budget_usd or self.spend.usd + est > self.budget_usd + 1e-12:
                raise BudgetExceeded(f"LLM budget {self.budget_usd:.4f} USD exhausted ({self.spend.usd:.4f} spent, {est:.4f} needed by this call)")
            if not self._holds or self.spend.usd + self._reserved + est <= self.budget_usd + 1e-12:
                self._reserved += est
                self._holds += 1
                return est, priced
            if not waited:  # pressure: calls in flight hold the rest (so one of them will settle and wake this one)
                waited = True
                self.spend.budget_waits += 1
            await self._settled.wait()

    async def _complete_paid(self, messages, tools, params, path, *, temperature, max_tokens, json_mode, seed, extra, sample_key=None) -> LLMResponse:
        if self.provider.tito:
            return await self._tito_complete(messages, tools, temperature=temperature, max_tokens=max_tokens, seed=seed, sample_key=sample_key)
        kwargs = self._request(messages, tools, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode, seed=seed, extra=extra)
        last: Exception | None = None
        t0 = time.monotonic()
        try:
            client = self._openai()
        except Exception as e:  # a missing key or bad base URL: retrying cannot help (it was retried 3x with backoff)
            raise LLMError(f"{self.model}: cannot build the client: {e}") from e
        attempt, limit = 0, self.max_retries
        while True:
            try:
                resp = await client.chat.completions.create(**kwargs)
                if not getattr(resp, "choices", None):  # OpenRouter can answer 200 with an error body and no choices
                    raise _NoChoices(f"no choices in response: {getattr(resp, 'error', None) or getattr(resp, 'model_extra', {})}")
                break
            except Exception as e:
                last = e
                status = getattr(e, "status_code", None)
                text = str(e).casefold()
                if status == 402 or (status == 429 and "insufficient_quota" in text):
                    raise BudgetExceeded(f"provider refused for credits ({status}): {e}") from e
                # an overflow in a 200 error body was retried and masked as an outage (M2 audit): it is the policy's overflow
                if is_context_overflow(e) or (isinstance(e, _NoChoices) and any(m in text for m in CONTEXT_OVERFLOW_MARKERS)):
                    raise ContextOverflow(f"{self.model}: {e}") from e
                if status is not None and 400 <= status < 500 and status not in (408, 409, 425, 429):  # retrying cannot help
                    raise RequestRejected(f"{self.model} rejected the request ({status}): {e}") from e
                if status == 429:
                    limit = max(limit, RATE_LIMIT_ATTEMPTS)
                attempt += 1
                if attempt >= limit:
                    err = LLMError(f"{self.model} failed after {attempt} attempts: {last}")
                    err.retry_after = _retry_after(e)  # M10 lab audit R-5 item 3: the last answer's Retry-After rides to the Failure
                    raise err from e
                await asyncio.sleep(_retry_delay(e, attempt - 1))
        choice = resp.choices[0]
        msg = choice.message
        calls = [{"id": tc.id, "name": tc.function.name, "arguments": _parse_arguments(tc.function.arguments)} for tc in msg.tool_calls or []]
        raw_message = {k: v for k, v in (msg.model_dump(exclude_none=True) if hasattr(msg, "model_dump") else {}).items()
                       if k not in ("role", "content", "tool_calls", "refusal", "annotations", "audio", "function_call")}
        u = resp.usage
        if u is None and self.require_usage:
            raise UsageBlank(f"{self.model} returned no usage (finish={choice.finish_reason}); spend cannot be accounted")
        if u is None:  # tolerant mode (B.5 row 2): the call happened and cost something; book the worst case as uncertain
            self.spend.blank_usage_calls += 1
            p, _ = self._prices()
            self.spend.uncertain_usd += (max_tokens / 1e6 * p[1] + _approx_tokens(messages) / 1e6 * p[0]) if p else 0.0
            usage = Usage(source="none")
        else:  # cached prompt tokens apart from the uncached ones, whatever the provider's shape (R-19)
            usage = Usage.model_validate(_usage_dict(u))
        cost = getattr(u, "cost", None) if u else None
        if cost is None and u is not None:
            extra_u = getattr(u, "model_extra", None) or {}
            cost = extra_u.get("cost")
        cost = float(cost) if cost is not None else self._price(usage)
        self.spend.usd += cost
        book_scoped(cost)
        self._max_seen = max(self._max_seen, cost)
        self.spend.calls += 1
        self.spend.prompt_tokens += usage.prompt_tokens
        self.spend.completion_tokens += usage.completion_tokens
        if (finish_reason(choice.finish_reason) == "length" and not calls and not (msg.content or "").strip() and usage.source != "none"
                and usage.output_tokens == 0):
            # the prompt filled the window: nothing was generated (engines that answer 200 instead of a 400; M2 audit)
            raise ContextOverflow(f"{self.model}: finish=length with 0 completion tokens ({usage.prompt_tokens} prompt tokens)")
        finish = choice.finish_reason
        if getattr(msg, "refusal", None) and not (msg.content or "").strip():
            finish = "refusal"  # the provider's refusal field: a labelled end, not an empty reply
        out = LLMResponse(text=msg.content or "", tool_calls=calls, usage=usage, cost_usd=cost, model=resp.model or self.model, raw_finish=finish,
                          raw_message=raw_message, latency_s=time.monotonic() - t0, response_id=getattr(resp, "id", None),
                          tokens=token_record(resp, choice, self.policy_version), sampling=SamplingParams.from_request(kwargs).model_dump())
        if self.cache_mode == "record":  # atomic: a concurrent reader of the same key never sees half a file
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.{id(out)}.tmp")
            tmp.write_text(json.dumps({"text": out.text, "tool_calls": out.tool_calls, "usage": out.usage, "model": out.model, "finish": out.raw_finish,
                                       "raw_message": raw_message, "sampling": out.sampling}, ensure_ascii=False, default=str))
            os.replace(tmp, path)
        return out

    async def _tito_complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, *, temperature: float, max_tokens: int,
                             seed: int | None, sample_key: str | None = None) -> LLMResponse:
        """``LLMProvider.tito``: ``train.world_env.EngineLLM`` over the engine's token-id completions endpoint. Prompts
        extend the served conversation token for token; the record is exactly what the engine sampled. One instance
        serves every episode of this client: ``sample_key`` (the episode) keys its conversations, and the seed goes with
        each call (M12 audit V1, V12)."""
        if getattr(self, "_tito", None) is None:
            # the engine stack lives in ``crucible.train`` (above actors): resolved through a pointer, never imported here
            from crucible.core.registry import load_impl

            self._tito = load_impl("crucible.train.world_env:tito_llm")(self.provider, self.model, timeout_s=self.timeout_s)
        t0 = time.monotonic()
        try:
            try:
                r = await self._tito.complete(messages, tools=tools, temperature=temperature, max_tokens=max_tokens, seed=seed, sample_key=sample_key)
            except LLMError as e:  # the engine wrapper keeps the transport error as its cause
                if isinstance(e.__cause__, httpx.HTTPError):
                    raise e.__cause__ from None
                raise
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400 and "maximum context length" in e.response.text.lower():
                raise ContextOverflow(f"{self.model}: {e.response.text[:300]}") from e
            raise LLMError(f"{self.model} (tito) failed: {e}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"{self.model} (tito) failed: {e}") from e
        r.tokens["policy_version"] = self.policy_version  # the served checkpoint's stamp (provider config / trainer bridge)
        r.sampling = r.sampling or SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed).model_dump()  # what the engine was sent
        self.spend.calls += 1
        self.spend.prompt_tokens += r.usage.get("prompt_tokens", 0)
        self.spend.completion_tokens += r.usage.get("completion_tokens", 0)
        r.latency_s = time.monotonic() - t0
        return r

    async def complete_json(self, messages: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        kw.setdefault("json_mode", True)
        r = await self.complete(messages, **kw)
        text = r.text.strip()
        if not text:
            raise LLMOutputError(f"empty completion (finish={r.raw_finish})")
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
            cut = " (cut off at max_tokens: shorten the output)" if r.finish == "length" else ""
            raise LLMOutputError(f"output is not valid JSON{cut}: {text[:300]}") from None
