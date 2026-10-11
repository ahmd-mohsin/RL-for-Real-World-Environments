"""LLM client for actors and judges: any OpenAI-compatible endpoint (OpenRouter, vLLM, SGLang).

- **Budget**: every client has a hard USD cap; exceeding it raises ``BudgetExceeded`` (failure kind
  ``crucible:credits_exhausted`` — an infrastructure failure, masked, never a 0 score).
- **Cache** (record / replay): responses are keyed by sha256(model, messages, tools, params) and stored
  under ``<data>/llm_cache``. ``CRUCIBLE_LLM_CACHE=replay`` makes a missing entry an error (CI);
  ``record`` (default) reads then writes; ``off`` disables. Replay makes simulated actors deterministic.
- **Accounting**: provider-reported cost when present (OpenRouter ``usage.cost``), else config prices.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from crucible.core.config import Config, get_config, resolve_secret
from crucible.core.ids import sha256_text


class BudgetExceeded(RuntimeError):
    failure_kind = "crucible:credits_exhausted"


class LLMError(RuntimeError):
    failure_kind: str | None = "provider_unavailable"


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
    usage: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    model: str = ""
    cached: bool = False
    raw_finish: str | None = None
    raw_message: dict[str, Any] = field(default_factory=dict)  # provider fields (reasoning, signatures) to replay verbatim
    latency_s: float = 0.0
    response_id: str | None = None
    tokens: dict[str, Any] | None = None  # token-in-token-out record when the endpoint returns it (prompt/completion ids, logprobs, policy_version)


@dataclass
class Spend:
    usd: float = 0.0
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    uncertain_usd: float = 0.0  # calls whose cost could not be read (blank usage in tolerant mode): an upper-bound estimate
    blank_usage_calls: int = 0


def _retry_delay(e: Exception, attempt: int, cap: float = 60.0) -> float:
    """Provider's Retry-After (seconds) when given, else capped exponential backoff with jitter (Karotte's agent
    loop; verifiers' ``backoff`` curve)."""
    import random

    headers = getattr(getattr(e, "response", None), "headers", None) or {}
    try:
        return min(float(headers.get("retry-after")), cap)
    except (TypeError, ValueError):
        return min(min(1.5 * 2**attempt, cap) * (0.5 + random.random()), cap)


def _approx_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(len(json.dumps(m, default=str)) for m in messages) // 4


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
                 cache_dir: Path | None = None, cache_mode: str | None = None, timeout_s: float = 120, max_retries: int = 3,
                 require_usage: bool = True):
        self.config = config or get_config()
        self.provider, self.model = self.config.model(model)
        self.budget_usd = self.config.llm_budget_usd if budget_usd is None else budget_usd
        self.cache_mode = cache_mode or os.environ.get("CRUCIBLE_LLM_CACHE", "record")
        self.cache_dir = cache_dir or (self.config.data_path() / "llm_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.require_usage = require_usage  # False: tolerate blank usage, booking an upper-bound estimate as uncertain spend
        # the served checkpoint's version (provider config), replaced by the trainer bridge on every weight update; stamped on each call's token record
        self.policy_version: int | str | None = self.provider.policy_version
        self.spend = Spend()
        self._client = None

    def _openai(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(base_url=self.provider.base_url, api_key=resolve_secret(self.provider.api_key), timeout=self.timeout_s,
                                       default_headers=self.provider.headers or None, max_retries=0)
        return self._client

    def _key(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, params: dict[str, Any]) -> str:
        return sha256_text(json.dumps({"model": self.model, "messages": messages, "tools": tools, "params": params}, sort_keys=True, ensure_ascii=False))

    def _price(self, usage: dict[str, int]) -> float:
        p = self.config.prices.get(self.model)
        if not p:
            return 0.0
        return usage.get("prompt_tokens", 0) / 1e6 * p[0] + usage.get("completion_tokens", 0) / 1e6 * p[1]

    async def complete(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, temperature: float = 0.0,
                       max_tokens: int = 800, json_mode: bool = False, seed: int | None = 0, sample_key: str | None = None, **extra: Any) -> LLMResponse:
        """``sample_key`` (e.g. episode id / sample index) separates rollouts in the record/replay cache: without it,
        N samples of one task would all replay the first completion (zero variance). Simulators and judges omit it."""
        params: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens, "json": json_mode, "seed": seed, **extra}
        key = self._key(messages, tools, {**params, **({"sample": sample_key} if sample_key else {})})
        path = self.cache_dir / key[:2] / f"{key}.json"
        if self.cache_mode in ("record", "replay") and path.exists():
            d = json.loads(path.read_text())
            self.spend.cached_calls += 1
            return LLMResponse(text=d["text"], tool_calls=d.get("tool_calls", []), usage=d.get("usage", {}), cost_usd=0.0, model=d.get("model", self.model), cached=True,
                               raw_finish=d.get("finish"), raw_message=d.get("raw_message", {}))
        if self.cache_mode == "replay":
            raise LLMError(f"cache miss in replay mode for {self.model} (key {key[:12]})")
        if self.spend.usd >= self.budget_usd:
            raise BudgetExceeded(f"LLM budget {self.budget_usd:.4f} USD exhausted ({self.spend.usd:.4f} spent)")
        if self.provider.tito:
            return await self._tito_complete(messages, tools, temperature=temperature, max_tokens=max_tokens, seed=seed)
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
        last: Exception | None = None
        t0 = time.monotonic()
        for attempt in range(self.max_retries):
            try:
                resp = await self._openai().chat.completions.create(**kwargs)
                if not getattr(resp, "choices", None):  # OpenRouter can answer 200 with an error body and no choices
                    raise _NoChoices(f"no choices in response: {getattr(resp, 'error', None) or getattr(resp, 'model_extra', {})}")
                break
            except Exception as e:
                last = e
                status = getattr(e, "status_code", None)
                text = str(e).casefold()
                if status == 402 or (status == 429 and "insufficient_quota" in text):
                    raise BudgetExceeded(f"provider refused for credits ({status}): {e}") from e
                if is_context_overflow(e):
                    raise ContextOverflow(f"{self.model}: {e}") from e
                if status is not None and 400 <= status < 500 and status not in (408, 409, 425, 429):  # retrying cannot help
                    raise RequestRejected(f"{self.model} rejected the request ({status}): {e}") from e
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(_retry_delay(e, attempt))
        else:
            raise LLMError(f"{self.model} failed after {self.max_retries} attempts: {last}")
        choice = resp.choices[0]
        msg = choice.message
        calls = [{"id": tc.id, "name": tc.function.name, "arguments": _parse_arguments(tc.function.arguments)} for tc in msg.tool_calls or []]
        raw_message = {k: v for k, v in (msg.model_dump(exclude_none=True) if hasattr(msg, "model_dump") else {}).items()
                       if k not in ("role", "content", "tool_calls", "refusal", "annotations", "audio", "function_call")}
        u = resp.usage
        if u is None and self.require_usage:
            raise UsageBlank(f"{self.model} returned no usage (finish={choice.finish_reason}); spend cannot be accounted")
        usage = {"prompt_tokens": getattr(u, "prompt_tokens", 0) or 0, "completion_tokens": getattr(u, "completion_tokens", 0) or 0} if u else {}
        if u is None:  # tolerant mode (B.5 row 2): the call happened and cost something; book the worst case as uncertain
            self.spend.blank_usage_calls += 1
            p = self.config.prices.get(self.model)
            self.spend.uncertain_usd += (max_tokens / 1e6 * p[1] + _approx_tokens(messages) / 1e6 * p[0]) if p else 0.0
            usage = {"blank": 1}
        cost = getattr(u, "cost", None) if u else None
        if cost is None and u is not None:
            extra_u = getattr(u, "model_extra", None) or {}
            cost = extra_u.get("cost")
        cost = float(cost) if cost is not None else self._price(usage)
        self.spend.usd += cost
        self.spend.calls += 1
        self.spend.prompt_tokens += usage.get("prompt_tokens", 0)
        self.spend.completion_tokens += usage.get("completion_tokens", 0)
        details = getattr(u, "completion_tokens_details", None) if u else None
        if details is not None and getattr(details, "reasoning_tokens", None):
            usage["reasoning_tokens"] = details.reasoning_tokens
        pdetails = getattr(u, "prompt_tokens_details", None) if u else None
        if pdetails is not None and getattr(pdetails, "cached_tokens", None):
            usage["cached_tokens"] = pdetails.cached_tokens
        out = LLMResponse(text=msg.content or "", tool_calls=calls, usage=usage, cost_usd=cost, model=resp.model or self.model, raw_finish=choice.finish_reason,
                          raw_message=raw_message, latency_s=time.monotonic() - t0, response_id=getattr(resp, "id", None),
                          tokens=token_record(resp, choice, self.policy_version))
        if self.cache_mode == "record":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"text": out.text, "tool_calls": out.tool_calls, "usage": usage, "model": out.model, "finish": out.raw_finish,
                                        "raw_message": raw_message}, ensure_ascii=False, default=str))
        return out

    async def _tito_complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, *, temperature: float, max_tokens: int,
                             seed: int | None) -> LLMResponse:
        """``LLMProvider.tito``: ``train.world_env.EngineLLM`` over the engine's token-id completions endpoint. Prompts
        extend the served conversation token for token; the record is exactly what the engine sampled."""
        if getattr(self, "_tito", None) is None:
            from transformers import AutoTokenizer

            from crucible.train.chat import HFChatCodec
            from crucible.train.engine import GenRequest
            from crucible.train.vllm_http import VLLMServerEngine
            from crucible.train.world_env import EngineLLM

            codec = HFChatCodec(AutoTokenizer.from_pretrained(self.model), template_kwargs=self.provider.template_kwargs)
            eng = VLLMServerEngine(self.provider.base_url.rstrip("/").removesuffix("/v1"), self.model, timeout_s=self.timeout_s,
                                   api_key=resolve_secret(self.provider.api_key) or None)
            calls = [0]

            async def gen(prompt: list[int], n: int, temp: float):
                calls[0] += 1
                return await eng.generate(GenRequest(rid="llm", call_id=f"c{calls[0]}", prompt_ids=prompt, max_tokens=n, temperature=temp,
                                                     stop_ids=codec.eos_ids, seed=seed))

            self._tito = EngineLLM(codec, gen, model=self.model)
        t0 = time.monotonic()
        try:
            r = await self._tito.complete(messages, tools=tools, temperature=temperature, max_tokens=max_tokens)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400 and "maximum context length" in e.response.text.lower():
                raise ContextOverflow(f"{self.model}: {e.response.text[:300]}") from e
            raise LLMError(f"{self.model} (tito) failed: {e}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"{self.model} (tito) failed: {e}") from e
        r.tokens["policy_version"] = self.policy_version  # the served checkpoint's stamp (provider config / trainer bridge)
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
            cut = " (cut off at max_tokens: shorten the output)" if r.raw_finish == "length" else ""
            raise LLMOutputError(f"output is not valid JSON{cut}: {text[:300]}") from None
