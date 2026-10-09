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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.core.config import Config, get_config, resolve_secret
from crucible.core.ids import sha256_text


class BudgetExceeded(RuntimeError):
    failure_kind = "crucible:credits_exhausted"


class LLMError(RuntimeError):
    failure_kind = "provider_unavailable"


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # [{id, name, arguments(dict)}]
    usage: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    model: str = ""
    cached: bool = False
    raw_finish: str | None = None


@dataclass
class Spend:
    usd: float = 0.0
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _retry_delay(e: Exception, attempt: int, cap: float = 60.0) -> float:
    """Provider's Retry-After (seconds) when given, else capped exponential backoff (Karotte's agent loop)."""
    headers = getattr(getattr(e, "response", None), "headers", None) or {}
    try:
        return min(float(headers.get("retry-after")), cap)
    except (TypeError, ValueError):
        return min(1.5 * 2**attempt, cap)


class LLMClient:
    def __init__(self, model: str = "sim-default", *, config: Config | None = None, budget_usd: float | None = None,
                 cache_dir: Path | None = None, cache_mode: str | None = None, timeout_s: float = 120, max_retries: int = 3):
        self.config = config or get_config()
        self.provider, self.model = self.config.model(model)
        self.budget_usd = self.config.llm_budget_usd if budget_usd is None else budget_usd
        self.cache_mode = cache_mode or os.environ.get("CRUCIBLE_LLM_CACHE", "record")
        self.cache_dir = cache_dir or (self.config.data_path() / "llm_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.max_retries = max_retries
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
                       max_tokens: int = 800, json_mode: bool = False, seed: int | None = 0, **extra: Any) -> LLMResponse:
        params: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens, "json": json_mode, "seed": seed, **extra}
        key = self._key(messages, tools, params)
        path = self.cache_dir / key[:2] / f"{key}.json"
        if self.cache_mode in ("record", "replay") and path.exists():
            d = json.loads(path.read_text())
            self.spend.cached_calls += 1
            return LLMResponse(text=d["text"], tool_calls=d.get("tool_calls", []), usage=d.get("usage", {}), cost_usd=0.0, model=d.get("model", self.model), cached=True,
                               raw_finish=d.get("finish"))
        if self.cache_mode == "replay":
            raise LLMError(f"cache miss in replay mode for {self.model} (key {key[:12]})")
        if self.spend.usd >= self.budget_usd:
            raise BudgetExceeded(f"LLM budget {self.budget_usd:.4f} USD exhausted ({self.spend.usd:.4f} spent)")
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if seed is not None:
            kwargs["seed"] = seed
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        kwargs.update(extra)
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = await self._openai().chat.completions.create(**kwargs)
                break
            except Exception as e:  # noqa: BLE001 - classify below
                last = e
                msg = str(e).lower()
                if "402" in msg or "credit" in msg or "insufficient" in msg or "budget" in msg:
                    raise BudgetExceeded(f"provider refused for credits: {e}") from e
                status = getattr(e, "status_code", None)
                if status in (400, 401, 403, 404, 422):  # the request is wrong, retrying cannot help
                    raise LLMError(f"{self.model} rejected the request ({status}): {e}") from e
                await asyncio.sleep(_retry_delay(e, attempt))
        else:
            raise LLMError(f"{self.model} failed after {self.max_retries} attempts: {last}")
        choice = resp.choices[0]
        msg = choice.message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"__unparseable__": tc.function.arguments}
            calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
        u = resp.usage
        usage = {"prompt_tokens": getattr(u, "prompt_tokens", 0) or 0, "completion_tokens": getattr(u, "completion_tokens", 0) or 0} if u else {}
        cost = getattr(u, "cost", None) if u else None
        if cost is None and u is not None:
            extra_u = getattr(u, "model_extra", None) or {}
            cost = extra_u.get("cost")
        cost = float(cost) if cost is not None else self._price(usage)
        self.spend.usd += cost
        self.spend.calls += 1
        self.spend.prompt_tokens += usage.get("prompt_tokens", 0)
        self.spend.completion_tokens += usage.get("completion_tokens", 0)
        out = LLMResponse(text=msg.content or "", tool_calls=calls, usage=usage, cost_usd=cost, model=resp.model or self.model, raw_finish=choice.finish_reason)
        if self.cache_mode == "record":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"text": out.text, "tool_calls": out.tool_calls, "usage": usage, "model": out.model, "finish": out.raw_finish}, ensure_ascii=False))
        return out

    async def complete_json(self, messages: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        kw.setdefault("json_mode", True)
        r = await self.complete(messages, **kw)
        text = r.text.strip()
        if not text:
            raise LLMError(f"empty completion (finish={r.raw_finish})")
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
            raise LLMError(f"output is not JSON: {text[:300]}") from None
