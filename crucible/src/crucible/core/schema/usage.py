"""Model-call accounting: token usage, finish reasons and stage timings (M0 audit 2026-10-14 R-19).

``Usage`` is one call's tokens, disjoint: ``input_tokens`` is the part of the prompt that was neither read from nor
written to the provider's prompt cache, so a prompt is ``input + cache_read + cache_write`` tokens, and
``reasoning_tokens`` are part of ``output_tokens``, never added again. Providers disagree on the shape: OpenAI chat's
``prompt_tokens`` includes cache reads (``prompt_tokens_details.cached_tokens``), Anthropic's ``input_tokens`` excludes
cache reads and writes, the Responses API's ``input_tokens`` includes reads (``input_tokens_details.cached_tokens``).
Every one of them validates to the same numbers. A dumped ``Usage`` also carries the totals ``prompt_tokens`` and
``completion_tokens``, so readers of a call record's usage dict keep their meaning. PATTERN: verifiers ``v1/types.py``
``Usage.from_openai`` / ``aggregate`` (MIT; no code copied), with Anthropic and Responses shapes, cache writes and a
source added.

``UsageTotals`` sums an attempt's calls with their coverage: how many reported usage, how many were estimated or had
none (their tokens are unknown, not 0), and how many were replayed from the record/replay cache (counted apart, never as
live tokens). ``FinishReason`` is why a call ended, one vocabulary over the providers' and engines' words
(``finish_reason``). ``Timing`` is an attempt's wall time per stage.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, get_args

from pydantic import Field, computed_field, field_validator, model_validator

from crucible.core.schema.base import CrucibleModel

UsageSource = Literal["provider", "engine", "estimate", "none"]  # none: the call happened but reported no usage
REPORTED: frozenset[str] = frozenset({"provider", "engine"})

FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "refusal", "abort", "error"]
FINISH_REASONS: frozenset[str] = frozenset(get_args(FinishReason))
_FINISH_ALIASES = {
    "eos": "stop", "eos_token": "stop", "stop_sequence": "stop", "function_call": "tool_calls",  # OpenAI chat, engines
    "end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls", "model_context_window_exceeded": "length",
    "pause_turn": "abort",  # Anthropic stop_reason (pause_turn: the server paused a long turn, to be resumed)
    "completed": "stop", "max_output_tokens": "length", "failed": "error", "cancelled": "abort",  # Responses status / reason
    "safety": "content_filter", "recitation": "content_filter", "prohibited_content": "content_filter",  # Gemini
}


def finish_reason(raw: Any) -> FinishReason | None:
    """A provider's or engine's finish word as a ``FinishReason``; None when absent or not known (Responses' bare
    ``incomplete`` says neither length nor filter: pass its ``incomplete_details.reason`` instead)."""
    if not isinstance(raw, str):
        return None
    word = raw.strip().casefold()
    out = word if word in FINISH_REASONS else _FINISH_ALIASES.get(word)
    return out  # type: ignore[return-value]


def finish_fields(raw: Any) -> dict[str, Any]:
    """A call record's finish: the normalized reason, and the provider's own word when it differs."""
    fin = finish_reason(raw)
    return {"finish": fin, **({"finish_raw": raw} if raw is not None and raw != fin else {})}


def _int(v: Any) -> int:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else 0


def _details(d: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    v = d.get(key)
    return v if isinstance(v, Mapping) else {}


def _read_usage(d: Mapping[str, Any]) -> dict[str, Any]:
    """Any provider's usage dict (or a dumped ``Usage``, or Crucible's pre-R-19 call usage) as ``Usage`` fields."""
    if "cache_read_tokens" in d or "cache_write_tokens" in d:  # a dumped Usage: its own keys, the derived totals dropped
        return {k: v for k, v in d.items() if k not in ("prompt_tokens", "completion_tokens")}
    pd, cd = _details(d, "prompt_tokens_details"), _details(d, "completion_tokens_details")
    idt, odt = _details(d, "input_tokens_details"), _details(d, "output_tokens_details")
    read = (_int(d.get("cache_read_input_tokens")) or _int(pd.get("cached_tokens")) or _int(idt.get("cached_tokens"))
            or _int(d.get("cached_tokens")) or _int(d.get("prompt_cache_hit_tokens")))  # Kimi and Crucible pre-R-19; DeepSeek
    write = _int(d.get("cache_creation_input_tokens")) or _int(pd.get("cache_write_tokens"))
    if "prompt_tokens" in d:  # OpenAI chat (and OpenRouter, vLLM, DeepSeek): the prompt includes its cached part
        inp = max(0, _int(d["prompt_tokens"]) - read - write)
    elif "input_tokens_details" in d or "total_tokens" in d:  # Responses: input_tokens includes cache reads
        inp = max(0, _int(d.get("input_tokens")) - read)
    else:  # Anthropic: input_tokens is the uncached part already
        inp = _int(d.get("input_tokens"))
    out = _int(d["completion_tokens"]) if "completion_tokens" in d else _int(d.get("output_tokens"))
    reasoning = d.get("reasoning_tokens", cd.get("reasoning_tokens", odt.get("reasoning_tokens")))
    has_count = any(k in d for k in ("prompt_tokens", "input_tokens", "completion_tokens", "output_tokens"))
    source = d.get("source") or ("provider" if has_count else "none")
    return {"input_tokens": inp, "cache_read_tokens": read, "cache_write_tokens": write, "output_tokens": out,
            "reasoning_tokens": _int(reasoning) if reasoning is not None else None, "source": source}


class Usage(CrucibleModel):
    """One model call's tokens (disjoint; see the module doc). Validates from any provider's usage dict."""

    input_tokens: int = Field(0, ge=0)
    cache_read_tokens: int = Field(0, ge=0)
    cache_write_tokens: int = Field(0, ge=0)
    output_tokens: int = Field(0, ge=0)
    reasoning_tokens: int | None = Field(None, ge=0, exclude_if=lambda v: v is None)  # part of output_tokens
    source: UsageSource = "provider"

    @model_validator(mode="before")
    @classmethod
    def _shapes(cls, data: Any) -> Any:
        return _read_usage(data) if isinstance(data, Mapping) else data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def prompt_tokens(self) -> int:
        """The whole prompt, cached or not (OpenAI's ``prompt_tokens``)."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @computed_field  # type: ignore[prop-decorator]
    @property
    def completion_tokens(self) -> int:
        return self.output_tokens


def _call_usage(c: Mapping[str, Any]) -> Usage:
    u = c.get("usage")
    return Usage.model_validate(u) if isinstance(u, Mapping) and u else Usage(source="none")


class UsageTotals(CrucibleModel):
    """An attempt's model calls summed: the tokens of the calls made live, and the coverage behind them."""

    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int | None = Field(None, exclude_if=lambda v: v is None)  # None: no call reported it
    calls: int = 0  # live calls (not replayed)
    reported: int = 0  # live calls whose provider or engine reported usage
    estimated: int = 0  # live calls whose usage is an estimate (chars/4)
    blank: int = 0  # live calls with no usage at all: their tokens are unknown, not 0
    replayed: int = 0  # calls answered from the record/replay cache: never in the totals above
    replayed_tokens: int = 0  # their prompt + completion tokens

    @classmethod
    def of(cls, calls: Iterable[Mapping[str, Any]] | Mapping[str, Iterable[Mapping[str, Any]]]) -> UsageTotals:
        """From call records (``{"usage": ..., "cached": ...}``, as policies keep them and ``calls.json`` stores them), or
        a mapping actor -> call records. ``cached`` marks a record/replay cache hit."""
        rows = [c for cs in calls.values() for c in cs or []] if isinstance(calls, Mapping) else list(calls)
        t = cls()
        reasoning: list[int] = []
        for c in rows:
            if not isinstance(c, Mapping):
                continue
            u = _call_usage(c)
            if c.get("cached"):
                t.replayed += 1
                t.replayed_tokens += u.prompt_tokens + u.completion_tokens
                continue
            t.calls += 1
            t.reported += u.source in REPORTED
            t.estimated += u.source == "estimate"
            t.blank += u.source == "none"
            t.input_tokens += u.input_tokens
            t.cache_read_tokens += u.cache_read_tokens
            t.cache_write_tokens += u.cache_write_tokens
            t.output_tokens += u.output_tokens
            if u.reasoning_tokens is not None:
                reasoning.append(u.reasoning_tokens)
        t.reasoning_tokens = sum(reasoning) if reasoning else None
        return t

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def completion_tokens(self) -> int:
        return self.output_tokens

    @property
    def coverage(self) -> float | None:
        """The share of live calls whose tokens were reported; None without live calls."""
        return self.reported / self.calls if self.calls else None


STAGES = ("seed", "agent", "collect", "verification", "cleanup")


class Timing(CrucibleModel):
    """An attempt's wall time per stage, in seconds; None = the stage did not run (PATTERN: verifiers ``v1/trace.py``
    ``Timing``, MIT). The stages run one after another, so ``wall_s`` is their sum. ``model`` is the part of ``agent``
    spent waiting on model calls (the sum of their latencies: concurrent calls can make it exceed ``agent``); the rest of
    ``agent`` is the World's and the tools'. A stage this model does not name is kept in ``extra``."""

    seed: float | None = Field(None, exclude_if=lambda v: v is None)
    agent: float | None = Field(None, exclude_if=lambda v: v is None)
    model: float | None = Field(None, exclude_if=lambda v: v is None)
    collect: float | None = Field(None, exclude_if=lambda v: v is None)
    verification: float | None = Field(None, exclude_if=lambda v: v is None)
    cleanup: float | None = Field(None, exclude_if=lambda v: v is None)
    extra: dict[str, float] = Field(default_factory=dict, exclude_if=lambda v: not v)

    @field_validator("extra")
    @classmethod
    def _not_a_stage(cls, v: dict[str, float]) -> dict[str, float]:
        if clash := sorted(set(v) & {*STAGES, "model"}):
            raise ValueError(f"stages {clash} have their own fields")
        return v

    @classmethod
    def from_stages(cls, stages: Mapping[str, Any], *, model: float | None = None) -> Timing:
        """From a harness's ``{stage: seconds}`` (``HarnessResult.timings``, a pre-R-19 record's ``meta["timings"]``)."""
        num = {k: float(v) for k, v in stages.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        model = num.pop("model", None) if model is None else model
        return cls(**{k: num[k] for k in STAGES if k in num}, model=model, extra={k: v for k, v in num.items() if k not in STAGES and k != "model"})

    @property
    def wall_s(self) -> float:
        return sum(v for k in STAGES if (v := getattr(self, k)) is not None) + sum(self.extra.values())

    def stages(self) -> dict[str, float]:
        """The stages as a harness reports them (``model`` is inside ``agent``, so it is not one)."""
        return {**{k: v for k in STAGES if (v := getattr(self, k)) is not None}, **self.extra}
