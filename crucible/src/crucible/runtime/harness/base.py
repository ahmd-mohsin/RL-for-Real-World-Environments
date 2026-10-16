"""The harness seam (research m9-m10/02 §3.2): one attempt of one sample, whatever runs the agent.

A harness gets an ``Attempt`` (sample key, attempt id, attempt directory, a budget lease) and returns a
``HarnessResult``: how the agent stopped, the grade (reward + gates, or ``None``), a classified failure if any, spend,
artifacts and timings. Harnesses never decide masking themselves beyond classifying a failure; the runner applies
``Failure.masks`` and the limit policy.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.core.schema.run import Failure
from crucible.core.schema.usage import UsageTotals
from crucible.runtime.keys import SampleKey


@dataclass
class Attempt:
    sample: SampleKey
    id: str
    n: int
    dir: Path
    task: Any  # a RunTask (pack-provided adapter)
    policy: Any  # a PolicyRef (pack/runner provided)
    ledger: Any = None
    scopes: list[str] = field(default_factory=list)


class HarnessResult(CrucibleModel):
    stop: str = "agent_done"
    truncated: bool = False
    reward: float | None = None
    gates: dict[str, int] = Field(default_factory=dict)
    failure: Failure | None = None
    steps: int = 0
    cost_usd: float = 0.0
    cost_uncertain_usd: float = 0.0
    tree_changed: bool | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    grade: dict[str, Any] = Field(default_factory=dict)
    calls: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)  # actor -> the policy's model calls (call ids, usage, token records)
    flags: list[dict[str, Any]] = Field(default_factory=list)  # level-1 hack suspicions (m13 C-4): copied to the rollout record
    format_errors: dict[str, int] = Field(default_factory=dict)  # model call id -> malformed tool calls it issued (m13 C-8)
    segments: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)  # actor -> context-compaction segments
    trajectory_tokens: dict[str, int] = Field(default_factory=dict)  # actor -> tokens that entered its context, before compaction
    usage: UsageTotals | None = None  # the attempt's own model calls summed, with coverage (R-19); None: none of its own known
    model_s: float | None = None  # their summed latency: the part of the agent stage spent waiting on the model (R-19)


def settle_usage(res: HarnessResult, calls: Mapping[str, Iterable[Mapping[str, Any]]], exclude: Collection[str] = ()) -> None:
    """The usage and model time of the attempt's own calls (R-19): never those restored from a checkpoint (``exclude``,
    paid by an earlier attempt) nor, from a re-grade or re-collect, those inherited from the attempt that ran the agent."""
    own = [c for cs in calls.values() for c in cs or [] if c.get("call_id") not in exclude]
    res.usage = UsageTotals.of(own)
    res.model_s = round(sum(float(c.get("latency_s") or 0.0) for c in own if not c.get("cached")), 3)


class Harness(Protocol):
    id: str  # name@version: part of the SampleKey (one harness per GRPO group)

    async def run(self, attempt: Attempt) -> HarnessResult: ...
