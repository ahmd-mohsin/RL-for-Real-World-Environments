"""The harness seam (research m9-m10/02 §3.2): one attempt of one sample, whatever runs the agent.

A harness gets an ``Attempt`` (sample key, attempt id, attempt directory, a budget lease) and returns a
``HarnessResult``: how the agent stopped, the grade (reward + gates, or ``None``), a classified failure if any, spend,
artifacts and timings. Harnesses never decide masking themselves beyond classifying a failure; the runner applies
``Failure.masks`` and the limit policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.core.schema.run import Failure
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


class Harness(Protocol):
    id: str  # name@version: part of the SampleKey (one harness per GRPO group)

    async def run(self, attempt: Attempt) -> HarnessResult: ...
