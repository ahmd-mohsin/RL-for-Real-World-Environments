"""The harness seam (research m9-m10/02 §3.2): one attempt of one sample, whatever runs the agent.

A harness gets an ``Attempt`` (sample key, attempt id, attempt directory, a budget lease) and returns a
``HarnessResult``: how the agent stopped, the grade (reward + gates, or ``None``), a classified failure if any, spend,
artifacts and timings. Harnesses never decide masking themselves beyond classifying a failure; the runner applies
``Failure.masks`` and the limit policy. A harness that persisted the attempt's work for a re-grade says so
(``regradable``, M10 lab audit R-3): only then is a failure after the agent phase retried, and only as a re-grade. It
also keeps ``Attempt.phase`` / ``Attempt.regradable`` current, so an attempt timeout after the agent phase is classified
the same way, and opens the grading's window on the Runner's clock (``Attempt.clock``, ``grading.grading_window``).
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
    # M10 lab audit R-3: what the harness updates while it runs, read by the Runner when its attempt clock cancels the
    # harness (its HarnessResult is lost then): the phase it is in (``agent``, ``collect``, ``verification``; a harness that
    # never sets it stays ``agent``) and whether the work is persisted for a re-grade (mirrors ``HarnessResult.regradable``).
    # A timeout after the agent phase is then a grade failure, retried only as a re-grade, never by sampling the policy again
    phase: str = "agent"
    regradable: bool = False
    # M10 lab audit R-3 review round 2: the Runner's attempt clock (its ``asyncio.Timeout``, None without
    # ``attempt_timeout_s``) and its length. A harness entering the verification phase opens the grading's own window on it
    # (``grading.grading_window``: the scoring budget defaults to the clock's length, and the clock is moved past that
    # budget and the control's), so a hang ends as ``verifier_timeout`` and is decided by the control, never cut by a clock
    # sized for the agent; ``agent_left`` is what the clock had left for the agent then (``grading.agent_clock``)
    clock: Any = None
    timeout_s: float | None = None
    agent_left: float | None = None


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
    # M10 lab audit R-3: this attempt's work is persisted for a grade-only retry (a stored ``artifacts/model.patch``, a
    # quiesced hand-off in ``verifier/handoff_ref.json``) and the next attempt re-grades it instead of running the agent;
    # ``RetryRule`` refuses a verification-stage retry without it (the policy is never sampled again because a grade failed)
    regradable: bool = False


class Harness(Protocol):
    id: str  # name@version: part of the SampleKey (one harness per GRPO group)

    async def run(self, attempt: Attempt) -> HarnessResult: ...
