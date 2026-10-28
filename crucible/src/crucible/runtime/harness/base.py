"""The harness seam (research m9-m10/02 §3.2): one attempt of one sample, whatever runs the agent.

A harness gets an ``Attempt`` (sample key, attempt id, attempt directory, a budget lease) and returns a
``HarnessResult``: how the agent stopped, the grade (reward + gates, or ``None``), a classified failure if any, spend,
artifacts and timings. Harnesses never decide masking themselves beyond classifying a failure; the runner applies
``Failure.masks`` and the limit policy. A harness that persisted the attempt's work for a re-grade says so
(``regradable``, M10 lab audit R-3): only then is a failure after the agent phase retried, and only as a re-grade. It
also keeps ``Attempt.phase`` / ``Attempt.regradable`` current, so an attempt timeout after the agent phase is classified
the same way, and opens the grading's window on the Runner's clock (``Attempt.clock``, ``grading.grading_window``).

M10 lab audit R-4: a harness's ``id`` names it in the ``SampleKey`` (``name@version#<config digest>``: the in-tree harnesses
digest what produces the work), and its ``identity_config()`` is the document the run's identity compares on a resume
(``runtime.identity``); a harness that holds state and declares none is refused before the run starts.
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
    # async-RL audit R3 (2026-10-28): the run this attempt belongs to, and the policies the harness made for it (``{actor:
    # Policy}``, set by the harness once ``policy.make`` returns), so ``Runner.inflight_min_version`` can read the versions of
    # the calls made so far while the attempt runs
    run_id: str | None = None
    policies: dict[str, Any] = field(default_factory=dict)


class HarnessResult(CrucibleModel):
    stop: str = "agent_done"
    truncated: bool = False
    reward: float | None = None
    gates: dict[str, int] = Field(default_factory=dict)
    gate_kinds: dict[str, str] = Field(default_factory=dict)  # gate -> "gate" (hard rule) | outcome / score kind (M6 R2)
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
    # M10 lab audit R-3: this attempt's work is persisted for a grade-only retry (a stored ``artifacts/model.patch``, a
    # quiesced hand-off in ``verifier/handoff_ref.json``) and the next attempt re-grades it instead of running the agent;
    # ``RetryRule`` refuses a verification-stage retry without it (the policy is never sampled again because a grade failed)
    regradable: bool = False


def settle_usage(res: HarnessResult, calls: Mapping[str, Iterable[Mapping[str, Any]]], exclude: Collection[str] = ()) -> None:
    """The usage and model time of the attempt's own calls (R-19): never those restored from a checkpoint (``exclude``,
    paid by an earlier attempt) nor, from a re-grade or re-collect, those inherited from the attempt that ran the agent."""
    own = [c for cs in calls.values() for c in cs or [] if c.get("call_id") not in exclude]
    res.usage = UsageTotals.of(own)
    res.model_s = round(sum(float(c.get("latency_s") or 0.0) for c in own if not c.get("cached")), 3)


class Harness(Protocol):
    @property
    def id(self) -> str: ...  # name@version#config digest: part of the SampleKey (one harness per GRPO group); read-only, so a property satisfies it

    # M10 lab audit R-4: the harness's configuration (and the content digests of what it grades with) as JSON. Required unless the
    # harness is stateless plain data: ``identity.describe`` refuses an object it cannot inspect, so a knob nobody declared is never
    # left out of the run's identity. A harness that is not a NativeHarness also puts a digest of its configuration in its ``id``
    def identity_config(self) -> dict[str, Any]: ...

    async def run(self, attempt: Attempt) -> HarnessResult: ...
