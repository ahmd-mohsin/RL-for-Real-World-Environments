"""Deterministic fault injectors for the M10 exit gate (research m9-m10/02 §6.2) and for CI.

Each injector produces an attempt that *would* be scored 0 if mishandled. Infrastructure classes must come out masked
(``reward=None``), excluded from pass@k, and retried; the policy controls must come out **scored 0**, never masked
(otherwise a classifier that masks everything would pass).

- ``FakeLLM``: a scripted tool-calling client; raises a chosen provider failure (blank usage, credits, overflow) at a chosen
  call on chosen attempts.
- ``FlakyGradeTask``: wraps a task so its grade comes back masked with a chosen infra reason (contaminated perf rounds,
  stale build, stale image, grader exception) on chosen attempts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from crucible.actors.llm import BudgetExceeded, ContextOverflow, LLMResponse, UsageBlank
from crucible.verify.runtime.plan import Check, Context, Gate, Plan
from crucible.verify.runtime.trusted import InfraError, Trusted


class FakeSpend:
    def __init__(self):
        self.usd = 0.0
        self.uncertain_usd = 0.0
        self.calls = 0


class FakeLLM:
    """Replies with ``script`` (LLMResponse objects or text) in order; ``fail_at`` = (call index, exception) raises there."""

    def __init__(self, script: list[Any], *, fail_at: tuple[int, Exception] | None = None, cost_per_call: float = 0.001, model: str = "fake"):
        self.script, self.fail_at, self.cost, self.model = list(script), fail_at, cost_per_call, model
        self.spend = FakeSpend()

    async def complete(self, messages, **kw):
        i = self.spend.calls
        self.spend.calls += 1
        if self.fail_at and i == self.fail_at[0]:
            raise self.fail_at[1]
        self.spend.usd += self.cost
        r = self.script.pop(0) if self.script else LLMResponse(text="done")
        return r if isinstance(r, LLMResponse) else LLMResponse(text=r)


def tool_reply(name: str, args: dict, i: int = 0) -> LLMResponse:
    return LLMResponse(text="", tool_calls=[{"id": f"c{i}", "name": name, "arguments": args}])


PROVIDER_FAILURES = {"usage_blank": UsageBlank("provider returned no usage"), "credits": BudgetExceeded("402 insufficient credits"),
                     "context_overflow": ContextOverflow("prompt exceeds the context window")}


class FlakyGradeTask:
    """Delegates to ``task``; on attempts whose grade dir contains one of ``on_attempts`` the grade is masked with
    ``reason`` (raised as InfraError inside a plan check, the way the M7 kit reports infra) or crashes in setup."""

    def __init__(self, task, reason: str, *, on_attempts: tuple[str, ...] = ("a1",), crash_setup: bool = False):
        self.task, self.reason, self.on, self.crash_setup = task, reason, on_attempts, crash_setup
        self.id, self.world_id = task.id, getattr(task, "world_id", "")

    def __getattr__(self, k):
        return getattr(self.task, k)

    def _hit(self, d: Path) -> bool:
        return any(f"/{a}/" in str(d) for a in self.on)

    async def _flaky(self, d: Path):
        def check(_):
            raise InfraError(self.reason)

        def setup(_):
            raise RuntimeError(self.reason)

        plan = Plan(self.id, [Gate("g")], [Check("g", "g", lambda c: True if self.crash_setup else check(c))], setup=setup if self.crash_setup else None)
        return await plan.arun(Context(Trusted(Path(d))))

    async def grade(self, world, d: Path):
        return await self._flaky(d) if self._hit(Path(d)) else await self.task.grade(world, d)

    async def grade_patch(self, patch: str, d: Path):
        return await self._flaky(d) if self._hit(Path(d)) else await self.task.grade_patch(patch, d)
