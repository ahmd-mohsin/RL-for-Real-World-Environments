"""Deterministic fault injectors for the M10 exit gate (research m9-m10/02 §6.2) and for CI.

Each injector produces an attempt that *would* be scored 0 if mishandled. Infrastructure classes must come out masked
(``reward=None``), excluded from pass@k, and retried; the policy controls must come out **scored 0**, never masked
(otherwise a classifier that masks everything would pass).

- ``FakeLLM``: a scripted tool-calling client; raises a chosen provider failure (blank usage, credits, overflow) at a chosen
  call on chosen attempts.
- ``FlakyGradeTask``: wraps a task so its grade comes back masked with a chosen infra reason (contaminated perf rounds,
  stale build, stale image, grader exception) on chosen attempts; its control grade (M10 lab audit R-3) is injected the
  same way.
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
        r = r if isinstance(r, LLMResponse) else LLMResponse(text=r)
        return r.model_copy(update={"cost_usd": self.cost}) if hasattr(r, "model_copy") else _with_cost(r, self.cost)


def _with_cost(r: LLMResponse, cost: float) -> LLMResponse:
    import dataclasses

    return dataclasses.replace(r, cost_usd=cost)


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

    async def grade_control(self, d: Path):
        """M10 lab audit R-3: the inner task's control, on the same, injected grader (a control graded under ``/a1/``
        masks like the attempt's own grades, so an injected outage fails its control: a task-blamed mask, never the
        policy's 0). None when the inner task has no control."""
        hook = getattr(self.task, "grade_control", None)
        if hook is None:
            return None
        return await self._flaky(d) if self._hit(Path(d)) else await hook(d)


class ReplayLLM:
    """A *stateless* tool-calling LLM that replays an action list (dicts with ``function_name``/``arguments`` or ``to``/``text``)
    through ``LLMToolPolicy``: the next action is found from the last assistant tool-call id still in the context
    (``rp<k>``), so it survives compaction (kept recent turns) and resume (restored context) exactly like a real model
    conditioned on its context would. A compaction request is answered with a summary; ``crash_at=(call, exc)`` raises.
    ``usage`` mirrors the context size, so accounting and budgets behave as with a real endpoint."""

    def __init__(self, actions: list[dict], *, crash_at: tuple[int, Exception] | None = None, model: str = "replay"):
        from crucible.core.schema.actions import openai_name

        self.actions, self.crash_at, self.model = actions, crash_at, model
        self.name = lambda q: openai_name(q)
        self.spend = FakeSpend()

    def _pos(self, messages: list[dict]) -> int:
        for m in reversed(messages):
            for tc in reversed(m.get("tool_calls") or []):
                if str(tc.get("id", "")).startswith("rp"):
                    return int(tc["id"][2:]) + 1
        return 0

    async def complete(self, messages, **kw):
        from crucible.actors.policy import COMPACT, approx_tokens

        i = self.spend.calls
        self.spend.calls += 1
        if self.crash_at and i == self.crash_at[0]:
            raise self.crash_at[1]
        usage = {"prompt_tokens": approx_tokens(messages), "completion_tokens": 40}
        if messages and messages[-1].get("content") == COMPACT:
            k = self._pos(messages)
            return LLMResponse(text=f"Summary: actions 0..{k - 1} of the plan are done; continue from action {k}.", usage=usage, model=self.model)
        k = self._pos(messages)
        if k >= len(self.actions):
            return LLMResponse(text="DONE", usage=usage, model=self.model)
        a = self.actions[k]
        if a.get("function_name"):
            tc = {"id": f"rp{k}", "name": self.name(a["function_name"]), "arguments": a.get("arguments") or {}}
        else:
            tc = {"id": f"rp{k}", "name": "send_message", "arguments": {"to": a["to"], "text": a.get("text", "")}}
        return LLMResponse(text="", tool_calls=[tc], usage=usage, model=self.model)
